import json
import logging
import os
import re
import sqlite3
import random
from typing import List, Optional

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("leadclaw")

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(title="LeadClaw API", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Environment variables (set all of these in Render) ────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")   # gemini-1.5-flash is the public name for 3.5 Flash
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL     = os.environ.get("GROQ_MODEL", "qwen-qwq-32b")         # Groq's hosted Qwen3 model

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ── SQLite database ────────────────────────────────────────────────────────
DB_FILE = "leadclaw.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users (
        email         TEXT PRIMARY KEY,
        workspace     TEXT,
        plan          TEXT DEFAULT 'solo',
        trials_left   INTEGER DEFAULT 3,
        gmail_token   TEXT,
        device_token  TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS verifications (
        email TEXT PRIMARY KEY,
        code  TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS disputes (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        email       TEXT,
        platform    TEXT,
        claim       TEXT,
        amount      REAL,
        valid       INTEGER,
        recovered   INTEGER DEFAULT 0,
        created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS feedback_entries (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        email      TEXT,
        workspace  TEXT,
        type       TEXT,
        message    TEXT,
        priority   TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.commit()
    conn.close()

init_db()

# ── Pydantic models ────────────────────────────────────────────────────────
class EmailRequest(BaseModel):
    email: str

class CodeVerifyRequest(BaseModel):
    email: str
    code: str

class WorkspaceSetupRequest(BaseModel):
    email: str
    workspace: str
    plan: str
    device_token: str

class FeedbackRequest(BaseModel):
    email: Optional[str] = ""
    workspace: Optional[str] = ""
    type: Optional[str] = "Feedback"
    message: str
    priority: Optional[str] = "medium"

class GmailTokenRequest(BaseModel):
    email: str
    gmail_token: str

class RecoverRequest(BaseModel):
    dispute_id: int
    email: str

# ── Policy constants ───────────────────────────────────────────────────────
PLATFORM_POLICY = {
    "Angi":      "Angi Lead Credit Policy",
    "Thumbtack": "Thumbtack Lead Price Credit Terms",
    "Porch":     "Porch Pro Lead Guarantee",
    "Networx":   "Networx Lead Dispute Terms",
}
PLATFORM_SUPPORT_EMAIL = {
    "Angi":      "leadcredits@angi.com",
    "Thumbtack": "support@thumbtack.com",
    "Porch":     "prosupport@porch.com",
    "Networx":   "support@networx.com",
}
NOT_ELIGIBLE_REASON = (
    "Pricing objections, competitor hiring, project postponements, and customer "
    "choices are non-refundable under marketplace policy."
)
NON_REFUNDABLE_RE = re.compile(
    r"competitor|hired|another contractor|chose another|price|pricing|expensive|quote|"
    r"changed\s+(?:\w+\s+)?mind|not interested|on hold|postponed|delay|cancelled|"
    r"budget|no answer|unresponsive|voicemail",
    re.I,
)
INVALID_SAFETY_RE = re.compile(
    r"hired (?:another|a|an|someone|a different)|"
    r"went (?:with|to) (?:a |another |the )?(?:competitor|contractor)|"
    r"project went to|chose (?:another|a different) (?:contractor|company)|"
    r"competitor|too expensive|over (?:our|my) budget|on hold|postponed",
    re.I,
)
SYSTEM_PROMPT = """You are an expert compliance auditor for home service lead marketplaces
(Angi, Thumbtack, Porch, Networx). Decide whether a lead qualifies for a credit refund.

ELIGIBLE (valid = true) — only when the LEAD ITSELF was defective:
- Permanently disconnected, wrong, or invalid phone number
- Customer is strictly outside the contractor's service area
- Duplicate lead in the same billing window
- Spam, bot, or fake submission
- Wrong trade category

NOT ELIGIBLE (valid = false) — whenever the outcome was a consumer or commercial choice:
- Homeowner hired a competitor or another contractor
- Homeowner rejected the price or quote, or said it was too expensive
- Customer changed their mind, postponed, cancelled, or put the project on hold
- Homeowner was unresponsive after a real initial contact

HARD RULE: If the notes or transcript mention a competitor being hired or a price/quote
rejection, valid MUST be false — even if other eligible reasons are present.

If valid is true: write a detailed, persuasive, professional dispute letter tailored to the
exact facts provided. Start with a Subject line, formal greeting, clear issue breakdown
(specific numbers, dates, error messages), policy citation, and a firm credit request.
Never use placeholder text like "See attached evidence."

If valid is false: one or two plain sentences for "reason" and "claim" must be empty string.

Respond ONLY in valid JSON with no markdown fences:
{"valid": true or false, "confidence": 0.95, "reason": "...", "claim": "..."}"""


# ── AI helpers ─────────────────────────────────────────────────────────────
def extract_json(text: str) -> dict:
    """Pull the first {...} block out of a string."""
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1:
        raise ValueError(f"No JSON found in: {text[:300]}")
    return json.loads(text[s:e + 1])


async def call_gemini(prompt_text: str, file_parts: list = None) -> dict:
    """Call Gemini (1.5 Flash / gemini-3.5-flash equivalent)."""
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT,
        generation_config={"temperature": 0.1, "response_mime_type": "application/json"},
    )
    contents = [prompt_text]
    if file_parts:
        contents.extend(file_parts)
    response = model.generate_content(contents)
    return extract_json(response.text or "")


async def call_groq(prompt_text: str) -> dict:
    """Call Groq API with Qwen3 4B (qwen-qwq-32b on Groq's hosted inference)."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set. Add it in Render environment variables.")
    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt_text},
                ],
                "temperature": 0.1,
                "max_tokens": 2048,
                "response_format": {"type": "json_object"},
            },
        )
        resp.raise_for_status()
        data = resp.json()
    content = data["choices"][0]["message"]["content"]
    return extract_json(content)


def not_eligible(cost: float, confidence: float = 0.88, reason: str = NOT_ELIGIBLE_REASON) -> dict:
    return {
        "valid": False, "confidence": confidence,
        "reason": reason, "claim": "", "amount": cost,
    }


# ── DB helpers ─────────────────────────────────────────────────────────────
def get_user(email: str = None, device_token: str = None) -> Optional[dict]:
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    if email:
        c.execute("SELECT email,workspace,plan,trials_left,gmail_token FROM users WHERE email=?", (email,))
    elif device_token:
        c.execute("SELECT email,workspace,plan,trials_left,gmail_token FROM users WHERE device_token=?", (device_token,))
    else:
        conn.close(); return None
    row = c.fetchone(); conn.close()
    if not row: return None
    return {"email": row[0], "workspace": row[1], "plan": row[2], "trials_left": row[3], "gmail_token": row[4]}


def deduct_trial(email: str, device_token: str):
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    c.execute(
        "UPDATE users SET trials_left = MAX(0, trials_left - 1) WHERE email=? OR device_token=?",
        (email, device_token),
    )
    conn.commit()
    c.execute("SELECT trials_left FROM users WHERE email=? OR device_token=?", (email, device_token))
    row = c.fetchone(); conn.close()
    return row[0] if row else 0


def plan_tier(plan: str) -> int:
    return {"solo": 1, "growth": 2, "crew": 3}.get(plan, 1)


# ── Routes: health ─────────────────────────────────────────────────────────
@app.get("/")
def health():
    return {
        "ok": True, "service": "LeadClaw API", "version": "4.0.0",
        "gemini_model": GEMINI_MODEL, "groq_model": GROQ_MODEL,
        "gemini_configured": bool(GEMINI_API_KEY),
        "groq_configured":   bool(GROQ_API_KEY),
    }


@app.get("/api/ai-check")
async def ai_check():
    """Debug route — remove or protect before production."""
    results = {}
    if GEMINI_API_KEY:
        try:
            m = genai.GenerativeModel(GEMINI_MODEL)
            r = m.generate_content("Reply with the single word: OK")
            results["gemini"] = {"ok": True, "reply": (r.text or "").strip()}
        except Exception as e:
            results["gemini"] = {"ok": False, "error": str(e)}
    else:
        results["gemini"] = {"ok": False, "error": "GEMINI_API_KEY not set"}
    if GROQ_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    json={"model": GROQ_MODEL, "messages": [{"role":"user","content":"Reply: OK"}], "max_tokens": 5},
                )
                results["groq"] = {"ok": True, "reply": resp.json()["choices"][0]["message"]["content"].strip()}
        except Exception as e:
            results["groq"] = {"ok": False, "error": str(e)}
    else:
        results["groq"] = {"ok": False, "error": "GROQ_API_KEY not set"}
    return results


# ── Routes: authentication ─────────────────────────────────────────────────
@app.post("/api/auth/send-code")
async def send_code(req: EmailRequest):
    code = f"{random.randint(100000, 999999)}"
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO verifications (email, code) VALUES (?,?)", (req.email, code))
    conn.commit(); conn.close()
    logger.info("Verification code for %s: %s", req.email, code)
    # TODO: replace logger.info with a real email send (SendGrid / Resend / SES)
    # Example with Resend:
    # async with httpx.AsyncClient() as client:
    #     await client.post("https://api.resend.com/emails",
    #         headers={"Authorization": f"Bearer {RESEND_API_KEY}"},
    #         json={"from":"noreply@leadclaw.com","to":req.email,
    #               "subject":"Your LeadClaw code","text":f"Your code: {code}"})
    return {"status": "success", "message": "Verification code sent.", "debug_code": code}


@app.post("/api/auth/verify-code")
async def verify_code(req: CodeVerifyRequest):
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    c.execute("SELECT code FROM verifications WHERE email=?", (req.email,))
    row = c.fetchone(); conn.close()
    if not row or row[0] != req.code:
        raise HTTPException(status_code=400, detail="Invalid or expired verification code.")
    return {"status": "success", "message": "Email verified."}


@app.post("/api/auth/setup-workspace")
async def setup_workspace(req: WorkspaceSetupRequest):
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    c.execute(
        """INSERT INTO users (email, workspace, plan, trials_left, device_token)
           VALUES (?,?,?, COALESCE((SELECT trials_left FROM users WHERE email=?), 3), ?)
           ON CONFLICT(email) DO UPDATE SET
             workspace=excluded.workspace, plan=excluded.plan, device_token=excluded.device_token""",
        (req.email, req.workspace, req.plan, req.email, req.device_token),
    )
    conn.commit(); conn.close()
    return {"status": "success", "workspace": req.workspace, "plan": req.plan}


@app.post("/api/auth/save-gmail-token")
async def save_gmail_token(req: GmailTokenRequest):
    """Store the Gmail OAuth access token for a user (Growth Pro and above)."""
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    c.execute("UPDATE users SET gmail_token=? WHERE email=?", (req.gmail_token, req.email))
    conn.commit(); conn.close()
    return {"status": "success"}


@app.get("/api/user")
async def get_user_info(email: str = None, device_token: str = None):
    user = get_user(email=email, device_token=device_token)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    return user


# ── Routes: dispute ────────────────────────────────────────────────────────
@app.post("/api/dispute")
@app.post("/api/dispute/")
async def analyze_dispute(
    platform:     str   = Form(...),
    notes:        str   = Form(""),
    transcript:   str   = Form(""),
    cost:         float = Form(45.00),
    model:        str   = Form("gemini"),
    email:        str   = Form(""),
    device_token: str   = Form("default"),
    files: List[UploadFile] = File(default=[]),
):
    text_check = f"{notes} {transcript}"
    policy     = PLATFORM_POLICY.get(platform, f"{platform} lead policy")

    # Trial / plan check
    user          = get_user(email=email, device_token=device_token)
    trials_left   = user["trials_left"] if user else 3
    user_plan     = user["plan"] if user else "solo"
    tier          = plan_tier(user_plan)
    is_unlimited  = tier >= 2 and user is not None

    if not is_unlimited and trials_left <= 0:
        raise HTTPException(status_code=402, detail="Trial limit reached. Please upgrade your plan.")

    prompt_text = (
        f"Marketplace Platform: {platform}\n"
        f"Policy: {policy}\n"
        f"Lead Cost: ${cost:.2f}\n\n"
        f"Contractor Notes:\n{notes or '(none)'}\n\n"
        f"Transcript:\n{transcript or '(none)'}"
    )

    try:
        if model == "groq":
            data = await call_groq(prompt_text)
        else:
            file_parts = []
            for f in files:
                raw = await f.read()
                if raw:
                    # Gemini SDK inline file data
                    n = (f.filename or "").lower()
                    mime = "audio/mpeg" if n.endswith(".mp3") else "audio/wav" if n.endswith(".wav") else "application/pdf"
                    file_parts.append({"mime_type": mime, "data": raw})
            data = await call_gemini(prompt_text, file_parts or None)

        valid = bool(data.get("valid"))

        # Safety net
        if valid and INVALID_SAFETY_RE.search(text_check):
            logger.info("Safety net overrode valid=True for: %.120s", text_check)
            result = not_eligible(cost)
        elif not valid:
            result = not_eligible(
                cost,
                confidence=float(data.get("confidence") or 0.88),
                reason=data.get("reason") or NOT_ELIGIBLE_REASON,
            )
        else:
            claim = str(data.get("claim") or "").strip()
            if not claim:
                raise ValueError("Model returned valid=True but produced no claim text.")
            result = {
                "valid":      True,
                "confidence": float(data.get("confidence") or 0.9),
                "reason":     data.get("reason", ""),
                "claim":      claim,
                "amount":     cost,
            }

        # Deduct trial and persist dispute
        if not is_unlimited:
            trials_left = deduct_trial(email, device_token)
        result["trials_left"] = trials_left

        if email:
            conn = sqlite3.connect(DB_FILE); c = conn.cursor()
            c.execute(
                "INSERT INTO disputes (email, platform, claim, amount, valid) VALUES (?,?,?,?,?)",
                (email, platform, result.get("claim",""), cost, int(result["valid"])),
            )
            result["dispute_id"] = c.lastrowid
            conn.commit(); conn.close()

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Dispute review failed (model=%s): %s", model, e)
        if NON_REFUNDABLE_RE.search(text_check):
            return not_eligible(cost)
        raise HTTPException(
            status_code=502,
            detail=f"AI review failed ({type(e).__name__}). Check GEMINI_API_KEY / GROQ_API_KEY in Render.",
        )


@app.post("/api/dispute/recover")
async def mark_recovered(req: RecoverRequest):
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    c.execute("UPDATE disputes SET recovered=1 WHERE id=? AND email=?", (req.dispute_id, req.email))
    conn.commit()
    c.execute("SELECT SUM(amount) FROM disputes WHERE email=? AND recovered=1", (req.email,))
    row = c.fetchone(); conn.close()
    return {"status": "success", "total_recovered": row[0] or 0.0}


@app.post("/api/dispute/estimate")
@app.post("/api/dispute/estimate/")
async def estimate_lead_cost(data: dict = Body(...)):
    platform = data.get("platform", "Angi")
    text     = data.get("text", "").lower()
    base, basis = 50, "general home services"
    if any(w in text for w in ["roof", "shingle", "gutter"]):   base, basis = 90, "roofing"
    elif any(w in text for w in ["plumb","leak","drain","heater"]): base, basis = 60, "plumbing"
    elif any(w in text for w in ["electric","wiring","panel"]):  base, basis = 55, "electrical"
    elif re.search(r"hvac|furnace|\bac\b|air condition|heating", text): base, basis = 70, "HVAC"
    elif any(w in text for w in ["lawn","landscap","tree","fence"]): base, basis = 40, "landscaping"
    mult   = {"Angi":1.1,"Thumbtack":0.9,"Porch":0.85,"Networx":1.0}.get(platform, 1.0)
    amount = round(base * mult, 2)
    return {"amount": amount, "low": round(amount*.75,2), "high": round(amount*1.25,2), "basis": f"{basis} on {platform}"}


# ── Routes: feedback ───────────────────────────────────────────────────────
@app.post("/api/feedback")
@app.post("/api/feedback/")
async def submit_feedback(req: FeedbackRequest):
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    c.execute(
        "INSERT INTO feedback_entries (email,workspace,type,message,priority) VALUES (?,?,?,?,?)",
        (req.email, req.workspace, req.type, req.message, req.priority),
    )
    conn.commit(); conn.close()
    logger.info("Feedback [%s/%s] from %s: %.200s", req.type, req.priority, req.email or "anon", req.message)
    return {"status": "success"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
