import json
import logging
import os
import re
import sqlite3
import random
from datetime import datetime, date
from typing import List, Optional

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from groq import Groq

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("leadclaw")

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(title="LeadClaw API", version="6.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Environment variables ──────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL     = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ── SQLite database ────────────────────────────────────────────────────────
DB_FILE = "leadclaw.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users (
        email         TEXT PRIMARY KEY,
        username      TEXT,
        password      TEXT,
        age           INTEGER,
        workspace     TEXT,
        plan          TEXT DEFAULT 'solo',
        trials_left   INTEGER DEFAULT 3,
        device_token  TEXT,
        created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
class RegisterRequest(BaseModel):
    email: str
    username: str
    password: str
    dob: str  # YYYY-MM-DD
    business: str

class PlanSelectRequest(BaseModel):
    email: str
    plan: str
    device_token: str

class FeedbackRequest(BaseModel):
    email: Optional[str] = ""
    workspace: Optional[str] = ""
    type: Optional[str] = "Feedback"
    message: str
    priority: Optional[str] = "medium"

class RecoverRequest(BaseModel):
    dispute_id: int
    email: str

# ── Policy constants & AI Prompts ──────────────────────────────────────────
PLATFORM_POLICY = {
    "Angi":      "Angi Lead Credit Policy",
    "Thumbtack": "Thumbtack Lead Price Credit Terms",
    "Porch":     "Porch Pro Lead Guarantee",
    "Networx":   "Networx Lead Dispute Terms",
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
exact facts provided. Start with a Subject line, formal greeting, clear issue breakdown,
policy citation, and a firm credit request.

If valid is false: one or two plain sentences for "reason" and "claim" must be empty string.

Respond ONLY in valid JSON format with these exact keys:
{"valid": true or false, "confidence": 0.95, "reason": "...", "claim": "..."}"""

def extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.endswith("```"):
        text = text[:-3]
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1:
        raise ValueError(f"No JSON found in: {text[:300]}")
    return json.loads(text[s:e + 1])

async def call_gemini(prompt_text: str, file_parts: list = None) -> dict:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set.")
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
    if not groq_client:
        raise ValueError("GROQ_API_KEY is not set.")
    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt_text},
        ],
        temperature=0.1,
        response_format={"type": "json_object"}
    )
    content = completion.choices[0].message.content
    return extract_json(content)

def not_eligible(cost: float, confidence: float = 0.88, reason: str = NOT_ELIGIBLE_REASON) -> dict:
    return {
        "valid": False, "confidence": confidence,
        "reason": reason, "claim": "", "amount": cost,
    }

def get_user(email: str = None, device_token: str = None) -> Optional[dict]:
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    if email:
        c.execute("SELECT email,username,age,workspace,plan,trials_left FROM users WHERE email=?", (email,))
    elif device_token:
        c.execute("SELECT email,username,age,workspace,plan,trials_left FROM users WHERE device_token=?", (device_token,))
    else:
        conn.close(); return None
    row = c.fetchone(); conn.close()
    if not row: return None
    return {"email": row[0], "username": row[1], "age": row[2], "workspace": row[3], "plan": row[4], "trials_left": row[5]}

def plan_tier(plan: str) -> int:
    return {"solo": 1, "growth": 2, "crew": 3}.get(plan, 1)

# ── Routes ─────────────────────────────────────────────────────────────────
@app.get("/")
def health():
    return {"ok": True, "service": "LeadClaw API", "version": "6.0.0"}

@app.post("/api/auth/register")
async def register_user(req: RegisterRequest):
    # 1. Enforce Gmail Only
    if not req.email.lower().endswith("@gmail.com"):
        raise HTTPException(status_code=400, detail="You must provide a valid @gmail.com email address.")

    # 2. Enforce Age 18+
    try:
        dob_date = datetime.strptime(req.dob, "%Y-%m-%d").date()
        today = date.today()
        calculated_age = today.year - dob_date.year - ((today.month, today.day) < (dob_date.month, dob_date.day))
        if calculated_age < 18:
            raise HTTPException(status_code=403, detail="You must be at least 18 years old to create an account.")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid Date of Birth format. Please select from the calendar picker.")

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        c.execute(
            """INSERT INTO users (email, username, password, age, workspace, plan, trials_left)
               VALUES (?, ?, ?, ?, ?, 'solo', 3)
               ON CONFLICT(email) DO UPDATE SET
                 username=excluded.username, password=excluded.password, age=excluded.age, workspace=excluded.workspace""",
            (req.email, req.username, req.password, calculated_age, req.business)
        )
        conn.commit()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        conn.close()
    return {"status": "success", "message": "Account created successfully."}

@app.post("/api/auth/select-plan")
async def select_plan(req: PlanSelectRequest):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("UPDATE users SET plan=?, device_token=? WHERE email=?", (req.plan, req.device_token, req.email))
    conn.commit()
    conn.close()
    return {"status": "success", "plan": req.plan}

@app.get("/api/admin/analytics")
async def get_customer_analytics():
    """Admin analytics route for querying SQLite customer records and dispute metrics."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute("SELECT COUNT(*) as total_users, AVG(age) as avg_age FROM users")
    stats = dict(c.fetchone())
    
    c.execute("SELECT plan, COUNT(*) as count FROM users GROUP BY plan")
    plan_distribution = [dict(row) for row in c.fetchall()]
    
    c.execute("SELECT email, username, age, workspace, plan, trials_left, created_at FROM users")
    users = [dict(row) for row in c.fetchall()]
    
    c.execute("SELECT id, email, platform, amount, valid, recovered, created_at FROM disputes")
    disputes = [dict(row) for row in c.fetchall()]
    
    conn.close()
    
    return {
        "status": "success",
        "summary": {
            "total_customers": stats["total_users"] or 0,
            "average_age": round(stats["avg_age"] or 0, 1)
        },
        "plan_distribution": plan_distribution,
        "users": users,
        "disputes": disputes
    }

@app.post("/api/dispute")
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
                    n = (f.filename or "").lower()
                    mime = "audio/mpeg" if n.endswith(".mp3") else "audio/wav" if n.endswith(".wav") else "application/pdf"
                    file_parts.append({"mime_type": mime, "data": raw})
            data = await call_gemini(prompt_text, file_parts or None)

        valid = bool(data.get("valid"))

        if valid and INVALID_SAFETY_RE.search(text_check):
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

        if not is_unlimited:
            conn = sqlite3.connect(DB_FILE); c = conn.cursor()
            c.execute("UPDATE users SET trials_left = MAX(0, trials_left - 1) WHERE email=? OR device_token=?", (email, device_token))
            c.execute("SELECT trials_left FROM users WHERE email=? OR device_token=?", (email, device_token))
            row = c.fetchone(); conn.close()
            trials_left = row[0] if row else 0

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
        logger.exception("Dispute review failed: %s", e)
        raise HTTPException(status_code=502, detail=f"AI review failed: {str(e)}")

@app.post("/api/dispute/recover")
async def mark_recovered(req: RecoverRequest):
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    c.execute("UPDATE disputes SET recovered=1 WHERE id=? AND email=?", (req.dispute_id, req.email))
    conn.commit()
    c.execute("SELECT SUM(amount) FROM disputes WHERE email=? AND recovered=1", (req.email,))
    row = c.fetchone(); conn.close()
    return {"status": "success", "total_recovered": row[0] or 0.0}

@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest):
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    c.execute(
        "INSERT INTO feedback_entries (email,workspace,type,message,priority) VALUES (?,?,?,?,?)",
        (req.email, req.workspace, req.type, req.message, req.priority),
    )
    conn.commit(); conn.close()
    return {"status": "success"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
