import json
import logging
import os
import re
import sqlite3
import random
from typing import List, Optional

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Body
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

# ── Environment variables ──────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL     = os.environ.get("GROQ_MODEL", "qwen-qwq-32b")

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

class GoogleAuthRequest(BaseModel):
    token: str

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
NOT_ELIGIBLE_REASON = (
    "Pricing objections, competitor hiring, project postponements, and customer "
    "choices are non-refundable under marketplace policy."
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

If valid is false: one or two plain sentences for "reason" and "claim" must be empty string.

Respond ONLY in valid JSON with no markdown fences:
{"valid": true or false, "confidence": 0.95, "reason": "...", "claim": "..."}"""

def extract_json(text: str) -> dict:
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1:
        raise ValueError(f"No JSON found in: {text[:300]}")
    return json.loads(text[s:e + 1])

async def call_gemini(prompt_text: str, file_parts: list = None) -> dict:
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

# ── Routes ─────────────────────────────────────────────────────────────────
@app.get("/")
def health():
    return {"ok": True, "service": "LeadClaw API", "version": "4.0.0"}

@app.post("/api/auth/send-code")
async def send_code(req: EmailRequest):
    code = f"{random.randint(100000, 999999)}"
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO verifications (email, code) VALUES (?,?)", (req.email, code))
    conn.commit(); conn.close()
    logger.info("Verification code for %s: %s", req.email, code)
    return {"status": "success", "message": "Verification code sent.", "debug_code": code}

@app.post("/api/auth/verify-code")
async def verify_code(req: CodeVerifyRequest):
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    c.execute("SELECT code FROM verifications WHERE email=?", (req.email,))
    row = c.fetchone(); conn.close()
    if not row or row[0] != req.code:
        raise HTTPException(status_code=400, detail="Invalid or expired verification code.")
    return {"status": "success", "message": "Email verified."}

@app.post("/api/auth/google")
async def google_auth(req: GoogleAuthRequest):
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"https://oauth2.googleapis.com/tokeninfo?id_token={req.token}")
        if resp.status_code != 200:
            raise HTTPException(status_code=400, detail="Invalid Google token.")
        data = resp.json()
        email = data.get("email")
        if not email:
            raise HTTPException(status_code=400, detail="Google token missing email.")
        return {"status": "success", "email": email}

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
    user = get_user(email=email, device_token=device_token)
    trials_left = user["trials_left"] if user else 3
    
    if trials_left <= 0:
        raise HTTPException(status_code=402, detail="Trial limit reached. Please upgrade your plan.")

    prompt_text = (
        f"Marketplace Platform: {platform}\n"
        f"Lead Cost: ${cost:.2f}\n\n"
        f"Contractor Notes:\n{notes or '(none)'}\n\n"
        f"Transcript:\n{transcript or '(none)'}"
    )

    try:
        data = await call_gemini(prompt_text)
        result = {
            "valid": bool(data.get("valid")),
            "confidence": float(data.get("confidence") or 0.9),
            "reason": data.get("reason", ""),
            "claim": str(data.get("claim") or "").strip(),
            "amount": cost,
        }
        
        conn = sqlite3.connect(DB_FILE); c = conn.cursor()
        c.execute("UPDATE users SET trials_left = MAX(0, trials_left - 1) WHERE email=? OR device_token=?", (email, device_token))
        c.execute("SELECT trials_left FROM users WHERE email=? OR device_token=?", (email, device_token))
        row = c.fetchone()
        result["trials_left"] = row[0] if row else 0
        conn.commit(); conn.close()
        return result
    except Exception as e:
        logger.exception("Dispute review failed: %s", e)
        raise HTTPException(status_code=502, detail=f"AI review failed: {str(e)}")

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
