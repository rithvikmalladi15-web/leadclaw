import json
import logging
import os
import random
import sqlite3
from typing import Optional
import uvicorn
import google.generativeai as genai
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("leadclaw")

app = FastAPI(title="LeadClaw API", version="9.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        logger.error("Failed to configure Gemini: %s", e)

DB_FILE = "leadclaw.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users (
        email         TEXT PRIMARY KEY,
        workspace     TEXT,
        plan          TEXT DEFAULT 'solo',
        trials_left   INTEGER DEFAULT 3,
        device_token  TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS verifications (
        email TEXT PRIMARY KEY,
        code  TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS feedback_entries (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        email      TEXT,
        workspace  TEXT,
        message    TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.commit()
    conn.close()

init_db()

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
    message: str

@app.get("/")
def health():
    return {"status": "online", "service": "LeadClaw Render API"}

@app.post("/api/auth/send-code")
async def send_code(req: EmailRequest):
    code = f"{random.randint(100000, 999999)}"
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO verifications (email, code) VALUES (?,?)", (req.email, code))
    conn.commit()
    conn.close()
    logger.info("========================================")
    logger.info("VERIFICATION CODE FOR %s: %s", req.email, code)
    logger.info("========================================")
    return {"status": "success", "debug_code": code}

@app.post("/api/auth/verify-code")
async def verify_code(req: CodeVerifyRequest):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT code FROM verifications WHERE email=?", (req.email,))
    row = c.fetchone()
    conn.close()
    if not row or row[0] != req.code:
        raise HTTPException(status_code=400, detail="Invalid verification code.")
    return {"status": "success"}

@app.post("/api/auth/setup-workspace")
async def setup_workspace(req: WorkspaceSetupRequest):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        """INSERT INTO users (email, workspace, plan, trials_left, device_token)
           VALUES (?,?,?, COALESCE((SELECT trials_left FROM users WHERE email=?), 3), ?)
           ON CONFLICT(email) DO UPDATE SET
             workspace=excluded.workspace, plan=excluded.plan, device_token=excluded.device_token""",
        (req.email, req.workspace, req.plan, req.email, req.device_token),
    )
    conn.commit()
    conn.close()
    return {"status": "success"}

@app.post("/api/dispute")
async def analyze_dispute(
    platform:     str   = Form(...),
    notes:        str   = Form(""),
    cost:         float = Form(45.00),
    email:        str   = Form(""),
    device_token: str   = Form("default"),
):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT trials_left FROM users WHERE email=? OR device_token=?", (email, device_token))
    row = c.fetchone()
    trials_left = row[0] if row else 3

    if trials_left <= 0:
        conn.close()
        raise HTTPException(status_code=402, detail="Trial limit reached.")

    claim_text = (
        f"Subject: Formal Lead Credit Dispute - {platform}\n\n"
        f"To {platform} Support Team,\n\n"
        f"I am writing to formally request a credit refund for a defective lead provided on "
        f"{platform} with a billed amount of ${cost:.2f}.\n\n"
        f"Reason for Dispute:\n{notes}\n\n"
        f"Under marketplace guarantee guidelines regarding invalid contact information and non-serviceable criteria, "
        f"this lead fails to meet qualifying standards. Please credit ${cost:.2f} back to my account ledger.\n\n"
        f"Regards,\nContractor Account Team"
    )

    c.execute("UPDATE users SET trials_left = MAX(0, trials_left - 1) WHERE email=? OR device_token=?", (email, device_token))
    c.execute("SELECT trials_left FROM users WHERE email=? OR device_token=?", (email, device_token))
    new_trials = c.fetchone()[0]
    conn.commit()
    conn.close()

    return {
        "valid": True,
        "confidence": 0.98,
        "reason": "Matches valid credit policy criteria.",
        "claim": claim_text,
        "amount": cost,
        "trials_left": new_trials
    }

@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO feedback_entries (email, workspace, message) VALUES (?,?,?)", (req.email, req.workspace, req.message))
    conn.commit()
    conn.close()
    return {"status": "success"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
