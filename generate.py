import os
import sqlite3
import random
import logging
from typing import List, Optional
from fastapi import FastAPI, Form, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LeadClaw")

app = FastAPI(title="LeadClaw Production API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_FILE = "leadclaw.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    email TEXT PRIMARY KEY,
                    workspace TEXT,
                    plan TEXT,
                    trials_left INTEGER,
                    device_token TEXT
                )''')
    c.execute('''CREATE TABLE IF NOT EXISTS verifications (
                    email TEXT PRIMARY KEY,
                    code TEXT
                )''')
    c.execute('''CREATE TABLE IF NOT EXISTS feedback_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT,
                    workspace TEXT,
                    message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )''')
    conn.commit()
    conn.close()

init_db()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

class EmailVerifyRequest(BaseModel):
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
async def root():
    return {"status": "ok", "app": "LeadClaw API", "storage": "sqlite"}

@app.post("/api/auth/send-code")
async def send_verification_code(req: EmailVerifyRequest):
    code = f"{random.randint(100000, 999999)}"
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO verifications (email, code) VALUES (?, ?)", (req.email, code))
    conn.commit()
    conn.close()
    logger.info(f"Verification code for {req.email}: {code}")
    return {"status": "success", "message": "Verification code sent.", "debug_code": code}

@app.post("/api/auth/verify-code")
async def verify_code(req: CodeVerifyRequest):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT code FROM verifications WHERE email = ?", (req.email,))
    row = c.fetchone()
    conn.close()
    
    if not row or row[0] != req.code:
        raise HTTPException(status_code=400, detail="Invalid verification code.")
    
    return {"status": "success", "message": "Email verified successfully."}

@app.post("/api/auth/setup-workspace")
async def setup_workspace(req: WorkspaceSetupRequest):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO users (email, workspace, plan, trials_left, device_token)
                 VALUES (?, ?, ?, COALESCE((SELECT trials_left FROM users WHERE email = ?), 3), ?)''',
              (req.email, req.workspace, req.plan, req.email, req.device_token))
    conn.commit()
    conn.close()
    return {"status": "success", "workspace": req.workspace, "plan": req.plan}

@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO feedback_entries (email, workspace, message) VALUES (?, ?, ?)", 
              (req.email, req.workspace, req.message))
    conn.commit()
    conn.close()
    logger.info(f"Feedback entry stored for {req.email or 'Anonymous'}")
    return {"status": "success", "message": "Feedback saved successfully."}

@app.post("/api/dispute")
async def create_dispute(
    platform: str = Form(...),
    notes: Optional[str] = Form(""),
    cost: Optional[float] = Form(0.0),
    device_token: Optional[str] = Form("default_device"),
    email: Optional[str] = Form("")
):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT trials_left FROM users WHERE device_token = ? OR email = ?", (device_token, email))
    row = c.fetchone()
    trials_left = row[0] if row else 3

    if trials_left > 0:
        trials_left -= 1
        c.execute("UPDATE users SET trials_left = ? WHERE device_token = ? OR email = ?", (trials_left, device_token, email))
        conn.commit()
    conn.close()

    claim_text = (
        f"To {platform} Support Team,\n\n"
        f"I am writing to formally request a credit/refund for lead charges incurred on our account. "
        f"Upon review, this lead meets qualifying refund criteria due to the following verified circumstances:\n\n"
        f"Details: {notes or 'Contact details were invalid or non-responsive upon multiple attempts.'}\n\n"
        f"We kindly request that the lead fee of ${cost:.2f} be credited back to our account per {platform} lead refund policies.\n\n"
        f"Thank you,\nLeadClaw Contractor Team"
    )

    return {
        "valid": True,
        "confidence": 0.95,
        "amount": cost if cost > 0 else 75.0,
        "claim": claim_text,
        "trial_left": trials_left
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
