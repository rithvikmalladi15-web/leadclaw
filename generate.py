import json
import logging
import os
import re
import sqlite3
import random
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, date
from typing import List, Optional

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from groq import Groq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("leadclaw")

app = FastAPI(title="LeadClaw API", version="14.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")

GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL     = os.environ.get("GROQ_MODEL", "qwen3-4b")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

SMTP_SENDER   = os.environ.get("SMTP_SENDER", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
SMTP_SERVER   = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT     = int(os.environ.get("SMTP_PORT", 587))

def send_reset_email(recipient: str, reset_code: str):
    if not SMTP_SENDER or not SMTP_PASSWORD:
        logger.warning("SMTP credentials not configured. Reset code: %s", reset_code)
        return
    try:
        msg = MIMEMultipart()
        msg["From"] = SMTP_SENDER
        msg["To"] = recipient
        msg["Subject"] = "LeadClaw Password Reset Request"
        body = f"Hello,\n\nYour LeadClaw password reset code is: {reset_code}\n\nEnter this code in the app to set a new password.\n\nBest regards,\nLeadClaw Team"
        msg.attach(MIMEText(body, "plain"))
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_SENDER, SMTP_PASSWORD)
            server.sendmail(SMTP_SENDER, recipient, msg.as_string())
    except Exception as e:
        logger.error("Failed to send reset email: %s", e)

DB_FILE = "leadclaw.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users (
        email         TEXT PRIMARY KEY,
        password      TEXT,
        age           INTEGER,
        workspace     TEXT,
        full_name     TEXT DEFAULT '',
        phone         TEXT DEFAULT '',
        acc_angi      TEXT DEFAULT '',
        acc_thumbtack TEXT DEFAULT '',
        acc_porch     TEXT DEFAULT '',
        acc_networx   TEXT DEFAULT '',
        plan          TEXT DEFAULT 'solo',
        trials_left   INTEGER DEFAULT 3,
        device_token  TEXT,
        reset_token   TEXT DEFAULT '',
        created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS disputes (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        email       TEXT,
        platform    TEXT,
        lead_phone  TEXT,
        lead_email  TEXT,
        claim       TEXT,
        amount      REAL,
        valid       INTEGER,
        recovered   INTEGER DEFAULT 0,
        created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.commit()
    conn.close()

init_db()

class RegisterRequest(BaseModel):
    email: str
    password: str
    dob: str
    business: str

class LoginRequest(BaseModel):
    email: str
    password: str

class ForgotPasswordRequest(BaseModel):
    email: str

class ResetPasswordRequest(BaseModel):
    email: str
    code: str
    new_password: str

class PlanSelectRequest(BaseModel):
    email: str
    plan: str
    device_token: str

class ProfileUpdateRequest(BaseModel):
    email: str
    password: Optional[str] = ""
    full_name: Optional[str] = ""
    phone: Optional[str] = ""
    acc_angi: Optional[str] = ""
    acc_thumbtack: Optional[str] = ""
    acc_porch: Optional[str] = ""
    acc_networx: Optional[str] = ""

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
exact facts provided, including account names, lead phone numbers, and policy citations.
Start with a Subject line, formal greeting, clear issue breakdown, policy citation, and a firm credit request.

If valid is false: one or two plain sentences for "reason" and "claim" must be empty string.

Respond ONLY in valid JSON format with these exact keys:
{"valid": true or false, "confidence": 0.95, "reason": "...", "claim": "..."}"""

def validate_password_strength(password: str):
    if len(password) < 7 or not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        raise HTTPException(
            status_code=400,
            detail="Password must be at least 7 characters long and contain at least 1 special character."
        )

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

async def call_gemini_3_5_flash(prompt_text: str, file_parts: list = None) -> dict:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not configured.")
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

async def call_groq_qwen3_4b(prompt_text: str) -> dict:
    if not groq_client:
        raise ValueError("GROQ_API_KEY is not configured.")
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

@app.get("/")
def health():
    return {"status": "healthy", "service": "LeadClaw API"}

@app.post("/api/auth/register")
async def register_user(req: RegisterRequest):
    missing_fields = []
    if not req.email.strip(): missing_fields.append("Email address")
    if not req.password.strip(): missing_fields.append("Password")
    if not req.dob.strip(): missing_fields.append("Date of Birth")
    if not req.business.strip(): missing_fields.append("Business Name")

    if missing_fields:
        raise HTTPException(status_code=400, detail=f"Please fill out: {', '.join(missing_fields)}")

    if not req.email.lower().endswith("@gmail.com"):
        raise HTTPException(status_code=400, detail="A valid @gmail.com email address is required.")

    validate_password_strength(req.password)

    try:
        dob_date = datetime.strptime(req.dob, "%Y-%m-%d").date()
        today = date.today()
        calculated_age = today.year - dob_date.year - ((today.month, today.day) < (dob_date.month, dob_date.day))
        if calculated_age < 18:
            raise HTTPException(status_code=403, detail="You must be at least 18 years old to create an account.")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid Date of Birth format.")

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO users (email, password, age, workspace, plan, trials_left) VALUES (?, ?, ?, ?, 'solo', 3) ON CONFLICT(email) DO UPDATE SET password=excluded.password, age=excluded.age, workspace=excluded.workspace",
            (req.email, req.password, calculated_age, req.business)
        )
        conn.commit()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        conn.close()
    return {"status": "success", "message": "Account created successfully."}

@app.post("/api/auth/login")
async def login_user(req: LoginRequest):
    missing = []
    if not req.email.strip(): missing.append("Email address")
    if not req.password.strip(): missing.append("Password")
    if missing:
        raise HTTPException(status_code=400, detail=f"Please fill out: {', '.join(missing)}")

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT email, password, trials_left, plan FROM users WHERE email=?", (req.email,))
    row = c.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=401, detail="No account found with this email address.")
    stored_email, stored_password, trials_left, plan = row
    if stored_password != req.password:
        raise HTTPException(status_code=401, detail="Incorrect password. Please try again.")

    return {"status": "success", "email": stored_email, "trials_left": trials_left, "plan": plan}

@app.post("/api/auth/forgot-password")
async def forgot_password(req: ForgotPasswordRequest):
    if not req.email.strip():
        raise HTTPException(status_code=400, detail="Please enter your email address.")

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT email FROM users WHERE email=?", (req.email,))
    row = c.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Email address not found in our records.")

    reset_code = "".join([str(random.randint(0, 9)) for _ in range(6)])
    c.execute("UPDATE users SET reset_token=? WHERE email=?", (reset_code, req.email))
    conn.commit()
    conn.close()

    send_reset_email(req.email, reset_code)
    return {"status": "success", "message": "Password reset code sent to your email."}

@app.post("/api/auth/reset-password")
async def reset_password(req: ResetPasswordRequest):
    validate_password_strength(req.new_password)
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT reset_token FROM users WHERE email=?", (req.email,))
    row = c.fetchone()
    if not row or row[0] != req.code:
        conn.close()
        raise HTTPException(status_code=400, detail="Invalid verification code.")

    c.execute("UPDATE users SET password=?, reset_token='' WHERE email=?", (req.new_password, req.email))
    conn.commit()
    conn.close()
    return {"status": "success", "message": "Password updated successfully."}

@app.post("/api/auth/select-plan")
async def select_plan(req: PlanSelectRequest):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("UPDATE users SET plan=?, device_token=? WHERE email=?", (req.plan, req.device_token, req.email))
    conn.commit()
    conn.close()
    return {"status": "success", "plan": req.plan}

@app.post("/api/user/profile")
async def update_profile(req: ProfileUpdateRequest):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    if req.password:
        validate_password_strength(req.password)
        c.execute(
            "UPDATE users SET password=?, full_name=?, phone=?, acc_angi=?, acc_thumbtack=?, acc_porch=?, acc_networx=? WHERE email=?",
            (req.password, req.full_name, req.phone, req.acc_angi, req.acc_thumbtack, req.acc_porch, req.acc_networx, req.email)
        )
    else:
        c.execute(
            "UPDATE users SET full_name=?, phone=?, acc_angi=?, acc_thumbtack=?, acc_porch=?, acc_networx=? WHERE email=?",
            (req.full_name, req.phone, req.acc_angi, req.acc_thumbtack, req.acc_porch, req.acc_networx, req.email)
        )
    conn.commit()
    conn.close()
    return {"status": "success"}

@app.get("/api/admin/analytics")
async def get_customer_analytics():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute("SELECT COUNT(*) as total_users, AVG(age) as avg_age FROM users")
    stats = dict(c.fetchone())
    
    c.execute("SELECT plan, COUNT(*) as count FROM users GROUP BY plan")
    plan_distribution = [dict(row) for row in c.fetchall()]
    
    c.execute("SELECT email, password, age, workspace, full_name, phone, acc_angi, acc_thumbtack, acc_porch, acc_networx, plan, trials_left, created_at FROM users")
    users = [dict(row) for row in c.fetchall()]
    
    c.execute("SELECT id, email, platform, lead_phone, lead_email, amount, valid, recovered, created_at FROM disputes")
    disputes = [dict(row) for row in c.fetchall()]
    
    conn.close()
    
    return {
        "status": "success",
        "summary": {"total_customers": stats["total_users"] or 0, "average_age": round(stats["avg_age"] or 0, 1)},
        "plan_distribution": plan_distribution,
        "users": users,
        "disputes": disputes
    }

@app.post("/api/dispute")
async def analyze_dispute(
    platform:     str   = Form(...),
    lead_phone:   str   = Form(""),
    lead_email:   str   = Form(""),
    notes:        str   = Form(""),
    transcript:   str   = Form(""),
    cost:         float = Form(45.00),
    model:        str   = Form("gemini"),
    email:        str   = Form(""),
    device_token: str   = Form("default"),
    files: List[UploadFile] = File(default=[]),
):
    missing_lead_info = []
    if not lead_phone.strip(): missing_lead_info.append("Lead Phone Number")
    if not lead_email.strip(): missing_lead_info.append("Lead Email Address")
    if missing_lead_info:
        raise HTTPException(status_code=400, detail=f"Please fill out: {', '.join(missing_lead_info)}")

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT plan, trials_left, full_name, workspace, acc_angi, acc_thumbtack, acc_porch, acc_networx FROM users WHERE email=? OR device_token=?", (email, device_token))
    row = c.fetchone()
    conn.close()

    plan = row[0] if row else "solo"
    trials_left = row[1] if row else 3
    is_unlimited = (plan == "growth")

    if not is_unlimited and trials_left <= 0:
        raise HTTPException(status_code=402, detail="Trial limit reached. Please upgrade your plan.")

    if files and plan == "solo":
        raise HTTPException(status_code=403, detail="Audio and PDF file reading is restricted to Growth Pro users. Please upgrade.")

    text_check = f"{notes} {transcript}"
    policy     = PLATFORM_POLICY.get(platform, f"{platform} lead policy")

    account_handle = ""
    if row:
        if platform == "Angi": account_handle = row[4]
        elif platform == "Thumbtack": account_handle = row[5]
        elif platform == "Porch": account_handle = row[6]
        elif platform == "Networx": account_handle = row[7]

    prompt_text = (
        f"Marketplace Platform: {platform}\n"
        f"Contractor Account ID: {account_handle or 'Default Workspace'}\n"
        f"Policy: {policy}\n"
        f"Lead Cost: ${cost:.2f}\n"
        f"Lead Phone: {lead_phone}\n"
        f"Lead Email: {lead_email}\n\n"
        f"Contractor Call Notes:\n{notes or '(none)'}\n\n"
        f"Transcript:\n{transcript or '(none)'}"
    )

    try:
        if model == "groq" and plan == "growth":
            data = await call_groq_qwen3_4b(prompt_text)
        else:
            file_parts = []
            for f in files:
                raw = await f.read()
                if raw:
                    n = (f.filename or "").lower()
                    mime = "audio/mpeg" if n.endswith(".mp3") else "audio/wav" if n.endswith(".wav") else "application/pdf"
                    file_parts.append({"mime_type": mime, "data": raw})
            data = await call_gemini_3_5_flash(prompt_text, file_parts or None)

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
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute("UPDATE users SET trials_left = MAX(0, trials_left - 1) WHERE email=? OR device_token=?", (email, device_token))
            c.execute("SELECT trials_left FROM users WHERE email=? OR device_token=?", (email, device_token))
            row_t = c.fetchone()
            conn.commit()
            conn.close()
            trials_left = row_t[0] if row_t else 0

        result["trials_left"] = trials_left

        if email:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute(
                "INSERT INTO disputes (email, platform, lead_phone, lead_email, claim, amount, valid) VALUES (?,?,?,?,?,?,?)",
                (email, platform, lead_phone, lead_email, result.get("claim",""), cost, int(result["valid"])),
            )
            result["dispute_id"] = c.lastrowid
            conn.commit()
            conn.close()

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Dispute review failed: %s", e)
        raise HTTPException(status_code=502, detail=f"AI review failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
