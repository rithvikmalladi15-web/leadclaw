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
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("leadclaw")

app = FastAPI(title="LeadClaw API", version="7.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "843627718759-1gs9i7afgfsnas1mjujinhg2nrvdom1v.apps.googleusercontent.com")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

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
    conn.commit()
    conn.close()

init_db()

class GoogleAuthRequest(BaseModel):
    token: str
    email: Optional[str] = None

class WorkspaceSetupRequest(BaseModel):
    email: str
    workspace: str
    plan: str
    device_token: str
    gmail_token: Optional[str] = None

@app.post("/api/auth/google")
async def google_auth(req: GoogleAuthRequest):
    # Verify token with Google's OAuth endpoint
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"https://oauth2.googleapis.com/tokeninfo?id_token={req.token}")
        if resp.status_code != 200:
            raise HTTPException(status_code=400, detail="Invalid Google token.")
        data = resp.json()
        email = data.get("email")
        if not email:
            raise HTTPException(status_code=400, detail="Google token missing email.")
        return {"status": "success", "email": email, "gmail_token": req.token}

@app.post("/api/auth/setup-workspace")
async def setup_workspace(req: WorkspaceSetupRequest):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        """INSERT INTO users (email, workspace, plan, trials_left, gmail_token, device_token)
           VALUES (?,?,?, COALESCE((SELECT trials_left FROM users WHERE email=?), 3), ?, ?)
           ON CONFLICT(email) DO UPDATE SET
             workspace=excluded.workspace, plan=excluded.plan, gmail_token=COALESCE(excluded.gmail_token, users.gmail_token), device_token=excluded.device_token""",
        (req.email, req.workspace, req.plan, req.email, req.gmail_token, req.device_token),
    )
    conn.commit()
    conn.close()
    return {"status": "success"}

# (Keep your /api/dispute and other routes from previous versions here)
