import os
from fastapi import FastAPI, Form, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from generate import process_lead_dispute

app = FastAPI(title="LeadClaw Backend")

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "online", "service": "LeadClaw API"}

@app.post("/api/dispute")
async def analyze_dispute(
    platform: str = Form(...),
    notes: Optional[str] = Form(""),
    transcript: Optional[str] = Form(""),
    cost: float = Form(0.0),
    model: str = Form("gemini"),
    files: List[UploadFile] = File(default=[])
):
    try:
        # Call your generation logic in generate.py
        result = process_lead_dispute(
            platform=platform,
            notes=notes,
            transcript=transcript,
            cost=cost,
            model=model
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
