import os
from typing import List, Optional
from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai

app = FastAPI(title="LeadClaw API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini API if key is available
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

class EstimateRequest(BaseModel):
    platform: str
    text: str

class FeedbackRequest(BaseModel):
    type: str
    message: str
    email: Optional[str] = ""
    workspace: Optional[str] = ""

@app.post("/api/dispute/estimate")
async def estimate_cost(req: EstimateRequest):
    text_lower = req.text.lower()
    base = 50
    if any(k in text_lower for k in ["roof", "shingle", "gutter"]):
        base = 90
    elif any(k in text_lower for k in ["plumb", "leak", "drain", "water heater", "pipe"]):
        base = 60
    elif any(k in text_lower for k in ["electric", "wiring", "panel", "outlet"]):
        base = 55
    elif any(k in text_lower for k in ["hvac", "furnace", "ac", "air condition", "heating"]):
        base = 70
    elif any(k in text_lower for k in ["lawn", "landscap", "tree", "fence"]):
        base = 40

    mult = {"Angi": 1.1, "Thumbtack": 0.9, "Porch": 0.85, "Networx": 1.0}.get(req.platform, 1.0)
    amount = round(base * mult)
    return {
        "amount": float(amount),
        "low": round(amount * 0.75),
        "high": round(amount * 1.25),
        "basis": f"Estimated based on lead service type for {req.platform}"
    }

@app.post("/api/dispute")
async def create_dispute(
    platform: str = Form(...),
    notes: Optional[str] = Form(""),
    transcript: Optional[str] = Form(""),
    cost: Optional[float] = Form(0.0),
    model: Optional[str] = Form("gemini"),
    files: List[UploadFile] = File([])
):
    combined_text = f"Notes: {notes}\nTranscript: {transcript}"
    text_lower = combined_text.lower()
    
    # Basic deterministic heuristic rule check for eligibility
    invalid_keywords = ["changed mind", "too expensive", "decided to wait", "hired someone else"]
    is_valid = True
    reason = ""
    
    for kw in invalid_keywords:
        if kw in text_lower:
            is_valid = False
            reason = f"Platform refund policy excludes claims where the customer '{kw}' after initial contact."
            break

    if not is_valid:
        return {
            "valid": False,
            "confidence": 0.88,
            "reason": reason
        }

    # Generate professional dispute claim text
    claim_text = (
        f"To {platform} Support Team,\n\n"
        f"I am writing to formally request a credit/refund for lead charges incurred on our account. "
        f"Upon review, this lead meets the qualifying refund criteria due to the following verified circumstances:\n\n"
        f"Details: {notes or 'Contact details were invalid or non-responsive upon multiple attempts.'}\n"
        f"{'Transcript Reference: ' + transcript[:200] + '...' if transcript else ''}\n\n"
        f"We kindly request that the lead fee of ${cost:.2f} be credited back to our contractor account per the {platform} Pro Lead Guarantee policy.\n\n"
        f"Thank you for your prompt assistance.\n"
        f"Sincerely,\nLeadClaw Pro Contractor"
    )

    return {
        "valid": True,
        "confidence": 0.94,
        "amount": cost if cost > 0 else 75.0,
        "claim": claim_text
    }

@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest):
    return {"status": "success", "message": "Feedback received successfully."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
