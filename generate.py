import json
import os
import re
from typing import List

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

app = FastAPI(title="LeadClaw API", version="1.0.0")

# Production CORS configuration (restrict or open based on deployment)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

SYSTEM_RULES = """You are an expert compliance auditor and lead marketplace policy specialist for Angi, Thumbtack, Porch, and Networx. 

Your task is to evaluate the contractor's notes, transcripts, and attached evidence to determine if the lead is genuinely defective or if it represents normal commercial drop-off.

POLICY GUIDELINES:
1. ELIGIBLE FOR REFUND (valid = true):
   - Contact failure: Permanently disconnected numbers, wrong numbers belonging to third parties, or undeliverable emails.
   - Geographic mismatch: Project address is strictly outside the contractor's service zip codes/radius.
   - Trade mismatch: Inquiries for services the contractor does not offer.
   - System/Duplicate errors: Duplicate leads sent within the billing window or bot/spam submissions.

2. INELIGIBLE FOR REFUND (valid = false):
   - Commercial / Consumer choices: Homeowner thought the price was too expensive, rejected estimates, or chose a competitor.
   - Postponements: Project put on hold or cancelled by the customer.
   - Unresponsiveness post-connection: Customer answered initially but stopped returning calls/voicemails later.

INSTRUCTIONS:
- Read the narrative context carefully to determine the true root cause.
- If valid is true, write a highly professional, persuasive, formal dispute claim letter tailored to the specific platform ("claim").
- If valid is false, provide a clear, policy-based explanation ("reason") and leave "claim" empty ("").

Respond STRICTLY in valid JSON with no extra markdown formatting outside the JSON block, using this exact schema:
{"valid": true or false, "confidence": 0.95, "reason": "...", "claim": "..."}"""

def mime_for(name, given):
    n = (name or "").lower()
    if n.endswith(".mp3"): return "audio/mpeg"
    if n.endswith(".wav"): return "audio/wav"
    if n.endswith(".pdf"): return "application/pdf"
    return given or "application/octet-stream"

@app.get("/")
def health():
    return {"status": "healthy", "service": "LeadClaw API"}

@app.post("/api/dispute")
@app.post("/api/dispute/")
async def analyze_dispute(
    platform: str = Form(...),
    notes: str = Form(""),
    transcript: str = Form(""),
    cost: float = Form(45.00),
    files: List[UploadFile] = File(default=[]),
):
    try:
        parts = [
            f"Marketplace Platform: {platform}\n"
            f"Lead Cost: ${cost:.2f}\n\n"
            f"Contractor Notes: {notes or '(none)'}\n\n"
            f"Transcript text: {transcript or '(none)'}"
        ]
        
        for f in files:
            data = await f.read()
            if data:
                parts.append(
                    types.Part.from_bytes(data=data, mime_type=mime_for(f.filename, f.content_type))
                )

        response = await client.aio.models.generate_content(
            model=MODEL,
            contents=parts,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_RULES,
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )

        raw_text = response.text.strip()
        
        # Robust JSON extraction using regex fallback in case of model formatting quirks
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            clean_json = match.group(0)
            data = json.loads(clean_json)
        else:
            raise ValueError("Model output did not contain valid JSON structure.")

        data["amount"] = cost
        return data

    except Exception as e:
        # Fallback response for unhandled runtime errors
        return {
            "valid": False,
            "confidence": 0.75,
            "reason": f"Unable to complete automated policy verification due to a processing error. Please review manually.",
            "claim": "",
            "amount": cost,
        }

@app.post("/api/dispute/estimate")
@app.post("/api/dispute/estimate/")
async def estimate_lead_cost(data: dict):
    platform = data.get("platform", "Angi")
    text = data.get("text", "").lower()

    base = 50
    basis = "general home services"
    if any(w in text for w in ["roof", "shingle", "gutter"]):
        base, basis = 90, "roofing"
    elif any(w in text for w in ["plumb", "leak", "drain", "water heater"]):
        base, basis = 60, "plumbing"
    elif any(w in text for w in ["electric", "wiring", "panel"]):
        base, basis = 55, "electrical"
    elif any(w in text for w in ["hvac", "furnace", "ac", "heating"]):
        base, basis = 70, "HVAC"
    elif any(w in text for w in ["lawn", "landscap", "tree"]):
        base, basis = 40, "landscaping"

    multiplier = {"Angi": 1.1, "Thumbtack": 0.9, "Porch": 0.85, "Networx": 1.0}.get(platform, 1.0)
    amount = round(base * multiplier, 2)

    return {
        "amount": amount,
        "low": round(amount * 0.75, 2),
        "high": round(amount * 1.25, 2),
        "basis": f"{basis} on {platform}"
    }
