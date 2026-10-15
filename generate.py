import json
import os
import re
from typing import List

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")

NOT_ELIGIBLE_REASON = (
    "Pricing, competitor hiring, project postponements, and customer choices are non-refundable under marketplace policy."
)

NON_REFUNDABLE_RE = re.compile(
    r"competitor|hired|another contractor|chose another|price|pricing|expensive|quote|"
    r"changed\s+(?:\w+\s+)?mind|not interested|on hold|postponed|delay|cancelled|"
    r"budget|lost job|financial|no answer|unresponsive|voicemail",
    re.I,
)

INVALID_SAFETY_RE = re.compile(
    r"hired (?:another|a|an|someone|a different)|"
    r"went (?:with|to) (?:a |another |the )?(?:competitor|contractor)|"
    r"project went to|chose (?:another|a different) (?:contractor|company)|"
    r"competitor|too expensive|over (?:our|my) budget|on hold|postponed",
    re.I,
)

SYSTEM_RULES = """You are an expert compliance auditor and policy analyst for home service lead marketplaces (Angi, Thumbtack, Porch, Networx). 

Your job is to deeply analyze the contractor's notes, transcripts, and attached evidence to determine if the lead itself was inherently defective at the time of delivery.

CRITICAL POLICY ANALYSIS PRINCIPLES:
1. FOCUS ON LEAD INTEGRITY (valid = true): The lead is ONLY valid for a refund if the marketplace delivered a broken, unreachable, or fundamentally miscategorized product. This includes:
   - Contact failure: Phone numbers that are permanently disconnected, wrong numbers belonging to entirely different people, or emails that instantly bounce back as undeliverable.
   - Geographic mismatch: The customer's project location is strictly outside the contractor's specified operating zip codes or service radius.
   - Categorization failure: The customer requested a trade or service that the contractor's profile does not offer (e.g., commercial walk-in freezer repair delivered to a residential-only electrical contractor).
   - System anomalies: Duplicate leads sent within the same billing window or obvious bot/spam submissions.

2. FOCUS ON COMMERCIAL & CUSTOMER CHOICES (valid = false): Do NOT approve refunds if the failure occurred *after* a valid connection was established due to normal consumer behavior or commercial negotiations. This includes:
   - Pricing or quote rejections: The homeowner thought the project was too expensive, rejected the estimate, or went with a lower bidder.
   - Competitor selection: The homeowner decided to hire another contractor or company.
   - Postponements & cancellations: The customer changed their mind, put the project on hold indefinitely, or postponed it due to personal reasons.
   - Communication gaps post-connection: The customer answered initially or was reachable, but later became unresponsive, failed to pick up voicemails, or didn't return calls.

INSTRUCTIONS:
- Do not just look for trigger words. Read the full narrative context to understand the root cause of why the job wasn't booked.
- If valid is true, write a detailed, professional, persuasive dispute letter customized to the exact facts ("claim").
- If valid is false, provide a clear, policy-based explanation ("reason") and leave "claim" as an empty string ("").

Respond STRICTLY in valid JSON with no markdown wrappers or backticks, using this exact schema:
{"valid": true or false, "confidence": 0.95, "reason": "...", "claim": "..."}"""

def mime_for(name, given):
    n = (name or "").lower()
    if n.endswith(".mp3"):
        return "audio/mpeg"
    if n.endswith(".wav"):
        return "audio/wav"
    if n.endswith(".pdf"):
        return "application/pdf"
    return given or "application/octet-stream"

def not_eligible(cost, confidence=0.88, reason=NOT_ELIGIBLE_REASON):
    return {
        "valid": False,
        "confidence": confidence,
        "reason": reason,
        "claim": "",
        "amount": cost,
    }

@app.get("/")
def health():
    return {"ok": True}

@app.post("/api/dispute")
@app.post("/api/dispute/")
async def analyze_dispute(
    platform: str = Form(...),
    notes: str = Form(""),
    transcript: str = Form(""),
    cost: float = Form(45.00),
    files: List[UploadFile] = File(default=[]),
):
    text_check = f"{notes} {transcript}"

    try:
        parts = [
            f"Marketplace Platform: {platform}\n"
            f"Lead cost: ${cost:.2f}\n\n"
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

        result_text = response.text.strip()
        if result_text.startswith("```json"):
            result_text = result_text[7:-3].strip()
        elif result_text.startswith("```"):
            result_text = result_text[3:-3].strip()

        data = json.loads(result_text)
        valid = bool(data.get("valid"))

        if valid and INVALID_SAFETY_RE.search(text_check):
            return not_eligible(cost)

        if not valid:
            return not_eligible(
                cost,
                confidence=float(data.get("confidence") or 0.88),
                reason=data.get("reason") or NOT_ELIGIBLE_REASON,
            )

        data["valid"] = True
        data["amount"] = cost
        return data

    except Exception:
        if NON_REFUNDABLE_RE.search(text_check):
            return not_eligible(cost)

        return {
            "valid": True,
            "confidence": 0.5,
            "reason": (
                f"AI review was unavailable, so this was not fully checked against {platform} "
                "policy. Review it before submitting."
            ),
            "claim": (
                f"Official Lead Credit Request\nPlatform: {platform}\n\n"
                f"Requesting credit of ${cost:.2f} due to a contact/system failure.\n\n"
                f"Details:\n{notes}"
            ),
            "amount": cost,
        }

@app.post("/api/dispute/estimate")
@app.post("/api/dispute/estimate/")
async def estimate_lead_cost(data: dict):
    platform = data.get("platform", "Angi")
    text = data.get("text", "").lower()

    base = 50
    if any(w in text for w in ["roof", "shingle", "gutter"]):
        base = 90
    elif any(w in text for w in ["plumb", "leak", "drain", "heater"]):
        base = 60
    elif any(w in text for w in ["electric", "wiring", "panel"]):
        base = 55
    elif re.search(r"hvac|furnace|\bac\b|air condition", text):
        base = 70
    elif any(w in text for w in ["lawn", "landscap", "tree"]):
        base = 40

    multiplier = {"Angi": 1.1, "Thumbtack": 0.9, "Porch": 0.85, "Networx": 1.0}.get(platform, 1.0)
    amount = round(base * multiplier)

    return {
        "amount": float(amount),
        "low": round(amount * 0.75),
        "high": round(amount * 1.25),
        "basis": f"home service standards on {platform}",
    }
