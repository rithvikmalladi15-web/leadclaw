import asyncio
import json
import os
import re
from typing import List

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

app = FastAPI(title="Lead Dispute AI Auditor", version="1.1.0")

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini client & model
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

NOT_ELIGIBLE_REASON = (
    "Pricing, competitor hiring, project postponements, and customer choices are non-refundable under marketplace policy."
)

# Expanded keyword check for fallbacks & safety filters
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

SYSTEM_RULES = """You are an expert compliance auditor and dispute specialist for home service lead marketplaces
(Angi, Thumbtack, Porch, Networx). Evaluate whether a contractor lead qualifies for a credit refund based on
strict platform terms of service.

ELIGIBLE (valid = true) ONLY when the lead itself contained an operational/system failure:
- Disconnected, wrong, or permanently invalid phone number / carrier error ("number not in service")
- Customer explicitly located outside the contractor's registered service area/zip code
- Verified duplicate lead charged within the policy window
- Spam, bot, or completely fake/malicious submission
- Wrong trade category mismatch (e.g., customer requested roofing when contractor only handles plumbing)

NOT ELIGIBLE (valid = false) whenever the homeowner made an active choice or the outcome was commercial:
- The homeowner hired a competitor, another contractor, or awarded the project elsewhere
- The homeowner rejected the price, quote, or stated the service was too expensive / over budget
- The customer changed their mind, put the project on hold, postponed, or no longer needs the work
- The homeowner was unresponsive, did not answer calls, or left voicemails after a valid connection was established

HARD RULES FOR THE CLAIM PARAGRAPH:
- If valid is true, "claim" MUST be an exceptionally polished, professional, and persuasive dispute letter tailored precisely to the facts in the contractor notes or transcript.
- Control the tone to be firm, polite, business-like, and strictly policy-aligned.
- Do NOT use generic placeholders like "See attached evidence".
- Explicitly detail exact error messages, wrong trade requests, disconnected numbers, or zip code mismatches.
- Format it cleanly starting with a Subject line, formal greeting, issue breakdown referencing platform terms, and a professional request for credit reissue.

If valid is false: "reason" must be a clear policy explanation and "claim" MUST be an empty string "".

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
    return {"status": "healthy", "service": "lead-dispute-auditor"}


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

        async def call_gemini():
            return await client.aio.models.generate_content(
                model=MODEL,
                contents=parts,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_RULES,
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
            )

        # 50-second timeout guardrail for API response hangs
        response = await asyncio.wait_for(call_gemini(), timeout=50.0)

        result_text = response.text.strip()
        if result_text.startswith("```json"):
            result_text = result_text[7:-3].strip()
        elif result_text.startswith("```"):
            result_text = result_text[3:-3].strip()

        data = json.loads(result_text)
        valid = bool(data.get("valid"))

        # Safety net override for non-refundable markers
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
                f"AI review was unavailable or timed out, so this was not fully checked against {platform} "
                "policy. Review it before submitting."
            ),
            "claim": (
                f"Subject: Formal Request for Lead Credit Reissue - ${cost:.2f}\n\n"
                f"Dear {platform} Support Team,\n\n"
                f"I am writing to formally request a credit refund for lead charges totaling ${cost:.2f} due to systemic connection and qualification failures.\n\n"
                f"Issue Details:\n{notes or transcript or 'Lead data failed to meet operational connection criteria.'}\n\n"
                "Under marketplace guidelines regarding invalid or unreachable contact details, I respectfully request that this credit be reviewed and reinstated to my account.\n\n"
                "Thank you for your prompt assistance.\n\n"
                "Sincerely,\nContractor Account Management"
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
