
import json
import logging
import os
import re
from typing import List

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini configuration
API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")

client = genai.Client(api_key=API_KEY) if API_KEY else None

NOT_ELIGIBLE_REASON = (
    "Pricing, competitor hiring, project postponements, and customer "
    "choices are non-refundable under marketplace policy."
)

NON_REFUNDABLE_RE = re.compile(
    r"competitor|hired|another contractor|chose another|price|pricing|"
    r"expensive|quote|changed\s+(?:\w+\s+)?mind|not interested|"
    r"on hold|postponed|delay|cancelled|budget|lost job|financial|"
    r"no answer|unresponsive|voicemail",
    re.I,
)

INVALID_SAFETY_RE = re.compile(
    r"hired\s+(?:another|a|an|someone|a different)|"
    r"went\s+(?:with|to)\s+(?:a\s+|another\s+|the\s+)?"
    r"(?:competitor|contractor)|"
    r"project went to|chose\s+(?:another|a different)\s+"
    r"(?:contractor|company)|competitor|too expensive|"
    r"over\s+(?:our|my)\s+budget|on hold|postponed",
    re.I,
)

SYSTEM_RULES = """
You are an expert lead-credit compliance analyst and professional
dispute writer for home-service lead marketplaces, including Angi,
Thumbtack, Porch, and Networx.

Your task is to evaluate the evidence and determine whether a lead
qualifies for a credit based on the lead's validity.

ELIGIBLE: valid = true, only when evidence supports that the lead
itself was invalid:
- Disconnected, wrong, or invalid phone number.
- Customer outside the contractor's service area.
- Duplicate lead.
- Spam, bot, or demonstrably fake submission.
- Wrong trade or service category.

NOT ELIGIBLE: valid = false:
- Homeowner hired a competitor or another contractor.
- Homeowner rejected the price or quote.
- Project was too expensive or over budget.
- Customer changed their mind, postponed the work, or no longer
  needs the service.
- Customer became unresponsive after a valid connection.
- Other commercial outcomes or customer choices.

EVIDENCE RULES:
- Read all notes and transcript text carefully.
- Consider the uploaded files when provided.
- Interpret statements in context; do not reject a lead just because
  a keyword appears.
- Never invent facts, dates, call attempts, error messages, or
  policy quotations.
- Do not claim an attachment proves something unless it supports it.
- If the evidence is insufficient or ambiguous, do not pretend the
  lead has been proven invalid. Explain the uncertainty.
- A customer choosing another contractor is not an invalid lead.

CLAIM GENERATION:
When valid = true, write a NEW, COMPLETE, professional claim based
on the actual evidence in this individual case.

Do not copy a fixed template. Do not merely insert the contractor's
notes into generic sentences.

Every claim must:
1. Start with a specific Subject line.
2. Include a professional greeting to the marketplace review team.
3. Identify the exact problem demonstrated by the evidence.
4. Explain the facts from the supplied notes, transcript, or files.
5. Explain why those facts support the applicable eligibility reason.
6. Request a review and a credit for the stated lead cost.
7. Use persuasive, natural, professional language.
8. Be concise but sufficiently detailed for a reviewer to understand
   this particular dispute.
9. Vary its structure and wording naturally according to the case.
10. Avoid unsupported statements, invented policy clauses, and
    generic filler.

Use only facts actually supplied. If the evidence identifies a
specific error message, phone issue, service-area mismatch, or wrong
trade, describe that detail precisely.

CONFIDENCE:
Return a number between 0 and 1 representing your estimated
confidence in the classification based on the evidence and policy
clarity. Reduce confidence when facts are missing, conflicting,
or ambiguous. This is an AI estimate, not a verified probability.

OUTPUT:
Return strictly valid JSON with exactly these fields:
{
  "valid": true or false,
  "confidence": 0.0,
  "reason": "case-specific explanation",
  "claim": "custom claim, or empty string"
}

If valid = false, claim MUST be an empty string.
Do not include markdown fences or any text outside the JSON.
"""


def mime_for(name, given):
    n = (name or "").lower()

    if n.endswith(".mp3"):
        return "audio/mpeg"
    if n.endswith(".wav"):
        return "audio/wav"
    if n.endswith(".pdf"):
        return "application/pdf"

    return given or "application/octet-stream"


def not_eligible(
    cost,
    confidence=0.0,
    reason=NOT_ELIGIBLE_REASON,
    ai_completed=False,
):
    return {
        "valid": False,
        "confidence": confidence,
        "reason": reason,
        "claim": "",
        "amount": cost,
        "ai_completed": ai_completed,
    }


def parse_ai_response(response_text):
    """Parse Gemini JSON without silently fabricating a result."""
    text = (response_text or "").strip()

    if text.startswith("```json"):
        text = text[7:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    elif text.startswith("```"):
        text = text[3:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()

    data = json.loads(text)

    if not isinstance(data, dict):
        raise ValueError("Gemini response must be a JSON object")

    if not isinstance(data.get("valid"), bool):
        raise ValueError("Missing or invalid 'valid' field")

    confidence = data.get("confidence")

    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0 <= confidence <= 1
    ):
        raise ValueError("Invalid confidence score")

    if not isinstance(data.get("reason"), str):
        raise ValueError("Missing or invalid reason")

    if not isinstance(data.get("claim"), str):
        raise ValueError("Missing or invalid claim")

    if data["valid"] and not data["claim"].strip():
        raise ValueError("Eligible result must contain a claim")

    if not data["valid"] and data["claim"].strip():
        raise ValueError("Ineligible result must have an empty claim")

    return data


@app.get("/")
def health():
    return {
        "ok": True,
        "ai_configured": client is not None,
        "model": MODEL,
    }


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

    if client is None:
        logger.error("GEMINI_API_KEY is not configured")

        return {
            "valid": False,
            "confidence": 0.0,
            "reason": "AI is unavailable because GEMINI_API_KEY is missing.",
            "claim": "",
            "amount": cost,
            "ai_completed": False,
        }

    try:
        parts = [
            (
                f"Marketplace Platform: {platform}\n"
                f"Lead cost: ${cost:.2f}\n\n"
                f"Contractor Notes:\n{notes or '(none)'}\n\n"
                f"Transcript:\n{transcript or '(none)'}\n\n"
                "Review the evidence and generate a case-specific "
                "classification and claim according to the system rules."
            )
        ]

        for uploaded_file in files:
            content = await uploaded_file.read()

            if content:
                content_type = mime_for(
                    uploaded_file.filename,
                    uploaded_file.content_type,
                )

                parts.append(
                    types.Part.from_bytes(
                        data=content,
                        mime_type=content_type,
                    )
                )

        response = await client.aio.models.generate_content(
            model=MODEL,
            contents=parts,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_RULES,
                response_mime_type="application/json",
                temperature=0.4,
            ),
        )

        data = parse_ai_response(response.text)

        # Conservative safety override for clear commercial outcomes.
        if data["valid"] and INVALID_SAFETY_RE.search(text_check):
            logger.warning(
                "AI eligibility overridden by safety rule"
            )

            return not_eligible(
                cost,
                confidence=data["confidence"],
                reason=NOT_ELIGIBLE_REASON,
                ai_completed=True,
            )

        if not data["valid"]:
            return not_eligible(
                cost,
                confidence=data["confidence"],
                reason=data["reason"],
                ai_completed=True,
            )

        return {
            "valid": True,
            "confidence": data["confidence"],
            "reason": data["reason"],
            "claim": data["claim"],
            "amount": cost,
            "ai_completed": True,
        }

    except Exception as exc:
        logger.exception("Gemini dispute analysis failed")

        # Never present a hardcoded claim as an AI-generated claim.
        # A failed analysis is not proof that the lead is invalid.
        return {
            "valid": False,
            "confidence": 0.0,
            "reason": (
                "AI analysis failed and no eligibility decision was "
                "completed. Please retry after checking the backend "
                f"logs. Error type: {type(exc).__name__}."
            ),
            "claim": "",
            "amount": cost,
            "ai_completed": False,
        }


@app.post("/api/dispute/estimate")
@app.post("/api/dispute/estimate/")
async def estimate_lead_cost(data: dict):
    platform = data.get("platform", "Angi")
    text = data.get("text", "").lower()

    base = 50

    if any(word in text for word in ["roof", "shingle", "gutter"]):
        base = 90
    elif any(word in text for word in ["plumb", "leak", "drain", "heater"]):
        base = 60
    elif any(word in text for word in ["electric", "wiring", "panel"]):
        base = 55
    elif re.search(r"hvac|furnace|\bac\b|air condition", text):
        base = 70
    elif any(word in text for word in ["lawn", "landscap", "tree"]):
        base = 40

    multiplier = {
        "Angi": 1.1,
        "Thumbtack": 0.9,
        "Porch": 0.85,
        "Networx": 1.0,
    }.get(platform, 1.0)

    amount = round(base * multiplier)

    return {
        "amount": float(amount),
        "low": round(amount * 0.75),
        "high": round(amount * 1.25),
        "basis": f"home service standards on {platform}",
    }
