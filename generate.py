import json
import os
import re
from typing import List

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

app = FastAPI()

# CORS: allow the Cloudflare Pages frontend (and any other origin).
# allow_credentials must stay False when allow_origins is ["*"].
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")

NOT_ELIGIBLE_REASON = (
    "Pricing, competitor hiring, and customer choices are non-refundable under marketplace policy."
)

# Keyword check used by the fallback when Gemini is unavailable.
NON_REFUNDABLE_RE = re.compile(
    r"competitor|hired|another contractor|chose another|price|pricing|expensive|quote|"
    r"changed\s+(?:\w+\s+)?mind|not interested",
    re.I,
)

# Competitor hiring is never refundable, even if the model says otherwise.
COMPETITOR_RE = re.compile(
    r"hired (?:another|a|an|someone|a different)|"
    r"went (?:with|to) (?:a |another |the )?(?:competitor|contractor)|"
    r"project went to|chose (?:another|a different) (?:contractor|company)|"
    r"competitor",
    re.I,
)

SYSTEM_RULES = """You are an expert compliance auditor for home service lead marketplaces
(Angi, Thumbtack, Porch, Networx). Decide whether a lead qualifies for a credit refund based on
strict platform policies.

ELIGIBLE (valid = true) only when the LEAD ITSELF was invalid:
- disconnected, wrong or invalid phone number
- customer outside the contractor's service area
- duplicate lead
- spam, bot or fake submission
- wrong trade category / service the contractor does not offer

NOT ELIGIBLE (valid = false) whenever the homeowner made a choice or the outcome was commercial:
- the homeowner hired a competitor or another contractor, or the project went to a competitor
- the homeowner rejected the price or quote, or said it was too expensive
- the customer changed their mind, no longer needs the work, or will do it themselves
- the homeowner is unresponsive or did not answer after a valid connection was made

HARD RULE: if the notes, transcript or any attached file says a competitor or another contractor
was hired, or that the price or quote was rejected, valid MUST be false, even if the notes also
list an eligible reason. Never produce a dispute claim for these cases.

If valid is false: "reason" is one or two plain sentences explaining why, and "claim" MUST be "".
If valid is true: "reason" is a short explanation and "claim" is a formal, professional dispute
statement ready to copy and paste, starting with a Subject line, naming the platform, stating the
credit amount, and using only facts from the provided material. Never invent facts.

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

        # Gemini 3.5 Flash handles the core policy reasoning
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
        # Clean potential markdown wrappers if the model includes them
        if result_text.startswith("```json"):
            result_text = result_text[7:-3].strip()
        elif result_text.startswith("```"):
            result_text = result_text[3:-3].strip()

        data = json.loads(result_text)
        valid = bool(data.get("valid"))

        # Safety net: never produce a claim when a competitor was hired.
        if valid and COMPETITOR_RE.search(text_check):
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
        # Fallback logic if Gemini hits an exception or rate limit
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
