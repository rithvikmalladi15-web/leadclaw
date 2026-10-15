import json
import logging
import os
import re
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("leadclaw")

app = FastAPI(title="LeadClaw API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
MODEL  = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

NOT_ELIGIBLE_REASON = (
    "Pricing, competitor hiring, project postponements, and customer choices "
    "are non-refundable under marketplace policy."
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

SYSTEM_RULES = """You are an expert compliance auditor and lead marketplace policy specialist
for Angi, Thumbtack, Porch, and Networx.

ELIGIBLE FOR REFUND (valid = true) — only when the LEAD ITSELF was defective:
- Permanently disconnected, wrong, or invalid phone number
- Customer is strictly outside the contractor's service area or zip codes
- Duplicate lead delivered within the same billing window
- Spam, bot, or fake submission
- Wrong trade category: the inquiry is for a service the contractor does not offer

NOT ELIGIBLE FOR REFUND (valid = false) — whenever the outcome was a consumer or commercial choice:
- Homeowner hired a competitor or another contractor, or the project went elsewhere
- Homeowner rejected the price, said it was too expensive, or would not accept a quote
- Customer changed their mind, put the project on hold, postponed, cancelled, or no longer needs work
- Homeowner was unresponsive or stopped returning calls after a valid initial connection

HARD RULES:
- If the notes, transcript, or files mention a competitor being hired or a price/quote rejection,
  valid MUST be false — even if the notes also list an eligible reason. Never produce a claim
  for these cases.
- If valid is true, "claim" MUST be a detailed, persuasive, professional dispute letter tailored
  to the specific facts provided. Start with a Subject line, formal greeting, issue breakdown,
  exact policy reference, and a polite but firm credit request. Do NOT use placeholders like
  "See attached evidence" — name the actual facts.
- If valid is false, "reason" is one or two plain-language sentences and "claim" MUST be "".

Respond STRICTLY in valid JSON with no markdown fences, using exactly this schema:
{"valid": true or false, "confidence": 0.95, "reason": "...", "claim": "..."}"""


def mime_for(name, given):
    n = (name or "").lower()
    if n.endswith(".mp3"):  return "audio/mpeg"
    if n.endswith(".wav"):  return "audio/wav"
    if n.endswith(".pdf"):  return "application/pdf"
    return given or "application/octet-stream"


def not_eligible(cost, confidence=0.88, reason=NOT_ELIGIBLE_REASON):
    return {"valid": False, "confidence": confidence,
            "reason": reason, "claim": "", "amount": cost}


# ── health ──────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {"ok": True, "service": "LeadClaw API", "model": MODEL}


@app.get("/api/ai-check")
async def ai_check():
    """Debug: open in browser to confirm Gemini is reachable. Remove before going to production."""
    try:
        r = await client.aio.models.generate_content(
            model=MODEL, contents="Reply with the single word: OK")
        return {"ok": True, "model": MODEL, "reply": (r.text or "").strip()}
    except Exception as e:
        return {"ok": False, "model": MODEL, "error": f"{type(e).__name__}: {e}"}


# ── dispute evaluation ───────────────────────────────────────────────────────

@app.post("/api/dispute")
@app.post("/api/dispute/")
async def analyze_dispute(
    platform:   str  = Form(...),
    notes:      str  = Form(""),
    transcript: str  = Form(""),
    cost:       float = Form(45.00),
    files:      List[UploadFile] = File(default=[]),
):
    text_check = f"{notes} {transcript}"

    try:
        parts = [
            f"Marketplace Platform: {platform}\n"
            f"Lead Cost: ${cost:.2f}\n\n"
            f"Contractor Notes:\n{notes or '(none)'}\n\n"
            f"Transcript:\n{transcript or '(none)'}"
        ]
        for f in files:
            data = await f.read()
            if data:
                parts.append(types.Part.from_bytes(
                    data=data, mime_type=mime_for(f.filename, f.content_type)))

        response = await client.aio.models.generate_content(
            model=MODEL,
            contents=parts,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_RULES,
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )

        raw = (response.text or "").strip()
        # Robust JSON extraction: handles stray text or markdown fences
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("Model returned no JSON object.")
        data = json.loads(raw[start:end+1])
        valid = bool(data.get("valid"))

        # Safety net: override a hallucinated "valid" when keywords say otherwise
        if valid and INVALID_SAFETY_RE.search(text_check):
            logger.info("Safety net overrode valid=True for non-refundable phrase.")
            return not_eligible(cost)

        if not valid:
            return not_eligible(
                cost,
                confidence=float(data.get("confidence") or 0.88),
                reason=data.get("reason") or NOT_ELIGIBLE_REASON,
            )

        claim = str(data.get("claim") or "").strip()
        if not claim:
            raise ValueError("Model marked the lead valid but produced no claim text.")

        return {"valid": True, "confidence": float(data.get("confidence") or 0.9),
                "reason": data.get("reason", ""), "claim": claim, "amount": cost}

    except Exception as e:
        logger.exception("Gemini dispute review failed: %s", e)
        if NON_REFUNDABLE_RE.search(text_check):
            return not_eligible(cost)
        raise HTTPException(
            status_code=502,
            detail=f"AI review failed ({type(e).__name__}). Check the server logs.")


# ── lead cost estimate ───────────────────────────────────────────────────────

@app.post("/api/dispute/estimate")
@app.post("/api/dispute/estimate/")
async def estimate_lead_cost(data: dict = Body(...)):
    platform = data.get("platform", "Angi")
    text     = data.get("text", "").lower()

    base, basis = 50, "general home services"
    if any(w in text for w in ["roof", "shingle", "gutter"]):
        base, basis = 90, "roofing"
    elif any(w in text for w in ["plumb", "leak", "drain", "water heater"]):
        base, basis = 60, "plumbing"
    elif any(w in text for w in ["electric", "wiring", "panel", "outlet"]):
        base, basis = 55, "electrical"
    elif re.search(r"hvac|furnace|\bac\b|air condition|heating", text):
        base, basis = 70, "HVAC"
    elif any(w in text for w in ["lawn", "landscap", "tree", "fence"]):
        base, basis = 40, "landscaping"

    mult   = {"Angi":1.1,"Thumbtack":0.9,"Porch":0.85,"Networx":1.0}.get(platform, 1.0)
    amount = round(base * mult, 2)
    return {"amount":amount, "low":round(amount*.75,2),
            "high":round(amount*1.25,2), "basis":f"{basis} on {platform}"}


# ── feedback ─────────────────────────────────────────────────────────────────

@app.post("/api/feedback")
@app.post("/api/feedback/")
async def feedback(payload: dict = Body(...)):
    logger.info(
        "FEEDBACK [%s / %s] from %s: %s",
        payload.get("type", "?"),
        payload.get("priority", "?"),
        payload.get("email", "anonymous"),
        (payload.get("message") or "")[:200],
    )
    # TODO: forward to email (SendGrid/Resend), Notion DB, or Slack webhook
    return {"ok": True}
