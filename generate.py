import json
import logging
import os
import re
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("leadclaw")

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(title="LeadClaw API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # lock this down to your Cloudflare Pages domain in production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── AI clients ─────────────────────────────────────────────────────────────
# Set GEMINI_API_KEY in your Render environment variables.
# Optionally set GEMINI_MODEL (default: gemini-2.5-flash).
# For the Qwen model the frontend can select, the request goes to Gemini
# unless you host Qwen separately and add a second client here.
gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
GEMINI_MODEL  = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
QWEN_ENDPOINT = os.environ.get("QWEN_ENDPOINT", "")  # e.g. http://localhost:11434/api/generate

# ── Constants ──────────────────────────────────────────────────────────────
NOT_ELIGIBLE_REASON = (
    "Pricing objections, competitor hiring, project postponements, and customer "
    "choices are non-refundable under marketplace policy."
)

# Catches non-refundable phrases in the fallback (when AI is unavailable).
NON_REFUNDABLE_RE = re.compile(
    r"competitor|hired|another contractor|chose another|price|pricing|expensive|quote|"
    r"changed\s+(?:\w+\s+)?mind|not interested|on hold|postponed|delay|cancelled|"
    r"budget|lost job|financial|no answer|unresponsive|voicemail",
    re.I,
)

# Safety net: overrides Gemini if it accidentally marks a clear non-refundable case as valid.
INVALID_SAFETY_RE = re.compile(
    r"hired (?:another|a|an|someone|a different)|"
    r"went (?:with|to) (?:a |another |the )?(?:competitor|contractor)|"
    r"project went to|chose (?:another|a different) (?:contractor|company)|"
    r"competitor|too expensive|over (?:our|my) budget|on hold|postponed",
    re.I,
)

PLATFORM_EMAIL = {
    "Angi":      "leadcredits@angi.com",
    "Thumbtack": "support@thumbtack.com",
    "Porch":     "prosupport@porch.com",
    "Networx":   "support@networx.com",
}
PLATFORM_POLICY = {
    "Angi":      "Angi Lead Credit Policy",
    "Thumbtack": "Thumbtack Lead Price Credit Terms",
    "Porch":     "Porch Pro Lead Guarantee",
    "Networx":   "Networx Lead Dispute Terms",
}

# ── Gemini system prompt ────────────────────────────────────────────────────
SYSTEM_RULES = """You are an expert compliance auditor for home service lead marketplaces
(Angi, Thumbtack, Porch, Networx). Your job is to decide whether a lead qualifies for a
credit refund and, if so, to write a professional dispute letter.

ELIGIBLE FOR REFUND (valid = true) — only when the LEAD ITSELF was defective:
- Disconnected, wrong, or permanently invalid phone number
- Customer is strictly outside the contractor's service area or ZIP codes
- Duplicate lead delivered within the same billing window
- Spam, bot, or fake submission with no real customer
- Wrong trade category: inquiry is for a service the contractor does not offer

NOT ELIGIBLE FOR REFUND (valid = false) — whenever the outcome was a consumer choice:
- Homeowner hired a competitor or another contractor
- Homeowner rejected the price, said it was too expensive, or refused a quote
- Customer changed their mind, put the project on hold, postponed, or cancelled
- Homeowner was unresponsive or stopped returning calls after a real first contact

HARD RULES:
1. If the notes, transcript, or any file mention a competitor being hired or a price/quote
   rejection, valid MUST be false — regardless of other content.
2. If valid is true, "claim" MUST be a detailed, persuasive, professional dispute letter
   written specifically around the facts provided. Start with a Subject line, use a formal
   greeting, describe the exact problem (specific error messages, invalid number, ZIP mismatch,
   etc.), cite the platform policy by name, and close with a polite but firm credit request.
   Never use placeholders like "See attached evidence" — reference the actual facts.
3. If valid is false, "reason" is one or two plain-language sentences explaining why.
   "claim" MUST be an empty string.

Respond ONLY in valid JSON with no markdown fences, matching this exact schema:
{"valid": true or false, "confidence": 0.95, "reason": "...", "claim": "..."}"""


# ── Helpers ────────────────────────────────────────────────────────────────
def mime_for(name: str, given: str) -> str:
    n = (name or "").lower()
    if n.endswith(".mp3"):  return "audio/mpeg"
    if n.endswith(".wav"):  return "audio/wav"
    if n.endswith(".pdf"):  return "application/pdf"
    return given or "application/octet-stream"


def not_eligible(cost: float, confidence: float = 0.88, reason: str = NOT_ELIGIBLE_REASON) -> dict:
    return {"valid": False, "confidence": confidence, "reason": reason, "claim": "", "amount": cost}


async def run_gemini(parts: list, model: str = GEMINI_MODEL) -> dict:
    """Call Gemini and return the parsed JSON response."""
    response = await gemini_client.aio.models.generate_content(
        model=model,
        contents=parts,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_RULES,
            response_mime_type="application/json",
            temperature=0.1,
        ),
    )
    raw = (response.text or "").strip()
    # Extract the JSON object even if there's stray text around it.
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in model output: {raw[:200]}")
    return json.loads(raw[start:end + 1])


async def run_qwen(prompt_text: str) -> dict:
    """
    Call a local Qwen model via Ollama's generate endpoint.
    Requires QWEN_ENDPOINT env var, e.g. http://localhost:11434/api/generate
    Falls back to Gemini if QWEN_ENDPOINT is not set.
    """
    if not QWEN_ENDPOINT:
        raise ValueError("QWEN_ENDPOINT is not configured. Set it in your Render environment.")

    import aiohttp
    payload = {
        "model": "qwen3:4b",
        "prompt": SYSTEM_RULES + "\n\n" + prompt_text,
        "stream": False,
        "format": "json",
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(QWEN_ENDPOINT, json=payload, timeout=aiohttp.ClientTimeout(total=90)) as resp:
            resp.raise_for_status()
            data = await resp.json()
    raw = (data.get("response") or "").strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON from Qwen: {raw[:200]}")
    return json.loads(raw[start:end + 1])


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {"ok": True, "service": "LeadClaw API", "version": "3.0.0", "model": GEMINI_MODEL}


@app.get("/api/ai-check")
async def ai_check():
    """
    Debug route. Open in a browser to confirm Gemini is reachable.
    Remove or protect this route before going to production.
    """
    try:
        r = await gemini_client.aio.models.generate_content(
            model=GEMINI_MODEL, contents="Reply with the single word: OK"
        )
        return {"ok": True, "model": GEMINI_MODEL, "reply": (r.text or "").strip()}
    except Exception as e:
        return {"ok": False, "model": GEMINI_MODEL, "error": f"{type(e).__name__}: {e}"}


@app.post("/api/dispute")
@app.post("/api/dispute/")
async def analyze_dispute(
    platform:   str   = Form(...),
    notes:      str   = Form(""),
    transcript: str   = Form(""),
    cost:       float = Form(45.00),
    model:      str   = Form("gemini"),         # "gemini" or "qwen"
    files:      List[UploadFile] = File(default=[]),
):
    """
    Main dispute evaluation endpoint.
    Accepts multipart/form-data from the frontend.
    Returns JSON: {valid, confidence, reason, claim, amount}
    """
    text_check = f"{notes} {transcript}"
    policy     = PLATFORM_POLICY.get(platform, f"{platform} lead policy")

    try:
        prompt_text = (
            f"Marketplace Platform: {platform}\n"
            f"Policy: {policy}\n"
            f"Lead Cost: ${cost:.2f}\n\n"
            f"Contractor Notes:\n{notes or '(none)'}\n\n"
            f"Transcript:\n{transcript or '(none)'}"
        )

        if model == "qwen":
            data = await run_qwen(prompt_text)
        else:
            # Build multimodal parts for Gemini
            parts = [prompt_text]
            for f in files:
                raw = await f.read()
                if raw:
                    parts.append(types.Part.from_bytes(
                        data=raw, mime_type=mime_for(f.filename, f.content_type)
                    ))
            data = await run_gemini(parts)

        valid = bool(data.get("valid"))

        # Safety net: catch hallucinated valid=True for clear non-refundable scenarios
        if valid and INVALID_SAFETY_RE.search(text_check):
            logger.info("Safety net overrode valid=True for non-refundable phrase in: %s", text_check[:120])
            return not_eligible(cost)

        if not valid:
            return not_eligible(
                cost,
                confidence=float(data.get("confidence") or 0.88),
                reason=data.get("reason") or NOT_ELIGIBLE_REASON,
            )

        claim = str(data.get("claim") or "").strip()
        if not claim:
            raise ValueError("Model returned valid=True but produced no claim text.")

        return {
            "valid":      True,
            "confidence": float(data.get("confidence") or 0.9),
            "reason":     data.get("reason", ""),
            "claim":      claim,
            "amount":     cost,
        }

    except Exception as e:
        logger.exception("Dispute review failed (model=%s): %s", model, e)
        # Still block obvious non-refundable reasons even when AI is down
        if NON_REFUNDABLE_RE.search(text_check):
            return not_eligible(cost)
        raise HTTPException(
            status_code=502,
            detail=f"AI review failed ({type(e).__name__}). Check the server logs and verify your GEMINI_API_KEY.",
        )


@app.post("/api/dispute/estimate")
@app.post("/api/dispute/estimate/")
async def estimate_lead_cost(data: dict = Body(...)):
    """
    Quick keyword-based lead cost estimator.
    Returns {amount, low, high, basis}.
    """
    platform = data.get("platform", "Angi")
    text     = data.get("text", "").lower()

    base, basis = 50, "general home services"
    if any(w in text for w in ["roof", "shingle", "gutter"]):
        base, basis = 90, "roofing"
    elif any(w in text for w in ["plumb", "leak", "drain", "water heater", "pipe"]):
        base, basis = 60, "plumbing"
    elif any(w in text for w in ["electric", "wiring", "panel", "outlet"]):
        base, basis = 55, "electrical"
    elif re.search(r"hvac|furnace|\bac\b|air condition|heating", text):
        base, basis = 70, "HVAC"
    elif any(w in text for w in ["lawn", "landscap", "tree", "fence"]):
        base, basis = 40, "landscaping"

    mult   = {"Angi": 1.1, "Thumbtack": 0.9, "Porch": 0.85, "Networx": 1.0}.get(platform, 1.0)
    amount = round(base * mult, 2)
    return {
        "amount": amount,
        "low":    round(amount * 0.75, 2),
        "high":   round(amount * 1.25, 2),
        "basis":  f"{basis} on {platform}",
    }


@app.post("/api/feedback")
@app.post("/api/feedback/")
async def feedback(payload: dict = Body(...)):
    """
    Receives feedback from the in-app modal.
    Logs it to Render's console. Wire to SendGrid / Slack / Notion in production.
    """
    logger.info(
        "FEEDBACK | type=%s | priority=%s | workspace=%s | email=%s | msg=%.200s",
        payload.get("type", "?"),
        payload.get("priority", "?"),
        payload.get("workspace", "?"),
        payload.get("email", "anonymous"),
        payload.get("message", ""),
    )
    # TODO: forward via email (SendGrid/Resend), post to Slack webhook, or insert into Notion DB
    return {"ok": True}
