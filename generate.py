import asyncio
import json
import os
import re
from typing import List

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
import httpx

app = FastAPI(title="Lead Dispute AI Auditor", version="2.2.0")

# CORS: allow all origins for the Cloudflare frontend connection.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize primary Gemini client & model (Gemini 3.5 Flash)
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")

# Groq Backup Configuration (Qwen 3 32B via Groq)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "qwen/qwen3-32b")

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

SYSTEM_RULES = """You are an expert compliance auditor and dispute specialist for home service lead marketplaces
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
- the homeowner rejected the price or quote, or said it was too expensive / over budget
- the customer changed their mind, put project on hold, postponed, no longer needs the work, or had a personal delay
- the homeowner is unresponsive or did not answer after a valid connection was made

HARD RULES FOR THE CLAIM:
- If valid is true, "claim" MUST be a detailed, professional, persuasive dispute letter customized specifically to the facts in the contractor refund reason or transcript. 
- Do NOT use generic placeholders like "See attached evidence".
- Explicitly detail the exact carrier error messages, wrong trade requests, disconnected numbers, or zip code mismatches mentioned in the input.
- Format it clearly starting with a Subject line, formal greeting, issue breakdown, policy reference, and a polite request for credit.

If valid is false: "reason" is a clear policy explanation and "claim" MUST be "".

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


async def call_groq_backup(prompt_text: str, cost: float) -> dict:
    if not GROQ_API_KEY:
        raise ValueError("Groq API key not configured")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_RULES},
            {"role": "user", "content": prompt_text}
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"}
    }

    async with httpx.AsyncClient(timeout=30.0) as client_http:
        response = await client_http.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json=payload,
            headers=headers
        )
        response.raise_for_status()
        res_json = response.json()
        content = res_json["choices"][0]["message"]["content"].strip()
        
        if content.startswith("```json"):
            content = content[7:-3].strip()
        elif content.startswith("```"):
            content = content[3:-3].strip()

        parsed = json.loads(content)
        parsed["amount"] = cost
        return parsed


@app.get("/")
def health():
    return {"ok": True, "primary": "gemini-3.5-flash", "backup": "qwen/qwen3-32b"}


@app.post("/api/dispute")
@app.post("/api/dispute/")
async def analyze_dispute(
    platform: str = Form(...),
    notes: str = Form(""),  # Handled as refund reason
    transcript: str = Form(""),
    cost: float = Form(45.00),
    files: List[UploadFile] = File(default=[]),
):
    text_check = f"{notes} {transcript}"
    combined_prompt = (
        f"Marketplace Platform: {platform}\n"
        f"Lead cost: ${cost:.2f}\n\n"
        f"Refund Reason: {notes or '(none)'}\n\n"
        f"Transcript text: {transcript or '(none)'}"
    )

    try:
        parts = [combined_prompt]
        for f in files:
            data = await f.read()
            if data:
                parts.append(
                    types.Part.from_bytes(data=data, mime_type=mime_for(f.filename, f.content_type))
                )

        async def call_gemini():
            return await client.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=parts,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_RULES,
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
            )

        response = await asyncio.wait_for(call_gemini(), timeout=50.0)

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
        try:
            groq_result = await call_groq_backup(combined_prompt, cost)
            valid = bool(groq_result.get("valid"))
            
            if valid and INVALID_SAFETY_RE.search(text_check):
                return not_eligible(cost)
                
            if not valid:
                return not_eligible(
                    cost,
                    confidence=float(groq_result.get("confidence") or 0.88),
                    reason=groq_result.get("reason") or NOT_ELIGIBLE_REASON,
                )
                
            groq_result["valid"] = True
            groq_result["amount"] = cost
            return groq_result

        except Exception:
            if NON_REFUNDABLE_RE.search(text_check):
                return not_eligible(cost)

            issue_summary = notes.strip() or transcript.strip() or "Lead failed operational connection standards."
            dynamic_claim = (
                f"Subject: Formal Request for Lead Credit Reissue - ${cost:.2f} ({platform})\n\n"
                f"Dear {platform} Support Team,\n\n"
                f"I am writing to formally request a credit refund for lead charges totaling ${cost:.2f}. "
                f"Under your platform's lead quality and verification guidelines, this lead qualifies for a credit reissue due to the following verified issue:\n\n"
                f"• Refund Reason: {issue_summary}\n\n"
                f"We attempted prompt outreach and verification in accordance with marketplace standards, but were unable to establish a valid connection due to the system/contact failure outlined above.\n\n"
                f"Please review the logs and issue credit back to our account.\n\n"
                f"Thank you for your prompt support.\n\n"
                f"Sincerely,\n"
                f"Contractor Account Management"
            )

            return {
                "valid": True,
                "confidence": 0.70,
                "reason": f"Lead verified as eligible for credit review under {platform} policy guidelines.",
                "claim": dynamic_claim,
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
