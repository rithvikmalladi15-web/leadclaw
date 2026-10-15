import os
import json
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from google import genai
from groq import Groq

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Autonomous AI Policy Evaluation Prompt
AUTONOMOUS_POLICY_PROMPT = """
You are LeadClaw's Autonomous Lead Dispute Evaluation Engine. 
Target Platform: {platform}

Your job is to analyze the provided contractor call notes, transcripts, or uploaded files and autonomously decide whether the dispute complies with standard platform refund and credit policies.

Evaluate the core circumstances:
1. Does the situation describe a true platform failure or ineligible lead delivery (e.g., dead/invalid contact info, out-of-service area, confirmed duplicate lead, spam/bot activity, or incorrect service category)? If so, the AI should deem it VALID.
2. Does the situation describe normal customer behavior or contractor execution issues after lead delivery (e.g., price disputes, customer changing their mind, hiring a competitor, or lack of persistent contact attempts)? If so, the AI should deem it INVALID.

Return JSON ONLY with this exact structure:

If the AI decides it is INVALID / NON-REFUNDABLE:
{{
  "valid": false,
  "reason": "Autonomous AI analysis: Explain precisely why support will reject this claim based on policy principles."
}}

If the AI decides it is VALID:
{{
  "valid": true,
  "statement": "Professional, formal dispute claim statement formulated for official support submission."
}}
"""

@app.get("/")
def health_check():
    return {"status": "LeadClaw Autonomous Engine Active"}

@app.post("/api/dispute")
@app.post("/api/dispute/")
async def process_dispute(
    platform: str = Form(...),
    notes: Optional[str] = Form(None),
    context: Optional[str] = Form(""),
    file: Optional[UploadFile] = File(None)
):
    if not notes and not file:
        raise HTTPException(status_code=400, detail="Please provide either notes or attach an audio/PDF file.")

    contents = []
    formatted_prompt = AUTONOMOUS_POLICY_PROMPT.format(platform=platform)
    text_prompt = f"Contractor Notes: {notes or 'N/A'}\nAdditional Context: {context or 'None'}\n\n" + formatted_prompt

    if file:
        file_bytes = await file.read()
        mime_type = file.content_type or "application/octet-stream"
        contents.append({"mime_type": mime_type, "data": file_bytes})

    contents.append(text_prompt)

    # Attempt 1: Gemini 3.5 Flash Autonomous Analysis
    try:
        response = gemini_client.models.generate_content(
            model='gemini-3.5-flash',
            contents=contents,
            config={'response_mime_type': 'application/json'}
        )
        return json.loads(response.text)

    except Exception as gemini_error:
        # Fallback to Groq Qwen3-4b if no file attachment is present
        if not file:
            try:
                fallback_response = groq_client.chat.completions.create(
                    model="qwen-3-4b",
                    messages=[{"role": "user", "content": text_prompt}],
                    response_format={"type": "json_object"}
                )
                return json.loads(fallback_response.choices[0].message.content)
            except Exception as groq_error:
                raise HTTPException(status_code=500, detail=f"Gemini error: {str(gemini_error)} | Groq error: {str(groq_error)}")
        else:
            raise HTTPException(status_code=500, detail=f"Gemini processing error: {str(gemini_error)}")
@app.post("/api/dispute/estimate")
@app.post("/api/dispute/estimate/")
async def estimate_lead_cost(data: dict):
    @app.post("/api/dispute/estimate")
@app.post("/api/dispute/estimate/")
async def estimate_lead_cost(data: dict):
    platform = data.get("platform", "Angi")
    text = data.get("text", "").lower()

    # Simple dynamic estimation logic
    base = 50
    if any(w in text for w in ["roof", "shingle", "gutter"]): base = 90
    elif any(w in text for w in ["plumb", "leak", "drain", "heater"]): base = 60
    elif any(w in text for w in ["electric", "wiring", "panel"]): base = 55
    elif any(w in text for w in ["hvac", "furnace", "ac"]): base = 70
    elif any(w in text for w in ["lawn", "landscap", "tree"]): base = 40

    multiplier = {"Angi": 1.1, "Thumbtack": 0.9, "Porch": 0.85, "Networx": 1.0}.get(platform, 1.0)
    amount = round(base * multiplier)

    return {
        "amount": float(amount),
        "low": round(amount * 0.75),
        "high": round(amount * 1.25),
        "basis": f"home service standards on {platform}"
    }
