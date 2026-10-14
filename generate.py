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

STRICT_SYSTEM_PROMPT = """
You are LeadClaw's Dispute Validation Engine. Your task is to rigorously validate contractor lead disputes against official platform credit policies.

CRITICAL REFUND POLICY RULES:
- VALID CLAIMS (ALLOW & FORMAT STATEMENT):
  * Phone number disconnected, invalid, fax machine, or non-working number.
  * Homeowner address/zip code is outside the contractor's explicit service radius.
  * Duplicate lead received within the platform's re-bill window.
  * Spam, bot, or automated test submission.
  * Wrong service category requested.

- NON-REFUNDABLE CLAIMS (STRICTLY BLOCK):
  * Homeowner changed their mind or decided not to do the project.
  * Homeowner hired a competitor or already completed the job.
  * Homeowner rejected contractor's price or quote.
  * Contractor called once and got no answer.

Return JSON ONLY with this exact structure:
If NON-REFUNDABLE:
{
  "valid": false,
  "reason": "Explicit explanation of why support will reject this claim based on policy rules."
}

If VALID:
{
  "valid": true,
  "statement": "Formal claim statement formatted specifically for submission."
}
"""

@app.get("/")
def health_check():
    return {"status": "LeadClaw Multimodal Engine Active"}

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
    text_prompt = f"Target Platform: {platform}\nContractor Notes: {notes or 'N/A'}\nAdditional Context: {context or 'None'}\n\n" + STRICT_SYSTEM_PROMPT

    if file:
        file_bytes = await file.read()
        mime_type = file.content_type or "application/octet-stream"
        contents.append({"mime_type": mime_type, "data": file_bytes})

    contents.append(text_prompt)

    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=contents,
            config={'response_mime_type': 'application/json'}
        )
        return json.loads(response.text)

    except Exception as gemini_error:
        if not file:
            try:
                fallback_response = groq_client.chat.completions.create(
                    model="qwen-2.5-32b",
                    messages=[{"role": "user", "content": text_prompt}],
                    response_format={"type": "json_object"}
                )
                return json.loads(fallback_response.choices[0].message.content)
            except Exception as groq_error:
                raise HTTPException(status_code=500, detail=f"Gemini error: {str(gemini_error)} | Groq error: {str(groq_error)}")
        else:
            raise HTTPException(status_code=500, detail=f"Gemini processing error: {str(gemini_error)}")
