import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from groq import Groq

app = FastAPI()

# Enable CORS for Cloudflare Pages frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

class DisputeRequest(BaseModel):
    platform: str
    notes: str
    context: str = ""

@app.get("/")
def health_check():
    return {"status": "LeadClaw Engine Active"}

@app.post("/api/dispute")
async def process_dispute(data: DisputeRequest):
    if not data.notes.strip():
        raise HTTPException(status_code=400, detail="Call notes are required.")

    prompt = f"""
You are LeadClaw's Dispute Validation Engine.
Target Platform: {data.platform}
Contractor Call Notes: "{data.notes}"
Additional Context: "{data.context}"

Task:
1. Evaluate whether these notes describe valid lead dispute criteria according to official {data.platform} lead credit policies.
   - VALID CRITERIA: Invalid or disconnected phone number, out of service area or zip code, duplicate lead received within platform window, spam or bot account, wrong service category requested.
   - INVALID CRITERIA: Customer did not buy, customer said price was too high, customer hired a competitor, customer changed their mind, contractor was unable to reach customer on a single attempt.

2. Return JSON ONLY with this exact structure (no markdown code blocks, raw JSON):
If INVALID:
{{
  "valid": false,
  "reason": "Clear explanation of why {data.platform} support will reject this specific claim based on their policy."
}}

If VALID:
{{
  "valid": true,
  "statement": "Professional, formal dispute claim statement formatted for {data.platform} support submission."
}}
"""

    # Attempt 1: Primary Engine (Gemini 2.5 Flash)
    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config={
                'response_mime_type': 'application/json'
            }
        )
        return json.loads(response.text)

    except Exception as gemini_error:
        # Attempt 2: Backup Engine (Groq Qwen)
        try:
            fallback_response = groq_client.chat.completions.create(
                model="qwen/qwen3.8-27b",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            return json.loads(fallback_response.choices[0].message.content)

        except Exception as groq_error:
            raise HTTPException(
                status_code=500, 
                detail=f"Both primary and backup engines failed. Gemini Error: {str(gemini_error)} | Groq Error: {str(groq_error)}"
            )
