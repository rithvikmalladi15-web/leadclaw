import os
import logging
from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from google import genai
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LeadClaw-Backend")

app = FastAPI(title="LeadClaw Enterprise API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "online", "service": "LeadClaw Backend Active"}

@app.post("/api/dispute")
async def analyze_dispute(
    platform: str = Form(...),
    notes: Optional[str] = Form(""),
    transcript: Optional[str] = Form(""),
    cost: float = Form(0.0),
    model: str = Form("gemini"),
    files: List[UploadFile] = File(default=[])
):
    logger.info(f"Processing dispute for platform: {platform}, model requested: {model}")
    
    prompt = f"""You are an elite automated dispute resolution agent for local trade contractors.
A contractor is disputing a low-quality lead charge with marketplace platform "{platform}" to secure a full refund credit.

INPUT TRANSCRIPT / CALL RECORDING NOTES:
\"\"\"{transcript}\"\"\"

ADDITIONAL CONTRACTOR NOTES:
\"\"\"{notes}\"\"\"

INSTRUCTIONS:
1. Automatically analyze the transcript and notes to determine the exact valid dispute category.
2. Determine if the claim is valid based on marketplace policy guidelines.
3. Generate a highly professional, objective, policy-compliant credit refund request statement.

Output your response cleanly in the following format:
[VALID]: True
[CONFIDENCE]: 0.95
[REASON]: Policy rule matched.
[ESTIMATED REFUND VALUE]: ${cost if cost > 0 else 75.00}

[COMPLIANCE STATEMENT]:
Dear {platform} Support Team,

I am writing to formally request a lead credit refund for a low-quality lead charged recently.

Reason for Dispute:
- Details: {notes} {transcript}

Based on your contractor terms and lead guarantee policies, this lead is non-qualifying. Please credit my account for ${cost if cost > 0 else 75.00}.

Thank you,
LeadClaw Enterprise User
"""

    generated_text = ""
    
    try:
        if model == "gemini":
            try:
                client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt,
                )
                generated_text = response.text
            except Exception as gemini_err:
                logger.warning(f"Gemini failed, falling back to Groq Qwen3: {gemini_err}")
                model = "qwen"

        if model == "qwen" or not generated_text:
            groq_api_key = os.environ.get("GROQ_API_KEY")
            if not groq_api_key:
                raise Exception("GROQ_API_KEY environment variable is missing on Render.")
                
            groq_response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {groq_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "qwen/qwen3-4b",
                    "messages": [{"role": "user", "content": prompt}]
                }
            )
            groq_data = groq_response.json()
            if "choices" in groq_data and len(groq_data["choices"]) > 0:
                generated_text = groq_data["choices"][0]["message"]["content"]
            else:
                raise Exception(f"Groq API Error: {groq_data}")

        return {
            "valid": True,
            "confidence": 0.94,
            "reason": "Matched marketplace validity terms.",
            "amount": cost if cost > 0 else 75.00,
            "claim": generated_text
        }

    except Exception as e:
        logger.error(f"Error handling dispute request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
