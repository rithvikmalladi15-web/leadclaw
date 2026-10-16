import os
import json
from groq import Groq

def process_lead_dispute(platform: str, notes: str, transcript: str, cost: float, model: str = "gemini"):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not configured on Render.")

    client = Groq(api_key=api_key)

    prompt = f"""
    You are an expert AI assistant for home contractors. 
    Analyze this disputed lead for platform: {platform}.
    Lead Cost: ${cost}
    Call Notes: {notes}
    Call Transcript: {transcript}

    Evaluate whether this lead qualifies for a credit or refund based on standard contractor platform policies (Angi, Thumbtack, Porch, Networx).
    You MUST output valid JSON format with these exact keys:
    - "valid": boolean (true or false)
    - "confidence": float between 0.0 and 1.0
    - "amount": numeric value of the credit at stake
    - "reason": string explaining why it qualifies or doesn't
    - "claim": string containing a fully formatted, professional dispute email/statement ready to be submitted.
    """

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are LeadClaw AI, an objective business assistant for home contractors. You must respond strictly in valid JSON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        response_text = completion.choices[0].message.content.strip()
        
        # Clean up markdown code blocks if the model wrapped them
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
            
        return json.loads(response_text.strip())

    except Exception as e:
        return {
            "valid": False,
            "confidence": 0.0,
            "amount": cost,
            "reason": f"AI evaluation error or invalid JSON response: {str(e)}",
            "claim": ""
        }
