import os
from groq import Groq

# Initialize Groq client securely using the server environment variable
groq_api_key = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=groq_api_key) if groq_api_key else None

def process_lead_dispute(platform: str, notes: str, transcript: str, cost: float, model: str = "gemini"):
    if not client:
        raise ValueError("Groq API key is not configured in the server environment variables.")

    # Format prompt for the contractor lead dispute
    prompt = f"""
    You are an expert AI assistant for home contractors. 
    Analyze this disputed lead for platform: {platform}.
    Lead Cost: ${cost}
    Call Notes: {notes}
    Call Transcript: {transcript}

    Evaluate whether this lead qualifies for a credit or refund based on standard contractor platform policies (Angi, Thumbtack, Porch, Networx).
    Provide your output in a structured format:
    - valid: true or false
    - confidence: decimal between 0.0 and 1.0
    - amount: numeric value of the credit at stake
    - reason: explanation of why it qualifies or doesn't
    - claim: a fully formatted, professional dispute email/statement ready to be submitted.
    """

    try:
        # Use a reliable cloud model via Groq API
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are LeadClaw AI, an objective business assistant for home contractors specializing in platform credit disputes."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        response_text = completion.choices[0].message.content
        
        # In production, ensure your function returns a dictionary matching your frontend expectations:
        # {"valid": True, "confidence": 0.95, "amount": cost, "reason": "...", "claim": "..."}
        return response_text

    except Exception as e:
        raise RuntimeError(f"Groq API inference failed: {str(e)}")
