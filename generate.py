import os
import logging
from typing import List, Optional
from fastapi import FastAPI, Form, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LeadClaw")

app = FastAPI(title="LeadClaw")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini API
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("Gemini API configured successfully for LeadClaw.")
else:
    logger.warning("GEMINI_API_KEY missing. Fallback heuristics enabled.")

class EstimateRequest(BaseModel):
    platform: str
    text: str

class FeedbackRequest(BaseModel):
    type: str
    message: str
    email: Optional[str] = ""
    workspace: Optional[str] = ""

class StripeCheckoutRequest(BaseModel):
    plan: str
    email: Optional[str] = ""

@app.get("/")
async def root():
    return {"status": "ok", "app": "LeadClaw API"}

@app.post("/api/stripe/create-checkout-session")
async def create_stripe_checkout(req: StripeCheckoutRequest):
    """
    Ready to hook up with Stripe Python library:
    import stripe
    stripe.api_key = os.environ.get("STRIPE_SECRET_KEY")
    """
    logger.info(f"Creating checkout session for plan: {req.plan}, email: {req.email}")
    
    # Placeholder return structure for Stripe Checkout URL integration
    # Once configured with your stripe secret key, return actual stripe.checkout.Session.create(...) url
    return {
        "status": "success",
        "url": None  # If null, frontend falls back to instant simulation or modal upgrade
    }

@app.post("/api/stripe/webhook")
async def stripe_webhook(request: Request):
    """
    Stripe Webhook listener endpoint for subscription updates, invoice payments, and renewals.
    """
    payload = await request.body()
    logger.info("Received webhook event from Stripe.")
    return {"status": "received"}

@app.post("/api/dispute/estimate")
async def estimate_cost(req: EstimateRequest):
    logger.info(f"Estimating cost for platform: {req.platform}")
    text_lower = req.text.lower()
    base = 50
    if any(k in text_lower for k in ["roof", "shingle", "gutter"]):
        base = 90
    elif any(k in text_lower for k in ["plumb", "leak", "drain", "water heater", "pipe"]):
        base = 60
    elif any(k in text_lower for k in ["electric", "wiring", "panel", "outlet"]):
        base = 55
    elif any(k in text_lower for k in ["hvac", "furnace", "ac", "air condition", "heating"]):
        base = 70
    elif any(k in text_lower for k in ["lawn", "landscap", "tree", "fence"]):
        base = 40

    mult = {"Angi": 1.1, "Thumbtack": 0.9, "Porch": 0.85, "Networx": 1.0}.get(req.platform, 1.0)
    amount = round(base * mult)
    return {
        "amount": float(amount),
        "low": round(amount * 0.75),
        "high": round(amount * 1.25),
        "basis": f"Estimated based on lead service type for {req.platform}"
    }

@app.post("/api/dispute")
async def create_dispute(
    platform: str = Form(...),
    notes: Optional[str] = Form(""),
    transcript: Optional[str] = Form(""),
    cost: Optional[float] = Form(0.0),
    model: Optional[str] = Form("gemini"),
    files: List[UploadFile] = File([])
):
    logger.info(f"Processing dispute for {platform} via LeadClaw using {model}")
    combined_text = f"Notes: {notes}\nTranscript: {transcript}"
    text_lower = combined_text.lower()
    
    # Eligibility check based on refund policy exclusions
    invalid_keywords = ["changed mind", "too expensive", "decided to wait", "hired someone else"]
    is_valid = True
    reason = ""
    
    for kw in invalid_keywords:
        if kw in text_lower:
            is_valid = False
            reason = f"Platform policy excludes claims where customer '{kw}' after initial contact."
            break

    if not is_valid:
        return {
            "valid": False,
            "confidence": 0.88,
            "reason": reason
        }

    # Format output claim text
    claim_text = (
        f"To {platform} Support Team,\n\n"
        f"I am writing to formally request a credit/refund for lead charges incurred on our account. "
        f"Upon review, this lead meets qualifying refund criteria due to the following verified circumstances:\n\n"
        f"Details: {notes or 'Contact details were invalid or non-responsive upon multiple attempts.'}\n"
        f"{'Transcript Reference: ' + transcript[:200] + '...' if transcript else ''}\n\n"
        f"We kindly request that the lead fee of ${cost:.2f} be credited back to our account per {platform} lead refund policies.\n\n"
        f"Thank you,\nLeadClaw Contractor Team"
    )

    return {
        "valid": True,
        "confidence": 0.94,
        "amount": cost if cost > 0 else 75.0,
        "claim": claim_text
    }

@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest):
    logger.info(f"Received feedback from workspace: {req.workspace}")
    return {"status": "success", "message": "Feedback received successfully."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
