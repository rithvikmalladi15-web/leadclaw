import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from google import genai
import requests

app = Flask(__name__, static_folder='../public', static_url_path='')
CORS(app)

@app.route('/')
def serve_index():
    return send_from_directory('../public', 'index.html')

@app.route('/api/generate', methods=['POST'])
def generate_dispute():
    try:
        data = request.get_json()
        platform = data.get('platform', 'Angi')
        transcript = data.get('transcript', '')
        user_notes = data.get('notes', '')

        gemini_key = os.environ.get("GEMINI_API_KEY")
        groq_key = os.environ.get("GROQ_API_KEY")

        prompt = f"""You are an expert dispute compliance officer for trade contractors.
Generate a formal, strictly policy-compliant credit refund request submission for marketplace platform "{platform}".

INPUT TRANSCRIPT / NOTES:
{transcript}

ADDITIONAL CONTRACTOR NOTES:
{user_notes}

STRICT FORMAT REQUIREMENT:
You must output your response in this exact structure to pass automated platform review:

[DISPUTE CATEGORY DETECTED]: <e.g., Disconnected Phone Number / Invalid Contact Info / Out of Service Radius>
[ESTIMATED REFUND VALUE]: $<Amount e.g. 75.00>

[COMPLIANCE STATEMENT]:
SUBJECT: Lead Credit Request - Policy Violation Claim ({platform})

To {platform} Vendor Support,

We are formally requesting a full lead credit / refund for the referenced lead under {platform}'s official Lead Quality and Verification Policy.

1. TIMELINE OF CONTACT: Outreach was initiated immediately (within the required platform response window) upon lead delivery.
2. VERIFIABLE POLICY VIOLATION: Upon dialing the provided phone number, the call failed validation due to a definitive carrier disconnect message ("The number you have dialed is no longer in service"). 
3. EVIDENCE: As documented in our communication logs, the contact information provided is non-operational, rendering connection impossible.

Because this lead fails standard contact verification criteria, we request a full account credit adjustment applied back to our balance. Thank you for your prompt assistance in resolving this invalid lead charge.
"""

        generated_text = ""
        ai_success = False

        if gemini_key:
            try:
                client = genai.Client(api_key=gemini_key)
                response = client.models.generate_content(
                    model='gemini-3.5-flash',
                    contents=prompt,
                )
                generated_text = response.text
                ai_success = True
            except Exception as e1:
                print("Gemini 3.5 Flash failed, trying fallback model...", e1)
                try:
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=prompt,
                    )
                    generated_text = response.text
                    ai_success = True
                except Exception as e2:
                    print("Gemini backup failed:", e2)

        if not ai_success and groq_key:
            try:
                groq_response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {groq_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "qwen/qwen3-32b",
                        "messages": [{"role": "user", "content": prompt}]
                    }
                )
                groq_data = groq_response.json()
                if "choices" in groq_data and len(groq_data["choices"]) > 0:
                    generated_text = groq_data["choices"][0]["message"]["content"]
                    ai_success = True
            except Exception as groq_err:
                print("Groq fallback error:", groq_err)

        if not ai_success:
            return jsonify({"error": "Unable to generate dispute. Please verify your terminal environment variables."}), 500

        return jsonify({"statement": generated_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)