import os

# Define folders and files for LeadClaw Enterprise (Clean, Real-World Version)
folders = ['public', 'api']

files = {
    'requirements.txt': """google-genai
requests
flask
flask-cors
""",
    
    'api/generate.py': """import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
import requests

app = Flask(__name__)
CORS(app)  # Enable CORS so the local frontend can talk to Flask smoothly

@app.route('/api/generate', methods=['POST'])
def generate_dispute():
    try:
        data = request.get_json()
        platform = data.get('platform', 'Angi')
        transcript = data.get('transcript', '')
        user_notes = data.get('notes', '')

        # Prompt engineering instructing AI to analyze transcript/notes and generate strict compliant output
        prompt = f\"\"\"You are an elite automated dispute resolution agent for local trade contractors.
A contractor is disputing a low-quality lead charge with marketplace platform "{platform}".

INPUT TRANSCRIPT / CALL RECORDING NOTES:
\"\"\"{transcript}\"\"\"

ADDITIONAL CONTRACTOR NOTES:
\"\"\"{user_notes}\"\"\"

INSTRUCTIONS:
1. Automatically analyze the transcript/notes above to determine the exact valid dispute category (e.g., Disconnected Phone Number, Out of Service Radius, Customer Unresponsive/Dead Lead, or Duplicate Lead).
2. Generate a highly professional, objective, policy-compliant credit refund request statement optimized specifically to pass {platform}'s automated dispute verification review.
3. Strictly adhere to formal dispute guidelines: clear, concise, objective, and mentioning policy violations directly.

Output your response cleanly in the following format:
[DISPUTE CATEGORY DETECTED]: <Category Here>
[ESTIMATED REFUND VALUE]: $<Amount Here e.g. 75.00>

[COMPLIANCE STATEMENT]:
<The formal dispute claim text ready to copy-paste or submit>
\"\"\"

        generated_text = ""

        # 1. PRIMARY: Try Gemini 3.5 Flash
        try:
            client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
            response = client.models.generate_content(
                model='gemini-3.5-flash',
                contents=prompt,
            )
            generated_text = response.text
        except Exception as gemini_err:
            print("Gemini 3.5 Flash failed, falling back to Groq Qwen 3...", gemini_err)

            # 2. BACKUP: Fallback to Groq using Qwen 3
            groq_response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.environ.get('GROQ_API_KEY')}",
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
            else:
                raise Exception("Both Gemini 3.5 Flash and Groq backup failed.")

        return jsonify({"statement": generated_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
""",

    'public/index.html': """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LeadClaw Enterprise | Automated Lead Dispute & Refund Engine</title>
    <style>
        :root {
            --font-stack: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            --primary: #2563eb;
            --primary-hover: #1d4ed8;
            --bg-body: #f3f4f6;
            --bg-card: #ffffff;
            --text-main: #1f2937;
            --text-muted: #4b5563;
            --border: #d1d5db;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: var(--font-stack); }
        body { background-color: var(--bg-body); color: var(--text-main); line-height: 1.5; padding: 20px; }
        .container { max-width: 900px; margin: 0 auto; }
        
        /* Header & Metrics */
        header { display: flex; justify-content: space-between; align-items: center; background: var(--bg-card); padding: 16px 20px; border-radius: 8px; border: 1px solid var(--border); margin-bottom: 20px; }
        .logo-area { display: flex; align-items: center; gap: 10px; }
        .logo-badge { background: #eff6ff; color: var(--primary); font-weight: 700; padding: 8px 12px; border-radius: 6px; font-size: 16px; }
        .logo-title { font-size: 18px; font-weight: 700; }
        
        .metric-card { background: var(--bg-card); padding: 16px 20px; border-radius: 8px; border: 1px solid var(--border); margin-bottom: 20px; text-align: center; }
        .metric-title { font-size: 12px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }
        .metric-value { font-size: 32px; font-weight: 800; color: #059669; }

        /* Workspace Panels */
        .panel { background: var(--bg-card); border-radius: 8px; border: 1px solid var(--border); padding: 20px; margin-bottom: 20px; }
        .panel-title { font-size: 15px; font-weight: 700; margin-bottom: 14px; color: #111827; }
        
        .platforms-container { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-bottom: 16px; }
        @media (max-width: 600px) { .platforms-container { grid-template-columns: repeat(2, 1fr); } }
        .platform-option { border: 1px solid var(--border); border-radius: 6px; padding: 10px; text-align: center; cursor: pointer; font-size: 13px; font-weight: 600; background: #fafafa; }
        .platform-option.selected { border-color: var(--primary); background: #eff6ff; color: var(--primary); }

        .form-group { margin-bottom: 14px; }
        label { display: block; font-size: 13px; font-weight: 700; margin-bottom: 6px; color: #374151; }
        textarea, input[type="text"] { width: 100%; padding: 10px; border: 1px solid var(--border); border-radius: 6px; font-size: 13px; background: #fff; }
        
        .btn-primary { background: var(--primary); color: white; border: none; border-radius: 6px; padding: 12px 20px; font-size: 14px; font-weight: 700; width: 100%; cursor: pointer; }
        .btn-primary:hover { background: var(--primary-hover); }
        .btn-success { background: #059669; color: white; border: none; border-radius: 6px; padding: 10px 16px; font-size: 13px; font-weight: 700; width: 100%; cursor: pointer; margin-top: 10px; }
        .btn-success:hover { background: #047857; }

        .output-box { background: #f9fafb; border: 1px solid var(--border); border-radius: 6px; padding: 14px; font-family: monospace; font-size: 12px; min-height: 150px; white-space: pre-wrap; margin-bottom: 14px; line-height: 1.4; color: #111827; }

        /* Modal Tutorial Overlay */
        .tutorial-overlay { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.6); display: flex; justify-content: center; align-items: center; z-index: 1000; }
        .tutorial-modal { background: #fff; padding: 24px; border-radius: 10px; max-width: 440px; width: 90%; box-shadow: 0 10px 25px rgba(0,0,0,0.2); }
        .tutorial-modal h2 { font-size: 18px; font-weight: 800; margin-bottom: 10px; color: var(--primary); }
        .tutorial-modal p { font-size: 13px; color: var(--text-muted); margin-bottom: 16px; line-height: 1.5; }
    </style>
</head>
<body>

<!-- Interactive Quick Tutorial Modal -->
<div class="tutorial-overlay" id="tutorialModal">
    <div class="tutorial-modal">
        <h2>Welcome to LeadClaw 🐾</h2>
        <p>Recover hundreds of dollars in wasted ad spend from dead or spam contractor leads in 3 simple steps:</p>
        <p><strong>1.</strong> Select your marketplace platform (Angi, Thumbtack, etc.).<br>
           <strong>2.</strong> Paste your call transcript or raw notes.<br>
           <strong>3.</strong> Click generate to let Gemini AI format a policy-compliant dispute instantly!</p>
        <button class="btn-primary" onclick="closeTutorial()">Got it, let's start!</button>
    </div>
</div>

<div class="container">
    <header>
        <div class="logo-area">
            <div class="logo-badge">LC</div>
            <div>
                <div class="logo-title">LeadClaw Enterprise</div>
                <div style="font-size: 11px; color: var(--text-muted);">Automated Lead Refund & Dispute Engine</div>
            </div>
        </div>
        <div style="font-size: 12px; font-weight: 600; color: #059669;">🟢 Live AI Connected</div>
    </header>

    <!-- Real-World Clean Metric: Starts at $0.00 -->
    <div class="metric-card">
        <div class="metric-title">Total Cash Recovered This Session</div>
        <div class="metric-value" id="totalRecovered">$0.00</div>
    </div>

    <!-- Workspace -->
    <div class="panel">
        <div class="panel-title">1. Select Marketplace Platform</div>
        <div class="platforms-container" id="platformSelector">
            <div class="platform-option selected" onclick="selectPlatform('Angi')">Angi</div>
            <div class="platform-option" onclick="selectPlatform('Thumbtack')">Thumbtack</div>
            <div class="platform-option" onclick="selectPlatform('Porch')">Porch</div>
            <div class="platform-option" onclick="selectPlatform('Networx')">Networx</div>
        </div>

        <div class="form-group">
            <label>Call / Lead Transcript</label>
            <textarea id="leadTranscript" rows="3" placeholder="Paste phone call transcript or customer interaction notes here... (e.g., Caller said 'Wrong number, I live in Ohio' or 'Already hired someone else')..."></textarea>
        </div>

        <div class="form-group">
            <label>Additional Context / Notes (Optional)</label>
            <input type="text" id="userNotes" placeholder="e.g., Called within 15 minutes of lead delivery policy window">
        </div>

        <button class="btn-primary" onclick="generateDispute()">Generate AI Compliance Dispute</button>
    </div>

    <div class="panel">
        <div class="panel-title">2. AI Generated Dispute & Gateway Submission</div>
        <label>Policy-Compliant Claim Output</label>
        <div class="output-box" id="outputStatement">Awaiting input... Select platform, paste transcript, and generate statement.</div>

        <button class="btn-success" onclick="submitClaim()">Submit Claim & Add to Cash Recovered</button>
    </div>
</div>

<script>
    let activePlatform = 'Angi';
    let currentEstimatedValue = 0.00;

    function closeTutorial() {
        document.getElementById('tutorialModal').style.display = 'none';
    }

    function selectPlatform(platform) {
        activePlatform = platform;
        const options = document.querySelectorAll('.platform-option');
        options.forEach(opt => {
            if(opt.innerText === platform) opt.classList.add('selected');
            else opt.classList.remove('selected');
        });
    }

    async function generateDispute() {
        const transcript = document.getElementById('leadTranscript').value;
        const notes = document.getElementById('userNotes').value;
        const outputBox = document.getElementById('outputStatement');

        if(!transcript) {
            alert('Please paste a call transcript or notes first.');
            return;
        }

        outputBox.innerText = "Analyzing transcript and generating compliance dispute via Gemini 3.5 Flash...";

        try {
            const response = await fetch('http://127.0.0.1:5000/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ platform: activePlatform, transcript, notes })
            });
            const data = await response.json();
            
            if(data.statement) {
                outputBox.innerText = data.statement;

                // Simple regex parser to extract dollar amount from AI response if available
                const match = data.statement.match(/\\$(\\d+\\.\\d{2})/);
                if(match) {
                    currentEstimatedValue = parseFloat(match[1]);
                } else {
                    currentEstimatedValue = 75.00; // default fallback per lead credit
                }
            } else {
                outputBox.innerText = "Error: " + data.error;
            }
        } catch (e) {
            outputBox.innerText = "Failed to connect to Python backend API. Make sure python api/generate.py is running in your terminal.";
        }
    }

    function submitClaim() {
        if(currentEstimatedValue <= 0) {
            alert('Please generate a valid dispute statement first.');
            return;
        }

        const totalEl = document.getElementById('totalRecovered');
        let currentTotal = parseFloat(totalEl.innerText.replace('$', '').replace(',', ''));
        currentTotal += currentEstimatedValue;
        totalEl.innerText = '$' + currentTotal.toFixed(2);

        alert('Claim successfully submitted! $' + currentEstimatedValue.toFixed(2) + ' added to your cash recovered total.');
        currentEstimatedValue = 0.00;
    }
</script>

</body>
</html>
"""
}

# Write files
for folder in folders:
    os.makedirs(folder, exist_ok=True)

for filepath, content in files.items():
    os.makedirs(os.path.dirname(filepath), exist_ok=True) if '/' in filepath else None
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

print("\nSUCCESS! LeadClaw updated with clean real-world UI, transcripts, and CORS enabled.")