import asyncio, hashlib, hmac, json, logging, os, re, secrets, time
from contextlib import contextmanager
from datetime import datetime, date
from typing import List, Optional

import httpx
import psycopg2
import psycopg2.extras
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from groq import Groq

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("leadclaw")
app = FastAPI(title="LeadClaw API", version="15.0.0")

FRONTEND_URL = os.environ.get("FRONTEND_URL", "https://leadclaw.pages.dev").rstrip("/")
app.add_middleware(CORSMiddleware, allow_origins=[FRONTEND_URL] if FRONTEND_URL else ["*"],
                   allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "qwen3-4b")
ADMIN_KEY = os.environ.get("ADMIN_KEY", "")
STRIPE_SECRET = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PRICES = {"solo": os.environ.get("STRIPE_PRICE_SOLO", ""), "growth": os.environ.get("STRIPE_PRICE_GROWTH", "")}
DATABASE_URL = os.environ.get("DATABASE_URL", "")
SESSION_TTL = 60 * 60 * 24 * 30
SOLO_MONTHLY_LIMIT = 25
MAX_FILE_BYTES = 15 * 1024 * 1024

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


class _Q:
    """Thin wrapper so queries can keep using ? placeholders and chained .fetchone()."""
    def __init__(self, cur):
        self.cur = cur

    def execute(self, sql, params=()):
        self.cur.execute(sql.replace("?", "%s"), params)
        return self.cur


@contextmanager
def db():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set.")
    conn = psycopg2.connect(DATABASE_URL, connect_timeout=15)
    try:
        yield _Q(conn.cursor(cursor_factory=psycopg2.extras.DictCursor))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with db() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY, password TEXT, age INTEGER, workspace TEXT,
            full_name TEXT DEFAULT '', phone TEXT DEFAULT '', acc_angi TEXT DEFAULT '',
            acc_thumbtack TEXT DEFAULT '', acc_porch TEXT DEFAULT '', acc_networx TEXT DEFAULT '',
            plan TEXT DEFAULT 'free', trials_left INTEGER DEFAULT 3, stripe_customer TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
        c.execute("""CREATE TABLE IF NOT EXISTS disputes (
            id SERIAL PRIMARY KEY, email TEXT, platform TEXT, lead_phone TEXT, lead_email TEXT,
            claim TEXT, reason TEXT DEFAULT '', amount DOUBLE PRECISION, valid INTEGER,
            recovered INTEGER DEFAULT 0, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
        c.execute("CREATE TABLE IF NOT EXISTS sessions (token TEXT PRIMARY KEY, email TEXT, created DOUBLE PRECISION)")
        c.execute("CREATE INDEX IF NOT EXISTS disputes_email_idx ON disputes (email)")


init_db()

# ---------- security helpers ----------
def hash_pw(pw: str) -> str:
    salt = secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 200_000).hex()
    return f"pbkdf2${salt}${h}"


def check_pw(pw: str, stored: str) -> bool:
    if stored and stored.startswith("pbkdf2$"):
        _, salt, h = stored.split("$")
        return hmac.compare_digest(hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 200_000).hex(), h)
    return bool(stored) and hmac.compare_digest(stored, pw)  # legacy plaintext; upgraded on login


def validate_password(pw: str):
    if len(pw) < 7 or not re.search(r"[!@#$%^&*(),.?\":{}|<>]", pw):
        raise HTTPException(400, "Password must be at least 7 characters and contain 1 special character.")


FAILS: dict = {}
def throttle(key: str):
    now = time.time()
    FAILS[key] = [t for t in FAILS.get(key, []) if now - t < 900]
    if len(FAILS[key]) >= 8:
        raise HTTPException(429, "Too many attempts. Try again in 15 minutes.")


def new_session(email: str) -> str:
    tok = secrets.token_urlsafe(32)
    with db() as c:
        c.execute("INSERT INTO sessions VALUES (?,?,?)", (tok, email, time.time()))
    return tok


def current_user(authorization: str = "") -> dict:
    tok = authorization.replace("Bearer ", "").strip()
    with db() as c:
        row = c.execute("SELECT u.* FROM sessions s JOIN users u ON u.email=s.email WHERE s.token=? AND s.created>?",
                        (tok, time.time() - SESSION_TTL)).fetchone()
    if not row:
        raise HTTPException(401, "Session expired. Please sign in again.")
    return dict(row)



# ---------- models ----------
class RegisterReq(BaseModel):
    email: str; password: str; dob: str; business: str; accept_terms: bool = False
class LoginReq(BaseModel):
    email: str; password: str
class ProfileReq(BaseModel):
    password: Optional[str] = ""; full_name: Optional[str] = ""; phone: Optional[str] = ""
    acc_angi: Optional[str] = ""; acc_thumbtack: Optional[str] = ""; acc_porch: Optional[str] = ""; acc_networx: Optional[str] = ""
class PlanReq(BaseModel):
    plan: str
class RecoveredReq(BaseModel):
    recovered: bool

# ---------- AI ----------
PLATFORM_POLICY = {"Angi": "Angi Lead Credit Policy", "Thumbtack": "Thumbtack Lead Price Credit Terms",
                   "Porch": "Porch Pro Lead Guarantee", "Networx": "Networx Lead Dispute Terms"}
NOT_ELIGIBLE_REASON = ("Pricing objections, competitor hiring, project postponements, and customer "
                       "choices are non-refundable under marketplace policy.")
INVALID_SAFETY_RE = re.compile(
    r"hired (?:another|a|an|someone|a different)|went (?:with|to) (?:a |another |the )?(?:competitor|contractor)|"
    r"project went to|chose (?:another|a different) (?:contractor|company)|competitor|too expensive|"
    r"over (?:our|my) budget|on hold|postponed", re.I)

SYSTEM_PROMPT = """You are an expert compliance auditor for home service lead marketplaces
(Angi, Thumbtack, Porch, Networx). Decide whether a lead qualifies for a credit refund.

ELIGIBLE (valid = true) only when the LEAD ITSELF was defective:
- Permanently disconnected, wrong, or invalid phone number
- Customer is strictly outside the contractor's service area
- Duplicate lead in the same billing window
- Spam, bot, or fake submission
- Wrong trade category

NOT ELIGIBLE (valid = false) when the outcome was a consumer or commercial choice:
- Homeowner hired a competitor, rejected the price, changed their mind, postponed, cancelled or put on hold
- Homeowner was unresponsive after a real initial contact

HARD RULE: If the notes or transcript mention a competitor being hired or a price/quote rejection, valid MUST be false.

If valid is true: write the dispute email in EXACTLY this structure, filling every field from the facts provided:
Subject: Urgent: Lead Credit Request - Account # <account id> - Lead ID <lead id>
Dear Partner Support Team,
a one-sentence formal request for a credit/refund for an invalid lead, then "LEAD IDENTIFICATION DATA" as bullet lines (Business Name, Platform Account ID, Lead ID Number, Date & Time Received, Lead Cost, Customer Phone, Customer Email),
then "REASON FOR DISPUTE (POLICY VIOLATION)" stating the lead qualifies for a credit under the platform's policy, with ONE bullet naming the single best-matching reason and explaining it with the contractor's exact facts (never invent facts, never claim anything not in the notes/transcript/evidence),
then "REQUEST" asking for the exact credit amount, a polite close, and a signature block with the contractor's name, business, phone and email. Use plain text with bullet characters, no markdown.
If valid is false: "reason" is one or two plain sentences and "claim" is an empty string.
Respond ONLY with JSON: {"valid": true or false, "confidence": 0.95, "reason": "...", "claim": "..."}"""


def extract_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1:
        raise ValueError("AI returned no JSON.")
    return json.loads(text[s:e + 1])


def _gemini(prompt, parts):
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not configured.")
    m = genai.GenerativeModel(GEMINI_MODEL, system_instruction=SYSTEM_PROMPT,
                              generation_config={"temperature": 0.1, "response_mime_type": "application/json"})
    return extract_json(m.generate_content([prompt, *parts]).text)


def _groq(prompt):
    if not groq_client:
        raise ValueError("GROQ_API_KEY is not configured.")
    r = groq_client.chat.completions.create(model=GROQ_MODEL, temperature=0.1,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        response_format={"type": "json_object"})
    return extract_json(r.choices[0].message.content)


async def run_ai(prompt, parts, prefer):
    order = [_groq, _gemini] if prefer == "groq" else [_gemini, _groq]
    err = None
    for fn in order:
        try:
            if fn is _gemini:
                return await asyncio.to_thread(_gemini, prompt, parts)
            if parts:  # groq cannot read audio/pdf
                continue
            return await asyncio.to_thread(_groq, prompt)
        except Exception as e:
            err = e; log.warning("%s failed: %s", fn.__name__, e)
    raise err or ValueError("No AI provider available.")


def template_claim(platform, policy, handle, phone, lead_email, cost, notes, reason):
    return (f"Subject: Lead Credit Request - {platform} - {phone}\n\nDear {platform} Support Team,\n\n"
            f"I am writing to request a credit of ${cost:.2f} for a lead delivered to my account"
            f"{f' ({handle})' if handle else ''}.\n\nLead details:\n- Phone: {phone}\n- Email: {lead_email}\n\n"
            f"Issue: {notes.strip() or reason}\n\nThis lead does not meet the standards of the {policy}. "
            f"Please review the lead and issue a credit to my account.\n\nThank you,\n[Your name]")


# ---------- auth ----------
@app.get("/")
def health():
    return {"status": "healthy", "service": "LeadClaw API"}


@app.post("/api/auth/register")
async def register(req: RegisterReq):
    email = req.email.strip().lower()
    if not (email and req.password and req.dob.strip() and req.business.strip()):
        raise HTTPException(400, "Please fill out all fields.")
    if not email.endswith("@gmail.com"):
        raise HTTPException(400, "A valid @gmail.com email address is required.")
    if not req.accept_terms:
        raise HTTPException(400, "You must accept the Terms & Policies.")
    validate_password(req.password)
    try:
        d = datetime.strptime(req.dob, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(400, "Invalid Date of Birth.")
    t = date.today()
    age = t.year - d.year - ((t.month, t.day) < (d.month, d.day))
    if age < 18:
        raise HTTPException(403, "You must be at least 18 years old.")
    with db() as c:
        if c.execute("SELECT 1 FROM users WHERE email=?", (email,)).fetchone():
            raise HTTPException(409, "An account with this email already exists. Please log in.")
        c.execute("INSERT INTO users (email,password,age,workspace,plan,trials_left) VALUES (?,?,?,?, 'free',3)",
                  (email, hash_pw(req.password), age, req.business.strip()))
    return {"status": "success", "token": new_session(email)}


@app.post("/api/auth/login")
async def login(req: LoginReq):
    email = req.email.strip().lower()
    throttle(email)
    with db() as c:
        row = c.execute("SELECT password FROM users WHERE email=?", (email,)).fetchone()
        if not row or not check_pw(req.password, row["password"]):
            FAILS.setdefault(email, []).append(time.time())
            raise HTTPException(401, "Incorrect email or password.")
        if not row["password"].startswith("pbkdf2$"):
            c.execute("UPDATE users SET password=? WHERE email=?", (hash_pw(req.password), email))
    return {"status": "success", "token": new_session(email)}


@app.post("/api/auth/logout")
async def logout(authorization: str = Header("")):
    with db() as c:
        c.execute("DELETE FROM sessions WHERE token=?", (authorization.replace("Bearer ", "").strip(),))
    return {"status": "success"}


# ---------- account ----------
@app.get("/api/me")
async def me(authorization: str = Header("")):
    u = current_user(authorization)
    with db() as c:
        used = c.execute("SELECT COUNT(*) FROM disputes WHERE email=? AND created_at>=date_trunc('month', CURRENT_TIMESTAMP)", (u["email"],)).fetchone()[0]
        ds = [dict(r) for r in c.execute("SELECT id,platform,lead_phone,lead_email,claim,reason,amount,valid,recovered,created_at FROM disputes WHERE email=? ORDER BY id DESC LIMIT 200", (u["email"],))]
    return {"email": u["email"], "plan": u["plan"], "trials_left": u["trials_left"], "workspace": u["workspace"],
            "full_name": u["full_name"], "phone": u["phone"], "acc_angi": u["acc_angi"], "acc_thumbtack": u["acc_thumbtack"],
            "acc_porch": u["acc_porch"], "acc_networx": u["acc_networx"], "month_used": used,
            "solo_limit": SOLO_MONTHLY_LIMIT, "billing": bool(u["stripe_customer"]), "disputes": ds,
            "eligible_total": sum(d["amount"] for d in ds if d["valid"]),
            "recovered_total": sum(d["amount"] for d in ds if d["recovered"])}


@app.post("/api/user/profile")
async def profile(req: ProfileReq, authorization: str = Header("")):
    u = current_user(authorization)
    with db() as c:
        c.execute("UPDATE users SET full_name=?,phone=?,acc_angi=?,acc_thumbtack=?,acc_porch=?,acc_networx=? WHERE email=?",
                  (req.full_name, req.phone, req.acc_angi, req.acc_thumbtack, req.acc_porch, req.acc_networx, u["email"]))
        if req.password:
            validate_password(req.password)
            c.execute("UPDATE users SET password=? WHERE email=?", (hash_pw(req.password), u["email"]))
    return {"status": "success"}


@app.post("/api/disputes/{dispute_id}/recovered")
async def mark_recovered(dispute_id: int, req: RecoveredReq, authorization: str = Header("")):
    u = current_user(authorization)
    with db() as c:
        c.execute("UPDATE disputes SET recovered=? WHERE id=? AND email=?", (int(req.recovered), dispute_id, u["email"]))
    return {"status": "success"}


# ---------- billing (Stripe) ----------
async def stripe_call(path, data):
    if not STRIPE_SECRET:
        raise HTTPException(503, "Billing is not configured yet.")
    async with httpx.AsyncClient(timeout=20) as h:
        r = await h.post(f"https://api.stripe.com/v1/{path}", data=data, auth=(STRIPE_SECRET, ""))
    if r.status_code >= 400:
        log.error("stripe: %s", r.text)
        raise HTTPException(502, "Payment provider error.")
    return r.json()


@app.post("/api/billing/checkout")
async def checkout(req: PlanReq, authorization: str = Header("")):
    u = current_user(authorization)
    if req.plan not in STRIPE_PRICES or not STRIPE_PRICES[req.plan]:
        raise HTTPException(400, "Unknown plan.")
    base = FRONTEND_URL or "http://localhost:8080"
    s = await stripe_call("checkout/sessions", {
        "mode": "subscription", "customer_email": u["email"], "client_reference_id": u["email"],
        "line_items[0][price]": STRIPE_PRICES[req.plan], "line_items[0][quantity]": "1",
        "metadata[plan]": req.plan, "success_url": f"{base}/?checkout=success", "cancel_url": f"{base}/?checkout=cancel"})
    return {"url": s["url"]}


@app.post("/api/billing/portal")
async def portal(authorization: str = Header("")):
    u = current_user(authorization)
    if not u["stripe_customer"]:
        raise HTTPException(400, "No active subscription.")
    s = await stripe_call("billing_portal/sessions", {"customer": u["stripe_customer"], "return_url": FRONTEND_URL or "http://localhost:8080"})
    return {"url": s["url"]}


@app.post("/api/billing/webhook")
async def webhook(request: Request, stripe_signature: str = Header("")):
    payload = await request.body()
    try:
        parts = dict(p.split("=", 1) for p in stripe_signature.split(","))
        expected = hmac.new(STRIPE_WEBHOOK.encode(), f"{parts['t']}.".encode() + payload, hashlib.sha256).hexdigest()
        ok = bool(STRIPE_WEBHOOK) and hmac.compare_digest(expected, parts["v1"]) and abs(time.time() - int(parts["t"])) < 600
    except Exception:
        ok = False
    if not ok:
        raise HTTPException(400, "Bad signature.")
    ev = json.loads(payload)
    obj = ev["data"]["object"]
    with db() as c:
        if ev["type"] == "checkout.session.completed":
            plan = (obj.get("metadata") or {}).get("plan")
            email = (obj.get("client_reference_id") or "").lower()
            if plan in STRIPE_PRICES and email:
                c.execute("UPDATE users SET plan=?, stripe_customer=? WHERE email=?", (plan, obj.get("customer") or "", email))
        elif ev["type"] == "customer.subscription.updated":
            status = obj.get("status")
            price = (((obj.get("items") or {}).get("data") or [{}])[0].get("price") or {}).get("id")
            plan = next((k for k, v in STRIPE_PRICES.items() if v and v == price), None)
            if status in ("canceled", "unpaid", "incomplete_expired"):
                c.execute("UPDATE users SET plan='free' WHERE stripe_customer=?", (obj.get("customer"),))
            elif status in ("active", "trialing", "past_due") and plan:
                c.execute("UPDATE users SET plan=? WHERE stripe_customer=?", (plan, obj.get("customer")))
        elif ev["type"] == "customer.subscription.deleted":
            c.execute("UPDATE users SET plan='free' WHERE stripe_customer=?", (obj.get("customer"),))
    return {"received": True}


# ---------- disputes ----------
@app.post("/api/dispute")
async def dispute(platform: str = Form(...), lead_phone: str = Form(""), lead_email: str = Form(""),
                  notes: str = Form(""), transcript: str = Form(""), cost: float = Form(45.0), lead_id: str = Form(""), received: str = Form(""),
                  model: str = Form("gemini"), authorization: str = Header(""),
                  files: List[UploadFile] = File(default=[])):
    u = current_user(authorization)
    if platform not in PLATFORM_POLICY:
        raise HTTPException(400, "Unknown platform.")
    if not lead_phone.strip() or not lead_email.strip():
        raise HTTPException(400, "Lead phone and email are required.")
    if not (0 < cost <= 10000):
        raise HTTPException(400, "Enter a valid lead cost.")
    plan = u["plan"]
    files = [f for f in files if f and f.filename]
    if plan == "free" and u["trials_left"] <= 0:
        raise HTTPException(402, "Free trial used up. Upgrade to keep filing disputes.")
    if plan == "solo":
        raise HTTPException(403, "Custom AI claims are a Growth Pro feature. Use the instant template.")
    if files and plan != "growth":
        raise HTTPException(403, "Audio/PDF evidence is a Growth Pro feature.")
    parts = []
    for f in files:
        raw = await f.read()
        if len(raw) > MAX_FILE_BYTES:
            raise HTTPException(413, "Evidence files must be under 15 MB.")
        n = f.filename.lower()
        if raw:
            parts.append({"mime_type": "audio/mpeg" if n.endswith(".mp3") else "audio/wav" if n.endswith(".wav") else "application/pdf", "data": raw})
    handle = u.get("acc_" + platform.lower(), "")
    policy = PLATFORM_POLICY[platform]
    prompt = (f"Marketplace Platform: {platform}\nContractor Account ID: {handle or 'Default Workspace'}\nPolicy: {policy}\n"
              f"Lead Cost: ${cost:.2f}\nLead ID: {lead_id or 'Not provided'}\nDate/Time Received: {received or 'Not provided'}\nBusiness Name: {u['workspace']}\nContractor Name: {u['full_name'] or 'Not provided'}\nContractor Phone: {u['phone'] or 'Not provided'}\nContractor Email: {u['email']}\nLead Phone: {lead_phone}\nLead Email: {lead_email}\n\n"
              f"Contractor Call Notes:\n{notes or '(none)'}\n\nTranscript:\n{transcript or '(none)'}")
    try:
        data = await run_ai(prompt, parts, "groq" if model == "groq" and plan == "growth" else "gemini")
    except Exception as e:
        log.exception("AI failed")
        raise HTTPException(502, "Our AI is temporarily unavailable. You were not charged a dispute — please retry.")
    valid = bool(data.get("valid")) and not INVALID_SAFETY_RE.search(f"{notes} {transcript}")
    ai_valid = bool(data.get("valid"))
    reason = (data.get("reason") or "").strip() if (valid or not ai_valid) else ""
    reason = reason or (NOT_ELIGIBLE_REASON if not valid else "Lead defect confirmed.")
    claim = ""
    if valid:
        claim = str(data.get("claim") or "").strip()
        if plan == "solo" or not claim:
            claim = template_claim(platform, policy, handle, lead_phone, lead_email, cost, notes, reason)
    with db() as c:
        if plan == "free":
            c.execute("UPDATE users SET trials_left=GREATEST(0,trials_left-1) WHERE email=?", (u["email"],))
        cur = c.execute("INSERT INTO disputes (email,platform,lead_phone,lead_email,claim,reason,amount,valid) VALUES (?,?,?,?,?,?,?,?) RETURNING id",
                        (u["email"], platform, lead_phone, lead_email, claim, reason, cost, int(valid))).fetchone()[0]
        left = c.execute("SELECT trials_left FROM users WHERE email=?", (u["email"],)).fetchone()[0]
    return {"valid": valid, "confidence": float(data.get("confidence") or 0.9), "reason": reason, "claim": claim,
            "amount": cost, "trials_left": left, "dispute_id": cur}


class RecordReq(BaseModel):
    platform: str; lead_phone: str; lead_email: str; cost: float; valid: bool; reason: str = ""; claim: str = ""


@app.post("/api/dispute/record")
async def record(req: RecordReq, authorization: str = Header("")):
    """Saves an instant-template dispute (generated in the browser) and enforces plan limits."""
    u = current_user(authorization)
    if req.platform not in PLATFORM_POLICY or not (0 < req.cost <= 10000):
        raise HTTPException(400, "Invalid dispute.")
    with db() as c:
        if u["plan"] == "free" and u["trials_left"] <= 0:
            raise HTTPException(402, "Free trial used up. Upgrade to keep filing disputes.")
        if u["plan"] == "solo":
            n = c.execute("SELECT COUNT(*) FROM disputes WHERE email=? AND created_at>=date_trunc('month', CURRENT_TIMESTAMP)", (u["email"],)).fetchone()[0]
            if n >= SOLO_MONTHLY_LIMIT:
                raise HTTPException(402, f"You've used all {SOLO_MONTHLY_LIMIT} disputes this month. Upgrade to Growth Pro for unlimited.")
        if u["plan"] == "free":
            c.execute("UPDATE users SET trials_left=GREATEST(0,trials_left-1) WHERE email=?", (u["email"],))
        cur = c.execute("INSERT INTO disputes (email,platform,lead_phone,lead_email,claim,reason,amount,valid) VALUES (?,?,?,?,?,?,?,?) RETURNING id",
                        (u["email"], req.platform, req.lead_phone[:40], req.lead_email[:120], req.claim[:8000], req.reason[:500], req.cost, int(req.valid))).fetchone()[0]
    return {"dispute_id": cur}


ADMIN_LOCK_SECONDS = 300
_admin_lock = {"until": 0.0}


@app.get("/api/admin/analytics")
async def analytics(x_admin_key: str = Header("")):
    now = time.time()
    if _admin_lock["until"] > now:  # one wrong key locks the admin area for 5 minutes, even for the right key
        raise HTTPException(429, detail={"message": "Admin is locked after a wrong key.", "retry_after": int(_admin_lock["until"] - now) + 1})
    if not ADMIN_KEY or not hmac.compare_digest(x_admin_key.encode(), ADMIN_KEY.encode()):
        _admin_lock["until"] = now + ADMIN_LOCK_SECONDS
        raise HTTPException(403, detail={"message": "Incorrect admin key.", "retry_after": ADMIN_LOCK_SECONDS})
    with db() as c:
        plans = [dict(r) for r in c.execute("SELECT plan, COUNT(*) AS count FROM users GROUP BY plan")]
        users = [dict(r) for r in c.execute(
            "SELECT u.email,u.workspace,u.plan,u.trials_left,u.created_at,"
            "(SELECT COUNT(*) FROM disputes d WHERE d.email=u.email) AS disputes FROM users u ORDER BY u.created_at DESC")]
        disputes = [dict(r) for r in c.execute(
            "SELECT id,email,platform,lead_phone,amount,valid,recovered,created_at FROM disputes ORDER BY id DESC LIMIT 300")]
        u = c.execute("SELECT COUNT(*) n, COALESCE(SUM((plan<>'free')::int),0) paid, COALESCE(SUM((created_at>=NOW()-INTERVAL '7 days')::int),0) new7 FROM users").fetchone()
        d = c.execute("SELECT COUNT(*) n, COALESCE(SUM(CASE WHEN valid=1 THEN amount END),0) el, COALESCE(SUM(CASE WHEN recovered=1 THEN amount END),0) rec FROM disputes").fetchone()
    return {"summary": {"users": u["n"], "paid": u["paid"], "new_7d": u["new7"], "disputes": d["n"],
                        "eligible_total": d["el"], "recovered_total": d["rec"]},
            "plan_distribution": plans, "users": users, "disputes": disputes}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
