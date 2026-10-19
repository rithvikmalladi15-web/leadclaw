import asyncio, hashlib, hmac, json, logging, os, re, secrets, time
from contextlib import contextmanager
from datetime import datetime, date
from typing import List, Optional

import httpx
import psycopg2
import psycopg2.extras
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import google.generativeai as genai
from groq import Groq

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("leadclaw")
app = FastAPI(title="LeadClaw API", version="16.0.0")

FRONTEND_URL = os.environ.get("FRONTEND_URL", "https://leadclaw.pages.dev").rstrip("/")


@app.middleware("http")
async def catch_all(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        log.exception("Unhandled error on %s", request.url.path)
        db_problem = isinstance(e, (psycopg2.Error, RuntimeError))
        return JSONResponse({"detail": "Database is not connected. Check DATABASE_URL on the server." if db_problem
                             else "Server error. Please try again shortly."}, status_code=500)


app.add_middleware(
    CORSMiddleware, allow_origins=[FRONTEND_URL] if FRONTEND_URL else ["*"],
    allow_origin_regex=r"^https://([a-z0-9-]+\.)?leadclaw\.pages\.dev$|^http://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "qwen3-4b")
ADMIN_KEY = os.environ.get("ADMIN_KEY", "")
TEST_EMAIL = "tester@leadclaw.test"  # built-in account used by the admin Testing dashboard
STRIPE_SECRET = os.environ.get("STRIPE_SECRET_KEY", "").strip()
STRIPE_WEBHOOK = os.environ.get("STRIPE_WEBHOOK_SECRET", "").strip()
TEAM_SEATS = int(os.environ.get("TEAM_SEATS", "5"))  # teammates a Pro owner can add
STRIPE_PRICES = {p: os.environ.get("STRIPE_PRICE_" + p.upper(), "").strip() for p in ("basic", "standard", "pro")}
STRIPE_KEY_OK = STRIPE_SECRET.startswith(("sk_", "rk_"))
if STRIPE_SECRET and not STRIPE_KEY_OK:
    log.error("STRIPE_SECRET_KEY must be a SECRET key starting with sk_ (or rk_), not a publishable pk_ key.")
for _p, _v in STRIPE_PRICES.items():
    if _v and not _v.startswith("price_"):
        log.error("STRIPE_PRICE_%s must be a Price ID starting with price_ (not a product ID).", _p.upper())

# Supabase Auth configuration (handles automated emails for free)
SUPABASE_URL = re.sub(r"/(rest|auth)/v1/?$", "", os.environ.get("SUPABASE_URL", "").strip().rstrip("/"))
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY") or os.environ.get("SUPABASE_SECRET_KEY", "")

TEST_MODE = os.environ.get("TEST_MODE", "") == "1"
TRIAL_DAYS = int(os.environ.get("TRIAL_DAYS", "7") or 0)
TRIAL_DISPUTES = int(os.environ.get("TRIAL_DISPUTES", "3") or 3)
STRIPE_TAX = os.environ.get("STRIPE_AUTOMATIC_TAX", "") == "1"  # turn on once Stripe Tax is set up
TRIAL_NO_CARD = os.environ.get("TRIAL_NO_CARD", "1") == "1"  # 1 = free trial without a card (default)
CREDITS = {"basic": 7, "standard": 15, "pro": -1}


def credits_left(plan, used):
    lim = CREDITS.get(plan)
    return 0 if lim is None else (-1 if lim < 0 else max(0, lim - used))


# Keeping your original DATABASE_URL for your tracking data
DATABASE_URL = os.environ.get("DATABASE_URL", "")
SESSION_TTL = 60 * 60 * 24 * 30
SOLO_MONTHLY_LIMIT = 25
MAX_FILE_BYTES = 15 * 1024 * 1024

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


class _Q:
    def __init__(self, cur):
        self.cur = cur

    def execute(self, sql, params=()):
        self.cur.execute(sql.replace("?", "%s"), params)
        return self.cur


_ready = {"ok": False}


@contextmanager
def db():
    if not _ready["ok"]:
        init_db()
    with _db_raw() as c:
        yield c


@contextmanager
def _db_raw():
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
    with _db_raw() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY, password TEXT, age INTEGER, workspace TEXT,
            full_name TEXT DEFAULT '', phone TEXT DEFAULT '', acc_angi TEXT DEFAULT '',
            acc_thumbtack TEXT DEFAULT '', acc_porch TEXT DEFAULT '', acc_networx TEXT DEFAULT '',
            plan TEXT DEFAULT 'none', trials_left INTEGER DEFAULT 3, stripe_customer TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
        c.execute("""CREATE TABLE IF NOT EXISTS disputes (
            id SERIAL PRIMARY KEY, email TEXT, platform TEXT, lead_phone TEXT, lead_email TEXT,
            claim TEXT, reason TEXT DEFAULT '', amount DOUBLE PRECISION, valid INTEGER,
            recovered INTEGER DEFAULT 0, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
        c.execute("CREATE TABLE IF NOT EXISTS sessions (token TEXT PRIMARY KEY, email TEXT, created DOUBLE PRECISION)")
        c.execute("CREATE INDEX IF NOT EXISTS disputes_email_idx ON disputes (email)")
        c.execute("CREATE TABLE IF NOT EXISTS team_members (member TEXT PRIMARY KEY, owner TEXT NOT NULL, added TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
        c.execute("CREATE TABLE IF NOT EXISTS email_verified (email TEXT PRIMARY KEY, verified_at DOUBLE PRECISION)")
        c.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS stripe_sub TEXT DEFAULT ''")
        c.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS on_trial INTEGER DEFAULT 0")
        c.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS trial_used INTEGER DEFAULT 0")
        c.execute("UPDATE users SET plan='none' WHERE plan='free'")
        c.execute("UPDATE users SET plan='basic' WHERE plan='solo'")
        c.execute("UPDATE users SET plan='pro' WHERE plan='growth'")
    _ready["ok"] = True


try:
    init_db()
except Exception as e:
    log.error("Database init failed: %s", e)


def hash_pw(pw: str) -> str:
    salt = secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 200_000).hex()
    return f"pbkdf2${salt}${h}"


def check_pw(pw: str, stored: str) -> bool:
    if stored and stored.startswith("pbkdf2$"):
        _, salt, h = stored.split("$")
        return hmac.compare_digest(hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 200_000).hex(), h)
    return bool(stored) and hmac.compare_digest(stored, pw)


def validate_password(pw: str):
    if len(pw) < 7 or len(pw) > 20 or not re.search(r"[!@#$%^&*(),.?\":{}|<>]", pw):
        raise HTTPException(400, "Password must be 7 to 20 characters and contain 1 special character.")


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
    u = dict(row)
    with db() as c:  # teammates of an active (paid) Pro owner get Pro access
        t = c.execute("SELECT m.owner FROM team_members m JOIN users o ON o.email=m.owner WHERE m.member=? AND o.plan='pro' AND COALESCE(o.on_trial,0)=0", (u["email"],)).fetchone()
    u["team_owner"] = t["owner"] if t else None
    if t:
        u["plan"], u["on_trial"] = "pro", 0
    return u


class RegisterReq(BaseModel):
    email: str; password: str; full_name: str; business: str; accept_terms: bool = False; supabase_token: str = ""
    acc_angi: str = ""; acc_thumbtack: str = ""; acc_porch: str = ""; acc_networx: str = ""
class LoginReq(BaseModel):
    email: str; password: str
class ProfileReq(BaseModel):
    password: Optional[str] = ""; full_name: Optional[str] = ""; phone: Optional[str] = ""
    acc_angi: Optional[str] = ""; acc_thumbtack: Optional[str] = ""; acc_porch: Optional[str] = ""; acc_networx: Optional[str] = ""
class PlanReq(BaseModel):
    plan: str
class RecoveredReq(BaseModel):
    recovered: bool
class SendCodeReq(BaseModel):
    email: str; full_name: str = ""
class VerifyCodeReq(BaseModel):
    email: str; code: str
class ForgotReq(BaseModel):
    email: str
class ResetReq(BaseModel):
    email: str; code: str; password: str

PLATFORM_POLICY = {"Angi": "Angi Lead Credit Policy", "Thumbtack": "Thumbtack Lead Price Credit Terms",
                   "Porch": "Porch Pro Lead Guarantee", "Networx": "Networx Lead Dispute Terms"}
NOT_ELIGIBLE_REASON = ("Pricing objections, competitor hiring, project postponements, and customer "
                       "choices are non-refundable under marketplace policy.")
INVALID_SAFETY_RE = re.compile(
    r"hired (?:another|a|an|someone|a different)|went (?:with|to) (?:a |another |the )?(?:competitor|contractor)|"
    r"project went to|chose (?:another|a different) (?:contractor|company)|competitor|too expensive|"
    r"over (?:our|my) budget|on hold|postponed", re.I)

SYSTEM_PROMPT = """You are an expert compliance auditor for home service lead marketplaces (Angi, Thumbtack, Porch, Networx).
Decide whether a lead qualifies for a credit refund under the marketplace's lead credit guidelines.

ELIGIBLE (valid = true) only when the LEAD ITSELF was defective:
- Permanently disconnected, wrong, or invalid phone number, or a number belonging to a third party with no knowledge of the project
- Customer is outside the contractor's service area
- Duplicate lead in the same billing window
- Spam, bot, or fake submission
- Wrong trade category
- Homeowner states they clicked by mistake and never requested the service

NOT ELIGIBLE (valid = false) when the outcome was a consumer or commercial choice:
- Homeowner hired a competitor, rejected the price, changed their mind, postponed, cancelled or put the project on hold
- Homeowner was unresponsive after a real initial contact

HARD RULE: if the notes or transcript mention a competitor being hired or a price/quote rejection, valid MUST be false.
Never invent facts. Use only what is in the notes, transcript and attached evidence.

If valid is true, write the dispute email in EXACTLY this structure, plain text with no markdown:
Subject: Urgent: Lead Credit Request - Account # <account id> - Lead ID <lead id>
Dear Partner Support Team,
One sentence formally requesting a credit/refund for an invalid lead charged to the account.
LEAD IDENTIFICATION DATA, as bullet lines starting with "- ": Business Name, Platform Account ID, Lead ID Number, Date & Time Received, Lead Cost, Customer Phone, Customer Email.
REASON FOR DISPUTE (POLICY VIOLATION): state that the lead qualifies for a credit under the platform's credit guarantee guidelines, then ONE bullet naming the single best-matching reason and explaining it with the contractor's exact facts.
REQUEST: ask for a credit of the exact lead cost, offer to provide more information, and close politely.
Signature block with the contractor's name, business, phone and email.

If valid is false: "reason" is one or two plain sentences and "claim" is an empty string.
Respond ONLY with JSON using exactly these keys: {"valid": true or false, "confidence": 0.95, "reason": "...", "claim": "..."}"""


def valid_account(v: str) -> bool:
    v = (v or "").strip()
    if re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]{2,}", v):
        return True
    digits = re.sub(r"\D", "", v)
    return bool(re.fullmatch(r"[\d\s().+-]+", v)) and 10 <= len(digits) <= 15


def extract_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1:
        raise ValueError("AI returned no JSON.")
    return json.loads(text[s:e + 1])


def _models(primary, env, default):
    out = [primary] + [x.strip() for x in os.environ.get(env, default).split(",") if x.strip()]
    return list(dict.fromkeys(out))


def _checked(d):
    if not isinstance(d, dict) or "valid" not in d:
        raise ValueError("AI response was missing the 'valid' field.")
    return d


def _gemini(prompt, parts):
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not configured.")
    err = None
    for name in _models(GEMINI_MODEL, "GEMINI_FALLBACK_MODELS", "gemini-2.5-flash,gemini-2.0-flash"):
        try:
            m = genai.GenerativeModel(name, system_instruction=SYSTEM_PROMPT,
                                      generation_config={"temperature": 0.1, "response_mime_type": "application/json"})
            return _checked(extract_json(m.generate_content([prompt, *parts]).text))
        except Exception as e:
            err = e; log.warning("gemini model %s failed: %s", name, e)
    raise err


def _groq(prompt):
    if not groq_client:
        raise ValueError("GROQ_API_KEY is not configured.")
    err = None
    for name in _models(GROQ_MODEL, "GROQ_FALLBACK_MODELS", "qwen/qwen3-32b,llama-3.3-70b-versatile"):
        try:
            r = groq_client.chat.completions.create(model=name, temperature=0.1,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                response_format={"type": "json_object"})
            return _checked(extract_json(r.choices[0].message.content))
        except Exception as e:
            err = e; log.warning("groq model %s failed: %s", name, e)
    raise err


async def run_ai(prompt, parts, prefer):
    order = [_groq, _gemini] if prefer == "groq" else [_gemini, _groq]
    err = None
    for fn in order:
        try:
            if fn is _gemini:
                return await asyncio.to_thread(_gemini, prompt, parts)
            if parts:
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


@app.get("/")
def health():
    try:
        with db() as c:
            c.execute("SELECT 1")
        database = "ok"
    except Exception as e:
        log.error("Health check DB error: %s", e)
        database = "error: check DATABASE_URL"
    checks = {"stripe": bool(STRIPE_KEY_OK and STRIPE_WEBHOOK.startswith("whsec_") and all(v.startswith("price_") for v in STRIPE_PRICES.values())),
              "email": bool(SUPABASE_URL and SUPABASE_KEY), "ai": bool(GEMINI_API_KEY or GROQ_API_KEY), "admin": bool(ADMIN_KEY)}
    return {"status": "healthy", "service": "LeadClaw API", "database": database, "configured": checks}


def gmail(v):
    v = v.strip().lower()
    if not re.fullmatch(r"[a-z0-9._%+-]+@gmail\.com", v):
        raise HTTPException(400, "A valid @gmail.com email address is required.")
    return v


# ---------- Auth via Supabase API ----------
async def supabase_post(path, body):
    if not (SUPABASE_URL and SUPABASE_KEY):
        raise HTTPException(503, "Email verification is not configured yet.")
    try:
        async with httpx.AsyncClient(timeout=20) as h:
            return await h.post(f"{SUPABASE_URL}/auth/v1/{path}", json=body,
                                headers={"apikey": SUPABASE_KEY, "Content-Type": "application/json"})
    except httpx.HTTPError as e:
        log.warning("Supabase request failed: %s", e)
        raise HTTPException(502, "Could not reach the email service. Please try again.")


@app.post("/api/auth/send-code")
async def send_code(req: SendCodeReq):
    email = gmail(req.email)
    k = "send:" + email
    throttle(k)
    FAILS.setdefault(k, []).append(time.time())
    with db() as c:
        if c.execute("SELECT 1 FROM users WHERE email=?", (email,)).fetchone():
            raise HTTPException(409, "An account with this email already exists. Please log in.")
    r = await supabase_post("otp", {"email": email, "create_user": True, "data": {"full_name": req.full_name.strip()[:80]}})
    if r.status_code >= 400:
        log.error("Supabase otp error: %s %s", r.status_code, r.text)
        raise HTTPException(429 if r.status_code == 429 else 502, "Could not send the verification email. Please wait a minute and try again.")
    return {"status": "sent"}


@app.post("/api/auth/verify-code")
async def verify_code(req: VerifyCodeReq):
    email = gmail(req.email)
    code = req.code.strip()
    k = "verify:" + email
    throttle(k)
    if not re.fullmatch(r"\d{6,10}", code):
        raise HTTPException(400, "Enter the full code from your email.")
    r = await supabase_post("verify", {"type": "email", "email": email, "token": code})
    if r.status_code >= 400:
        FAILS.setdefault(k, []).append(time.time())
        raise HTTPException(400, "That code is incorrect or has expired.")
    with db() as c:
        c.execute("INSERT INTO email_verified (email, verified_at) VALUES (?,?) ON CONFLICT (email) DO UPDATE SET verified_at=EXCLUDED.verified_at", (email, time.time()))
    return {"status": "verified"}


@app.post("/api/auth/forgot")
async def forgot(req: ForgotReq):
    email = gmail(req.email)
    k = "forgot:" + email
    throttle(k)
    FAILS.setdefault(k, []).append(time.time())
    with db() as c:
        exists = c.execute("SELECT 1 FROM users WHERE email=?", (email,)).fetchone()
    if exists:
        r = await supabase_post("otp", {"email": email, "create_user": True})
        if r.status_code >= 400:
            log.error("Supabase reset otp error: %s %s", r.status_code, r.text)
            raise HTTPException(429 if r.status_code == 429 else 502, "Could not send the reset email. Please wait a minute and try again.")
    return {"status": "sent"}


@app.post("/api/auth/reset")
async def reset_password(req: ResetReq):
    email = gmail(req.email)
    code = req.code.strip()
    k = "reset:" + email
    throttle(k)
    validate_password(req.password)
    if not re.fullmatch(r"\d{6,10}", code):
        raise HTTPException(400, "Enter the full code from your email.")
    r = await supabase_post("verify", {"type": "email", "email": email, "token": code})
    if r.status_code >= 400:
        FAILS.setdefault(k, []).append(time.time())
        raise HTTPException(400, "That code is incorrect or has expired.")
    with db() as c:
        if not c.execute("SELECT 1 FROM users WHERE email=?", (email,)).fetchone():
            raise HTTPException(400, "That code is incorrect or has expired.")
        c.execute("UPDATE users SET password=? WHERE email=?", (hash_pw(req.password), email))
        c.execute("DELETE FROM sessions WHERE email=?", (email,))
    return {"status": "success"}


@app.post("/api/auth/register")
async def register(req: RegisterReq):
    email = gmail(req.email)
    if not (req.full_name.strip() and req.business.strip() and req.password):
        raise HTTPException(400, "Please fill out all required fields.")
    if not req.accept_terms:
        raise HTTPException(400, "You must accept the Terms & Conditions.")
    validate_password(req.password)
    
    with db() as c:
        if c.execute("SELECT 1 FROM users WHERE email=?", (email,)).fetchone():
            raise HTTPException(409, "An account with this email already exists. Please log in.")

    ok = False
    if req.supabase_token and SUPABASE_URL and SUPABASE_KEY:
        try:
            async with httpx.AsyncClient(timeout=20) as hc:
                r = await hc.get(f"{SUPABASE_URL}/auth/v1/user",
                                 headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {req.supabase_token}"})
            su = r.json() if r.status_code == 200 else {}
            ok = (su.get("email") or "").lower() == email and bool(su.get("email_confirmed_at") or su.get("confirmed_at"))
        except Exception as e:
            log.warning("Supabase user check failed: %s", e)
    if not ok:
        with db() as c:
            v = c.execute("SELECT verified_at FROM email_verified WHERE email=?", (email,)).fetchone()
        ok = bool(v) and time.time() - v[0] <= 3600
    if not ok:
        raise HTTPException(400, "Please verify your email first.")

    with db() as c:
        c.execute("INSERT INTO users (email,password,workspace,full_name,acc_angi,acc_thumbtack,acc_porch,acc_networx,plan) VALUES (?,?,?,?,?,?,?,?,'none')",
                  (email, hash_pw(req.password), req.business.strip()[:120], req.full_name.strip()[:80],
                   req.acc_angi.strip()[:120], req.acc_thumbtack.strip()[:120], req.acc_porch.strip()[:120], req.acc_networx.strip()[:120]))
        c.execute("DELETE FROM email_verified WHERE email=?", (email,))
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
    with db() as c:
        total = c.execute("SELECT COUNT(*) FROM disputes WHERE email=?", (u["email"],)).fetchone()[0]
    return {"on_trial": bool(u.get("on_trial")), "trial_limit": TRIAL_DISPUTES,
            "trial_left": max(0, TRIAL_DISPUTES - total) if u.get("on_trial") else None,
            "email": u["email"], "plan": u["plan"], "credits_left": credits_left(u["plan"], used), "credits_limit": CREDITS.get(u["plan"]), "team_owner": u.get("team_owner"), "workspace": u["workspace"],
            "full_name": u["full_name"], "phone": u["phone"], "acc_angi": u["acc_angi"], "acc_thumbtack": u["acc_thumbtack"],
            "acc_porch": u["acc_porch"], "acc_networx": u["acc_networx"], "month_used": used,
            "solo_limit": SOLO_MONTHLY_LIMIT, "billing": bool(u["stripe_customer"]), "disputes": ds,
            "eligible_total": sum(d["amount"] for d in ds if d["valid"]),
            "recovered_total": sum(d["amount"] for d in ds if d["recovered"])}


class TeamReq(BaseModel):
    email: str


def _team_owner(u):
    return u["plan"] == "pro" and not u.get("team_owner") and not u.get("on_trial")


@app.get("/api/team")
async def team_list(authorization: str = Header("")):
    u = current_user(authorization)
    owner = _team_owner(u)
    members = []
    if owner:
        with db() as c:
            members = [{"email": r["member"], "full_name": r["full_name"] or ""} for r in c.execute(
                "SELECT m.member, COALESCE((SELECT x.full_name FROM users x WHERE x.email=m.member),'') AS full_name FROM team_members m WHERE m.owner=? ORDER BY m.added", (u["email"],))]
    return {"is_owner": owner, "owner": u.get("team_owner"), "limit": TEAM_SEATS, "members": members}


@app.post("/api/team/add")
async def team_add(req: TeamReq, authorization: str = Header("")):
    u = current_user(authorization)
    if not _team_owner(u):
        raise HTTPException(403, "Team seats are available to paid Pro subscribers.")
    email = gmail(req.email)
    if email == u["email"]:
        raise HTTPException(400, "You are already the owner of this workspace.")
    with db() as c:
        if c.execute("SELECT COUNT(*) FROM team_members WHERE owner=?", (u["email"],)).fetchone()[0] >= TEAM_SEATS:
            raise HTTPException(400, f"All {TEAM_SEATS} team seats are in use. Remove a teammate to add another.")
        if c.execute("SELECT 1 FROM team_members WHERE member=?", (email,)).fetchone():
            raise HTTPException(409, "That person is already on a team.")
        c.execute("INSERT INTO team_members (member, owner) VALUES (?,?)", (email, u["email"]))
    return await team_list(authorization)


@app.post("/api/team/remove")
async def team_remove(req: TeamReq, authorization: str = Header("")):
    u = current_user(authorization)
    with db() as c:
        c.execute("DELETE FROM team_members WHERE member=? AND (owner=? OR member=?)", (req.email.strip().lower(), u["email"], u["email"]))
    return await team_list(authorization)


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


# ---------- billing ----------
async def stripe_call(path, data):
    if not STRIPE_SECRET:
        raise HTTPException(503, "Billing is not configured yet.")
    if not STRIPE_KEY_OK:
        raise HTTPException(503, "Billing is misconfigured: the server needs a Stripe secret key (sk_...), not a publishable key.")
    async with httpx.AsyncClient(timeout=20) as h:
        r = await h.post(f"https://api.stripe.com/v1/{path}", data=data, auth=(STRIPE_SECRET, ""))
    if r.status_code >= 400:
        try:
            err = r.json().get("error", {})
        except Exception:
            err = {}
        code, msg = err.get("code") or err.get("type") or "unknown", err.get("message", "")
        log.error("stripe %s -> %s %s: %s", path, r.status_code, code, msg)
        hint = {401: "the Stripe key is invalid or incomplete", 403: "the Stripe key lacks the required permissions"}.get(r.status_code)
        if code == "resource_missing":
            hint = "a Stripe price was not found. Check the STRIPE_PRICE_* values and that the key and prices are in the same mode"
        raise HTTPException(502, f"Billing setup problem: {hint}." if hint else f"Payment provider error ({code}).")
    return r.json()


def stripe_cust(u):
    """Reuse the Stripe customer when one exists so repeat purchases never create duplicates."""
    return {"customer": u["stripe_customer"]} if u.get("stripe_customer") else {"customer_email": u["email"]}


@app.post("/api/billing/checkout")
async def checkout(req: PlanReq, authorization: str = Header("")):
    u = current_user(authorization)
    if req.plan not in CREDITS:
        raise HTTPException(400, "Unknown plan.")
    if TRIAL_NO_CARD and TRIAL_DISPUTES > 0 and u["plan"] == "none" and not u.get("trial_used") and not u.get("stripe_customer"):
        with db() as c:
            c.execute("UPDATE users SET plan=?, on_trial=1, trial_used=1 WHERE email=?", (req.plan, u["email"]))
        return {"url": None, "activated": True}
    if not STRIPE_SECRET and TEST_MODE:
        with db() as c:
            c.execute("UPDATE users SET plan=?, on_trial=? WHERE email=?", (req.plan, 1 if TRIAL_DAYS > 0 else 0, u["email"]))
        return {"url": None, "activated": True}
    if not STRIPE_PRICES.get(req.plan):
        raise HTTPException(503, "Billing is not configured yet.")
    if STRIPE_SECRET and u.get("stripe_sub") and u.get("stripe_customer") and u["plan"] in CREDITS:
        # already subscribed: switch plans in the Stripe billing portal so nobody is billed twice
        p = await stripe_call("billing_portal/sessions", {"customer": u["stripe_customer"], "return_url": FRONTEND_URL or "http://localhost:8080"})
        return {"url": p["url"]}
    base = FRONTEND_URL or "http://localhost:8080"
    s = await stripe_call("checkout/sessions", {
        "mode": "subscription", "allow_promotion_codes": "true", **stripe_cust(u), "client_reference_id": u["email"],
        "line_items[0][price]": STRIPE_PRICES[req.plan], "line_items[0][quantity]": "1",
        "metadata[plan]": req.plan, **({"automatic_tax[enabled]": "true", "billing_address_collection": "required"} if STRIPE_TAX else {}), **({"subscription_data[trial_period_days]": str(TRIAL_DAYS)} if TRIAL_DAYS > 0 and not TRIAL_NO_CARD else {}), "success_url": f"{base}/?checkout=success", "cancel_url": f"{base}/?checkout=cancel"})
    return {"url": s["url"]}


@app.post("/api/billing/start-plan")
async def start_plan(authorization: str = Header("")):
    u = current_user(authorization)
    if not u.get("on_trial"):
        return {"status": "active"}
    if TRIAL_NO_CARD and not u.get("stripe_sub") and STRIPE_SECRET and STRIPE_PRICES.get(u["plan"]):
        base = FRONTEND_URL or "http://localhost:8080"
        sess = await stripe_call("checkout/sessions", {
            "mode": "subscription", "allow_promotion_codes": "true", **stripe_cust(u), "client_reference_id": u["email"],
            "line_items[0][price]": STRIPE_PRICES[u["plan"]], "line_items[0][quantity]": "1", "metadata[plan]": u["plan"],
            **({"automatic_tax[enabled]": "true", "billing_address_collection": "required"} if STRIPE_TAX else {}),
            "success_url": f"{base}/?checkout=success", "cancel_url": f"{base}/?checkout=cancel"})
        return {"url": sess["url"]}
    if STRIPE_SECRET and u.get("stripe_sub"):
        sub = await stripe_call(f"subscriptions/{u['stripe_sub']}", {"trial_end": "now"})
        if sub.get("status") != "active":
            raise HTTPException(402, "We could not charge your card. Update your payment method under Manage subscription.")
    elif not TEST_MODE:
        raise HTTPException(503, "Billing is not configured yet.")
    with db() as c:
        c.execute("UPDATE users SET on_trial=0 WHERE email=?", (u["email"],))
    return {"status": "active"}


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
                c.execute("UPDATE users SET plan=?, stripe_customer=?, stripe_sub=?, on_trial=? WHERE email=?", (plan, obj.get("customer") or "", obj.get("subscription") or "", 1 if obj.get("payment_status") == "no_payment_required" else 0, email))
        elif ev["type"] == "customer.subscription.updated":
            status = obj.get("status")
            price = (((obj.get("items") or {}).get("data") or [{}])[0].get("price") or {}).get("id")
            plan = next((k for k, v in STRIPE_PRICES.items() if v and v == price), None)
            if status in ("canceled", "unpaid", "incomplete_expired"):
                c.execute("UPDATE users SET plan='none', on_trial=0 WHERE stripe_customer=?", (obj.get("customer"),))
            elif status in ("active", "trialing", "past_due") and plan:
                c.execute("UPDATE users SET plan=?, on_trial=? WHERE stripe_customer=?", (plan, 1 if status == "trialing" else 0, obj.get("customer")))
        elif ev["type"] == "customer.subscription.deleted":
            c.execute("UPDATE users SET plan='none', on_trial=0 WHERE stripe_customer=?", (obj.get("customer"),))
    return {"received": True}


# ---------- disputes ----------
@app.post("/api/dispute")
async def dispute(platform: str = Form(...), lead_phone: str = Form(""), lead_email: str = Form(""),
                  notes: str = Form(""), transcript: str = Form(""), cost: float = Form(45.0), lead_id: str = Form(""), received: str = Form(""), account: str = Form(""),
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
    if plan not in CREDITS:
        raise HTTPException(402, "Choose a plan to generate dispute letters.")
    if u.get("on_trial"):
        with db() as c:
            done = c.execute("SELECT COUNT(*) FROM disputes WHERE email=?", (u["email"],)).fetchone()[0]
        if done >= TRIAL_DISPUTES:
            raise HTTPException(402, f"Your free trial has ended after {TRIAL_DISPUTES} disputes. Start your plan to continue.")
    with db() as c:
        used = c.execute("SELECT COUNT(*) FROM disputes WHERE email=? AND created_at>=date_trunc('month', CURRENT_TIMESTAMP)", (u["email"],)).fetchone()[0]
    if credits_left(plan, used) == 0:
        raise HTTPException(402, f"You have used all {CREDITS[plan]} AI credits this month. Upgrade for more.")
    if not valid_account(account):
        raise HTTPException(400, f"Enter the email or phone number of your {platform} account.")
    if files and plan != "pro":
        raise HTTPException(403, "Audio/PDF evidence is a Pro feature.")
    parts = []
    for f in files:
        raw = await f.read()
        if len(raw) > MAX_FILE_BYTES:
            raise HTTPException(413, "Evidence files must be under 15 MB.")
        n = f.filename.lower()
        if raw:
            parts.append({"mime_type": "audio/mpeg" if n.endswith(".mp3") else "audio/wav" if n.endswith(".wav") else "application/pdf", "data": raw})
    handle = account.strip()
    policy = PLATFORM_POLICY[platform]
    prompt = (f"Marketplace Platform: {platform}\nContractor Account ID: {handle or 'Default Workspace'}\nPolicy: {policy}\n"
              f"Lead Cost: ${cost:.2f}\nLead ID: {lead_id or 'Not provided'}\nDate/Time Received: {received or 'Not provided'}\nBusiness Name: {u['workspace']}\nContractor Name: {u['full_name'] or 'Not provided'}\nContractor Phone: {u['phone'] or 'Not provided'}\nContractor Email: {u['email']}\nLead Phone: {lead_phone}\nLead Email: {lead_email}\n\n"
              f"Contractor Call Notes:\n{notes or '(none)'}\n\nTranscript:\n{transcript or '(none)'}")
    try:
        data = await run_ai(prompt, parts, "groq" if model == "groq" and plan == "pro" else "gemini")
    except Exception as e:
        log.exception("AI failed")
        raise HTTPException(502, "Our AI is temporarily unavailable. You were not charged a dispute — please retry.")
    valid = bool(data.get("valid")) and not INVALID_SAFETY_RE.search(f"{notes} {transcript}")
    ai_valid = bool(data.get("valid"))
    reason = (data.get("reason") or "").strip() if (valid or not ai_valid) else ""
    reason = reason or ("Lead defect confirmed." if valid else "Not eligible under marketplace policy.")
    claim = ""
    if valid:
        claim = str(data.get("claim") or "").strip()
        if not claim:
            claim = template_claim(platform, policy, handle, lead_phone, lead_email, cost, notes, reason)
    with db() as c:
        cur = c.execute("INSERT INTO disputes (email,platform,lead_phone,lead_email,claim,reason,amount,valid) VALUES (?,?,?,?,?,?,?,?) RETURNING id",
                        (u["email"], platform, lead_phone, lead_email, claim, reason, cost, int(valid))).fetchone()[0]
        left = credits_left(plan, used + 1)
    return {"valid": valid, "confidence": float(data.get("confidence") or 0.9), "reason": reason, "claim": claim,
            "amount": cost, "credits_left": left, "dispute_id": cur}


ADMIN_LOCK_SECONDS = 300
_admin_lock = {"until": 0.0}


def admin_guard(key: str):
    now = time.time()
    if _admin_lock["until"] > now:
        raise HTTPException(429, detail={"message": "Admin is locked after a wrong key.", "retry_after": int(_admin_lock["until"] - now) + 1})
    if not ADMIN_KEY or not hmac.compare_digest(key.encode(), ADMIN_KEY.encode()):
        _admin_lock["until"] = now + ADMIN_LOCK_SECONDS
        raise HTTPException(403, detail={"message": "Incorrect admin key.", "retry_after": ADMIN_LOCK_SECONDS})


@app.get("/api/admin/ai-test")
async def ai_test(x_admin_key: str = Header("")):
    admin_guard(x_admin_key)
    probe = "Marketplace Platform: Angi\nLead Cost: $75.00\nContractor Call Notes:\nPermanently disconnected phone number. Called twice with no answer."
    out = {}
    for name, fn in (("gemini", lambda: _gemini(probe, [])), ("groq", lambda: _groq(probe))):
        t = time.time()
        try:
            d = await asyncio.to_thread(fn)
            out[name] = {"ok": True, "valid": d.get("valid"), "seconds": round(time.time() - t, 1)}
        except Exception as e:
            out[name] = {"ok": False, "error": str(e)[:240], "seconds": round(time.time() - t, 1)}
    return out


@app.get("/api/admin/analytics")
async def analytics(x_admin_key: str = Header("")):
    admin_guard(x_admin_key)
    with db() as c:
        plans = [dict(r) for r in c.execute("SELECT plan, COUNT(*) AS count FROM users GROUP BY plan")]
        users = [dict(r) for r in c.execute(
            "SELECT u.email,u.full_name,u.phone,u.workspace,u.plan,u.on_trial,u.created_at,"
            "(SELECT COUNT(*) FROM disputes d WHERE d.email=u.email) AS disputes FROM users u ORDER BY u.created_at DESC")]
        disputes = [dict(r) for r in c.execute(
            "SELECT id,email,platform,lead_phone,amount,valid,recovered,created_at FROM disputes ORDER BY id DESC LIMIT 300")]
        u = c.execute("SELECT COUNT(*) n, COALESCE(SUM((plan IN ('basic','standard','pro'))::int),0) paid, COALESCE(SUM((created_at>=NOW()-INTERVAL '7 days')::int),0) new7 FROM users WHERE email <> ?", (TEST_EMAIL,)).fetchone()
        d = c.execute("SELECT COUNT(*) n, COALESCE(SUM(CASE WHEN valid=1 THEN amount END),0) el, COALESCE(SUM(CASE WHEN recovered=1 THEN amount END),0) rec FROM disputes WHERE email <> ?", (TEST_EMAIL,)).fetchone()
    users = [x for x in users if x["email"] != TEST_EMAIL]
    disputes = [x for x in disputes if x["email"] != TEST_EMAIL]
    return {"summary": {"users": u["n"], "paid": u["paid"], "new_7d": u["new7"], "disputes": d["n"],
                        "eligible_total": d["el"], "recovered_total": d["rec"]},
            "plan_distribution": plans, "users": users, "disputes": disputes}


class TestPlanReq(BaseModel):
    plan: str = "none"
    on_trial: bool = False
    reset_disputes: bool = False


@app.post("/api/admin/test-session")
async def admin_test_session(x_admin_key: str = Header("")):
    admin_guard(x_admin_key)
    with db() as c:
        c.execute("INSERT INTO users (email,password,workspace,full_name,plan) VALUES (?,?,?,?,'none') ON CONFLICT (email) DO NOTHING",
                  (TEST_EMAIL, hash_pw(os.urandom(18).hex()), "Test Workspace", "Test Account"))
    return {"token": new_session(TEST_EMAIL)}


@app.post("/api/admin/test-plan")
async def admin_test_plan(req: TestPlanReq, x_admin_key: str = Header("")):
    admin_guard(x_admin_key)
    if req.plan not in ("none", "basic", "standard", "pro"):
        raise HTTPException(400, "Unknown plan.")
    with db() as c:
        c.execute("UPDATE users SET plan=?, on_trial=? WHERE email=?", (req.plan, 1 if (req.on_trial and req.plan != "none") else 0, TEST_EMAIL))
        if req.reset_disputes:
            c.execute("DELETE FROM disputes WHERE email=?", (TEST_EMAIL,))
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
