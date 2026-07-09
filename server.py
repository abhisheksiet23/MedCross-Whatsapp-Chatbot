"""
TBCare - FastAPI WhatsApp Webhook Server v4b

Key improvements:
- _send_reply uses AI-provided buttons/list from decision JSON
  (fully language-aware — no hardcoded English labels)
- WhatsApp number auto-detection on MOBILE step
- parse_incoming returns msg_id so special button IDs are handled cleanly
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, BackgroundTasks
from fastapi.responses import PlainTextResponse

from fastapi import HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
from collections import Counter
import sqlite3
import re

from config import WHATSAPP_VERIFY_TOKEN
import whatsapp
from main import handle_whatsapp_message

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    import os
    from config  import LOG_PATH
    from storage import init_db

    handlers = [logging.FileHandler(LOG_PATH, encoding="utf-8")]
    if os.getenv("DEBUG", "").lower() in ("1", "true"):
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        handlers=handlers,
    )
    init_db()
    log.info("TBCare WhatsApp server started")
    yield
    log.info("TBCare WhatsApp server stopped")


app = FastAPI(title="TBCare WhatsApp Bot", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
BASE_DIR = Path(__file__).resolve().parent

def get_conn():
    from config import DB_PATH, LOG_PATH as LOG_PATH_STR
    db_path = BASE_DIR / DB_PATH
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

def _get_log_path():
    from config import LOG_PATH
    return BASE_DIR / LOG_PATH


#  SEND REPLY — uses AI-generated buttons/list from decision JSON
#  Fully language-aware: AI generates all text in patient's language

def _send_reply(phone: str, reply: str, decision: dict) -> None:
    """
    Route to the correct WhatsApp message type.
    decision["message_type"] = "text" | "buttons" | "list"
    decision["buttons"]       = list of {id, title} — already in patient's language
    decision["list_sections"] = list of sections with rows
    decision["list_button_label"] = label for the list opener button
    decision["last_slots"]    = raw slot list (fallback for SLOT_SELECTION)
    """
    msg_type = decision.get("message_type", "text")

    try:
        # Buttons 
        if msg_type == "buttons":
            buttons = decision.get("buttons", [])
            if buttons:
                whatsapp.send_button_message(phone, reply, buttons)
                return

        # List 
        elif msg_type == "list":
            sections = decision.get("list_sections", [])
            label    = decision.get("list_button_label", "View Options")

            # Fallback: if AI didn't provide list_sections for SLOT_SELECTION,
            # build them from raw slots
            if not sections and decision.get("last_slots"):
                sections = [{
                    "title": "Available Slots",
                    "rows": [
                        {"id": f"slot_{i}", "title": s["slotLabel"][:24]}
                        for i, s in enumerate(decision["last_slots"][:10])
                    ]
                }]

            if sections:
                whatsapp.send_list_message(phone, reply, label, sections)
                return

    except Exception as e:
        log.error("Interactive send failed (type=%s): %s", msg_type, e)

    # Fallback: plain text 
    try:
        whatsapp.send_message(phone, reply)
    except Exception as e:
        log.error("Plain text fallback also failed: %s", e)



#  WHATSAPP NUMBER AUTO-DETECTION
#  When AI is at MOBILE step, offer patient's WhatsApp number as quick option

def _send_mobile_step(phone: str, reply: str, detected_language: str) -> None:
    """
    Instead of asking patient to type their number, show their WhatsApp
    number as a one-tap option. Language-aware button labels.
    """
    wa_number = phone[-10:]   # strip country code, keep last 10 digits

    # Language-aware button labels
    lang = detected_language or "English"
    if lang in ("Hindi",):
        yes_label  = f"✅ {wa_number} उपयोग करें"
        diff_label = "📱 दूसरा नंबर"
        body_note  = f"\n\n📱 आपका WhatsApp नंबर: *{wa_number}*\nक्या यही नंबर उपयोग करें?"
    elif lang == "Hinglish":
        yes_label  = f"✅ {wa_number} Use Karo"
        diff_label = "📱 Doosra Number"
        body_note  = f"\n\n📱 Aapka WhatsApp number: *{wa_number}*\nKya yahi use karein?"
    elif lang == "Tamil":
        yes_label  = f"✅ {wa_number} பயன்படுத்து"
        diff_label = "📱 வேறு எண்"
        body_note  = f"\n\n📱 உங்கள் WhatsApp எண்: *{wa_number}*"
    elif lang == "Telugu":
        yes_label  = f"✅ {wa_number} వాడండి"
        diff_label = "📱 వేరే నంబర్"
        body_note  = f"\n\n📱 మీ WhatsApp నంబర్: *{wa_number}*"
    elif lang == "Bengali":
        yes_label  = f"✅ {wa_number} ব্যবহার করুন"
        diff_label = "📱 অন্য নম্বর"
        body_note  = f"\n\n📱 আপনার WhatsApp নম্বর: *{wa_number}*"
    elif lang == "Marathi":
        yes_label  = f"✅ {wa_number} वापरा"
        diff_label = "📱 वेगळा नंबर"
        body_note  = f"\n\n📱 तुमचा WhatsApp नंबर: *{wa_number}*"
    elif lang == "Gujarati":
        yes_label  = f"✅ {wa_number} વાપરો"
        diff_label = "📱 અલગ નંબર"
        body_note  = f"\n\n📱 તમારો WhatsApp નંબર: *{wa_number}*"
    else:
        # English default
        yes_label  = f"✅ Use {wa_number}"
        diff_label = "📱 Different number"
        body_note  = f"\n\n📱 Your WhatsApp number is *{wa_number}*\nWould you like to use this?"

    # Button title max 20 chars — truncate if needed
    yes_label = yes_label[:20]

    full_body = reply + body_note
    whatsapp.send_button_message(phone, full_body, [
        {"id": "use_wa_number",        "title": yes_label},
        {"id": "use_different_number", "title": diff_label[:20]},
    ])



#  WEBHOOK HANDLERS

@app.get("/webhook")
async def verify_webhook(request: Request):
    params    = dict(request.query_params)
    mode      = params.get("hub.mode")
    token     = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")
    if mode == "subscribe" and token == WHATSAPP_VERIFY_TOKEN:
        log.info("Webhook verified")
        return PlainTextResponse(content=challenge)
    log.warning("Verification failed")
    return Response(status_code=403)


@app.get("/")
async def root_verify(request: Request):
    params    = dict(request.query_params)
    mode      = params.get("hub.mode")
    token     = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")
    if mode == "subscribe" and token == WHATSAPP_VERIFY_TOKEN:
        log.info("Webhook verified via /")
        return PlainTextResponse(content=challenge)
    return Response(status_code=404)


@app.post("/webhook")
async def receive_message(request: Request, background_tasks: BackgroundTasks):
    payload = await request.json()
    background_tasks.add_task(_handle_payload, payload)
    return Response(status_code=200)


def _handle_payload(payload: dict) -> None:
    try:
        result = whatsapp.parse_incoming(payload)
        if result is None:
            return

        phone, user_text, msg_id = result

        # Mark as read 
        try:
            mid = payload["entry"][0]["changes"][0]["value"]["messages"][0]["id"]
            whatsapp.mark_as_read(mid)
        except Exception:
            pass

        # Handle WhatsApp number button responses 
        if msg_id == "use_wa_number":
            # Patient tapped "Use XXXXXXXXXX" — send their WA number to AI
            user_text = phone[-10:]
            log.info("Patient chose to use WA number: %s", user_text)

        elif msg_id == "use_different_number":
            # Patient wants to enter a different number — tell AI to ask
            user_text = "I want to use a different phone number"
            log.info("Patient wants a different number")

        # ── Process message ───────────────────────────────────────────────────
        replies, decision = handle_whatsapp_message(phone, user_text)
        step     = decision.get("step", "UNKNOWN")
        lang     = decision.get("detected_language", "English")

        for reply_text in replies:
            if not reply_text:
                continue

            # Special handling: MOBILE step → auto-show WhatsApp number
            if step == "MOBILE":
                _send_mobile_step(phone, reply_text, lang)
            else:
                _send_reply(phone, reply_text, decision)

    except Exception as e:
        log.error("Error in _handle_payload: %s", e, exc_info=True)


# POST /initiate — send first message to a patient 
@app.post("/initiate")
async def initiate_conversation(request: Request):
    body     = await request.json()
    phone    = body.get("phone")
    template = body.get("template", "tbcare_welcome")
    if not phone:
        return Response(content="phone required", status_code=400)
    try:
        whatsapp.send_template_message(to=phone, template_name=template)
        return {"status": "sent", "to": phone}
    except Exception as e:
        return Response(content=str(e), status_code=500)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "TBCare WhatsApp Bot v4b"}

@app.get("/dashboard")
def dashboard():
    conn = get_conn()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    total = cur.execute("SELECT COUNT(*) FROM leads").fetchone()[0]
    completed = cur.execute(
        "SELECT COUNT(*) FROM leads WHERE conversation_done=1"
    ).fetchone()[0]
    booked = cur.execute(
        "SELECT COUNT(*) FROM leads WHERE booking_confirmed=1"
    ).fetchone()[0]
    active = total - completed

    conn.close()

    return {
        "total_leads": total,
        "completed_conversations": completed,
        "active_conversations": active,
        "bookings_confirmed": booked,
    }


@app.get("/leads")
def leads(
    page: int = 1,
    limit: int = 50,
    all: bool = False,
):
    conn = get_conn()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    total = cur.execute("SELECT COUNT(*) FROM leads").fetchone()[0]

    if all:
        rows = cur.execute(
            "SELECT * FROM leads ORDER BY updated_at DESC"
        ).fetchall()
    else:
        offset = (page - 1) * limit
        rows = cur.execute(
            "SELECT * FROM leads ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()

    conn.close()

    return {
        "total": total,
        "page": page,
        "limit": limit,
        "count": len(rows),
        "data": [dict(r) for r in rows],
    }


@app.get("/leads/{session_id}")
def lead(session_id: str):
    conn = get_conn()
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM leads WHERE session_id=?",
        (session_id,),
    ).fetchone()
    conn.close()

    if not row:
        raise HTTPException(404, "Lead not found")

    return dict(row)


@app.get("/conversations/{session_id}")
def conversations(session_id: str):
    conn = get_conn()
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        """
        SELECT *
        FROM conversation_logs
        WHERE session_id=?
        ORDER BY timestamp
        """,
        (session_id,),
    ).fetchall()

    conn.close()

    return [dict(r) for r in rows]


LOG_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} [\d:,]+)\s+"
    r"(?P<level>\w+)\s+"
    r"(?P<module>\w+)\s+"
    r"(?P<message>.*)$"
)


@app.get("/logs")
def logs(
    level: str | None = None,
    module: str | None = None,
    q: str | None = None,
    limit: int = 200,
):
    if not LOG_PATH.exists():
        raise HTTPException(404, "chatbot.log not found")

    out = []

    with LOG_PATH.open("r", encoding="utf-8", errors="ignore") as f:
        for line in reversed(f.readlines()):
            m = LOG_RE.match(line.strip())
            if not m:
                continue

            item = m.groupdict()

            if level and item["level"] != level:
                continue

            if module and item["module"] != module:
                continue

            if q and q.lower() not in line.lower():
                continue

            out.append(item)

            if len(out) >= limit:
                break

    return out


@app.get("/logs/errors")
def error_logs(limit: int = 100):
    return logs(level="ERROR", limit=limit)


@app.get("/search")
def search(q: str):
    conn = get_conn()
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        """
        SELECT *
        FROM leads
        WHERE
            session_id LIKE ?
            OR patient_name LIKE ?
            OR phone_number LIKE ?
            OR mobile LIKE ?
            OR api_lead_id LIKE ?
            OR appointment_id LIKE ?
        """,
        tuple([f"%{q}%"] * 6),
    ).fetchall()

    conv = []
    if rows:
        sid = rows[0]["session_id"]
        conv = conn.execute(
            "SELECT * FROM conversation_logs WHERE session_id=? ORDER BY timestamp",
            (sid,),
        ).fetchall()

    conn.close()

    matching_logs = []
    if LOG_PATH.exists():
        with LOG_PATH.open("r", encoding="utf-8", errors="ignore") as f:
            matching_logs = [
                line.rstrip() for line in f if q.lower() in line.lower()
            ][:100]

    return {
        "leads": [dict(r) for r in rows],
        "conversation": [dict(r) for r in conv],
        "logs": matching_logs,
    }


@app.get("/statistics")
def statistics():
    conn = get_conn()
    conn.row_factory = sqlite3.Row

    diseases = Counter(
        r["disease_name"] or "Unknown"
        for r in conn.execute("SELECT disease_name FROM leads")
    )

    patient_types = Counter(
        r["patient_type"] or "Unknown"
        for r in conn.execute("SELECT patient_type FROM leads")
    )

    conn.close()

    return {
        "diseases": diseases,
        "patient_types": patient_types,
    }