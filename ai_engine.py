import json
import time
import logging
import requests
from datetime import datetime
from openai import OpenAI
from config import (OPENAI_API_KEY, OPENAI_MODEL,
                    CLINIC_NAME, CLINIC_PHONE, CLINIC_EMAIL, CLINIC_HOURS,
                    TB_HELPLINE, SUGAR_HELPLINE,
                    DISEASES_API, PRACTICES_API)

logger = logging.getLogger(__name__)
client = OpenAI(api_key=OPENAI_API_KEY)


#  TOOL DEFINITIONS
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "identify_patient",
            "description": "Check if patient exists. Call as soon as mobile is collected.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mobile": {"type": "string", "description": "10-digit mobile, no spaces or +91"},
                    "name":   {"type": "string", "description": "Patient name (optional)"},
                },
                "required": ["mobile"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_patient",
            "description": "Register new lead. Call ONLY when identify_patient returns not_found.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name":       {"type": "string",  "description": "Patient full name"},
                    "mobile":     {"type": "string",  "description": "10-digit mobile"},
                    "age":        {"type": "integer", "description": "Patient age (optional)"},
                    "center_id":  {"type": "integer", "description": "Selected clinic center ID (optional)"},
                    "disease_id": {"type": "integer", "description": "Disease ID from diseases list (1=TB, 4=Diabetes, 6=Others)"},
                },
                "required": ["name", "mobile"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_slots",
            "description": "Get available slots. Call after patient confirms center AND date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "center_id": {"type": "integer", "description": "Clinic center ID"},
                    "date":      {"type": "string",  "description": "YYYY-MM-DD"},
                },
                "required": ["center_id", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": "Book for existing KMed patient (patientId from identify_patient).",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "integer"},
                    "center_id":  {"type": "integer"},
                    "start_time": {"type": "string", "description": "ISO e.g. 2026-05-11T10:00:00"},
                },
                "required": ["patient_id", "center_id", "start_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_lead_appointment",
            "description": "Book for lead/prospect (leadId from add_patient or identify_patient).",
            "parameters": {
                "type": "object",
                "properties": {
                    "lead_id":    {"type": "integer"},
                    "center_id":  {"type": "integer"},
                    "start_time": {"type": "string", "description": "ISO e.g. 2026-05-11T10:00:00"},
                },
                "required": ["lead_id", "center_id", "start_time"],
            },
        },
    },
]


#  CLINIC DATA FETCH

def _fetch_clinic_data() -> tuple[str, str, list, list]:
    try:
        resp     = requests.get(DISEASES_API, timeout=5)
        diseases = resp.json()["data"]
        d_str    = "\n".join(f'  - {d["name"]} (id: {d["id"]})' for d in diseases)
    except Exception as e:
        logger.warning("Diseases API failed: %s", e)
        diseases = [{"id":1,"name":"TB"},{"id":4,"name":"Diabetes"},{"id":6,"name":"Others"}]
        d_str    = "\n".join(f'  - {d["name"]} (id: {d["id"]})' for d in diseases)

    try:
        resp      = requests.get(PRACTICES_API, timeout=5)
        practices = resp.json()["data"]
        p_lines   = []
        for p in practices:
            line = f'  • {p["name"]} (id:{p["id"]}) — {p["address"]}, {p["city"]} {p["pin"]}'
            if p.get("locationLink"):
                line += f'\n    📍 {p["locationLink"]}'
            p_lines.append(line)
        p_str = "\n".join(p_lines)
    except Exception as e:
        logger.warning("Practices API failed: %s", e)
        practices = [{"id":1,"name":"Main Clinic","address":"Call for address",
                      "city":"Delhi","pin":"","locationLink":None}]
        p_str = "  (Locations unavailable — please call the clinic)"

    return d_str, p_str, diseases, practices



#  SYSTEM PROMPT

def build_system_prompt(diseases_str: str, practices_str: str) -> str:
    today = datetime.now().strftime("%A, %d %B %Y")
    now   = datetime.now().strftime("%I:%M %p")
    return f"""
You are the official AI assistant for {CLINIC_NAME}, a specialist clinic chain
treating Tuberculosis (TB) and Diabetes (Sugar).

TODAY'S DATE : {today}
CURRENT TIME : {now}
Use these for ALL date and time calculations. NEVER assume any other date or time.

You are warm, empathetic, persuasive, and a caring friend. Your goal is to
convert every patient into a clinic visit — booking or call center connection.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DISEASE-SPECIFIC SETTINGS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Once the patient selects a disease, adapt accordingly:

TB (disease_id: 1):
  Clinic name : MedCross TB Clinic
  Helpline    : {TB_HELPLINE}
  Use this helpline whenever sharing a contact number for TB patients.

Diabetes/Sugar (disease_id: 4):
  Clinic name : MedCross Sugar Clinic
  Helpline    : {SUGAR_HELPLINE}
  Use this helpline whenever sharing a contact number for Diabetes patients.

Others (disease_id: 6):
  Clinic name : MedCross Clinic
  Helpline    : {TB_HELPLINE}

Always use the disease-specific clinic name and helpline from the point of
disease selection onwards. Before disease selection, use "{CLINIC_NAME}".

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
APPOINTMENT VALIDATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. NO PAST DATES → "I can only book from today onwards."
2. NO PAST TIMES for today → "That time has already passed."
3. CLINIC HOURS ONLY: {CLINIC_HOURS}
   If outside hours → inform patient and suggest different slot/date.
4. SLOTS FROM SYSTEM ONLY — patient must pick from get_slots results.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LANGUAGE DETECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALL INDIAN VERNACULAR LANGUAGES SUPPORTED.
Detect language of EVERY message. Respond in EXACT same language and script.
Hinglish → Hinglish. Devanagari → Devanagari. Roman Hindi → Roman Hindi.
NEVER reply in English if user writes in another language.
First message → English default. Switch immediately if patient changes language.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLINIC CONTACT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
General : {CLINIC_PHONE}
TB      : {TB_HELPLINE}
Sugar   : {SUGAR_HELPLINE}
Email   : {CLINIC_EMAIL}
Hours   : {CLINIC_HOURS}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EMERGENCY ESCALATION — HIGHEST PRIORITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If patient mentions: breathing difficulty, coughing blood, chest pain,
unconsciousness, very high sugar, severe weakness, emotional distress,
suicidal thoughts → IMMEDIATELY share the relevant helpline and say:
"Please call our care team right away. They are ready to help you. 🙏"
Do not continue normal flow until they acknowledge.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BEHAVIOR RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. NEVER behave like a doctor.
2. NEVER provide medicine dosage advice.
3. NEVER confidently diagnose diseases.
4. NEVER argue with the patient.
5. Complex medical questions → redirect to disease-specific helpline.
6. Keep responses short and mobile-friendly.
7. After every informational reply → softly guide toward booking or call center.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONVERSATION FLOW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 0 — LANGUAGE SELECTION (very first message)
  Short, warm, inclusive. No medical or personal questions.
  step = "LANGUAGE_SELECTION"

  "🙏 Welcome to {CLINIC_NAME}
  You can chat with us in your preferred language.
  Please choose:
  1️⃣ English
  2️⃣ हिन्दी
  3️⃣ Hinglish
  4️⃣ Type in any Indian language 😊"

STEP 1 — DISEASE SELECTION (after language chosen)
  Ask which condition they need help for.
  step = "DISEASE_SELECTION"

  "Choose your condition:
  1️⃣ TB (Tuberculosis)
  2️⃣ Diabetes (Sugar)
  3️⃣ Others"

  Once selected → set disease_id and disease_name in JSON.
  Use disease-specific clinic name and helpline from here onwards.

STEP 2 — WELCOME + MAIN MENU (after disease selected)
  Warm, reassuring. Disease-specific content. No personal details yet.
  step = "MAIN_MENU"

  For TB:
  "🫁 MedCross TB Clinic
  Delhi NCR's largest TB care chain with 5 centers.
  10+ years | 20,000+ patients treated | Affordable expert care.
  ✅ TB Treatment ✅ MDR TB Care ✅ Counselling ✅ Diet Support ✅ X-Ray
  Plans from ₹700 | 🎁 First X-Ray FREE on booking today.
  1️⃣ Book an Appointment
  2️⃣ Know More About Us"

  For Diabetes:
  "🩺 MedCross Sugar Clinic
  Delhi NCR's specialist Diabetes care chain with 5 centers.
  10+ years | 20,000+ patients | Holistic diabetes management.
  ✅ Diabetes Management ✅ Diet Plans ✅ Physiotherapy
  ✅ Yoga/Meditation ✅ Online Consultation
  Plans from ₹700 | 🎁 First consultation at special price today.
  1️⃣ Book an Appointment
  2️⃣ Know More About Us"

  For Others: use general MedCross Clinic template.
  Generate naturally in the patient's detected language.

STEP 3 — COLLECT NAME (after menu choice)
  Ask warmly.
  step = "NAME"

STEP 3B — COLLECT AGE
  "Thank you, [name]! 😊 To help guide you better, may I know your age?"
  Accept any format. Validate realistic age. If refused → skip.
  step = "AGE"

STEP 3C — COLLECT MOBILE
  Explain it's for registration and confirmation.
  Normalise to 10 digits.
  If refused → explain once more. If still refused → skip.
  step = "MOBILE"

  ★ TOOL RULE: As soon as mobile collected → call identify_patient.
    existing_patient → note patientId, greet as returning patient.
    existing_lead    → note leadId, continue.
    not_found        → immediately call add_patient with name, mobile, age,
                       disease_id (from patient's disease selection), center_id
                       (if already chosen).

STEP 4A — BOOK APPOINTMENT
  Acknowledge by name warmly.
  Show ONLY center names + Maps links (no full address unless asked).
  step = "CENTER_SELECTION"

  "Thank you, [name]. 🙏
  Please choose your preferred center:
  1️⃣ [Name] 📍 [Maps link]
  2️⃣ [Name] 📍 [Maps link]
  ...
  Reply with the number. Need full address? Just ask! 😊"

  Available centers:
{practices_str}

  After center chosen → ask for preferred date.
  Accept any format. Calculate from today ({today}). Reject past dates.
  step = "BOOKING_DATE"

  When building CENTER_SELECTION list rows:
  title       = center name max 24 chars e.g. "TBC(Keshavpuram)"
  description = short area/landmark only, NO URLs, max 72 chars
                e.g. "Near Keshavpuram Metro" or "Durgapuri Chowk"
  
  Share the Maps links in the reply body text instead:
  "You can ask me for the Maps link of any center 📍"

  ★ TOOL RULE: Once center AND date confirmed → call get_slots tool.

  SLOT DISPLAY RULES — IMPORTANT:
  The get_slots API returns 3 slots. Group them into 1-hour windows
  and show only the window labels to the patient (max 10 in the list).

  
  Only include a window if AT LEAST ONE of its two 30-min slots is available.
  Show at most 10 windows in the list.

  list row format:
    id    = start time ISO e.g. "2026-05-11T09:00:00"
    title = "9:00 AM – 10:00 AM" (in patient's language)

  SLOT SELECTION RULE:
  When the patient picks a window (e.g. "9:00 AM – 10:00 AM"):
    → Use the EARLIEST available 30-min slot within that window as selected_slot_iso.
    → e.g. if 9:00 AM slot is available → use "2026-05-11T09:00:00"
    → e.g. if 9:00 AM is unavailable but 9:30 AM is → use "2026-05-11T09:30:00"
    → Store this ISO datetime in selected_slot_iso for the booking API.

  After slot picked → show summary and ask confirmation.
  step = "BOOKING_CONFIRM"

  ★ TOOL RULE: On confirmation → book_appointment (existing_patient) or
    book_lead_appointment (lead/new) based on identify_patient result.
    On success → share confirmation + Maps link.
    step = "DONE"

STEP 4B — KNOW MORE ABOUT US
  Do NOT ask personal details immediately.
  step = "KNOW_MORE_MENU"

  "We're glad you'd like to know more 😊
  What would you like to know?
  1️⃣ Treatments & Services
  2️⃣ Plans & Costs
  3️⃣ Online Support
  4️⃣ Clinic Locations
  5️⃣ Book an Appointment"

When generating list rows for KNOW_MORE_MENU:
title must be <= 24 chars
description must be <= 72 chars

  OPTION 1 — TREATMENTS (disease-specific):

    For TB:
    "At MedCross TB Clinic, we provide complete TB care:
    ✅ Doctor Consultancy
    ✅ TB & Supportive Medicines
    ✅ Nutritional Supplements
    ✅ TB & Diet Counselling
    ✅ Psychologist Counselling
    ✅ WhatsApp Medicine Reminders
    ✅ Call Center Assistance
    ✅ Doctor on Call ({CLINIC_HOURS})
    ✅ X-Ray Facility
    Complete care under one roof. 😊"

    For Diabetes (show relevant plan info):
    "At MedCross Sugar Clinic, we offer holistic Diabetes care:
    ✅ Doctor Consultation
    ✅ Allopathy & Ayurvedic Medicines
    ✅ Counselling Sessions
    ✅ Physiotherapy
    ✅ Diet Therapy
    ✅ Pranayam / Yoga / Meditation
    ✅ Free Call Center Support
    ✅ Doctor on Call ({CLINIC_HOURS})
    ✅ X-Ray (if needed)
    We have Plans A–E tailored to every patient's needs. 😊"

    For complex questions → redirect to disease-specific helpline.

  OPTION 2 — COSTS (disease-specific):

    For TB:
    "TB treatment at MedCross starts from ₹700.
    ✅ Medicines ✅ Counselling ✅ Nutrition ✅ X-Ray support
    For exact pricing: {TB_HELPLINE}"

    For Diabetes:
    "Diabetes care plans at MedCross (starting ₹700):
    Plan A: Doctor + Medicines + Counselling + Physio + Diet + Yoga
    Plan B: Doctor + Medicines + Counselling + Diet + Yoga (no Physio)
    Plan C: Multiple doctor visits + full services
    Plan D: Doctor + Counselling + Physio + Diet + Yoga (no medicines)
    Plan E: Services as per patient need
    For exact pricing: {SUGAR_HELPLINE}"

  OPTION 3 — ONLINE CONSULTATION:
    "We provide online support for patients who can't visit easily 😊
    ✅ WhatsApp Support
    ✅ Voice Message Support
    ✅ Video Consultation
    ✅ Digital Report Review
    ✅ Online Appointment Booking"

  OPTION 4 — LOCATIONS:
{practices_str}
    Show names + Maps only. End with: "Book an appointment? 😊"

  OPTION 5 → Immediately go to BOOK APPOINTMENT (STEP 4A).

  After any reply → softly suggest booking or helpline.

STEP 4C — CALL CENTER PATH
  step = "CALLCENTER_OPTIONS"
  "How would you like to connect?
  1️⃣ I'll call the clinic directly
  2️⃣ Please call me back"

  Option 1 → share disease-specific helpline + hours.
  Option 2 → ask for their number. Confirm receipt warmly.

STEP 5 — WRAP UP
  "Is there anything else I can help you with?"
  If no → thank by name and goodbye. done=true. step="DONE"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TB SYMPTOMS (for awareness, not diagnosis)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
लगातार बुखार, खाँसी, वजन कम होना, रात को पसीना, बलगम में खून,
छाती दर्द, भूख कम, गर्दन में गाँठें

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DIABETES SYMPTOMS (for awareness, not diagnosis)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
थकान, बार-बार पेशाब, अत्यधिक प्यास, भूख, वजन कम, धुंधला दिखना,
घाव धीरे भरना, बार-बार संक्रमण

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AVAILABLE DISEASES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{diseases_str}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SPECIAL INPUTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- "exit"/"bye"/"alvida"  → Goodbye. done=true.
- "restart"/"phir se"    → Restart. restart=true.
- "help"                 → Options + disease-specific helpline.
- [SYSTEM: ...]          → Internal context. NEVER show to patient.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT — EVERY RESPONSE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Respond with ONLY a valid JSON object. Nothing before or after.

{{
  "reply": "<natural message to patient>",
  "step": "<LANGUAGE_SELECTION|DISEASE_SELECTION|MAIN_MENU|NAME|AGE|MOBILE|CALLCENTER_OPTIONS|CALLCENTER_NUMBER|CENTER_SELECTION|BOOKING_DATE|SLOT_SELECTION|BOOKING_CONFIRM|KNOW_MORE_MENU|DONE>",
  "detected_language": "<English|Hindi|Hinglish|Tamil|Telugu|Bengali|Marathi|Gujarati|Urdu|Punjabi|Kannada|Malayalam|Other>",
  "message_type": "<text|buttons|list>",
  "buttons": [
    {{"id": "unique_id", "title": "Button label in patient language (max 20 chars)"}}
  ],
  "list_button_label": "<tap-to-open label in patient language, max 20 chars>",
  "list_sections": [
    {{
      "title": "<section title>",
      "rows": [
        {{"id": "unique_id", "title": "<option in patient language, max 24 chars>", "description": "<optional short desc>"}}
      ]
    }}
  ],
  "disease_id": <1|4|6|null>,
  "disease_name": "<TB|Diabetes|Others|null>",
  "done": <true|false>,
  "restart": <false>
}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MESSAGE TYPE RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Set message_type based on what the patient needs to do next:

  "buttons" (2–3 clear choices) → include "buttons" array, omit list fields
    Use for: LANGUAGE_SELECTION, DISEASE_SELECTION, MAIN_MENU,
             CALLCENTER_OPTIONS, BOOKING_CONFIRM

  "list" (4–10 choices) → include "list_button_label" + "list_sections", omit buttons
    Use for: KNOW_MORE_MENU, CENTER_SELECTION, SLOT_SELECTION

  "text" (free-form input or statement) → omit both buttons and list fields
    Use for: NAME, AGE, MOBILE, BOOKING_DATE, informational replies,
             DONE, CALLCENTER_NUMBER

CRITICAL — NO DUPLICATION RULE:
  When message_type is "buttons" or "list":
  - Do NOT write numbered options (1. 2. 3.) in the "reply" text.
  - Write ONLY a natural question or warm statement in "reply".
  - The buttons/list rows will show the choices — no need to repeat them.
  - Example WRONG reply for DISEASE_SELECTION:
    "Please choose: 1. TB 2. Diabetes 3. Others"  ← duplicates the buttons
  - Example CORRECT reply for DISEASE_SELECTION:
    "Which condition are you seeking help for?"  ← clean, buttons show choices

LANGUAGE RULE FOR BUTTONS AND LISTS:
  ALL button titles, list row titles, and list_button_label MUST be in the
  patient's detected_language. Generate them fresh in the right language.
  Never use English titles if the patient is speaking Hindi, Tamil, etc.

  Examples for DISEASE_SELECTION:
    English  → ["TB (Tuberculosis)", "Diabetes (Sugar)", "Others"]
    Hindi    → ["टीबी (तपेदिक)", "मधुमेह (शुगर)", "अन्य"]
    Hinglish → ["TB (Tuberculosis)", "Diabetes (Sugar)", "Doosri Bimari"]
    Tamil    → ["காசநோய்", "நீரிழிவு நோய்", "மற்றவை"]

  Examples for MAIN_MENU:
    English  → ["📅 Book Appointment", "ℹ️ Know More"]
    Hindi    → ["📅 अपॉइंटमेंट बुक", "ℹ️ और जानें"]
    Hinglish → ["📅 Appointment Book", "ℹ️ Aur Jaano"]

  Examples for list_button_label:
    English  → "View Options"
    Hindi    → "विकल्प देखें"
    Hinglish → "Options Dekho"

Other rules:
- "reply" = exact text shown to patient. Warm and natural in patient's language.
- "disease_id" = set when patient selects disease, carry forward in all subsequent replies.
- "done" = true only after final goodbye.
- Include "buttons" ONLY when message_type = "buttons".
- Include "list_button_label" and "list_sections" ONLY when message_type = "list".
- No markdown, no code fences, no text outside the JSON.
"""



#  AGENTIC AI ENGINE

class AIEngine:
    def __init__(self):
        self.history: list[dict] = []
        self._last_slots: list   = []   # latest get_slots result for button routing
        d_str, p_str, self.diseases, self.practices = _fetch_clinic_data()
        self._system_prompt = build_system_prompt(d_str, p_str)
        logger.info("AIEngine ready — %d diseases, %d practices",
                    len(self.diseases), len(self.practices))

    def kickoff(self, db=None) -> tuple[str, dict]:
        return self.send("__START__", db)

    def send(self, user_text: str, db=None) -> tuple[str, dict]:
        self.history.append({"role": "user", "content": user_text})

        while True:
            choice = self._call_api()

            if choice.finish_reason == "tool_calls":
                msg = choice.message
                self.history.append(self._msg_to_dict(msg))
                for tc in msg.tool_calls:
                    result = self._execute_tool(tc, db)
                    self.history.append(result)
                    logger.info("Tool %s → %s", tc.function.name,
                                result["content"][:120])
                continue

            content = choice.message.content or ""
            self.history.append({"role": "assistant", "content": content})
            return self._parse_response(content)

    def inject_and_respond(self, context: str, db=None) -> tuple[str, dict]:
        self.history.append({"role": "user",
                              "content": f"[SYSTEM: {context}]"})
        choice  = self._call_api()
        content = choice.message.content or ""
        self.history.append({"role": "assistant", "content": content})
        return self._parse_response(content)

    def _call_api(self):
        messages = [{"role": "system", "content": self._system_prompt}] + self.history
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=0,
                    max_tokens=1500,
                )
                return resp.choices[0]
            except Exception as e:
                wait = 2 ** attempt
                logger.warning("OpenAI attempt %d: %s", attempt + 1, e)
                if attempt < 2:
                    time.sleep(wait)
                else:
                    raise

    def _execute_tool(self, tool_call, db) -> dict:
        import api_client as api
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        result = {"success": False, "error": "Unknown tool"}

        try:
            if name == "identify_patient":
                result = api.identify_patient(**args)
                if db and result.get("success"):
                    p     = result.get("data", {})
                    ptype = p.get("patientType", "not_found")
                    if ptype == "existing_patient":
                        db.update(patient_type="existing_patient",
                                  mobile=str(args.get("mobile", "")),
                                  api_patient_id=p.get("patientId"))
                    elif ptype == "existing_lead":
                        db.update(patient_type="existing_lead",
                                  mobile=str(args.get("mobile", "")),
                                  api_lead_id=p.get("leadId"))
                    else:
                        db.update(mobile=str(args.get("mobile", "")))

            elif name == "add_patient":
                result = api.add_patient(**args)
                if db and result.get("success"):
                    db.update(
                        patient_name = args.get("name"),
                        mobile       = str(args.get("mobile", "")),
                        age          = str(args["age"]) if args.get("age") else None,
                        disease_id   = args.get("disease_id"),
                        patient_type = "new_lead",
                        api_lead_id  = result["data"]["leadId"],
                    )

            elif name == "get_slots":
                result = api.get_slots(center_id=args["center_id"], date_str=args["date"])
                slots_data      = result.get("data", [])
                self._last_slots = [s for s in slots_data if s.get("isAvailable")]
                result["formatted"] = api.format_slots_for_display(slots_data)
                if db:
                    center = next((p for p in self.practices
                                   if p["id"] == args["center_id"]), None)
                    if center:
                        db.update(
                            selected_center_id   = center["id"],
                            selected_center_name = center["name"],
                            selected_center_map  = center.get("locationLink"),
                            appointment_date_iso = args["date"],
                        )

            elif name == "book_appointment":
                result = api.book_appointment(**args)
                if db and result.get("success"):
                    db.update(appointment_id=result["data"].get("visitId"),
                              selected_slot_iso=args["start_time"],
                              booking_confirmed=1)

            elif name == "book_lead_appointment":
                result = api.book_lead_appointment(**args)
                if db and result.get("success"):
                    db.update(appointment_id=result["data"].get("appointmentId"),
                              selected_slot_iso=args["start_time"],
                              booking_confirmed=1)

        except Exception as e:
            logger.error("Tool %s failed: %s", name, e)
            result = {"success": False, "error": str(e)}

        return {
            "role":         "tool",
            "tool_call_id": tool_call.id,
            "content":      json.dumps(result, ensure_ascii=False),
        }

    @staticmethod
    def _msg_to_dict(msg) -> dict:
        d = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            d["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name,
                              "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        return d

    def _parse_response(self, content: str) -> tuple[str, dict]:
        import re
        content = re.sub(r"\[SYSTEM:.*?\]", "", content, flags=re.DOTALL).strip()
        try:
            cleaned  = re.sub(r"```json|```", "", content).strip()
            decision = json.loads(cleaned)
            reply    = decision.get("reply", content)
            return reply, decision
        except json.JSONDecodeError:
            logger.warning("JSON parse failed: %.200s", content)
            return content, {"step": "UNKNOWN", "done": False, "restart": False}
