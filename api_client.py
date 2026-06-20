"""
TBCare - API Client
Handles all calls to the LeadProspect backend API.

CENTER_MAP: maps center_id (from practices API) to the correct
doctorId and locationId needed for slots and booking APIs.
Add new centers here as the clinic expands.
"""
import logging
import requests
from datetime import datetime, timedelta
from config import BASE_URL, CHATBOT_SOURCE_ID

logger = logging.getLogger(__name__)


# Center mapping 
# Key   = center_id (id from practices API)
# Value = doctorId and locationId required by slots/book APIs
CENTER_MAP = {
    2: {"doctor_id": 4,   "location_id": 1},
    4: {"doctor_id": 29,  "location_id": 2},
    8: {"doctor_id": 106, "location_id": 3},
    9: {"doctor_id": 159, "location_id": 4},
}


def _resolve_center(center_id: int) -> tuple[int, int]:
    """
    Returns (doctor_id, location_id) for a given center_id.
    Falls back to (0, center_id) if center not in CENTER_MAP.
    """
    mapping = CENTER_MAP.get(center_id)
    if mapping:
        return mapping["doctor_id"], mapping["location_id"]
    logger.warning("No mapping found for center_id=%s — using fallback", center_id)
    return 0, center_id   # 0 = auto-assign doctor, center_id as location fallback


# Internal helpers 
def _get(endpoint: str, params: dict = None) -> dict:
    url = f"{BASE_URL}{endpoint}"
    logger.info("API GET  → %s | params: %s", endpoint, params)
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    result = resp.json()
    logger.info("API GET  ← %s | status: %s | response: %s",
                endpoint, resp.status_code, str(result)[:200])
    return result


def _post(endpoint: str, payload: dict) -> dict:
    url = f"{BASE_URL}{endpoint}"
    logger.info("API POST → %s | payload: %s", endpoint, payload)
    resp = requests.post(url, json=payload, timeout=10)
    resp.raise_for_status()
    result = resp.json()
    logger.info("API POST ← %s | status: %s | response: %s",
                endpoint, resp.status_code, str(result)[:200])
    return result


# 1. Identify Patient 
def identify_patient(mobile: str = None, name: str = None,
                     kmedid: str = None) -> dict:
    """
    3-layer check: KMed ID → mobile → name.
    Returns patientType: existing_patient | existing_lead | not_found
    """
    params = {}
    if mobile: params["mobile"] = str(mobile).strip()
    if name:   params["name"]   = name
    if kmedid: params["kmedid"] = kmedid

    try:
        result = _get("/identify", params)
        logger.info("Identify result: %s",
                    result.get("data", {}).get("patientType"))
        return result
    except Exception as e:
        logger.error("identify_patient failed: %s", e)
        return {"success": False, "data": {"patientType": "not_found"}}


# 2. Register New Lead 
def add_patient(name: str, mobile: str, age: int = None,
                disease_id: int = None, center_id: int = None,
                note: str = "Chatbot enquiry") -> dict:
    """
    Register a new lead from the chatbot.
    Returns { leadId, name, mobile }
    """
    payload = {
        "name":     name,
        "mobile":   int(str(mobile).strip()),
        "note":     note,
        "sourceId": CHATBOT_SOURCE_ID,
    }
    if age        is not None: payload["age"]              = int(age)
    if disease_id is not None: payload["disease"]          = disease_id
    if center_id  is not None: payload["referralCenterId"] = center_id

    try:
        result = _post("/addpatient", payload)
        logger.info("add_patient: leadId=%s smsSent=%s",
                    result.get("data", {}).get("leadId"), result.get("smsSent"))
        return result
    except Exception as e:
        logger.error("add_patient failed: %s", e)
        return {"success": False, "data": {}}


# 3. Get Available Slots 
def get_slots(center_id: int, date_str: str) -> dict:
    """
    Fetch available 30-min slots for a center + date.
    center_id = practices API id
    date_str  = YYYY-MM-DD
    Doctor and location are resolved automatically from CENTER_MAP.
    """
    doctor_id, location_id = _resolve_center(center_id)

    params = {
        "practiceId": center_id,
        "locationId": location_id,
        "doctorId":   doctor_id,
        "date":       date_str,
    }
    try:
        result = _get("/slots", params)
        available = [s for s in result.get("data", []) if s.get("isAvailable")]
        logger.info("get_slots centerId=%s locationId=%s doctorId=%s date=%s: %d available",
                    center_id, location_id, doctor_id, date_str, len(available))
        return result
    except Exception as e:
        logger.error("get_slots failed: %s", e)
        return {"success": False, "data": []}


# 4. Book Appointment — Existing Patient 
def book_appointment(patient_id: int, center_id: int, start_time: str,
                     note: str = "Booked via Chatbot") -> dict:
    """
    Book for an existing KMed patient.
    start_time = ISO 8601 e.g. "2026-05-07T10:00:00"
    Returns { visitId, appointmentDateTime, doctorName, patientName }
    """
    doctor_id, location_id = _resolve_center(center_id)
    end_time = (datetime.fromisoformat(start_time) + timedelta(minutes=30)).isoformat()

    payload = {
        "patientId":   patient_id,
        "practiceId":  center_id,
        "locationId":  location_id,
        "doctorId":    doctor_id,
        "startTime":   start_time,
        "endTime":     end_time,
        "scheduledBy": 1,
        "note":        note,
    }
    try:
        result = _post("/book", payload)
        logger.info("book_appointment: visitId=%s doctor=%s",
                    result.get("data", {}).get("visitId"),
                    result.get("data", {}).get("doctorName"))
        return result
    except Exception as e:
        logger.error("book_appointment failed: %s", e)
        return {"success": False, "data": {}}


# 5. Book Appointment — Lead / Prospect 
def book_lead_appointment(lead_id: int, center_id: int, start_time: str,
                          note: str = "Prospect appointment via Chatbot") -> dict:
    """
    Book for a lead/prospect patient.
    lead_id    = ReferredPatientByDoctor.row_id from add_patient or identify
    start_time = ISO 8601 e.g. "2026-05-07T11:00:00"
    Returns { appointmentId, leadId, leadName, appointmentDateTime, doctorName }
    """
    doctor_id, location_id = _resolve_center(center_id)
    end_time = (datetime.fromisoformat(start_time) + timedelta(minutes=30)).isoformat()

    payload = {
        "leadPatientId": lead_id,
        "practiceId":    center_id,
        "locationId":    location_id,
        "doctorId":      doctor_id,
        "startTime":     start_time,
        "endTime":       end_time,
        "note":          note,
    }
    try:
        result = _post("/book-lead", payload)
        logger.info("book_lead_appointment: appointmentId=%s doctor=%s",
                    result.get("data", {}).get("appointmentId"),
                    result.get("data", {}).get("doctorName"))
        return result
    except Exception as e:
        logger.error("book_lead_appointment failed: %s", e)
        return {"success": False, "data": {}}


#  Helpers 
def format_slots_for_display(slots_data: list) -> str:
    """Format API slot list into a clean numbered string for the AI to present."""
    available = [s for s in slots_data if s.get("isAvailable")]
    if not available:
        return "No slots available for this date. Please try another date."
    lines = [f"  {i+1}. {s['slotLabel']}" for i, s in enumerate(available)]
    return "\n".join(lines)


def find_location_by_name(name: str, practices: list) -> dict | None:
    """Find a practice dict by partial case-insensitive name match."""
    name_lower = name.lower()
    for p in practices:
        if name_lower in p["name"].lower() or p["name"].lower() in name_lower:
            return p
    return None
