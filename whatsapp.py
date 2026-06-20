"""
TBCare - WhatsApp API Layer v4b
parse_incoming now returns (phone, text, msg_id) — msg_id enables
server.py to detect button/list selections by their ID, not just title.
"""
import logging
import requests
import json
from config import WHATSAPP_TOKEN, WHATSAPP_PHONE_NUMBER_ID, WHATSAPP_API_VERSION

logger    = logging.getLogger(__name__)
GRAPH_URL = f"https://graph.facebook.com/{WHATSAPP_API_VERSION}"


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type":  "application/json",
    }


# 1. Plain text 
def send_message(to: str, text: str) -> dict:
    url     = f"{GRAPH_URL}/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "to": to, "type": "text",
        "text": {"body": text},
    }
    try:
        resp = requests.post(url, json=payload, headers=_headers(), timeout=10)
        resp.raise_for_status()
        logger.info("Text sent to %s: %.60s…", to, text)
        return resp.json()
    except Exception as e:
        logger.error("send_message failed: %s", e)
        raise


# 2. Button message (max 3) 
def send_button_message(to: str, body: str, buttons: list[dict]) -> dict:
    """
    buttons = [{"id": "unique_id", "title": "Label (max 20 chars)"}, ...]
    Title is shown on the button. ID comes back in parse_incoming as msg_id.
    """
    url     = f"{GRAPH_URL}/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "to": to, "type": "interactive",
        "interactive": {
            "type": "button",
            "body": {"text": body},
            "action": {
                "buttons": [
                    {"type": "reply",
                     "reply": {"id": b["id"], "title": b["title"][:20]}}
                    for b in buttons[:3]
                ]
            }
        }
    }
    try:
        resp = requests.post(url, json=payload, headers=_headers(), timeout=10)
        resp.raise_for_status()
        logger.info("Button msg sent to %s (%d buttons)", to, len(buttons))
        return resp.json()
    except Exception as e:
        logger.error("send_button_message failed: %s", e)
        return send_message(to, body)   # fallback


# 3. List message (max 10 items) 
def send_list_message(to: str, body: str,
                      button_label: str, sections: list[dict]) -> dict:
    """
    sections = [{"title": "...", "rows": [{"id":"..","title":"..","description":".."}]}]
    """
    url     = f"{GRAPH_URL}/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "to": to, "type": "interactive",
        "interactive": {
            "type": "list",
            "body": {"text": body},
            "action": {
                "button":   button_label[:20],
                "sections": sections,
            }
        }
    }
    try:
        resp = requests.post(url, json=payload, headers=_headers(), timeout=10)
        resp.raise_for_status()
        logger.info("List msg sent to %s", to)
        return resp.json()
    except Exception as e:
        logger.error("send_list_message failed: %s", e)
        logger.error("Failed payload: %s", json.dumps(payload, ensure_ascii=False))
        logger.error("WhatsApp response: %s", resp.text if resp else "no response")
        return send_message(to, body)   # fallback


# 4. Template (initiate conversation) 
def send_template_message(to: str, template_name: str,
                          language_code: str = "en") -> dict:
    url     = f"{GRAPH_URL}/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "to": to, "type": "template",
        "template": {"name": template_name, "language": {"code": language_code}},
    }
    try:
        resp = requests.post(url, json=payload, headers=_headers(), timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error("send_template_message failed: %s", e)
        raise


# 5. Mark as read 
def mark_as_read(message_id: str) -> None:
    url     = f"{GRAPH_URL}/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    payload = {"messaging_product": "whatsapp",
               "status": "read", "message_id": message_id}
    try:
        requests.post(url, json=payload, headers=_headers(), timeout=5)
    except Exception:
        pass


# 6. Parse incoming 
def parse_incoming(payload: dict) -> tuple[str, str, str | None] | None:
    """
    Returns (phone, text, msg_id) or None.

    msg_id is the button/list item ID for interactive messages, None for text.
    server.py uses msg_id to detect special actions (e.g. "use_wa_number")
    without parsing the user-visible title text.
    """
    try:
        entry    = payload["entry"][0]
        changes  = entry["changes"][0]
        value    = changes["value"]

        if "messages" not in value:
            return None

        message  = value["messages"][0]
        phone    = message["from"]
        msg_type = message.get("type")

        if msg_type == "text":
            return phone, message["text"]["body"].strip(), None

        if msg_type == "interactive":
            interactive = message["interactive"]

            if interactive["type"] == "button_reply":
                br = interactive["button_reply"]
                logger.info("Button reply from %s: id=%s title=%s",
                            phone, br["id"], br["title"])
                return phone, br["title"], br["id"]

            if interactive["type"] == "list_reply":
                lr = interactive["list_reply"]
                logger.info("List reply from %s: id=%s title=%s",
                            phone, lr["id"], lr["title"])
                return phone, lr["title"], lr["id"]

        logger.info("Unhandled type '%s' from %s", msg_type, phone)
        return None

    except (KeyError, IndexError) as e:
        logger.warning("Could not parse payload: %s", e)
        return None
