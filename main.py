"""
TBCare - Conversation Orchestrator v4b
process_message and handle_whatsapp_message now return the full decision dict
so server.py can use AI-generated button/list content directly.
"""
import sys
import uuid
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from rich.console import Console
console = Console()
log = logging.getLogger(__name__)


def _setup_logging():
    from config import LOG_PATH
    handlers = [logging.FileHandler(LOG_PATH, encoding="utf-8")]
    if os.getenv("DEBUG", "").lower() in ("1", "true"):
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        handlers=handlers,
    )


def _validate_env():
    from config import OPENAI_API_KEY
    if not OPENAI_API_KEY:
        console.print("\n[bright_red]OPENAI_API_KEY not set in .env[/bright_red]\n")
        sys.exit(1)



#  CORE: process one message

def process_message(user_text: str, ai, db) -> tuple[list[str], bool, bool, dict]:
    """
    Returns: (replies, done, restart, decision)
    decision contains step, message_type, buttons, list_sections, etc.
    server.py uses decision directly for WhatsApp message type routing.
    """
    try:
        reply, decision = ai.send(user_text, db)
    except Exception as e:
        log.error("AI error: %s", e)
        fallback = {"step": "UNKNOWN", "message_type": "text",
                    "done": False, "restart": False}
        return ["Sorry, I had a technical issue. Please try again."], False, False, fallback

    db.log("assistant", reply)

    # Persist disease info when selected
    disease_id   = decision.get("disease_id")
    disease_name = decision.get("disease_name")
    if disease_id and disease_name:
        db.update(disease_id=disease_id, disease_name=disease_name)

    done    = bool(decision.get("done"))
    restart = bool(decision.get("restart"))
    return [reply], done, restart, decision



#  WHATSAPP: called by server.py per webhook message

def handle_whatsapp_message(phone: str,
                            user_text: str) -> tuple[list[str], dict]:
    """
    Returns: (replies, decision)
    decision carries step, message_type, buttons, list_sections, last_slots
    so server.py can send the right WhatsApp message format.
    """
    from ai_engine import AIEngine
    from storage   import LeadStorage

    db = LeadStorage.get_or_create_by_phone(phone)
    db.update(phone_number = phone)
    db.log("user", user_text)

    history, _, _ = db.load_session()
    ai            = AIEngine()
    ai.history    = history

    # First ever message from this number → send kickoff welcome
    if not ai.history:
        try:
            reply, decision = ai.kickoff(db)
            db.log("assistant", reply)
            db.save_session(ai.history, {}, {})
            db.close()
            decision["last_slots"] = []
            return [reply], decision
        except Exception as e:
            log.error("Kickoff error: %s", e)
            db.close()
            fallback = {"step": "LANGUAGE_SELECTION", "message_type": "text",
                        "done": False, "restart": False, "last_slots": []}
            return ["Welcome to MedCross! Technical issue — please try again."], fallback

    replies, done, restart, decision = process_message(user_text, ai, db)

    if done or restart:
        db.update(conversation_done=1)

    # Attach latest slots so server.py can build list message if needed
    decision["last_slots"] = ai._last_slots

    db.save_session(ai.history, {}, {})
    db.close()
    return replies, decision



#  CLI: python main.py

def run_session() -> bool:
    from ai_engine import AIEngine
    from storage   import LeadStorage
    import ui

    db = LeadStorage(str(uuid.uuid4()))
    ai = AIEngine()

    ui.clear_and_header()
    ui.thinking()

    try:
        reply, _ = ai.kickoff(db)
    except Exception as e:
        ui.error(f"Could not connect to OpenAI: {e}")
        db.close()
        return False

    ui.bot_say(reply)
    db.log("assistant", reply)

    while True:
        user_input = ui.get_input()
        if not user_input:
            continue

        db.log("user", user_input)
        ui.thinking()

        replies, done, restart, decision = process_message(user_input, ai, db)
        for r in replies:
            ui.bot_say(r)

        if restart:
            db.close()
            return True
        if done:
            ui.divider()
            db.close()
            return False

    db.close()
    return False


def main():
    _setup_logging()
    _validate_env()

    from storage import init_db
    init_db()

    try:
        restart = True
        while restart:
            restart = run_session()
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Goodbye! 👋[/yellow]\n")
    except Exception as e:
        logging.critical("Fatal: %s", e, exc_info=True)
        console.print(f"\n[bright_red]Fatal: {e}[/bright_red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
