"""
TBCare - Storage
"""
import json
import uuid
import sqlite3
import logging
from datetime import datetime, timedelta
from config import DB_PATH

logger = logging.getLogger(__name__)
SESSION_TIMEOUT_DAYS = 7


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS leads (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id          TEXT    NOT NULL UNIQUE,
            phone_number        TEXT,
            patient_name        TEXT,
            mobile              TEXT,
            age                 TEXT,
            disease_id          INTEGER,
            disease_name        TEXT,
            patient_type        TEXT,
            api_patient_id      INTEGER,
            api_lead_id         INTEGER,
            action_taken        TEXT,
            callback_number     TEXT,
            selected_center_name  TEXT,
            selected_center_id    INTEGER,
            selected_center_map   TEXT,
            appointment_date      TEXT,
            appointment_date_iso  TEXT,
            selected_slot_iso     TEXT,
            appointment_id        INTEGER,
            booking_confirmed     INTEGER DEFAULT 0,
            ai_history          TEXT,
            ai_collected        TEXT,
            session_state       TEXT,
            conversation_done   INTEGER DEFAULT 0,
            created_at          TEXT    NOT NULL,
            updated_at          TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS conversation_logs (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role       TEXT NOT NULL,
            content    TEXT NOT NULL,
            timestamp  TEXT NOT NULL
        );
    """)
    conn.commit()
    conn.close()


class LeadStorage:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self._conn = sqlite3.connect(DB_PATH)
        self._conn.row_factory = sqlite3.Row
        now = datetime.now().isoformat()
        self._conn.execute(
            "INSERT OR IGNORE INTO leads (session_id, created_at, updated_at) VALUES (?,?,?)",
            (session_id, now, now),
        )
        self._conn.commit()

    def update(self, **fields) -> None:
        if not fields:
            return
        fields["updated_at"] = datetime.now().isoformat()
        cols   = ", ".join(f"{k}=?" for k in fields)
        values = list(fields.values()) + [self.session_id]
        self._conn.execute(f"UPDATE leads SET {cols} WHERE session_id=?", values)
        self._conn.commit()

    def get(self) -> dict:
        row = self._conn.execute(
            "SELECT * FROM leads WHERE session_id=?", (self.session_id,)
        ).fetchone()
        return dict(row) if row else {}

    def log(self, role: str, content: str) -> None:
        self._conn.execute(
            "INSERT INTO conversation_logs (session_id, role, content, timestamp)"
            " VALUES (?,?,?,?)",
            (self.session_id, role, content, datetime.now().isoformat()),
        )
        self._conn.commit()

    def save_session(self, history: list, collected: dict, s: dict) -> None:
        self.update(
            ai_history    = json.dumps(history,   ensure_ascii=False),
            ai_collected  = json.dumps(collected, ensure_ascii=False),
            session_state = json.dumps(s,         ensure_ascii=False),
        )

    def load_session(self) -> tuple[list, dict, dict]:
        row = self.get()
        history   = json.loads(row["ai_history"])    if row.get("ai_history")    else []
        collected = json.loads(row["ai_collected"])  if row.get("ai_collected")  else {}
        s         = json.loads(row["session_state"]) if row.get("session_state") else {}
        return history, collected, s

    def close(self):
        self._conn.close()

    @classmethod
    def get_or_create_by_phone(cls, phone: str) -> "LeadStorage":
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT session_id, conversation_done, updated_at FROM leads"
            " WHERE phone_number=? ORDER BY created_at DESC LIMIT 1",
            (phone,)
        ).fetchone()
        conn.close()

        if row and not row["conversation_done"]:
            last_activity = datetime.fromisoformat(row["updated_at"])
            if datetime.now() - last_activity < timedelta(days=SESSION_TIMEOUT_DAYS):
                return cls(row["session_id"])

        new_id   = str(uuid.uuid4())
        instance = cls(new_id)
        instance.update(phone_number=phone)
        return instance
