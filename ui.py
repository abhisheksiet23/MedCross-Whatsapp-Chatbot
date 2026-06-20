"""
TBCare V2 - Terminal UI
All Rich display helpers. Zero business logic.
"""
import time
from rich.console import Console
from rich.panel   import Panel
from rich.text    import Text
from rich.table   import Table
from rich.rule    import Rule
from rich         import box

console = Console()

C_BOT    = "bright_cyan"
C_USER   = "bright_yellow"
C_GOOD   = "bright_green"
C_BAD    = "bright_red"
C_DIM    = "grey50"
C_CLINIC = "green3"
C_INFO   = "steel_blue1"


def clear_and_header() -> None:
    console.clear()
    title = Text(justify="center")
    title.append("🏥  TBCare Clinic Chain\n", style="bold white")
    title.append("Doctor Referral AI Assistant\n", style=f"italic {C_BOT}")
    title.append("─" * 44 + "\n", style=C_DIM)
    title.append("Type 'exit' to quit  ·  'restart' to start over  ·  'help' for options", style=C_DIM)
    console.print(Panel(title, border_style=C_CLINIC, box=box.DOUBLE, padding=(1, 6)))
    console.print()


def bot_say(message: str) -> None:
    """Display bot message with a slight delay for natural pacing."""
    console.print(
        Panel(
            Text(message, style="white"),
            title=f"[{C_BOT}]🏥 TBCare Assistant[/{C_BOT}]",
            border_style=C_BOT,
            padding=(1, 2),
        )
    )
    time.sleep(0.2)


def get_input() -> str:
    console.print(f"\n[{C_USER}]▶  You:[/{C_USER}] ", end="")
    return console.input("").strip()


def thinking() -> None:
    console.print(f"[{C_DIM}]\n🔄  Thinking…[/{C_DIM}]")


def success(msg: str) -> None:
    console.print(f"\n[{C_GOOD}]✅  {msg}[/{C_GOOD}]")


def error(msg: str) -> None:
    console.print(f"\n[{C_BAD}]❌  {msg}[/{C_BAD}]")


def divider() -> None:
    console.print(Rule(style=C_DIM))


def show_lead_summary(data: dict) -> None:
    """Show collected patient data as a neat table (for dev/debug)."""
    tbl = Table(box=box.SIMPLE, border_style=C_DIM, show_header=False)
    tbl.add_column("Field", style=C_INFO)
    tbl.add_column("Value", style="white")
    for label, key in [
        ("Name",       "patient_name"),
        ("Age",        "age"),
        ("City",       "city"),
        ("Condition",  "disease_type"),
        ("Referred By","referral_doctor"),
        ("Ref Code",   "referral_code"),
        ("Booking",    "appointment_date"),
        ("Time",       "appointment_time"),
    ]:
        val = data.get(key)
        if val:
            tbl.add_row(label, str(val))
    if tbl.row_count:
        console.print("\n", tbl)
