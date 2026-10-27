"""Standard-library CSV loaders. Loading only - no joins or business logic.

Resolving accounts, interpreting usage and finding related tickets is part of
the task. Note the realistic noise described in data/README.md.
"""
import csv
import math
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


def _read_csv(name: str) -> list[dict[str, str]]:
    """Read a task CSV as row dictionaries, preserving blank cells as empty strings."""
    with (DATA / name).open("r", newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def _number(value: str):
    """Parse numeric CSV fields without converting blanks into zero."""
    value = value.strip()
    if not value:
        return None
    try:
        number = float(value)
    except ValueError:
        return value
    if not math.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


def load_accounts() -> list[dict[str, str]]:
    """Load account rows with every source field preserved as text."""
    return _read_csv("accounts.csv")


def load_tickets() -> list[dict[str, str]]:
    """Load ticket rows with blank fields preserved as empty strings."""
    return _read_csv("tickets.csv")


def load_usage() -> list[dict]:
    """Load usage rows, keeping identifiers/months textual and metrics numeric."""
    rows = _read_csv("usage_metrics.csv")
    text_fields = {"account_id", "month"}
    for row in rows:
        for field in row:
            if field not in text_fields:
                row[field] = _number(row[field])
    return rows


def load_review_set() -> list[str]:
    return [l.strip() for l in (DATA / "review_set.txt").read_text().splitlines() if l.strip()]


AS_OF_DATE = "2026-10-20"  # treat this as "today" for all date calculations
