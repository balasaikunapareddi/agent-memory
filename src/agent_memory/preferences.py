"""Deterministic preference extraction.

Preference learning is code, not prompt: a small set of explicit rules
reads stated preferences out of a user message and stores them as
structured key/value pairs. The rules are intentionally narrow — they
only fire on clear statements ("I prefer email over calls", "my
timezone is America/Chicago") so the agent never invents a preference
the user did not state.
"""

from __future__ import annotations

import re

_CHANNELS = {
    "email": "email",
    "e-mail": "email",
    "call": "phone",
    "calls": "phone",
    "phone": "phone",
    "text": "sms",
    "texts": "sms",
    "sms": "sms",
}

_RULES: list[tuple[re.Pattern[str], str]] = [
    # "I prefer email over calls" / "prefer texts instead of calls"
    (
        re.compile(
            r"prefer\s+(email|e-mail|calls?|phone|texts?|sms)\s+(?:over|instead of|to)\b",
            re.IGNORECASE,
        ),
        "contact_channel",
    ),
    # "my timezone is America/Chicago" / "timezone: US/Eastern"
    (
        re.compile(r"time\s*zone\s*(?:is|:)\s*([A-Za-z_/+-]+)", re.IGNORECASE),
        "timezone",
    ),
    # "call me Sam"
    (re.compile(r"\bcall me\s+([A-Z][a-zA-Z'-]+)", re.IGNORECASE), "preferred_name"),
    # "my email is sam@example.com"
    (
        re.compile(r"my email is\s+([\w.+-]+@[\w-]+\.[\w.]+)", re.IGNORECASE),
        "email",
    ),
    # "my phone is 555-123-4567"
    (
        re.compile(r"my phone(?: number)? is\s+([\d\-+() ]{7,})", re.IGNORECASE),
        "phone",
    ),
]


def extract_preferences(text: str) -> dict[str, str]:
    """Return the preferences explicitly stated in one user message."""
    found: dict[str, str] = {}
    for pattern, key in _RULES:
        match = pattern.search(text)
        if not match:
            continue
        value = match.group(1).strip()
        if key == "contact_channel":
            value = _CHANNELS[value.lower()]
        if key == "phone":
            value = value.strip().rstrip(".")
        found[key] = value
    return found
