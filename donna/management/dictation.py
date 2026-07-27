"""Stage 8.5 — Dictation Loop (step-by-step task learning).

Pairs a spoken/typed command with the latest Florence OCR visual state and
files it on the Blackboard ``dictation_sessions`` table.
"""

from __future__ import annotations

import re
from typing import Any

from donna.memory.blackboard import (
    is_dictation_mode,
    read_perception_ocr_text,
    record_dictation_session,
    set_dictation_mode,
)

_DICTATE_RE = re.compile(
    r"^\s*(?:hey\s+)?donna\b[\s,.\-!:]*",
    re.IGNORECASE,
)
_DICTATE_KEYWORD_RE = re.compile(
    r"\bdictate\b",
    re.IGNORECASE,
)
_STRIP_DICTATE_RE = re.compile(
    r"^\s*(?:hey\s+)?donna\b[\s,.\-!:]*|"
    r"^\s*dictate\b[\s,.\-!:]*|"
    r"\bdictate\b[\s,.\-!:]*",
    re.IGNORECASE,
)

DICTATION_ACK = "Dictation logged."


def mentions_dictate(text: str) -> bool:
    """True when the utterance contains the dictate keyword."""
    return bool(_DICTATE_KEYWORD_RE.search(text or ""))


def should_handle_dictation(text: str, *, db_path=None) -> bool:
    """Keyword hit or GUI-forced dictation mode."""
    if is_dictation_mode(db_path=db_path):
        return True
    return mentions_dictate(text)


def strip_dictate_wrapper(text: str) -> str:
    """Remove wake / dictate wrappers; keep the step instruction."""
    body = (text or "").strip()
    body = _DICTATE_RE.sub("", body, count=1).strip()
    # Drop leading/trailing dictate tokens while preserving inner words.
    body = re.sub(r"^\s*dictate\b[\s,.\-!:]*", "", body, flags=re.I).strip()
    body = re.sub(r"\bdictate\b[\s,.\-!:]*", " ", body, flags=re.I)
    body = re.sub(r"\s+", " ", body).strip(" ,.-")
    return body or (text or "").strip()


def _capture_visual_reference(*, force_ocr: bool = True) -> str:
    """Prefer typed ``perception.ocr``; optionally force Florence once."""
    visual = read_perception_ocr_text() or ""
    if visual.strip():
        return visual.strip()
    if not force_ocr:
        return ""
    try:
        from donna.tools.visual_tools import ocr_with_region

        obs = str(ocr_with_region(query="") or "")
        # Prefer re-read of typed topic after publish.
        visual = read_perception_ocr_text() or obs
    except Exception as exc:  # noqa: BLE001
        visual = f"(ocr_unavailable: {exc})"
    return (visual or "").strip()


def handle_dictation(
    text: str,
    *,
    db_path=None,
    force_ocr: bool = True,
) -> dict[str, Any]:
    """Log one dictation step + OCR visual reference; return session row + ack."""
    command = strip_dictate_wrapper(text)
    visual = _capture_visual_reference(force_ocr=force_ocr)
    row = record_dictation_session(
        command,
        visual_state_reference=visual,
        status="recorded",
        db_path=db_path,
    )
    return {
        "ok": True,
        "ack": DICTATION_ACK,
        "command_text": command,
        "visual_chars": len(visual),
        "session": row,
    }


def toggle_dictation_mode(active: bool | None = None, *, db_path=None) -> bool:
    """Flip or set GUI dictation latch; return new active state."""
    if active is None:
        active = not is_dictation_mode(db_path=db_path)
    return set_dictation_mode(bool(active), db_path=db_path)
