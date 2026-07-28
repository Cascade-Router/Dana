"""English argument normalization for tool execution."""

from __future__ import annotations

import re
import unicodedata
from typing import Any


def contains_non_latin_script(text: str) -> bool:
    """True when text includes Arabic-script code points (legacy helper)."""
    return bool(re.search(r"[\u0600-\u06FF]", text or ""))


# Back-compat alias used by older call sites (always safe; English release).
contains_language = contains_non_latin_script


def detect_lang(text: str) -> str:
    """Language detector stub — public release is English-only."""
    _ = text
    return "en"


def normalize_text(text: str) -> str:
    """NFC + whitespace fold for tool argument matching."""
    if not text:
        return ""
    out = unicodedata.normalize("NFC", text)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def normalize_tool_arguments(arguments: dict[str, Any], source_lang: str) -> dict[str, Any]:
    """Normalize string args for tool dispatch (English-only release)."""
    _ = source_lang
    cleaned: dict[str, Any] = {}
    for key, value in arguments.items():
        if isinstance(value, str):
            s = normalize_text(value)
            if re.fullmatch(r"[A-Za-z_]+", s):
                s = s.lower()
            cleaned[str(key)] = s
        else:
            cleaned[str(key)] = value
    return cleaned
