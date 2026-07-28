"""STT vocabulary middleware — fix known Whisper phonetic mangling before the LLM."""

from __future__ import annotations

import json
import re
from pathlib import Path

_VOCAB_PATH = Path(__file__).resolve().parent / "vocabulary.json"
_phrase_map: dict[str, str] | None = None
_phrase_patterns: list[tuple[re.Pattern[str], str]] | None = None


def _load_vocabulary() -> dict[str, str]:
    global _phrase_map, _phrase_patterns
    if _phrase_map is not None and _phrase_patterns is not None:
        return _phrase_map
    mapping: dict[str, str] = {}
    try:
        raw = json.loads(_VOCAB_PATH.read_text(encoding="utf-8"))
        mapping = {
            str(k).strip().lower(): str(v)
            for k, v in (raw.get("phrase_corrections") or {}).items()
            if str(k).strip() and str(v).strip()
        }
    except Exception:
        mapping = {}
    # Longest keys first so multi-word phrases win over fragments.
    patterns: list[tuple[re.Pattern[str], str]] = []
    for src, dst in sorted(mapping.items(), key=lambda kv: len(kv[0]), reverse=True):
        patterns.append(
            (re.compile(rf"(?<!\w){re.escape(src)}(?!\w)", re.IGNORECASE), dst)
        )
    _phrase_map = mapping
    _phrase_patterns = patterns
    return mapping


def reload_vocabulary() -> None:
    """Clear cache so tests / hot-edits pick up vocabulary.json changes."""
    global _phrase_map, _phrase_patterns
    _phrase_map = None
    _phrase_patterns = None
    _load_vocabulary()


def correct_titan_codename_stt(text: str) -> str:
    """Repair Whisper ``JSON``/``Jason`` slips when the user means Titan Protocol."""
    out = (text or "").strip()
    if not out:
        return out
    titan_ctx = re.compile(
        r"\b(?:initiative|protocol|supervisor|activate|deploy|engage|launch|"
        r"start|run|vanguard|titan|self[- ]?improvement)\b",
        re.I,
    )
    json_jason_cmd = re.compile(
        r"\b(?:activate|start|run|deploy|engage|launch)\s+(?:the\s+)?(?:json|jason)\b",
        re.I,
    )
    if not titan_ctx.search(out) and not json_jason_cmd.search(out):
        return out
    out = re.sub(r"\bjson\s+initiative\b", "Titan initiative", out, flags=re.I)
    out = re.sub(r"\bjson\s+protocol\b", "Titan Protocol", out, flags=re.I)
    out = re.sub(r"\bjson\s+supervisor\b", "Titan supervisor", out, flags=re.I)
    out = re.sub(r"\bJSON\b", "Titan", out)
    out = re.sub(r"\bJason\b", "Titan", out, flags=re.I)
    return out


def strip_trailing_punctuation_hallucinations(text: str) -> str:
    """Drop STT tail-noise punctuation without changing the spoken words."""
    out = (text or "").strip()
    if not out:
        return out
    # Whisper sometimes appends punctuation-only tails from room reflections.
    out = re.sub(r"\s+[\.,!?;:]+$", "", out)
    out = re.sub(r"[\.,!?;:]{2,}$", "", out)
    out = re.sub(r"\s+[\]\)\}\"'`]+$", "", out)
    out = re.sub(r"[\]\)\}\"'`]+$", "", out)
    return out.strip()


def correct_stt(text: str) -> str:
    """Apply phrase corrections from ``vocabulary.json`` (case-insensitive)."""
    out = (text or "").strip()
    if not out:
        return out
    _load_vocabulary()
    assert _phrase_patterns is not None
    for pattern, repl in _phrase_patterns:
        out = pattern.sub(repl, out)
    out = correct_titan_codename_stt(out)
    return strip_trailing_punctuation_hallucinations(out)
