"""Text cleanup before speech synthesis: markdown, code blocks and newlines
turned into something Piper can read aloud (see tts_worker.synthesize_to_file).
"""

from __future__ import annotations

import re


def strip_code_blocks_for_tts(text: str) -> str:
    """Replace markdown fenced code with a short spoken placeholder.

    Prevents Piper from reading raw Python/JSON aloud (long TTS ceiling crash).
    """
    from dana.audio import speech_state as state

    raw = text or ""
    if "```" not in raw:
        return raw
    out = state._CODE_FENCE_TTS_RE.sub("[Code block generated]", raw)
    out = state._CODE_FENCE_TTS_UNCLOSED_RE.sub("[Code block generated]", out)
    out = re.sub(r"(?:\s*\[Code block generated\]\s*){2,}", " [Code block generated] ", out)
    return out.strip()


def sanitize_text_for_tts(text: str) -> str:
    """Strip markdown emphasis/code markers before Piper synthesis.

    Returns empty string when nothing speakable remains (caller must skip TTS).
    Newlines become pause punctuation so OCR/vision dumps are not rushed.
    """
    from dana.audio import speech_state as state

    out = strip_code_blocks_for_tts(text or "")
    out = state._TTS_MD_MARKERS_RE.sub("", out)
    # Florence / vision dumps often arrive as newline-joined tokens with no stops.
    out = re.sub(r"[\r\n]+", ". ", out)
    out = re.sub(r"\s+", " ", out).strip()
    # Soft-break very long comma-less runs so Piper inserts breath pauses.
    if len(out) > 160 and not re.search(r"[.!?]", out):
        out = re.sub(r"(.{40,80}?)\s+", r"\1. ", out)
    if not out or state._PUNCT_OR_SPACE_ONLY_RE.match(out):
        return ""
    return out


__all__ = ("sanitize_text_for_tts", "strip_code_blocks_for_tts")
