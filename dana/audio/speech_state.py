"""Speech (STT + TTS text) state split out of ``dana.core.shared_state``.

``dana.audio.stt`` and the TTS text sanitizers in ``dana.audio.tts_manager``
need only these names. Importing them from here instead
of ``shared_state`` keeps the live server from loading the legacy voice/vision
stack (vault daemon client, YOLO agents) just to transcribe or speak.
``shared_state`` re-exports the shared objects, so legacy readers of
``state.stop_event`` etc. still see the same instances.
"""

from __future__ import annotations

import re
import threading
from typing import Any, Optional

# Optional Arabic-script detection (unused for English-only TTS routing).
ARABIC_SCRIPT_RE = re.compile(r"[؀-ۿ]")


# ---------------------------------------------------------------------------
# Whisper STT
# ---------------------------------------------------------------------------

# Shared Whisper bundle for wake-phrase verification (set by conversation_worker).
whisper_bundle_lock = threading.Lock()
whisper_bundle: Optional[tuple[Any, Any, Any, Any]] = None  # reassigned
# Set when background Whisper load finishes (success or failure).
whisper_ready = threading.Event()
_whisper_load_error: Optional[str] = None  # reassigned

# Global shutdown signal shared by the STT, mic and TTS threads.
stop_event = threading.Event()

# ---------------------------------------------------------------------------
# Whisper hallucination filters (constants)
# ---------------------------------------------------------------------------

# Common Whisper-tiny hallucinations on silence / static.
WHISPER_HALLUCINATIONS = {
    "",
    ".",
    ",",
    "!",
    "?",
    "...",
    "…",
    "you",
    "the",
    "a",
    "i",
    "oh",
    "uh",
    "um",
    "hmm",
    "thanks",
    "thank you",
    "thank you.",
    "thanks for watching",
    "thanks for watching.",
    "subscribe",
    "subscribe.",
    "bye",
    "bye.",
    "goodbye",
    "goodbye.",
    "okay",
    "ok",
    "yes",
    "no",
    "hello",
    "hi",
    "hey",
    "music",
    "applause",
    "laughter",
    "www.youtube.com",
    "please subscribe",
    "like and subscribe",
}

# Ambient-noise artifacts that must be discarded silently (no LLM, no apology TTS).
WHISPER_AMBIENT_SILENT = frozenset(
    {
        "",
        ".",
        ",",
        "!",
        "?",
        "...",
        "…",
        "thanks",
        "thank you",
        "thank you.",
        "thanks.",
        "thanks for watching",
        "thanks for watching.",
        "thank you for watching",
        "thank you for watching.",
        "bye",
        "bye.",
        "goodbye",
        "goodbye.",
        "subscribe",
        "subscribe.",
        "please subscribe",
        "like and subscribe",
        "music",
        "applause",
        "laughter",
        "www.youtube.com",
        "thanks for listening",
        "thank you for listening",
    }
)

_PUNCT_OR_SPACE_ONLY_RE = re.compile(r"^[\s\W_]+$", re.UNICODE)

# TTS text sanitizers (dana.audio.tts_manager.strip_code_blocks_for_tts / sanitize_text_for_tts).
_CODE_FENCE_TTS_RE = re.compile(r"```[\w+-]*\n?[\s\S]*?```", re.MULTILINE)
_CODE_FENCE_TTS_UNCLOSED_RE = re.compile(r"```[\w+-]*\n?[\s\S]*$", re.MULTILINE)
_TTS_MD_MARKERS_RE = re.compile(r"`+|\*{1,3}|_{2,}")
