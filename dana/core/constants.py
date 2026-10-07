"""App-wide tunable constants, extracted verbatim from ``dana.core_agent``.

These are pure, never-reassigned literals (verified: none of them appear in
any ``global`` statement in ``core_agent.py``) read across every bucket of
the eventual decomposition — vision (YOLO/tracker), agent-loop (Ollama),
audio (wake-word/VAD/Whisper/TTS/barge-in). Centralizing them here means
Phase 3 (audio) and Phase 4 (agent-loop) can each import config values from
one neutral module without depending on each other or on core_agent.py.
"""

from __future__ import annotations

import os
import re

from dana.paths import YOLO_WEIGHTS_PATH

MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"  # retained for optional future vision
WHISPER_ID = "distil-whisper/distil-small.en"
WHISPER_TASK = "transcribe"
YOLO_WEIGHTS = str(YOLO_WEIGHTS_PATH)
FRAME_SIZE = (640, 480)  # (width, height)
YOLO_CONF = 0.35
# Tracker samples active_vision_tool into a maxlen-60 thumbnail buffer @ ~1 fps.
TRACKER_SLEEP_SEC = 0.25
TRACKER_BUFFER_INTERVAL_S = 1.0

# Local Ollama conversational brain
OLLAMA_URL = "http://localhost:11434/api/chat"
# Prefer DANA_OLLAMA_MODEL / OLLAMA_MODEL so Suite 11 can pin Qwen2.5-Coder.
OLLAMA_MODEL = (
    (os.environ.get("DANA_OLLAMA_MODEL") or os.environ.get("OLLAMA_MODEL") or "")
    .strip()
    or "qwen2.5-coder:7b"
)
OLLAMA_TIMEOUT_SEC = 180.0

# Intent-router keywords for dynamic vision tool switching.
SCREEN_KEYWORDS = ["screen", "monitor", "code", "display", "desktop"]
CAMERA_KEYWORDS = ["camera", "room", "me", "face", "physical", "look at me"]

# Audio / wake-word constants
SAMPLE_RATE = 16000
WAKE_PHRASE_VERIFY = False  # skip Whisper second gate; openWakeWord score onset starts session
# Legacy bias string — kept for echo detection only. Live STT must NOT pass
# initial_prompt / prompt_ids (ticket-log regurgitation / context loops).
WHISPER_INITIAL_PROMPT = (
    "Dana, Titan initiative, Titan Protocol, Titan supervisor, "
    "activate Titan, Vanguard Protocol, "
    "bye, quit, exit, lockdown, lock yourself."
)
# Discard transcripts denser than a realistic speaking rate (hallucinated dumps).
WHISPER_MAX_WORDS_PER_SEC = 5.0
# Post-STT repairs for known proper nouns Distil-Whisper / Whisper often mangle.
_STT_NAME_FIXES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bAmir\s*[- ]?\s*Hosein\b", re.I), "Amirhosein"),
    (re.compile(r"\bAmir\s+Hussain\b", re.I), "Amirhosein"),
    (re.compile(r"\bAmir\s+Hussein\b", re.I), "Amirhosein"),
    (re.compile(r"\bAMIRHOSEN\b", re.I), "Amirhosein"),
    (re.compile(r"\bAmirhos(?:e|ei|i)n\b", re.I), "Amirhosein"),
    (re.compile(r"\bAmy\s+Hors(?:e)?t\b", re.I), "Amirhosein"),
    (re.compile(r"\bAmi\s+Hosein\b", re.I), "Amirhosein"),
    (re.compile(r"\bNarius\b", re.I), "Narges"),
    (re.compile(r"\bNarjis\b", re.I), "Narges"),
    (re.compile(r"\bAR[- ]?GES\b", re.I), "Narges"),
    (re.compile(r"\bNarg(?:es|is|ez)\b", re.I), "Narges"),
    # Accent / phoneme swaps Whisper-base makes on short questions.
    (re.compile(r"\bwife'?s?\s+saying\b", re.I), "wife's name"),
    (re.compile(r"\bwife\s+saying\b", re.I), "wife's name"),
    (re.compile(r"\bpartner'?s?\s+saying\b", re.I), "partner's name"),
    (re.compile(r"\bwhy time'?s on\b", re.I), "what time it is"),
    (re.compile(r"\bwhat time'?s on\b", re.I), "what time it is"),
)
# Quiet-mic / headphone gain: soft speech often lands at rms≈0.0001–0.01.
WHISPER_TARGET_RMS = 0.05
WHISPER_GAIN_RMS_CEIL = 0.02
# Absolute floor for Whisper gain (headphone conversational levels).
WHISPER_MIN_RMS_FOR_GAIN = 0.00005
WHISPER_MAX_GAIN = 64.0
# First-order DC blocker pole (closer to 1.0 = lower cutoff). ~0.995 @ 16 kHz
# removes mic DC offset / rumble that otherwise keeps VAD from silence_cutoff.
DC_BLOCKER_R = 0.995
SENDGRID_MAIL_URL = "https://api.sendgrid.com/v3/mail/send"
