"""Speech-to-text tunables read by ``dana.audio`` (Whisper model, sample
rate, transcript filters and gain) — plain literals, never reassigned.
"""

from __future__ import annotations

import re

WHISPER_ID = "distil-whisper/distil-small.en"
WHISPER_TASK = "transcribe"

# Audio constants
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
