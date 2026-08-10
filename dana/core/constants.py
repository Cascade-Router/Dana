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
# Default Whisper language; overridden at runtime from settings.json (English-first).
WHISPER_LANGUAGE = "english"
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
WAKE_CHUNK = 1280  # 80 ms @ 16 kHz
# Local dana.onnx is sticky on this mic (can sit at ~0.99 on hush).
# Require a real onset: score must rise from low -> high, not stay pegged.
WAKE_THRESHOLD = 0.65
WAKE_MIN_CONSECUTIVE = 3  # ~240 ms of consecutive high scores
WAKE_ONSET_BELOW = 0.45  # must have been below this recently before a hit
WAKE_ONSET_LOOKBACK = 12  # ~1 s of score history
WAKE_PHRASE_WINDOW_CHUNKS = 18  # ~1.44 s rolling buffer for phrase verify
WAKE_PHRASE_VERIFY = False  # skip Whisper second gate; openWakeWord score onset starts session
WAKE_COOLDOWN_SEC = 6.0
WAKE_PHRASE_TOKENS = ("dana", "hey dana", "hey, dana")
# Whisper often mishears "Dana" as these; treat as wake confirmations.
# Include "don't know" / donald / donut to cut false negatives on quiet mics;
# OpenWakeWord score+onset still gate hush false-positives.
WAKE_PHRASE_ALIASES = (
    "dana",
    "hey dana",
    "donald",
    "hey donald",
    "donut",
    "don t know",
    "dont know",
    "don know",
    "dana dana",
    "dawn",
    "hey dawn",
)
# Remaining silence / noise transcripts that must never confirm a wake.
WAKE_PHRASE_REJECT = (
    "i don t know",
    "do not know",
    "i do not know",
    "i dont know",
    "dunno",
    "i dunno",
)
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
# Silero VAD window @ 16 kHz (512 samples ≈ 32 ms).
VAD_FRAME_SAMPLES = 512
VAD_FRAME_MS = int(round(1000.0 * VAD_FRAME_SAMPLES / SAMPLE_RATE))
# Natural cadence: 1.5s tolerates conversational mid-sentence pauses.
VAD_SILENCE_MS = 1500
VAD_MAX_SECONDS = 10.0  # initial wake failsafe (empty-room timeout)
FOLLOWUP_VAD_MAX_SECONDS = 9.0  # silence timeout while waiting for a follow-up
VAD_MIN_SPEECH_MS = 120
VAD_PRE_ROLL_FRAMES = 10  # keep ~320 ms before speech onset @ 32 ms frames
# After short wake ack ("Yes?"): let speakers clear, then discard queued echo frames.
POST_ACK_VAD_GRACE_SEC = 0.6
POST_ACK_SETTLE_SEC = POST_ACK_VAD_GRACE_SEC
POST_ACK_FLUSH_SEC = POST_ACK_VAD_GRACE_SEC
POST_ACK_IGNORE_ONSET_MS = 200.0  # residual onset ignore after grace/flush
FOLLOWUP_FLUSH_SEC = 0.05
# First-order DC blocker pole (closer to 1.0 = lower cutoff). ~0.995 @ 16 kHz
# removes mic DC offset / rumble that otherwise keeps VAD from silence_cutoff.
DC_BLOCKER_R = 0.995
# Barge-in while Dana speaks: Silero probability (not raw RMS) gates interrupt.
BARGE_IN_RMS = 0.12  # retained for adaptive helpers / logging only
BARGE_IN_MIN_SPEECH_MS = 350.0
# Suppress barge-in for the first 400ms after TtsWorker begins a play turn
# (speaker pop / room echo at playback onset).
BARGE_IN_PLAYBACK_GRACE_MS = 400.0
BARGE_IN_SETTLE_MS = BARGE_IN_PLAYBACK_GRACE_MS
BARGE_IN_CHUNK_MS = 50.0  # TTS write chunk size (interrupt granularity)
BARGE_IN_AMBIENT_MULT = 80.0  # threshold >= ambient_rms * this
# Stream barge-in: require strong Silero speech for N consecutive frames (~128ms).
BARGE_IN_SILERO_THRESHOLD = 0.85
BARGE_IN_SILERO_CONSEC_FRAMES = 4
# Sharp RMS spike retained as diagnostic floor only (not used to interrupt).
STREAM_BARGE_RMS = 0.09
MIC_AMBIENT_DEAD_RMS = 1e-4  # probe below this -> soft gain / adaptive floors
# Skip OpenWakeWord predict() on near-silence / dead virtual mics (phantom wakes).
DEAD_MIC_RMS_FLOOR = 0.0001
# TTS recovery: max wait for queue drain; hard cap per Piper utterance (synth+play).
# Long vision/OCR summaries need headroom — 18s was aborting mid-sentence.
TTS_IDLE_WAIT_TIMEOUT = 12.0
TTS_UTTERANCE_MAX_SECONDS = 90.0
# Soft-split long replies into independent spool items (~15-20s Piper each).
TTS_CHUNK_MAX_CHARS = 280
MIN_SPEECH_RMS = 0.01  # after peak-normalize; reject near-silence hallucinations
WAKEWORD_MODELS = ["dana"]
SENDGRID_MAIL_URL = "https://api.sendgrid.com/v3/mail/send"
