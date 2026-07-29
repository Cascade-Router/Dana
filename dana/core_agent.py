"""
CAMGRASPER - Offline Voice-to-Voice Donna assistant.

Pipeline (4 threads + agent keep-alive loop + GUI main thread):
  1. Tracker   - YOLOv8n rolling buffer (~2s) from active_vision_tool + live ROI overlay
  2. WakeWord  - OpenWakeWord custom "donna.onnx" on mic @ 16 kHz
  3. Conversation - VAD -> Whisper STT -> tool_router -> YOLO + Ollama LLM -> TTS
  4. Audio     - offline TTS via piper-tts (en_US HFC female)

UI:
  - Windows system tray icon (Open Settings / Quit)
  - CustomTkinter Live Trace window (header mode + pipeline TraceCells; Stats/Audio tabs)

Dual-engine cascade:
  - Eyes: YOLO spatial labels from ScreenAgent / VideoAgent
  - Brain: local Ollama chat API (llama3.2 / 3B)

Audio devices are configured via settings.json (interactive first-run setup or GUI).
Long-term user profile is stored in an AES-256 encrypted vault (donna_memory.enc).

Triggers:
  - Say 'Donna' to wake (then multi-turn follow-up without wake word)
  - Create .trigger_ask (empty = listen; non-empty = inject transcript)
  - Enqueue tasks in CAMGRASPER/execution_jail/task_queue.json then wake — bypasses Whisper
  - Tray Quit / Ctrl+C to quit

Setup:
  python -m dana.core_agent --download   # one-time Whisper/OWW cache
  python audio_diagnostics.py             # verify mic/speaker/TTS
  python -m dana.core_agent
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import queue
import re
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path
from typing import Any, Optional, Union
from collections import deque

# Bootstrap BEFORE package imports: running ``python dana/core_agent.py`` puts
# ``dana/`` on sys.path[0], which breaks ``import dana`` and root modules.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Windows taskbar identity — must run before CustomTkinter/Tk root creation.
if sys.platform == "win32":
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "CascadeRouter.Donna.DesktopAgent.1.0"
        )
    except Exception:  # noqa: BLE001
        pass

# Absolute Windows console kill-switch: CREATE_NO_WINDOW + STARTUPINFO hide +
# mutate python.exe → pythonw.exe (class patch so asyncio can still subclass).
if os.name == "nt":
    _original_popen = subprocess.Popen
    _pythonw_path = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")

    def _coerce_pythonw_cmd(cmd0: Any) -> Any:
        if not os.path.isfile(_pythonw_path):
            return cmd0
        if isinstance(cmd0, (list, tuple)) and cmd0:
            cmd = list(cmd0)
            head = str(cmd[0])
            if (
                head == sys.executable
                or os.path.normcase(head) == os.path.normcase(sys.executable)
                or os.path.basename(head).lower() == "python.exe"
            ):
                cmd[0] = _pythonw_path
            return cmd
        if isinstance(cmd0, str):
            if cmd0 == sys.executable or cmd0.startswith(sys.executable):
                return cmd0.replace(sys.executable, _pythonw_path, 1)
            if os.path.basename(cmd0.split(" ", 1)[0]).lower() == "python.exe":
                return cmd0.replace(cmd0.split(" ", 1)[0], _pythonw_path, 1)
        return cmd0

    class _PatchedPopen(_original_popen):  # type: ignore[valid-type,misc]
        def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            # 1. Force CREATE_NO_WINDOW
            kwargs["creationflags"] = int(kwargs.get("creationflags") or 0) | 0x08000000

            # 2. Force STARTUPINFO invisibility cloak
            if "startupinfo" not in kwargs or kwargs.get("startupinfo") is None:
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = int(getattr(subprocess, "SW_HIDE", 0))
                kwargs["startupinfo"] = startupinfo

            # 3. Intercept sys.executable / python.exe and mutate to pythonw.exe
            if args:
                first = _coerce_pythonw_cmd(args[0])
                args = (first,) + args[1:]
            if "args" in kwargs and kwargs["args"] is not None:
                kwargs["args"] = _coerce_pythonw_cmd(kwargs["args"])

            super().__init__(*args, **kwargs)

    subprocess.Popen = _PatchedPopen  # type: ignore[misc, assignment]

# Force multiprocessing workers onto windowless pythonw.exe (CreateProcess bypasses Popen).
import multiprocessing

if os.name == "nt":
    _pythonw_path = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
    if os.path.isfile(_pythonw_path):
        multiprocessing.set_executable(_pythonw_path)


def _nt_hide_console_if_mp_child() -> None:
    """Hide console only inside multiprocessing children — never the main agent terminal."""
    if os.name != "nt":
        return
    try:
        if multiprocessing.current_process().name == "MainProcess":
            return
        import ctypes

        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 0)
    except Exception:  # noqa: BLE001
        pass

import customtkinter as ctk
import cv2
import numpy as np
import pystray
import requests
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from PIL import Image, ImageDraw
from piper import PiperVoice
from piper.config import SynthesisConfig

from vision_tools import ScreenAgent, VideoAgent
from dana.paths import (
    ENV_PATH,
    PROJECT_ROOT as _PATHS_PROJECT_ROOT,
    SETTINGS_PATH as _SETTINGS_PATH,
    TEMP_REPLY_WAV,
    TEXT_INJECTION_PATH,
    TRIGGER_ASK_PATH,
    TTS_MODELS_DIR as _TTS_DIR,
    WAKEWORD_ONNX,
    YOLO_WEIGHTS_PATH,
    chdir_project_root,
)
# TEXT_INJECTION_PATH kept for legacy migrate; ingestion uses task_queue.json.

# Keep bootstrap string and dana.paths.PROJECT_ROOT in sync.
PROJECT_ROOT = os.path.abspath(str(_PATHS_PROJECT_ROOT))
import ingest  # noqa: E402 — repo-root input.txt → task_queue.json converter
from dana.secure_memory import SecureMemory, default_vault_path
from dana.vault_service import VaultClient
from dana.tools import ToolCall, ToolValidationError, get_broker
from dana.agentic import (
    CHAT_MEMORY_CLEARED_ACK,
    CHAT_MEMORY_WINDOW_K,
    REACT_MAX_ITERS,
    build_lightweight_chat_system_prompt,
    chat_memory_size,
    clear_chat_memory,
    get_donna_mode,
    mode_switch_spoken_ack,
    parse_clear_chat_memory,
    parse_mode_switch,
    requires_tool_graph,
    run_lightweight_chat,
    run_react_loop,
    set_donna_mode,
)
from dana.logging import (
    CONVERSATION_LOG_PATH,
    enable_runtime_file_logging,
    log,
    log_conversation,
    log_debug,
    log_exception,
)
from dana.sanitize import sanitize_tool_trace
from dana.prompts.spatial_synthesis import build_agent_system_prompt, spatial_focus_hint
from spatial_context import SPATIAL_AGGREGATOR

load_dotenv(ENV_PATH)
load_dotenv()

# ---------------------------------------------------------------------------
# Singleton lock (keep socket open for process lifetime)
# ---------------------------------------------------------------------------

_SINGLETON_PORT = 47474
_singleton_socket: Optional[socket.socket] = None
_tray_icon: Optional[pystray.Icon] = None
_gui_instance: Optional["DonnaGUI"] = None
_agent_loop_thread: Optional[threading.Thread] = None
# Last mic ambient probe — drives adaptive VAD / barge-in floors for quiet headsets.
_mic_ambient_rms: float = 0.0

# Live Trace telemetry (background threads → Tk main thread via Queue only).
gui_telemetry_queue: queue.Queue = queue.Queue()
_TRACE_MODE_COLORS: dict[str, str] = {
    "chat": "#00E676",
    "developer": "#FB8C00",  # execution / developer
    "vision": "#3B82F6",
    "research": "#F59E0B",
    "dictation": "#9C27B0",
}
_TRACE_IDLE_COLOR = "#9CA3AF"
_UI_CANVAS = "#121218"
_UI_CARD = "#1E1E2E"
_UI_CARD_BORDER = "#2A2A3C"
_UI_GHOST = "#2A2A3C"
_UI_MUTED = "#9AA0A6"
# Stage 8.9.8 — unified cybernetic accent (replaces default CTk blue).
_UI_ACCENT = "#00ADB5"
_UI_ACCENT_HOVER = "#008E95"
# ASCII-only — emoji tofu glyphs rendered as broken purple boxes on Win fonts.
_TRACE_STATUS_ICONS: dict[str, str] = {
    "active": "[~]",
    "completed": "[OK]",
    "bypassed": "[--]",
}


def emit_trace(
    stage: str,
    status: str,
    message: str,
    mode: str | None = None,
) -> None:
    """Push one Live Trace event (thread-safe; UI drains on Tk main thread)."""
    payload = {
        "stage": str(stage or "").strip() or "stage",
        "status": str(status or "active").strip().lower(),
        "message": str(message or "").strip(),
        "mode": (str(mode).strip().lower() if mode else None),
    }
    if payload["status"] not in _TRACE_STATUS_ICONS:
        payload["status"] = "active"
    try:
        gui_telemetry_queue.put_nowait(payload)
    except Exception:  # noqa: BLE001
        pass
    # Canonical bus for LiveTracePanel (never touches Tk from worker threads).
    try:
        from dana.ui.trace_bus import emit_trace_event

        status_l = payload["status"]
        et = "node_enter" if status_l == "active" else "node_exit"
        emit_trace_event(
            et,
            node=payload["stage"],
            message=payload["message"],
            mode=payload["mode"] or "",
            payload=payload["message"],
        )
    except Exception:  # noqa: BLE001
        pass


def enforce_singleton() -> None:
    """Bind a local TCP port so only one Donna process can run."""
    global _singleton_socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # Do not set SO_REUSEADDR — that would defeat the singleton check.
        sock.bind(("127.0.0.1", _SINGLETON_PORT))
        sock.listen(1)
    except OSError:
        print("[System] Donna is already running.", flush=True)
        try:
            sock.close()
        except OSError:
            pass
        sys.exit(0)
    _singleton_socket = sock


# ---------------------------------------------------------------------------
# Piper TTS model download (offline voices)
# ---------------------------------------------------------------------------

_PROJECT_DIR = PROJECT_ROOT
TTS_MODELS_DIR = str(_TTS_DIR)
PIPER_TEMP_WAV = str(TEMP_REPLY_WAV)
# Pre-rendered canned UX acknowledgments (skip live Piper during LLM load).
AUDIO_CACHE_DIR = Path(PROJECT_ROOT) / "dana" / "assets" / "audio_cache"
# Canonical UX phrases → WAV filenames. Lookup uses fuzzy keys (lower + no punct).
_CANNED_UX_WAV_FILES: dict[str, str] = {
    "The ticket is on the board.": "the_ticket_is_on_the_board.wav",
    "Yes?": "yes.wav",
    "Standing by.": "standing_by.wav",
    "I didn't catch that.": "i_didnt_catch_that.wav",
    "Donna is ready.": "donna_is_ready.wav",
    "Developer mode active.": "developer_mode_active.wav",
    "Chat mode active.": "chat_mode_active.wav",
    "Vision mode active.": "vision_mode_active.wav",
    "Research mode active.": "research_mode_active.wav",
    "Memory cleared.": "memory_cleared.wav",
}
# Runtime / conversation log paths live in dana.logging (re-exported above).
PIPER_EN_ONNX = os.path.join(TTS_MODELS_DIR, "en_US-hfc_female-medium.onnx")
PIPER_EN_JSON = os.path.join(TTS_MODELS_DIR, "en_US-hfc_female-medium.onnx.json")
# length_scale > 1.0 slows speech (VITS). Default 1.25 keeps OCR/vision reads intelligible.
try:
    PIPER_LENGTH_SCALE = float(os.environ.get("DONNA_PIPER_LENGTH_SCALE", "1.25"))
except ValueError:
    PIPER_LENGTH_SCALE = 1.25
PIPER_LENGTH_SCALE = max(0.8, min(2.0, PIPER_LENGTH_SCALE))
# Incomplete localization voices are disabled for the public release.
# Related local Piper assets remain gitignored under tts_models/.
PIPER_MODEL_URLS: tuple[tuple[str, str], ...] = (
    (
        PIPER_EN_ONNX,
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/hfc_female/medium/en_US-hfc_female-medium.onnx",
    ),
    (
        PIPER_EN_JSON,
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/hfc_female/medium/en_US-hfc_female-medium.onnx.json",
    ),
)
_piper_voice_cache: dict[str, PiperVoice] = {}
DONNA_WAKEWORD_ONNX = str(WAKEWORD_ONNX)


def _download_file(url: str, dest: str) -> None:
    """Stream-download url to dest (atomic replace)."""
    print(f"[TTS] Downloading Piper model -> {os.path.basename(dest)} ...", flush=True)
    with requests.get(url, stream=True, timeout=180) as resp:
        resp.raise_for_status()
        tmp_path = dest + ".partial"
        with open(tmp_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1024 * 256):
                if chunk:
                    fh.write(chunk)
        os.replace(tmp_path, dest)
    print(
        f"[TTS] Saved {os.path.basename(dest)} "
        f"({os.path.getsize(dest) / (1024 * 1024):.1f} MB)",
        flush=True,
    )


def download_piper_models() -> None:
    """Download the English Piper voice into tts_models/ if missing."""
    os.makedirs(TTS_MODELS_DIR, exist_ok=True)
    for dest, url in PIPER_MODEL_URLS:
        if os.path.isfile(dest) and os.path.getsize(dest) > 0:
            continue
        try:
            _download_file(url, dest)
        except Exception as exc:  # noqa: BLE001
            try:
                if os.path.isfile(dest + ".partial"):
                    os.remove(dest + ".partial")
            except OSError:
                pass
            raise RuntimeError(
                f"Failed to download Piper model from {url}: {exc}"
            ) from exc


from openwakeword.model import Model as OpenWakeWordModel


MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"  # retained for optional future vision
WHISPER_ID = "distil-whisper/distil-small.en"
# Default Whisper language; overridden at runtime from settings.json (English-first).
WHISPER_LANGUAGE = "english"
WHISPER_TASK = "transcribe"
YOLO_WEIGHTS = str(YOLO_WEIGHTS_PATH)
FRAME_SIZE = (640, 480)  # (width, height)
MAX_NEW_TOKENS = 50  # legacy SmolVLM cap (unused in Ollama cascade)
YOLO_CONF = 0.35
# Tracker samples active_vision_tool into a maxlen-5 buffer every ~2s (low CPU).
TRACKER_SLEEP_SEC = 0.25
TRACKER_BUFFER_INTERVAL_S = 2.0

# Local Ollama conversational brain
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2"
OLLAMA_TIMEOUT_SEC = 180.0

# Intent-router keywords for dynamic vision tool switching.
SCREEN_KEYWORDS = ["screen", "monitor", "code", "display", "desktop"]
CAMERA_KEYWORDS = ["camera", "room", "me", "face", "physical", "look at me"]

# Audio / wake-word constants
SAMPLE_RATE = 16000
WAKE_CHUNK = 1280  # 80 ms @ 16 kHz
# Local donna.onnx is sticky on this mic (can sit at ~0.99 on hush).
# Require a real onset: score must rise from low -> high, not stay pegged.
WAKE_THRESHOLD = 0.80
WAKE_MIN_CONSECUTIVE = 3  # ~240 ms of consecutive high scores
WAKE_ONSET_BELOW = 0.45  # must have been below this recently before a hit
WAKE_ONSET_LOOKBACK = 12  # ~1 s of score history
WAKE_PHRASE_WINDOW_CHUNKS = 18  # ~1.44 s rolling buffer for phrase verify
WAKE_PHRASE_VERIFY = False  # skip Whisper second gate; openWakeWord score onset starts session
WAKE_COOLDOWN_SEC = 6.0
WAKE_PHRASE_TOKENS = ("donna", "hey donna", "hey, donna")
# Whisper often mishears "Donna" as these; treat as wake confirmations.
# Include "don't know" / donald / donut to cut false negatives on quiet mics;
# OpenWakeWord score+onset still gate hush false-positives.
WAKE_PHRASE_ALIASES = (
    "donna",
    "hey donna",
    "donald",
    "hey donald",
    "donut",
    "don t know",
    "dont know",
    "don know",
    "donna donna",
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
    "Donna, Titan initiative, Titan Protocol, Titan supervisor, "
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
# Quiet-mic gain: real Sonar speech often lands at rms≈0.003.
WHISPER_TARGET_RMS = 0.05
WHISPER_GAIN_RMS_CEIL = 0.015
WHISPER_MIN_RMS_FOR_GAIN = 0.0015
WHISPER_MAX_GAIN = 12.0
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
# Barge-in while Donna speaks: Silero probability (not raw RMS) gates interrupt.
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
MIC_AMBIENT_DEAD_RMS = 1e-4  # probe below this → soft gain / adaptive floors
# TTS recovery: max wait for queue drain; hard cap per Piper utterance (synth+play).
# Long vision/OCR summaries need headroom — 18s was aborting mid-sentence.
TTS_IDLE_WAIT_TIMEOUT = 12.0
TTS_UTTERANCE_MAX_SECONDS = 90.0
# Soft-split long replies into independent spool items (~15–20s Piper each).
TTS_CHUNK_MAX_CHARS = 280
MIN_SPEECH_RMS = 0.01  # after peak-normalize; reject near-silence hallucinations
WAKEWORD_MODELS = ["donna"]
AUDIO_INPUT_DEVICE: Optional[int] = None  # resolved at startup
AUDIO_INPUT_RATE: int = SAMPLE_RATE  # device native rate; resampled to 16 kHz
AUDIO_OUTPUT_DEVICE: Optional[int] = None  # TTS playback via sounddevice
SENDGRID_MAIL_URL = "https://api.sendgrid.com/v3/mail/send"

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

latest_frame_lock = threading.Lock()
latest_frame: Optional[np.ndarray] = None  # BGR, 640x480

latest_dets_lock = threading.Lock()
latest_dets: list[tuple[np.ndarray, str, float]] = []

# Dynamic vision tool calling (ScreenAgent / VideoAgent).
screen_tool = ScreenAgent()
camera_tool = VideoAgent()
active_vision_lock = threading.Lock()
active_vision_tool: Union[ScreenAgent, VideoAgent] = screen_tool

# Sliding-window chat memory for Ollama (last 6 messages = 3 user + 3 assistant).
conversation_history: list[dict[str, str]] = []
conversation_history_lock = threading.Lock()
HISTORY_MAX_MESSAGES = 6

# Decrypted long-term profile (AES vault); injected into Ollama system prompt.
donna_profile: dict[str, Any] = {}
donna_vault: Optional["SecureMemory"] = None
# High-frequency identity keys prefetched post-unlock (skip ReAct vault tools).
VAULT_HOT_CACHE: dict[str, str] = {}

# Optional Arabic-script detection (unused for English-only TTS routing).
ARABIC_SCRIPT_RE = re.compile(r"[\u0600-\u06FF]")

# Short-term spatial memory so flickering detections still answer "where is X?"
spatial_memory_lock = threading.Lock()
spatial_memory: dict[str, float] = {}  # label -> last_seen monotonic time
SPATIAL_MEMORY_SEC = 2.5

# Wake word / .trigger_ask starts a conversational turn.
is_recording = threading.Event()
# Serialize mic *open/close* for the single ingest producer only.
mic_lock = threading.Lock()
# Legacy name kept for call sites: producer-ready / stream healthy.
wake_mic_released = threading.Event()
wake_mic_released.set()
# Device acquisition / first-read hang guards (Windows MME can block forever).
MIC_STREAM_OPEN_TIMEOUT_S = 2.5
MIC_STREAM_READ_TIMEOUT_S = 1.5
MIC_DEVICE_SETTLE_S = 0.08
# Producer-consumer mic path: one InputStream → shared 16 kHz VAD frames.
AUDIO_BUFFER_MAX_FRAMES = 100  # ~3s @ 30ms — drop oldest on overflow
audio_buffer_queue: queue.Queue = queue.Queue(maxsize=AUDIO_BUFFER_MAX_FRAMES)
mic_ingest_ready = threading.Event()
mic_ingest_restart = threading.Event()
_mic_ingest_thread: Optional[threading.Thread] = None

# Shared Whisper bundle for wake-phrase verification (set by conversation_worker).
whisper_bundle_lock = threading.Lock()
whisper_bundle: Optional[tuple[Any, Any, Any, Any]] = None
# Set when background Whisper load finishes (success or failure).
whisper_ready = threading.Event()
_whisper_load_error: Optional[str] = None

# Conversation phase: idle | listening | followup | transcribing | thinking
ui_state_lock = threading.Lock()
ui_state = "idle"

# Latest Whisper transcript (logged for headless debugging).
subtitle_lock = threading.Lock()
subtitle_text = ""

# Optional injected question from .trigger_ask file contents (automation / tests).
injected_question_lock = threading.Lock()
injected_question: Optional[str] = None
_injected_source: str = "inject"
_injected_already_logged: bool = False

# TTS Output Spooler — producers push (text, interruptible); consumer owns PortAudio.
# ``interruptible=False`` = UI ack exemption (no self-barge-in on speaker bleed).
# Stage 8.8 — spool items are (text, interruptible, agent_id).
tts_queue: queue.Queue[Optional[tuple[str, bool, str]]] = queue.Queue(maxsize=16)
speech_queue = tts_queue  # backward-compatible alias
# Serialize TTS enqueue / flush mutations.
_tts_enqueue_lock = threading.Lock()
_speech_enqueue_lock = _tts_enqueue_lock  # alias
# Exclusive PortAudio output lifecycle (open → write chunks → close / stop).
playback_lock = threading.RLock()
# Max phrases allowed to pile up while a stream already owns the speaker.
_SPEECH_MAX_PENDING_WHILE_BUSY = 3
# Max time to defer a dequeued phrase while the user is speaking (VAD).
_TTS_HOLD_FOR_VAD_MAX_S = 12.0
# Set while tts_worker is actively rendering/playing TTS (mic must stay idle).
tts_busy = threading.Event()
# Barge-in: set by VAD when user speaks over TTS; checked in the playback chunk loop.
tts_interrupt_event = threading.Event()
# Process-wide barge-in controller (shares ``tts_interrupt_event``).
from dana.audio.tts_worker import get_tts_worker as _get_tts_worker  # noqa: E402

_tts_barge = _get_tts_worker(barge_in_event=tts_interrupt_event)
# True while ``record_utterance`` owns the microphone (barge-in watcher must stand down).
vad_capture_active = threading.Event()
# Cleared until conversation_worker's Ollama warm-up finishes (gates wake-word arming).
ollama_ready = threading.Event()
# Boot coordination: ready audio plays only when all three are set.
piper_voices_ready = threading.Event()
wakeword_armed = threading.Event()
# Stage 8.9.7 — soft engine ignition (clear = STANDBY; set = ACTIVE).
# Distinct from stop_event / STOP DONNA (hard exit).
engine_engaged = threading.Event()
_boot_ready_audio_lock = threading.Lock()
_boot_ready_audio_played = False
# Shared OpenWakeWord model for stream-barge during TTS (set by wakeword_worker).
_shared_wakeword_model: Any = None
_shared_wakeword_token: str = "donna"
# Set when the TTS spooler is drained and nothing is playing.
speech_idle = threading.Event()
speech_idle.set()
_tts_worker_thread: Optional[threading.Thread] = None
# One "Let me check" per conversational turn (router + ReAct share this).
_tool_working_ack_sent = threading.Event()
stop_event = threading.Event()
# PortAudio / hardware fault signal: Audio thread -> Main (soft recovery).
audio_hardware_fault = threading.Event()
_audio_hardware_fault_lock = threading.Lock()
_audio_hardware_fault_detail: str = ""
TRIGGER_FILE = str(TRIGGER_ASK_PATH)
SETTINGS_FILE = str(_SETTINGS_PATH)
MEMORY_FILE = default_vault_path()
MEMORY_SALT = b"donna_secure_salt"
PBKDF2_ITERATIONS = 390_000
vault_client = VaultClient()

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

_CODE_FENCE_TTS_RE = re.compile(r"```[\w+-]*\n?[\s\S]*?```", re.MULTILINE)
_CODE_FENCE_TTS_UNCLOSED_RE = re.compile(r"```[\w+-]*\n?[\s\S]*$", re.MULTILINE)
_TTS_MD_MARKERS_RE = re.compile(r"`+|\*{1,3}|_{2,}")
_PUNCT_OR_SPACE_ONLY_RE = re.compile(r"^[\s\W_]+$", re.UNICODE)


def compile_and_append_voice_prompt(raw_transcript: str) -> str:
    """Meta-Planner: compile Whisper text, append to execution_jail/input.txt.

    On any compiler failure, appends and returns the raw transcript instead.
    Never raises into the audio / conversation loop.
    """
    raw = (raw_transcript or "").strip()
    if not raw:
        return raw_transcript or ""
    text = raw
    try:
        from dana.swarm.compiler_node import compile_voice_to_prompt

        compiled = compile_voice_to_prompt(raw)
        if (compiled or "").strip():
            text = compiled.strip()
    except Exception as exc:  # noqa: BLE001
        log("Compiler", f"compile_voice_to_prompt failed — using raw transcript ({exc})")
        text = raw
    try:
        target = TEXT_INJECTION_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(text.rstrip() + "\n\n")
        preview = text if len(text) <= 160 else text[:157] + "..."
        log("Compiler", f'Appended compiled prompt to input.txt: "{preview}"')
    except Exception as exc:  # noqa: BLE001
        log("Compiler", f"WARNING: input.txt append failed ({exc})")
    return text


def emit_live_transcript(
    speaker: str,
    text: str,
    *,
    agent_id: str | None = None,
) -> None:
    """Thread-safe bridge from audio/LLM workers into the Dashboard transcript."""
    gui = _gui_instance
    if gui is None:
        return
    try:
        gui.log_transcript(speaker, text, agent_id=agent_id)
    except Exception as exc:  # noqa: BLE001
        log("UI", f"WARNING: live transcript update failed ({exc})")


def set_ui_state(state: str) -> None:
    global ui_state
    with ui_state_lock:
        ui_state = state
    SPATIAL_AGGREGATOR.set_ui_state(state)
    log_debug("UI", f"State -> {state}")
    # Visual cue: tray icon turns green while VAD is actively listening.
    try:
        update_tray_icon_for_state(state)
    except Exception:  # noqa: BLE001
        pass


def get_ui_state() -> str:
    with ui_state_lock:
        return ui_state


def set_subtitle(text: str) -> None:
    global subtitle_text
    with subtitle_lock:
        subtitle_text = text
    if text:
        log_debug("UI", f"Subtitle -> {text}")


def set_injected_question(
    text: str,
    *,
    source: str = "inject",
    already_logged: bool = False,
) -> None:
    """Queue text as the next user utterance (skips mic / Whisper).

    ``source="text"`` marks Dashboard silent-chat injections for transcript labeling.
    ``already_logged=True`` skips a second Live Transcript echo.
    """
    global injected_question, _injected_source, _injected_already_logged
    with injected_question_lock:
        injected_question = text
        _injected_source = (source or "inject").strip() or "inject"
        _injected_already_logged = bool(already_logged)


def clear_injected_question() -> None:
    global injected_question, _injected_source, _injected_already_logged
    with injected_question_lock:
        injected_question = None
        _injected_source = "inject"
        _injected_already_logged = False


def pop_injected_question() -> Optional[str]:
    """Pop pending inject text (legacy API — discards source metadata)."""
    text, _source, _logged = pop_injected_question_ex()
    return text


def pop_injected_question_ex() -> tuple[Optional[str], str, bool]:
    """Pop inject text with ``(text, source, already_logged)``."""
    global injected_question, _injected_source, _injected_already_logged
    with injected_question_lock:
        text = injected_question
        source = _injected_source
        already_logged = _injected_already_logged
        injected_question = None
        _injected_source = "inject"
        _injected_already_logged = False
        return text, source, already_logged


def _clear_text_injection_file(path: Path | None = None) -> None:
    """Truncate the interceptor file so the same text cannot re-fire forever."""
    target = path or TEXT_INJECTION_PATH
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("", encoding="utf-8")
    except OSError as exc:
        log("Interceptor", f"WARNING: failed to clear {target}: {exc}")


def pop_text_injection(*, path: Path | None = None) -> Optional[str]:
    """Deprecated legacy reader for ``input.txt``.

    Production ingestion uses ``execution_jail/task_queue.json`` via
    :func:`dana.tools.broker.dispatch_pending_tasks`. When ``path`` is omitted,
    any leftover ``input.txt`` content is migrated into the queue and ``None``
    is returned. Explicit ``path=`` (unit tests) still reads + clears that file.
    """
    if path is None:
        try:
            from dana.tools.task_queue import migrate_legacy_input_txt

            migrated = migrate_legacy_input_txt()
            if migrated:
                preview = migrated if len(migrated) <= 160 else migrated[:157] + "..."
                log(
                    "TaskQueue",
                    f'Migrated legacy input.txt into task_queue.json: "{preview}"',
                )
        except Exception as exc:  # noqa: BLE001
            log("TaskQueue", f"WARNING: legacy input.txt migrate failed: {exc}")
        return None

    target = path
    try:
        if not target.is_file():
            return None
        raw = target.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as exc:
        log("Interceptor", f"WARNING: could not read {target}: {exc}")
        return None

    text = (raw or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    if not text:
        return None

    _clear_text_injection_file(target)
    preview = text if len(text) <= 160 else text[:157] + "..."
    log("Interceptor", f'Bypassing Whisper. Injecting text: "{preview}"')
    return text


def is_whisper_prompt_echo(text: str) -> bool:
    """True when STT regurgitated the Whisper initial_prompt / fixture bias.

    Quiet-mic turns used to invent ``read the file project_omega_status.txt``
    (and similar bias phrases), which the broker then treated as a real file
    read and spoke the confidential Omega fixture aloud.
    """
    cleaned = re.sub(r"\s+", " ", (text or "").strip().lower())
    if not cleaned:
        return False
    bias = re.sub(r"\s+", " ", WHISPER_INITIAL_PROMPT.lower())
    if cleaned == bias or cleaned in bias or bias in cleaned:
        return True
    # Legacy bias tokens that must never become a live transcript on their own.
    legacy_markers = (
        "project_omega_status",
        "project omega",
        "file_jail_enforcer",
        "confidential status report",
        "draft_cursor_prompt",
        "patch_ledger",
    )
    hits = sum(1 for m in legacy_markers if m in cleaned)
    if hits >= 2:
        return True
    # Isolated Omega filename "read" with no other user intent → treat as echo.
    if "project_omega" in cleaned and re.search(
        r"\bread\s+(?:the\s+)?file\b", cleaned
    ):
        # Allow only when the user also states a clear non-bias intent verb+object.
        if not re.search(
            r"\b(summarize|explain|what\s+does|tell\s+me\s+about|status\s+of)\b",
            cleaned,
        ):
            return True
    return False


def is_punctuation_or_whitespace_only(text: str) -> bool:
    """True when transcript is only punctuation / whitespace (no letters or digits)."""
    raw = (text or "").strip()
    if not raw:
        return True
    if ARABIC_SCRIPT_RE.search(raw) or re.search(r"[A-Za-z0-9]", raw):
        return False
    return bool(_PUNCT_OR_SPACE_ONLY_RE.fullmatch(raw))


def is_whisper_rate_hallucination(text: str, duration_s: float) -> bool:
    """True when word density exceeds a realistic human speaking rate.

    Word count is ``len(transcript.split())`` — never ``len(transcript)``
    (character length). Rate is strictly ``word_count / audio_duration_s``;
    values ``> WHISPER_MAX_WORDS_PER_SEC`` (5.0) return True.
    """
    transcript = (text or "").strip()
    if not transcript:
        return False
    # Unit-correct: words, not characters.
    word_count = len(transcript.split())
    if word_count <= 0:
        return False
    dur = float(duration_s)
    if dur <= 0.0:
        # No usable duration — only reject dense dumps on near-zero audio.
        return word_count >= 8
    rate = word_count / dur
    return rate > float(WHISPER_MAX_WORDS_PER_SEC)


def is_whisper_hallucination(
    text: str,
    *,
    audio_duration_s: Optional[float] = None,
) -> bool:
    """Reject empty, ultra-short, bracketed non-speech, or known silence hallucinations.

    Short non-Latin script utterances are allowed when they contain letters.
    Do not hardcode language-specific spam tokens here — diagnose low-SNR
    captures via STT debug logs instead.
    When ``audio_duration_s`` is provided, also apply the duration-to-word
    sanity check (words/sec above ``WHISPER_MAX_WORDS_PER_SEC``).
    """
    raw = (text or "").strip()
    if not raw:
        return True

    if is_punctuation_or_whitespace_only(raw):
        return True

    if is_whisper_prompt_echo(raw):
        return True

    if audio_duration_s is not None and is_whisper_rate_hallucination(
        raw, float(audio_duration_s)
    ):
        return True

    # Breathing / non-speech often lands as [sigh], (breathing), [BLANK_AUDIO], etc.
    if re.fullmatch(r"(?:\s*[\(\[][^\)\]]*[\)\]]\s*)+", raw):
        return True
    paren_stripped = re.sub(r"[\(\[][^\)\]]*[\)\]]", "", raw).strip(" .,!?;:\"'`-")
    if not paren_stripped:
        return True

    # Non-Latin script: keep short real phrases; only drop pure noise/punct.
    if ARABIC_SCRIPT_RE.search(raw):
        letters = ARABIC_SCRIPT_RE.findall(raw)
        return len(letters) < 1

    cleaned = raw.lower().strip(" .,!?;:\"'`")
    if cleaned in WHISPER_HALLUCINATIONS or cleaned in WHISPER_AMBIENT_SILENT:
        return True

    # Explicit non-speech tags Whisper emits even without brackets.
    non_speech = {
        "sigh",
        "breathing",
        "breath",
        "inhale",
        "exhale",
        "cough",
        "laughter",
        "blank_audio",
        "silence",
        "music",
        "applause",
    }
    if cleaned in non_speech:
        return True

    words = [w for w in cleaned.replace("-", " ").split() if w]
    if len(words) < 2:
        return True

    # Extra phrase-level traps Whisper-tiny loves on noise.
    bad_phrases = (
        "thank you for watching",
        "thanks for watching",
        "please subscribe",
        "like and subscribe",
        "see you next time",
        "don't forget to subscribe",
        "i'm going to be playing with you",
        "i am going to be playing with you",
        "i'm going to play with you",
        "i am going to play with you",
        "thanks for listening",
        "thank you for listening",
        "subtitles by",
        "transcript by",
        "draft_cursor_prompt",
        "patch ledger",
        "the ticket is on the board",
    )
    return any(p in cleaned for p in bad_phrases)


_STANDBY_PHRASES = frozenset(
    {
        "stand by",
        "standby",
        "go to sleep",
        "stop listening",
        "shut up",
        "bye",
        "quit",
        "exit",
        "stop",
        "goodbye",
        "good bye",
        "",
        "",
    }
)
_STANDBY_TAIL_WORDS = frozenset(
    {"bye", "quit", "exit", "stop", "goodbye", "standby"}
)

_CLEAR_CONTEXT_PHRASES = frozenset(
    {
        "clear context",
        "clear the context",
        "kill context",
        "kill your context",
        "kill the context",
        "forget that",
        "forget this",
        "forget everything",
        "start over",
        "reset memory",
        "reset context",
        "wipe context",
        "wipe memory",
        "new conversation",
        "fresh start",
        "   ",
        " ",
        " ",
    }
)


_LOCKDOWN_PHRASES = frozenset(
    {
        "lockdown",
        "lock down",
        "lock yourself",
        "secure the system",
    }
)

_TIME_PHRASES = (
    "what time is it right now",
    "what's the time right now",
    "what is the time right now",
    "what time of the day is it",
    "what time of day is it",
    "can you tell me what time of the day is it",
    "can you tell me what time it is",
    "tell me the time",
    "what's the time",
    "what is the time",
    "what time is it",
    "current time",
)


def is_engine_engaged() -> bool:
    """Stage 8.9.7 — True when Dashboard ENGAGE has armed the LangGraph engine."""
    return bool(engine_engaged.is_set())


def set_engine_engaged(active: bool) -> None:
    """Arm (True) or soft-standby (False) the conversational engine."""
    if active:
        engine_engaged.set()
    else:
        engine_engaged.clear()


def _warm_heavy_runtime_assets() -> None:
    """Stage 8.9.7 — background warm after ENGAGE (GUI already interactive).

    Imports LangGraph wiring only. Florence-2 / YOLO stay JIT on first vision
    call so Standby boot never blocks on VRAM-heavy models.
    """
    try:
        import dana.agentic_react_graph  # noqa: F401

        log("Main", "Heavy warm: agentic_react_graph imported (Florence remains JIT)")
    except Exception as exc:  # noqa: BLE001
        log("Main", f"Heavy warm skipped: {exc}")


def is_standby_command(text: str) -> bool:
    """True if STT is an explicit standby / sleep system command (bypass LLM)."""
    raw = (text or "").strip()
    if not raw:
        return False
    # Collapsed ASCII form for EN phrases (handles "And bye.").
    ascii_norm = re.sub(r"\s+", " ", raw.lower()).strip(" .,!?;:\"'`")
    if ascii_norm in _STANDBY_PHRASES:
        return True
    for phrase in sorted(_STANDBY_PHRASES, key=len, reverse=True):
        if " " in phrase and (
            ascii_norm == phrase or ascii_norm.endswith(" " + phrase)
        ):
            return True
    words = [w for w in re.split(r"\s+", ascii_norm) if w]
    if words and words[-1].strip(".,!?;:\"'`") in _STANDBY_TAIL_WORDS:
        return True
    # Exact phrase match after light whitespace normalize.
    fa_norm = re.sub(r"\s+", " ", raw).strip(" .,!?;:\"'`")
    return fa_norm in _STANDBY_PHRASES


def is_clear_context_command(text: str) -> bool:
    """True if STT asks to wipe the short-term conversation memory window."""
    raw = (text or "").strip()
    if not raw:
        return False
    ascii_norm = re.sub(r"\s+", " ", raw.lower()).strip(" .,!?;:\"'`")
    if ascii_norm in _CLEAR_CONTEXT_PHRASES:
        return True
    for phrase in sorted(_CLEAR_CONTEXT_PHRASES, key=len, reverse=True):
        if ascii_norm == phrase or ascii_norm.endswith(" " + phrase):
            return True
        # Allow wake-prefixed forms: "Donna, clear context"
        if ascii_norm.startswith(phrase + " ") or f" {phrase}" in f" {ascii_norm}":
            # Avoid matching unrelated sentences that merely contain a substring
            # of a multi-word phrase mid-word; require phrase as contiguous tokens.
            if phrase in ascii_norm:
                return True
    fa_norm = re.sub(r"\s+", " ", raw).strip(" .,!?;:\"'`")
    return fa_norm in _CLEAR_CONTEXT_PHRASES


def flush_conversation_memory(*, reason: str = "manual") -> int:
    """Wipe the sliding short-term history (Memory window N/6). Returns prior turn count.

    Also runs the custom-tools context-wipe failsafe (delete Desktop custom_tools
    ``.py`` files, unregister, clear ``sys.modules``).
    """
    global conversation_history
    with conversation_history_lock:
        prior = [
            m
            for m in conversation_history
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]
        n = len(prior)
        conversation_history.clear()
    log("Conversation", f"Memory window flushed ({reason}); cleared {n} msgs")
    log_conversation("System", f"Context cleared ({reason}); wiped {n} msgs")
    try:
        from dana.tools.registry import wipe_custom_tools

        wiped = wipe_custom_tools(reason=f"context_wipe:{reason}")
        if wiped:
            log("Conversation", f"Custom tools wipe companion: {wiped!r}")
    except Exception as exc:  # noqa: BLE001
        log("Conversation", f"WARNING: custom tools wipe failed ({exc})")
    return n


def clear_context_spoken_reply(text: str = "") -> str:
    """Ack phrase after flushing short-term memory."""
    from dana.settings import resolve_reply_lang

    if resolve_reply_lang(text or "") == "fa":
        return " —    ‌."
    return "Okay — fresh start. Context cleared."


def is_lockdown_command(text: str) -> bool:
    """True if STT is an explicit vault lockdown / kill-switch command."""
    raw = (text or "").strip()
    if not raw:
        return False
    ascii_norm = re.sub(r"\s+", " ", raw.lower()).strip(" .,!?;:\"'`")
    if ascii_norm in _LOCKDOWN_PHRASES:
        return True
    for phrase in sorted(_LOCKDOWN_PHRASES, key=len, reverse=True):
        if ascii_norm == phrase or ascii_norm.endswith(" " + phrase):
            return True
    return False


def is_time_command(text: str) -> bool:
    """True if STT is a wall-clock question (deterministic fast-path; bypass LLM)."""
    raw = (text or "").strip()
    if not raw:
        return False
    ascii_norm = re.sub(r"\s+", " ", raw.lower()).strip(" .,!?;:\"'`")
    ascii_norm = ascii_norm.replace("whats", "what's")
    for phrase in _TIME_PHRASES:
        if ascii_norm == phrase or ascii_norm.endswith(" " + phrase):
            return True
        if phrase in ascii_norm and len(ascii_norm) <= len(phrase) + 12:
            return True
    return False


def wall_clock_spoken_reply() -> str:
    """Format local wall clock for TTS (no LLM)."""
    from datetime import datetime

    current_time = datetime.now().strftime("%I:%M %p").lstrip("0")
    return f"It is {current_time}."


def populate_vault_hot_cache(client: Optional["VaultClient"] = None) -> None:
    """Prefetch core identity keys into VAULT_HOT_CACHE after a successful unlock."""
    global VAULT_HOT_CACHE
    client = client if client is not None else vault_client
    user_name = "Amirhosein"
    family_partner = "Narges"
    try:
        raw = client.read_memory("user_name")
        if raw is not None and str(raw).strip():
            user_name = str(raw).strip()
    except Exception:  # noqa: BLE001
        pass
    try:
        raw = client.read_memory("family_partner")
        if raw is not None and str(raw).strip():
            family_partner = str(raw).strip()
    except Exception:  # noqa: BLE001
        pass
    VAULT_HOT_CACHE = {
        "user_name": user_name,
        "family_partner": family_partner,
    }
    log(
        "Memory",
        f"Vault hot-cache ready "
        f"(user_name={user_name!r}, family_partner={family_partner!r}).",
    )


def execute_lockdown_shutdown() -> None:
    """Speak lockdown ack, purge vault RAM key, hard-kill the process."""
    global donna_vault, donna_profile, vault_client, VAULT_HOT_CACHE

    log("Security", "Lockdown fast-path triggered — purging vault and exiting.")
    try:
        # Prefer spooler; fall back to direct play if the worker is already dead.
        flush_tts_queue()
        enqueue_speech("Initiating lockdown. Vault secured. Goodbye.")
        wait_for_speech_idle(timeout=4.0)
    except Exception as exc:  # noqa: BLE001
        print(f"[Security] Lockdown TTS failed: {exc}", flush=True)
        log("Security", f"WARNING: lockdown TTS failed ({exc})")
        try:
            _synthesize_and_play(
                "Initiating lockdown. Vault secured. Goodbye.",
                AUDIO_OUTPUT_DEVICE,
            )
        except Exception:
            pass

    try:
        vault_client.lock_vault()
        print(
            "[Security] Vault session purged. Password will be required next boot.",
            flush=True,
        )
        log("Security", "Vault daemon RAM key + sessions purged.")
    except Exception as exc:  # noqa: BLE001
        print(f"[Security] Failed to contact daemon for purge: {exc}", flush=True)
        log("Security", f"ERROR: lockdown purge failed ({exc})")

    try:
        if donna_vault is not None:
            donna_vault.lock()
    except Exception:  # noqa: BLE001
        pass
    donna_vault = None
    donna_profile = {}
    VAULT_HOT_CACHE = {}

    # Force immediate termination of all threads and GUI.
    os._exit(0)


def is_silent_non_speech_transcript(text: str) -> bool:
    """STT artifacts that must return to listening with no LLM and no apology TTS.

    Covers empty / punctuation-only transcripts, bracketed non-speech, and known
    Whisper ambient-noise hallucinations (e.g. \"Thank you\", \"Bye\").
    """
    raw = (text or "").strip()
    if not raw:
        return True
    if is_punctuation_or_whitespace_only(raw):
        return True
    if re.fullmatch(r"(?:\s*[\(\[][^\)\]]*[\)\]]\s*)+", raw):
        return True
    stripped = re.sub(r"[\(\[][^\)\]]*[\)\]]", "", raw).strip(" .,!?;:\"'`-")
    if not stripped:
        return True
    cleaned = raw.lower().strip(" .,!?;:\"'`")
    if cleaned in WHISPER_AMBIENT_SILENT:
        return True
    ambient_phrases = (
        "thank you for watching",
        "thanks for watching",
        "please subscribe",
        "like and subscribe",
        "thanks for listening",
        "thank you for listening",
        "see you next time",
        "subtitles by",
        "transcript by",
    )
    return any(p == cleaned or p in cleaned for p in ambient_phrases)


def correct_known_stt_names(text: str) -> str:
    """Repair Whisper mangling of known household names / common phrases."""
    from dana.tools.stt_corrector import correct_stt

    out = correct_stt(text or "")
    if not out:
        return out
    for pattern, repl in _STT_NAME_FIXES:
        out = pattern.sub(repl, out)
    # Collapse "Amirhosein, Amirhosein" / "Narges, and Narges" after repair.
    out = re.sub(r"\b(Amirhosein)(?:\s*,\s*(?:and\s+)?|\s+and\s+|\s+)\1\b", r"\1", out, flags=re.I)
    out = re.sub(r"\b(Narges)(?:\s*,\s*(?:and\s+)?|\s+and\s+|\s+)\1\b", r"\1", out, flags=re.I)
    return out


def select_device():
    import torch

    # Intel/x86 AVX CPU path for PyTorch ops (no-op / ignored on unsupported builds).
    try:
        torch.backends.mkldnn.enabled = True
    except Exception:
        pass

    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        log(
            "Main",
            f"Accelerator: CUDA ({name}) - YOLO + Whisper on cuda:0; brain=Ollama",
        )
        return device
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        log("Main", "Accelerator: Apple MPS (CUDA unavailable)")
        return torch.device("mps")
    log(
        "Main",
        "Accelerator: CPU fallback (MKLDNN enabled). "
        "Install CUDA wheels: pip install torch torchvision "
        "--index-url https://download.pytorch.org/whl/cu126",
    )
    return torch.device("cpu")


def select_dtype(device):
    import torch

    if device.type == "cuda":
        major, _minor = torch.cuda.get_device_capability(0)
        if major >= 8 and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def yolo_device_arg(device) -> str | int:
    if device.type == "cuda":
        return 0
    if device.type == "mps":
        return "mps"
    return "cpu"


def strip_code_blocks_for_tts(text: str) -> str:
    """Replace markdown fenced code with a short spoken placeholder.

    Prevents Piper from reading raw Python/JSON aloud (long TTS ceiling crash).
    """
    raw = text or ""
    if "```" not in raw:
        return raw
    out = _CODE_FENCE_TTS_RE.sub("[Code block generated]", raw)
    out = _CODE_FENCE_TTS_UNCLOSED_RE.sub("[Code block generated]", out)
    out = re.sub(r"(?:\s*\[Code block generated\]\s*){2,}", " [Code block generated] ", out)
    return out.strip()


def sanitize_text_for_tts(text: str) -> str:
    """Strip markdown emphasis/code markers before Piper synthesis.

    Returns empty string when nothing speakable remains (caller must skip TTS).
    Newlines become pause punctuation so OCR/vision dumps are not rushed.
    """
    out = strip_code_blocks_for_tts(text or "")
    out = _TTS_MD_MARKERS_RE.sub("", out)
    # Florence / vision dumps often arrive as newline-joined tokens with no stops.
    out = re.sub(r"[\r\n]+", ". ", out)
    out = re.sub(r"\s+", " ", out).strip()
    # Soft-break very long comma-less runs so Piper inserts breath pauses.
    if len(out) > 160 and not re.search(r"[.!?]", out):
        out = re.sub(r"(.{40,80}?)\s+", r"\1. ", out)
    if not out or _PUNCT_OR_SPACE_ONLY_RE.match(out):
        return ""
    return out


def _safe_sd_stop(*, where: str = "", blocking: bool = True) -> None:
    """Stop PortAudio playback under ``playback_lock`` (best-effort)."""
    acquired = False
    try:
        acquired = (
            playback_lock.acquire(blocking=blocking)
            if blocking
            else playback_lock.acquire(blocking=False)
        )
        if not acquired:
            log_debug("Audio", f"sd.stop deferred (lock busy) where={where or '-'}")
            return
        t0 = time.perf_counter()
        try:
            sd.stop()
        except Exception as exc:  # noqa: BLE001
            log_debug("Audio", f"sd.stop ignored ({where or '-'}): {exc}")
        else:
            log_debug(
                "Audio",
                f"sd.stop ok where={where or '-'} dt_ms={(time.perf_counter() - t0) * 1000.0:.1f}",
            )
    finally:
        if acquired:
            playback_lock.release()


def chunk_text_for_tts(text: str, *, max_chars: int = TTS_CHUNK_MAX_CHARS) -> list[str]:
    """Split long speakable text into sentence-sized chunks under ``max_chars``.

    Keeps short UX phrases intact. Prevents the 90s watchdog from aborting a
    single monolithic Piper render of OCR / multi-paragraph answers.
    """
    raw = sanitize_text_for_tts(text or "")
    if not raw:
        return []
    limit = max(80, int(max_chars))
    if len(raw) <= limit:
        return [raw]
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw) if s.strip()]
    if not sentences:
        sentences = [raw]
    chunks: list[str] = []
    buf = ""
    for sentence in sentences:
        if not buf:
            candidate = sentence
        else:
            candidate = f"{buf} {sentence}"
        if len(candidate) <= limit:
            buf = candidate
            continue
        if buf:
            chunks.append(buf)
        if len(sentence) <= limit:
            buf = sentence
        else:
            # Hard-wrap oversized sentence on spaces.
            words = sentence.split()
            buf = ""
            for word in words:
                piece = word if not buf else f"{buf} {word}"
                if len(piece) <= limit:
                    buf = piece
                else:
                    if buf:
                        chunks.append(buf)
                    buf = word
    if buf:
        chunks.append(buf)
    return chunks or [raw]


def enqueue_speech(
    text: str,
    *,
    interruptible: bool | None = None,
    agent_id: str | None = None,
) -> None:
    """Producer API: push text into the TTS spooler and return immediately.

    Never opens PortAudio or blocks on Piper — the ``tts_worker`` owns playback.
    Caps pending phrases while busy so stream handlers cannot hammer the device.
    Long utterances are chunked into sequential spool items (sentence-level) so
    playback can start before LangGraph finishes the full turn.

    ``interruptible=False`` marks UI acknowledgments (wake "Yes?", mode acks) so
    VAD/wake barge-in cannot cut them (Apple-style self-barge exemption).
    When omitted, canned UX cache hits default to uninterruptible.

    Stage 8.8 — ``agent_id`` selects a multi-voice profile (broker/moa/jason/…).
    """
    pieces = chunk_text_for_tts(text or "")
    if not pieces:
        with _tts_enqueue_lock:
            if tts_queue.empty() and not tts_busy.is_set():
                speech_idle.set()
        return

    try:
        from dana.audio.multi_voice_tts import (
            normalize_agent_id,
            set_active_tts_agent,
        )

        aid = normalize_agent_id(agent_id) if agent_id else "broker"
        if agent_id:
            set_active_tts_agent(aid)
    except Exception:  # noqa: BLE001
        aid = (agent_id or "broker").strip().lower() or "broker"

    with _tts_enqueue_lock:
        for piece in pieces:
            if interruptible is None:
                # Canned UX WAVs (Yes?, mode active, …) are uninterruptible by default.
                try:
                    piece_interruptible = canned_ux_cache_path(piece) is None
                except Exception:  # noqa: BLE001
                    piece_interruptible = True
            else:
                piece_interruptible = bool(interruptible)
            speech_idle.clear()
            pending = tts_queue.qsize()
            if tts_busy.is_set() and pending >= _SPEECH_MAX_PENDING_WHILE_BUSY:
                log_debug(
                    "TTS",
                    f"spool busy — drop overflow chars={len(piece)} pending={pending}",
                )
                break
            try:
                tts_queue.put_nowait((piece, piece_interruptible, aid))
                log_debug(
                    "TTS",
                    f"spooled chars={len(piece)} interruptible={piece_interruptible} "
                    f"agent={aid} pending={tts_queue.qsize()} "
                    f"busy={tts_busy.is_set()} vad={vad_capture_active.is_set()}",
                )
            except queue.Full:
                log_debug("TTS", f"spool full — drop chars={len(piece)}")
                break


def _parse_tts_spool_item(item: Any) -> tuple[str, bool, str]:
    """Normalize queue items to ``(text, interruptible, agent_id)``."""
    if isinstance(item, tuple) and item:
        text = str(item[0] or "")
        flag = bool(item[1]) if len(item) > 1 else True
        agent = str(item[2] or "broker") if len(item) > 2 else "broker"
        return text, flag, agent
    return str(item or ""), True, "broker"


def flush_tts_queue() -> int:
    """Instantly dump pending system messages (barge-in / interrupt).

    Uses the internal deque clear under the queue mutex so the agent does not
    keep talking after being cut off.
    """
    with _tts_enqueue_lock:
        with tts_queue.mutex:
            n = len(tts_queue.queue)
            tts_queue.queue.clear()
            tts_queue.not_full.notify_all()
        if n:
            log_debug("TTS", f"Flushed {n} pending spool item(s)")
        return n


def flush_speech_queue() -> int:
    """Alias for ``flush_tts_queue`` (legacy call sites)."""
    return flush_tts_queue()


def _bind_tts_barge_controller() -> None:
    """Wire core_agent callbacks into the shared ``TtsWorker`` (idempotent)."""

    def _reset_stream() -> None:
        try:
            from dana.agentic import reset_stream_sentence_tts

            reset_stream_sentence_tts()
        except Exception:  # noqa: BLE001
            pass

    _tts_barge.bind(
        flush_fn=flush_tts_queue,
        sd_stop_fn=_safe_sd_stop,
        set_ui_fn=set_ui_state,
        reset_stream_fn=_reset_stream,
        busy_fn=tts_busy.is_set,
    )


def interrupt_tts(*, reason: str = "barge_in", force: bool = False) -> int:
    """Hard-stop TTS: latch barge-in, flush spool, abort PortAudio stream."""
    _bind_tts_barge_controller()
    return int(
        _tts_barge.interrupt(reason=reason, set_listening=True, force=force)
    )


def _wait_tts_clear_of_user_speech(text: str) -> bool:
    """Hold a dequeued phrase while VAD capture is active.

    Returns False if the phrase should be discarded (interrupt / shutdown /
    hold timeout) instead of spoken.
    """
    if not vad_capture_active.is_set():
        return True
    log_debug(
        "TTS",
        f"holding spool item while user speaks chars={len(text)} "
        f"(max={_TTS_HOLD_FOR_VAD_MAX_S:.0f}s)",
    )
    deadline = time.perf_counter() + _TTS_HOLD_FOR_VAD_MAX_S
    while vad_capture_active.is_set() and not stop_event.is_set():
        if tts_interrupt_event.is_set():
            log_debug("TTS", "discard held spool item (barge-in during VAD hold)")
            return False
        if time.perf_counter() >= deadline:
            log(
                "TTS",
                "WARNING: discard held spool item — user speech exceeded hold window",
            )
            return False
        time.sleep(0.05)
    if stop_event.is_set() or tts_interrupt_event.is_set():
        return False
    return True


def reset_tts_audio_state(
    reason: str = "",
    *,
    ui_state: str = "idle",
    flush_queue: bool = True,
) -> int:
    """Force-release TTS / PortAudio locks after timeout or hung Piper playback.

    Without this, ``speech_idle`` stays cleared and ``tts_busy`` may remain set,
    which permanently blocks the wake-word listener.
    """
    _bind_tts_barge_controller()
    if flush_queue:
        dropped = interrupt_tts(reason=f"reset:{reason or 'unspecified'}")
    else:
        tts_interrupt_event.set()
        _safe_sd_stop(where=f"reset_tts:{reason or 'unspecified'}")
        dropped = 0
    tts_busy.clear()
    speech_idle.set()
    # Do not clear vad_capture_active here — record_utterance may still own the mic.
    try:
        set_ui_state(ui_state)
    except Exception:  # noqa: BLE001
        pass
    if reason:
        log(
            "Audio",
            f"TTS state reset ({reason}); flushed {dropped} queued item(s); "
            f"-> {ui_state}/listening",
        )
    return dropped


def wait_for_speech_idle(timeout: float = 20.0) -> None:
    """Block until queued TTS has finished playing (or timeout + hard recovery)."""
    if speech_idle.wait(timeout=timeout):
        return
    try:
        reset_tts_audio_state(f"timed out waiting for TTS after {timeout:.1f}s")
    finally:
        # Guarantee wake-word gates even if interrupt/flush races a worker.
        tts_busy.clear()
        speech_idle.set()


def report_audio_hardware_fault(exc: BaseException, *, where: str = "audio") -> None:
    """Signal Main that PortAudio/hardware failed so it can soft-recover before freeze."""
    global _audio_hardware_fault_detail
    detail = f"{where}: {type(exc).__name__}: {exc}"
    with _audio_hardware_fault_lock:
        _audio_hardware_fault_detail = detail
    audio_hardware_fault.set()
    log_exception("Audio", f"TTS Engine Failure / PortAudio fault ({where})", exc=exc)


def consume_audio_hardware_fault() -> str:
    """Return and clear the pending hardware-fault detail (empty if none)."""
    global _audio_hardware_fault_detail
    if not audio_hardware_fault.is_set():
        return ""
    with _audio_hardware_fault_lock:
        detail = _audio_hardware_fault_detail
        _audio_hardware_fault_detail = ""
    audio_hardware_fault.clear()
    return detail


def soft_recover_audio_hardware(detail: str = "") -> None:
    """Main-loop soft restart after PaErrorCode: release locks and log device state."""
    reason = detail or "PortAudio hardware fault"
    log("Main", f"Audio hardware fault — soft restart ({reason})")
    reset_tts_audio_state(f"hardware fault soft-restart: {reason}", ui_state="idle")
    flush_audio_buffer_queue()
    try:
        list_input_devices()
        list_output_devices()
    except Exception as exc:  # noqa: BLE001
        log_exception("Main", "Failed listing audio devices during soft restart", exc=exc)
    try:
        # Nudge PortAudio to drop stale streams.
        sd.stop()
    except Exception:  # noqa: BLE001
        pass
    request_mic_ingest_restart()
    ensure_mic_ingest_thread()
    log("Main", "Audio soft restart complete — returning to idle/listening")


def _is_portaudio_error(exc: BaseException) -> bool:
    """True for sounddevice PortAudioError or messages carrying PaErrorCode."""
    name = type(exc).__name__
    if name == "PortAudioError":
        return True
    msg = str(exc)
    return "PaErrorCode" in msg or "PortAudio" in msg


def speak_tool_working_ack(call: ToolCall, reply_lang: str) -> None:
    """Short TTS filler as soon as we know a tool will run (before slow LLM/search)."""
    if _tool_working_ack_sent.is_set():
        return
    _tool_working_ack_sent.set()
    tool_id = getattr(call, "tool_id", "") or ""
    if reply_lang == "fa":
        phrase = {
            "web_search": "  .",
            "describe_spatial_scene": "  .",
            "read_vault_memory": " ‌   .",
            "read_clipboard_context": " ‌  .",
            "run_terminal_command": "    .",
            "shell_execute": "  .",
            "file_editor": "  .",
            "python_repl": "  .",
            "flush_memory": "  ‌   ‌.",
            "publish_tool_to_general": "      ‌.",
            "open_application": "    ‌.",
            "read_local_file": "   .",
            "read_system_architecture": " .",
            "dispatch_research_swarm": "    ‌.",
            "dispatch_watchdog": "   ‌.",
            "kill_watchdog": "    ‌.",
            "save_script_to_library": "      ‌.",
        }.get(tool_id, " .")
    else:
        phrase = {
            "web_search": "Let me check.",
            "describe_spatial_scene": "Let me look.",
            "read_vault_memory": "Let me check my memory.",
            "read_clipboard_context": "Let me check the clipboard.",
            "run_terminal_command": "Let me run that in the terminal.",
            "shell_execute": "Running that in the local shell.",
            "file_editor": "Working on that file.",
            "python_repl": "Running that in the Python sandbox.",
            "flush_memory": "Okay — wiping short-term memory.",
            "publish_tool_to_general": "Okay — promoting that tool to general.",
            "open_application": "Okay — opening that now.",
            "read_local_file": "Let me read that file.",
            "read_system_architecture": "Let me see.",
            "dispatch_research_swarm": "Sending that to the research swarm.",
            "dispatch_watchdog": "Okay — deploying a watchdog.",
            "kill_watchdog": "Okay — stopping that watchdog.",
            "save_script_to_library": "Okay — saving that script to the library.",
        }.get(tool_id, "Let me see.")
    log_debug("Conversation", f'Tool working ack ({tool_id}): "{phrase}"')
    set_subtitle(phrase)
    # Fire-and-forget so Piper plays while Ollama / web_search run on this thread.
    # Short filler acks are uninterruptible (avoid self-barge from speaker bleed).
    enqueue_speech(phrase, interruptible=False)


def spatial_zone(cx: float, cy: float, frame_w: int = FRAME_SIZE[0], frame_h: int = FRAME_SIZE[1]) -> str:
    """Map a point to a 3x3 spatial label for 640x480 (or given) frames."""
    # X-axis: Left (< 213), Center (213-426), Right (> 426) on 640-wide frames.
    x_left = frame_w / 3.0
    x_right = 2.0 * frame_w / 3.0
    # Y-axis: Top (< 160), Center (160-320), Bottom (> 320) on 480-tall frames.
    y_top = frame_h / 3.0
    y_bottom = 2.0 * frame_h / 3.0

    if cx < x_left:
        x_pos = "left"
    elif cx > x_right:
        x_pos = "right"
    else:
        x_pos = "center"

    if cy < y_top:
        y_pos = "top"
    elif cy > y_bottom:
        y_pos = "bottom"
    else:
        y_pos = "center"

    if x_pos == "center" and y_pos == "center":
        return "center"
    if x_pos == "center":
        return y_pos
    if y_pos == "center":
        return x_pos
    return f"{y_pos}-{x_pos}"


def parse_yolo_results(results: Any) -> tuple[list[str], list[tuple[np.ndarray, str, float]]]:
    """Return spatial labels like 'bottle (top-left)' plus drawable detections."""
    labels: list[str] = []
    dets: list[tuple[np.ndarray, str, float]] = []
    if not results:
        return labels, dets

    result = results[0]
    names = result.names
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return labels, dets

    # Prefer actual frame size from the result if available.
    frame_w, frame_h = FRAME_SIZE
    try:
        shape = getattr(result, "orig_shape", None)
        if shape is not None and len(shape) >= 2:
            frame_h, frame_w = int(shape[0]), int(shape[1])
    except Exception:
        pass

    for box in boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        name = str(names.get(cls_id, cls_id))
        xyxy = box.xyxy[0].detach().cpu().numpy()
        x1, y1, x2, y2 = (float(v) for v in xyxy)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        zone = spatial_zone(cx, cy, frame_w, frame_h)
        spatial_label = f"{name} ({zone})"
        labels.append(spatial_label)
        dets.append((xyxy, spatial_label, conf))
    return labels, dets


def remember_spatial_labels(labels: list[str]) -> None:
    now = time.monotonic()
    with spatial_memory_lock:
        for label in labels:
            spatial_memory[label] = now
        stale = [k for k, ts in spatial_memory.items() if now - ts > SPATIAL_MEMORY_SEC]
        for key in stale:
            del spatial_memory[key]


def get_spatial_memory_labels() -> list[str]:
    now = time.monotonic()
    with spatial_memory_lock:
        alive = [(label, ts) for label, ts in spatial_memory.items() if now - ts <= SPATIAL_MEMORY_SEC]
        alive.sort(key=lambda row: row[1], reverse=True)
        return [label for label, _ts in alive]


def format_class_list(labels: list[str] | set[str]) -> str:
    """Join spatial anchors; keep same-class objects in different zones."""
    if isinstance(labels, set):
        items = sorted(labels)
    else:
        # Preserve order; dedupe identical full labels only.
        items = list(dict.fromkeys(labels))
    return ", ".join(items) if items else "none detected"


def format_vision_context_for_llm(labels: list[str] | set[str] | str | None) -> str:
    """Natural Visual Context sentence for ReAct injection (empty if none)."""
    from dana.prompts.spatial_synthesis import format_vision_context

    return format_vision_context(labels)



# ---------------------------------------------------------------------------
# Secure encrypted memory vault (AES-256 via Fernet + PBKDF2)
# ---------------------------------------------------------------------------

# SecureMemory lives in dana.secure_memory.py (shared with vault daemon).



def email_recovery_key(recovery_key: str) -> None:
    """Optionally email the backup recovery key via SendGrid v3 API (.env credentials)."""
    choice = input(
        "Would you like to email your Backup Recovery Key to yourself? (y/n): "
    ).strip().lower()
    if choice not in ("y", "yes"):
        print(
            f"\n[Memory] Backup Recovery Key (save this somewhere safe):\n{recovery_key}\n",
            flush=True,
        )
        return

    sendgrid_api_key = (os.getenv("SENDGRID_API_KEY") or "").strip()
    sendgrid_from_email = (os.getenv("SENDGRID_FROM_EMAIL") or "").strip()
    if not sendgrid_api_key or not sendgrid_from_email:
        print(
            "[System] SENDGRID_API_KEY or SENDGRID_FROM_EMAIL missing from .env. "
            "Skipping email recovery setup.",
            flush=True,
        )
        print(
            f"\n[Memory] Backup Recovery Key (save this somewhere safe):\n{recovery_key}\n",
            flush=True,
        )
        log(
            "Memory",
            "WARNING: SendGrid env vars missing; printed recovery key to terminal.",
        )
        return

    destination = input(
        "Enter Destination Email Address (where the key should be sent): "
    ).strip()
    if not destination:
        print(
            "[Memory] No destination email provided. Skipping send.",
            flush=True,
        )
        print(
            f"\n[Memory] Backup Recovery Key (save this somewhere safe):\n{recovery_key}\n",
            flush=True,
        )
        return

    payload = {
        "personalizations": [
            {
                "to": [{"email": destination}],
            }
        ],
        "from": {"email": sendgrid_from_email},
        "subject": "Dānā: Secure Memory Recovery Key",
        "content": [
            {
                "type": "text/plain",
                "value": (
                    "Your Donna Backup Recovery Key is below.\n"
                    "Store it offline. Anyone with this key can unlock Donna's memory vault.\n\n"
                    f"{recovery_key}\n"
                ),
            }
        ],
    }
    headers = {
        "Authorization": f"Bearer {sendgrid_api_key}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(
            SENDGRID_MAIL_URL,
            headers=headers,
            json=payload,
            timeout=30,
        )
        if resp.status_code == 202:
            print("[Memory] Recovery key emailed successfully via SendGrid.", flush=True)
            log("Memory", "Recovery key emailed successfully via SendGrid.")
            return
        print(
            f"[Memory] SendGrid rejected the request "
            f"(HTTP {resp.status_code}): {resp.text[:300]}",
            flush=True,
        )
        print(
            f"\n[Memory] Backup Recovery Key (save this somewhere safe):\n{recovery_key}\n",
            flush=True,
        )
        log(
            "Memory",
            f"WARNING: SendGrid HTTP {resp.status_code}; printed recovery key.",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[Memory] Email failed: {exc}", flush=True)
        print(
            f"\n[Memory] Backup Recovery Key (save this somewhere safe):\n{recovery_key}\n",
            flush=True,
        )
        log("Memory", f"WARNING: recovery email failed ({exc})")


def unlock_donna_memory() -> SecureMemory:
    """Unlock via in-RAM vault daemon (Option B). Password only if resume fails.

    Daemon handshake (try_resume_session) ALWAYS runs before any password prompt.
    """
    global donna_profile, donna_vault, vault_client
    vault_client = VaultClient()
    try:
        vault_client.ensure_ready()
    except Exception as exc:  # noqa: BLE001
        print(f"[Memory Error] Vault daemon unavailable: {exc}", flush=True)
        log("Memory", f"ERROR vault daemon: {exc}")
        raise SystemExit(1) from exc

    # --- HARD GATE: resume first; password ONLY in the else branch ---
    resumed = False
    try:
        resumed = bool(vault_client.try_resume_session())
    except Exception as exc:  # noqa: BLE001
        log("Memory", f"try_resume_session failed ({exc})")
        resumed = False

    if resumed:
        donna_profile = dict(vault_client.profile)
        vault = SecureMemory(path=MEMORY_FILE)
        try:
            from dana.vault_service import _rpc
            import base64 as _b64

            resp = _rpc(
                {
                    "op": "export_data_key",
                    "session_token": vault_client.session_token,
                },
                timeout=5.0,
            )
            if resp.get("ok"):
                key = _b64.urlsafe_b64decode(resp["data_key_b64"].encode("ascii"))
                vault.unlock_with_data_key(key)
                vault.profile = dict(donna_profile)
        except Exception as exc:  # noqa: BLE001
            log("Memory", f"WARNING: local vault hydrate skipped ({exc})")
        donna_vault = vault
        populate_vault_hot_cache(vault_client)
        log(
            "Memory",
            f"Vault unlocked via daemon session "
            f"(keys={len(donna_profile)}; token cached in RAM daemon).",
        )
        print(
            "[Memory] Vault unlocked via daemon session (keys cached in RAM).",
            flush=True,
        )
        return vault

    # else: daemon locked → resolve credential (env → keyring → TTY prompt)
    from dana.tools.vault import VaultCredentialsMissing, _get_master_key

    prompt = "Enter Master Password (or pasted Recovery Key) to unlock Dānā: "
    vault_exists = os.path.isfile(MEMORY_FILE)

    try:
        password = _get_master_key(prompt=prompt)
    except VaultCredentialsMissing as exc:
        print(f"[Memory Error] {exc}", flush=True)
        log("Memory", f"ERROR: {exc}")
        raise SystemExit(1) from exc

    if not vault_exists:
        recovery_key = secrets.token_urlsafe(32)
        try:
            donna_profile = vault_client.unlock(
                password, create=True, recovery_key=recovery_key
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[Memory Error] Could not create vault: {exc}", flush=True)
            log("Memory", f"ERROR creating vault: {exc}")
            raise SystemExit(1) from exc
        email_recovery_key(recovery_key)
    else:
        try:
            donna_profile = vault_client.unlock(password, create=False)
        except Exception as exc:  # noqa: BLE001
            print(f"[Memory Error] {exc}", flush=True)
            log("Memory", f"ERROR: {exc}")
            raise SystemExit(1) from exc

    vault = SecureMemory(path=MEMORY_FILE)
    try:
        from dana.vault_service import _rpc
        import base64 as _b64

        resp = _rpc(
            {
                "op": "export_data_key",
                "session_token": vault_client.session_token,
            },
            timeout=5.0,
        )
        if resp.get("ok"):
            key = _b64.urlsafe_b64decode(resp["data_key_b64"].encode("ascii"))
            vault.unlock_with_data_key(key)
            vault.profile = dict(donna_profile)
    except Exception as exc:  # noqa: BLE001
        log("Memory", f"WARNING: local vault hydrate failed ({exc})")

    donna_vault = vault
    populate_vault_hot_cache(vault_client)
    log(
        "Memory",
        f"Vault unlocked via daemon session "
        f"(keys={len(donna_profile)}; token cached in RAM daemon).",
    )
    return vault


def reset_donna_vault() -> None:
    """Authorize with master/recovery credential, then wipe the encrypted vault."""
    if not os.path.isfile(MEMORY_FILE):
        print("No vault found.", flush=True)
        log("Memory", "No vault found (--reset-vault).")
        raise SystemExit(0)

    from dana.tools.vault import VaultCredentialsMissing, _get_master_key

    try:
        password = _get_master_key(
            prompt=(
                "Enter Master Password (or Recovery Key) to authorize vault deletion: "
            )
        )
    except VaultCredentialsMissing as exc:
        print(f"[Security] ACCESS DENIED. {exc}", flush=True)
        raise SystemExit(1) from exc

    vault = SecureMemory()
    try:
        vault.unlock(password)
    except ValueError:
        print("[Security] ACCESS DENIED. Incorrect password.", flush=True)
        log("Memory", "ACCESS DENIED on --reset-vault (bad credential).")
        raise SystemExit(1)
    except Exception as exc:  # noqa: BLE001
        print("[Security] ACCESS DENIED. Incorrect password.", flush=True)
        log("Memory", f"ACCESS DENIED on --reset-vault ({exc}).")
        raise SystemExit(1)

    # Credential verified — safe to wipe.
    for path in (MEMORY_FILE, MEMORY_FILE + ".tmp"):
        try:
            os.remove(path)
            log("Memory", f"Deleted {path} (--reset-vault).")
        except FileNotFoundError:
            continue
        except OSError as exc:
            print(f"[Security] Could not delete {path}: {exc}", flush=True)
            raise SystemExit(1) from exc

    print("[Security] Vault successfully wiped.", flush=True)
    log("Memory", "Vault successfully wiped after authorized --reset-vault.")


# ---------------------------------------------------------------------------
# Dynamic audio configuration (settings.json)
# ---------------------------------------------------------------------------

def _device_rate(index: int) -> int:
    try:
        return int(round(float(sd.query_devices()[index]["default_samplerate"])))
    except Exception:
        return SAMPLE_RATE


def _validate_mic_id(mic_id: int) -> bool:
    devices = sd.query_devices()
    if mic_id < 0 or mic_id >= len(devices):
        return False
    return int(devices[mic_id].get("max_input_channels", 0)) >= 1


def _validate_speaker_id(speaker_id: int) -> bool:
    devices = sd.query_devices()
    if speaker_id < 0 or speaker_id >= len(devices):
        return False
    return int(devices[speaker_id].get("max_output_channels", 0)) >= 1


def save_audio_settings(mic_id: int, speaker_id: int) -> None:
    # Preserve non-audio flags (e.g. enable_dynamic_tool_synthesis) across audio saves.
    payload: dict[str, Any] = {}
    if os.path.isfile(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as fh:
                existing = json.load(fh)
            if isinstance(existing, dict):
                payload.update(existing)
        except Exception:  # noqa: BLE001
            pass
    payload["mic_id"] = int(mic_id)
    payload["speaker_id"] = int(speaker_id)
    if "enable_dynamic_tool_synthesis" not in payload:
        payload["enable_dynamic_tool_synthesis"] = True
    if "assistant_language" not in payload:
        payload["assistant_language"] = "en"
    if "whisper_language" not in payload:
        payload["whisper_language"] = "english"
    with open(SETTINGS_FILE, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    log("Audio", f"Saved audio settings -> {os.path.basename(SETTINGS_FILE)}")


def interactive_audio_setup() -> tuple[int, int]:
    """First-run terminal wizard: pick mic + speaker, persist to settings.json."""
    print("\n=== Donna first-run audio setup ===", flush=True)
    print(
        "No settings.json found. Let's configure your microphone and speakers.\n",
        flush=True,
    )

    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    print("Available INPUT devices:", flush=True)
    print(f"{'Index':<7} {'Rate':<8} {'Ch':<4} {'HostAPI':<18} Name", flush=True)
    print("-" * 72, flush=True)
    for idx, dev in enumerate(devices):
        if int(dev.get("max_input_channels", 0)) < 1:
            continue
        try:
            api = str(hostapis[int(dev["hostapi"])]["name"])
        except Exception:
            api = "?"
        rate = int(round(float(dev.get("default_samplerate", 0))))
        print(
            f"{idx:<7} {rate:<8} {int(dev['max_input_channels']):<4} "
            f"{api:<18} {dev.get('name', '')}",
            flush=True,
        )

    print("\nAvailable OUTPUT devices:", flush=True)
    print(f"{'Index':<7} {'Rate':<8} {'Ch':<4} {'HostAPI':<18} Name", flush=True)
    print("-" * 72, flush=True)
    for idx, dev in enumerate(devices):
        if int(dev.get("max_output_channels", 0)) < 1:
            continue
        try:
            api = str(hostapis[int(dev["hostapi"])]["name"])
        except Exception:
            api = "?"
        rate = int(round(float(dev.get("default_samplerate", 0))))
        print(
            f"{idx:<7} {rate:<8} {int(dev['max_output_channels']):<4} "
            f"{api:<18} {dev.get('name', '')}",
            flush=True,
        )

    while True:
        raw = input(
            "\nPlease enter the ID of your preferred Microphone: "
        ).strip()
        try:
            mic_id = int(raw)
        except ValueError:
            print("Please enter a numeric device index.", flush=True)
            continue
        if not _validate_mic_id(mic_id):
            print(f"Invalid microphone ID: {mic_id}", flush=True)
            continue
        break

    while True:
        raw = input(
            "Please enter the ID of your preferred Speaker/Headphones: "
        ).strip()
        try:
            speaker_id = int(raw)
        except ValueError:
            print("Please enter a numeric device index.", flush=True)
            continue
        if not _validate_speaker_id(speaker_id):
            print(f"Invalid speaker ID: {speaker_id}", flush=True)
            continue
        break

    save_audio_settings(mic_id, speaker_id)
    print(
        f"Saved settings.json (mic={mic_id}, speaker={speaker_id}). "
        "Delete settings.json to re-run this wizard.\n",
        flush=True,
    )
    return mic_id, speaker_id


def load_audio_settings() -> tuple[int, int, int]:
    """Load mic/speaker from settings.json, or run interactive setup.

    Returns (mic_id, speaker_id, mic_native_rate).
    """
    if not os.path.isfile(SETTINGS_FILE):
        mic_id, speaker_id = interactive_audio_setup()
        return mic_id, speaker_id, _device_rate(mic_id)

    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        mic_id = int(cfg["mic_id"])
        speaker_id = int(cfg["speaker_id"])
    except Exception as exc:  # noqa: BLE001
        log("Audio", f"WARNING: settings.json unreadable ({exc}); re-running setup.")
        mic_id, speaker_id = interactive_audio_setup()
        return mic_id, speaker_id, _device_rate(mic_id)

    if not _validate_mic_id(mic_id) or not _validate_speaker_id(speaker_id):
        log(
            "Audio",
            f"WARNING: settings.json devices invalid "
            f"(mic={mic_id}, speaker={speaker_id}); re-running setup.",
        )
        mic_id, speaker_id = interactive_audio_setup()
        return mic_id, speaker_id, _device_rate(mic_id)

    devices = sd.query_devices()
    log(
        "Audio",
        f"Loaded settings.json -> mic [{mic_id}] {devices[mic_id].get('name')} | "
        f"speaker [{speaker_id}] {devices[speaker_id].get('name')}",
    )
    return mic_id, speaker_id, _device_rate(mic_id)


def build_donna_system_prompt(
    yolo_labels: list[str],
    profile: Optional[dict[str, Any]] = None,
    user_text: str = "",
) -> str:
    """System prompt: SpatialIR synthesis guide + ReAct protocol + language lock."""
    if profile is None:
        profile = donna_profile
    if profile:
        try:
            from dana.vault_service import profile_for_prompt

            flat = profile_for_prompt(profile)
            profile_summary = json.dumps(flat, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            flat = {}
            try:
                profile_summary = json.dumps(profile, ensure_ascii=False, separators=(",", ":"))
            except Exception:
                profile_summary = str(profile)
    else:
        flat = {}
        profile_summary = "No long-term user profile stored yet."
    spatial_block = SPATIAL_AGGREGATOR.synthesize_prompt_block()
    labels = yolo_labels or SPATIAL_AGGREGATOR.label_list()
    from dana.settings import resolve_reply_lang

    reply_lang = resolve_reply_lang(user_text)
    prompt = build_agent_system_prompt(
        spatial_block=spatial_block,
        labels_csv=format_class_list(labels),
        profile_summary=profile_summary,
        reply_lang=reply_lang,
        timezone=str(flat.get("timezone") or "") or None,
        home_city=str(flat.get("home_city") or "") or None,
        home_region=str(flat.get("home_region") or "") or None,
        vault_hot_cache=VAULT_HOT_CACHE or None,
    )
    # Inject distilled lessons_learned when the intent matches a prior failure domain.
    try:
        broker = get_broker()
        if vault_client is not None and vault_client.session_token:

            def _lessons_provider():
                from dana.reflector import load_lessons

                return load_lessons(vault_client)

            broker.set_lessons_provider(_lessons_provider)
        if user_text:
            prompt = broker.augment_system_prompt(prompt, user_text)
    except Exception:  # noqa: BLE001
        pass
    return prompt


def build_voice_prompt(yolo_labels: list[str], whisper_text: str) -> str:
    """Legacy SmolVLM prompt (unused by Ollama cascade; kept for reference)."""
    vision = format_vision_context_for_llm(yolo_labels)
    vision_line = f"{vision} " if vision else ""
    return (
        "System context: You are Dānā, a helpful AI assistant. "
        f"{vision_line}"
        f"User asks: '{whisper_text}'. "
        "Respond in exactly one complete, natural sentence."
    )


def _keyword_hit(text_l: str, keywords: list[str]) -> bool:
    """Match multi-word phrases via substring; single words via word boundaries."""
    for key in keywords:
        if " " in key:
            if key in text_l:
                return True
        elif re.search(rf"\b{re.escape(key)}\b", text_l):
            return True
    return False


# ---------------------------------------------------------------------------
# Thread 1 - YOLO tracker (pulls frames from active_vision_tool)
# ---------------------------------------------------------------------------

def tracker_worker(device) -> None:
    _nt_hide_console_if_mp_child()
    global latest_frame, latest_dets

    from dana.tracker import (
        FRAME_BUFFER_INTERVAL_S,
        get_yolo_model,
        map_box_to_screen,
        primary_monitor_geometry,
        push_frame,
        should_push_frame,
        seconds_since_last_push,
        yolo_is_loaded,
    )

    log(
        "Tracker",
        f"Idle (JIT YOLO) — will load {YOLO_WEIGHTS} on Vision mode or first detect.",
    )
    yolo_dev = yolo_device_arg(device)
    buf_interval = float(TRACKER_BUFFER_INTERVAL_S or FRAME_BUFFER_INTERVAL_S)
    log(
        "Tracker",
        f"Rolling buffer every {buf_interval:.1f}s from "
        f"active_vision_tool.get_frame() (maxlen=5).",
    )

    frames = 0
    while not stop_event.is_set():
        # Wait out the ~2s cadence before grabbing (avoids busy mss polling).
        if not should_push_frame(interval_s=buf_interval):
            rem = buf_interval - seconds_since_last_push()
            if rem == float("inf"):
                rem = 0.0
            time.sleep(max(0.05, min(TRACKER_SLEEP_SEC, max(0.0, rem))))
            continue

        with active_vision_lock:
            tool = active_vision_tool
        tool_name = "camera" if tool is camera_tool else "screen"

        try:
            frame = tool.get_frame()
        except Exception as exc:  # noqa: BLE001
            log("Tracker", f"WARNING: {tool_name} get_frame failed ({exc})")
            time.sleep(0.05)
            continue

        if frame is None:
            time.sleep(0.05)
            continue

        with latest_frame_lock:
            latest_frame = frame

        try:
            mode = get_donna_mode()
        except Exception:  # noqa: BLE001
            mode = "chat"

        run_yolo = mode == "vision" or yolo_is_loaded()
        monitor = primary_monitor_geometry() if tool_name == "screen" else None

        dets: list = []
        labels: list[str] = []
        if run_yolo:
            try:
                yolo = get_yolo_model(YOLO_WEIGHTS)
                results = yolo.predict(
                    source=frame,
                    conf=YOLO_CONF,
                    device=yolo_dev,
                    verbose=False,
                )
                _, dets = parse_yolo_results(results)
            except Exception as exc:  # noqa: BLE001
                log("Tracker", f"WARNING: YOLO predict failed: {exc}")
                dets = []

            with latest_dets_lock:
                latest_dets = dets

            labels = [name for _, name, _ in dets]
            remember_spatial_labels(labels)
            SPATIAL_AGGREGATOR.set_vision_source(tool_name)
            SPATIAL_AGGREGATOR.update_from_dets(
                dets, frame_shape=getattr(frame, "shape", None)
            )

            if dets and tool_name == "screen":
                try:
                    from dana.vision.overlay import update_roi

                    best = max(dets, key=lambda d: float(d[2]))
                    xyxy, label, _conf = best
                    shape = getattr(frame, "shape", None)
                    fw = (
                        int(shape[1])
                        if shape is not None and len(shape) >= 2
                        else FRAME_SIZE[0]
                    )
                    fh = (
                        int(shape[0])
                        if shape is not None and len(shape) >= 2
                        else FRAME_SIZE[1]
                    )
                    screen_box = map_box_to_screen(
                        xyxy, frame_wh=(fw, fh), monitor=monitor
                    )
                    if screen_box is not None:
                        update_roi(screen_box, label)
                except Exception as exc:  # noqa: BLE001
                    log_debug("Tracker", f"ROI overlay update skipped ({exc})")

        push_frame(
            frame,
            source=tool_name,
            dets=list(dets),
            monitor=monitor,
            force=True,
        )

        frames += 1
        if frames % 15 == 0:
            from dana.tracker import buffer_len

            log_debug(
                "Tracker",
                f"Alive - {frames} samples via {tool_name}; "
                f"buffer={buffer_len()}/5; last=[{format_class_list(labels)}]",
            )

        time.sleep(TRACKER_SLEEP_SEC)

    log("Tracker", "Stopped.")
    try:
        from dana.vision.overlay import clear_roi

        clear_roi()
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Thread 3 - Wake word (OpenWakeWord — Donna only)
# ---------------------------------------------------------------------------

def wake_score_hit(
    prediction: dict[str, Any],
    *,
    require_token: str = "donna",
    threshold: float = WAKE_THRESHOLD,
) -> Optional[str]:
    """Return matched wake-word key if score crosses threshold for require_token."""
    token = (require_token or "donna").lower()
    for key, score in prediction.items():
        try:
            value = float(score)
        except (TypeError, ValueError):
            continue
        key_l = str(key).lower()
        if value >= threshold and token in key_l:
            return f"{key}={value:.2f}"
    return None


def _normalize_wake_text(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _wake_text_matches_donna(normalized: str) -> bool:
    """True if Whisper text is Donna or a known Donna mishearing."""
    if not normalized:
        return False
    if any(token in normalized for token in WAKE_PHRASE_TOKENS):
        return True
    # Exact / near-exact alias match (avoid accepting long unrelated sentences).
    if normalized in WAKE_PHRASE_ALIASES:
        return True
    for alias in WAKE_PHRASE_ALIASES:
        if normalized == alias or normalized.startswith(alias + " ") or normalized.endswith(" " + alias):
            return True
        # Short wake buffers are often just the misheard phrase.
        if len(normalized) <= len(alias) + 4 and alias in normalized:
            return True
    return False


def wake_phrase_confirmed(audio_16k: np.ndarray) -> bool:
    """Second gate: Whisper must hear Donna / Hey Donna in the wake buffer.

    When WAKE_PHRASE_VERIFY is False, openWakeWord score+onset alone starts the session.
    """
    if not WAKE_PHRASE_VERIFY:
        return True
    if audio_16k.size < SAMPLE_RATE // 4:
        return False

    with whisper_bundle_lock:
        bundle = whisper_bundle
    if bundle is None:
        # Whisper not loaded yet — keep energy/score gates only.
        return True

    processor, model, device, dtype = bundle
    try:
        import torch

        audio_prep = prepare_audio_for_whisper(audio_16k.astype(np.float32))
        inputs = processor(
            audio_prep,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )
        moved = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                if value.is_floating_point():
                    moved[key] = value.to(device=device, dtype=dtype)
                else:
                    moved[key] = value.to(device=device)
            else:
                moved[key] = value
        _sanitize_whisper_generation_config(model)
        gen_kwargs = _whisper_generate_kwargs(
            max_new_tokens=32,
            language="english",
            task="transcribe",
            model=model,
        )
        with torch.no_grad():
            generated_ids = model.generate(
                **moved,
                **gen_kwargs,
            )
        text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]
    except Exception as exc:  # noqa: BLE001
        log("WakeWord", f"WARNING: phrase verify failed ({exc}); allowing score gate only")
        return True

    normalized = _normalize_wake_text(text)
    if any(rej == normalized or rej in normalized for rej in WAKE_PHRASE_REJECT):
        log("WakeWord", f"Phrase verify REJECT (noise alias) -> \"{text.strip()}\"")
        print(f"[Debug] Wake phrase verify: \"{text.strip()}\" -> REJECT", flush=True)
        return False
    if _wake_text_matches_donna(normalized):
        log("WakeWord", f"Phrase verify PASS -> \"{text.strip()}\"")
        print(f"[Debug] Wake phrase verify: \"{text.strip()}\" -> PASS", flush=True)
        return True
    # Anything else (incl. short Whisper mishears like "Oh no.") is inconclusive:
    # OpenWakeWord score+onset already fired; only the explicit noise aliases above
    # are hard-rejected ("don't know" on hush).
    log(
        "WakeWord",
        f"Phrase verify inconclusive -> \"{text.strip()}\"; allowing score+energy gate",
    )
    print(
        f"[Debug] Wake phrase verify: \"{text.strip()}\" -> INCONCLUSIVE (allow)",
        flush=True,
    )
    return True


def wakeword_worker() -> None:
    _nt_hide_console_if_mp_child()
    global AUDIO_INPUT_DEVICE, AUDIO_INPUT_RATE

    # Prefer custom Donna model. If missing, temporarily use stock Alexa so we can
    # live-debug the mic path (Donna model is not listening at all when absent).
    wake_token = "donna"
    model_paths: list[str]
    if os.path.isfile(DONNA_WAKEWORD_ONNX):
        model_paths = [DONNA_WAKEWORD_ONNX]
        log("WakeWord", f"Loading OpenWakeWord model: {DONNA_WAKEWORD_ONNX}")
        try:
            from openwakeword.utils import download_models

            # Feature extractors (melspec/embedding) live in the package resources.
            download_models()
        except Exception as exc:  # noqa: BLE001
            log("WakeWord", f"WARNING: could not refresh OWW feature models ({exc})")
    else:
        print(
            "[Warning] donna.onnx not found! Temporary Alexa wake-word enabled for "
            "mic debugging. Say 'Alexa' (not Donna). Place donna.onnx in the project "
            "root to switch back.",
            flush=True,
        )
        log(
            "WakeWord",
            "WARNING: donna.onnx missing — temporary Alexa debug wake-word active.",
        )
        try:
            import openwakeword
            from openwakeword.utils import download_models

            models_dir = os.path.join(
                os.path.dirname(openwakeword.__file__), "resources", "models"
            )
            alexa_path = os.path.join(models_dir, "alexa_v0.1.onnx")
            if not os.path.isfile(alexa_path):
                log("WakeWord", "Downloading OpenWakeWord ONNX models for Alexa debug...")
                download_models()
            if not os.path.isfile(alexa_path):
                print(
                    "[Warning] Alexa debug model also missing. Voice wake-word disabled. "
                    "Use manual triggers (.trigger_ask).",
                    flush=True,
                )
                while not stop_event.is_set():
                    time.sleep(1)
                return
            model_paths = [alexa_path]
            wake_token = "alexa"
        except Exception as exc:  # noqa: BLE001
            print(
                f"[Warning] Could not load debug wake model ({exc}). "
                "Voice wake-word disabled. Use manual triggers.",
                flush=True,
            )
            log("WakeWord", f"WARNING: debug wake load failed ({exc})")
            while not stop_event.is_set():
                time.sleep(1)
            return

    try:
        oww = OpenWakeWordModel(
            wakeword_models=model_paths,
            inference_framework="onnx",
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"[Warning] Failed to load wake model ({exc}). "
            "Voice wake-word disabled. Use manual triggers.",
            flush=True,
        )
        log("WakeWord", f"WARNING: wake model load failed ({exc})")
        while not stop_event.is_set():
            time.sleep(1)
        return

    log("WakeWord", f"Models ready: {list(getattr(oww, 'models', {}).keys())}")
    global _shared_wakeword_model, _shared_wakeword_token
    _shared_wakeword_model = oww
    _shared_wakeword_token = wake_token
    if wake_token == "donna":
        print("Say 'Donna' to wake.", flush=True)
        listen_msg = "Donna"
    else:
        print("DEBUG: Say 'Alexa' to wake (temporary until donna.onnx is added).", flush=True)
        listen_msg = "Alexa (debug)"
    log(
        "WakeWord",
        f"Listening for {listen_msg} on mic [{AUDIO_INPUT_DEVICE}] @ {AUDIO_INPUT_RATE} Hz "
        "(or .trigger_ask)...",
    )
    print(
        f"[Debug] WakeWord using device={AUDIO_INPUT_DEVICE} "
        f"rate={AUDIO_INPUT_RATE} threshold={WAKE_THRESHOLD} "
        f"consec={WAKE_MIN_CONSECUTIVE} onset_below={WAKE_ONSET_BELOW} token={wake_token}",
        flush=True,
    )

    if not mic_ingest_ready.wait(timeout=8.0):
        log(
            "WakeWord",
            "WARNING: MicIngest not ready after 8s — continuing (will wait on queue)",
        )

    # Do not arm wake triggers until Ollama warm-up finishes (avoids CPU/TTS fights).
    log("WakeWord", "Waiting for Ollama warm-up before arming listener...")
    if not ollama_ready.wait(timeout=180.0):
        log(
            "WakeWord",
            "WARNING: Ollama warm-up not signaled after 180s — arming wake-word anyway",
        )
        ollama_ready.set()
    log("WakeWord", "Ollama ready — wake-word listener armed")
    wakeword_armed.set()
    maybe_play_boot_ready_audio()

    cooldown_until = 0.0
    next_rms_log = 0.0
    consecutive_hits = 0
    score_history: deque[float] = deque(maxlen=WAKE_ONSET_LOOKBACK)
    audio_ring: deque[np.ndarray] = deque(maxlen=WAKE_PHRASE_WINDOW_CHUNKS)
    next_sticky_reset = 0.0
    # Assemble WAKE_CHUNK (80ms) from shared VAD frames (30ms).
    wake_accum: list[np.ndarray] = []
    wake_accum_samples = 0

    def _reset_wake_accum() -> None:
        nonlocal wake_accum_samples
        wake_accum.clear()
        wake_accum_samples = 0

    def _pull_wake_audio() -> Optional[np.ndarray]:
        """Consumer: build one WAKE_CHUNK from audio_buffer_queue frames."""
        nonlocal wake_accum_samples
        while wake_accum_samples < WAKE_CHUNK:
            if (
                tts_busy.is_set()
                or is_recording.is_set()
                or vad_capture_active.is_set()
                or get_ui_state() != "idle"
            ):
                return None
            frame = get_mic_frame(timeout=0.2)
            if frame is None:
                return None
            wake_accum.append(frame)
            wake_accum_samples += int(frame.size)
        merged = np.concatenate(wake_accum).astype(np.float32, copy=False)
        audio = merged[:WAKE_CHUNK].copy()
        remainder = merged[WAKE_CHUNK:]
        wake_accum.clear()
        wake_accum_samples = 0
        if remainder.size:
            wake_accum.append(remainder)
            wake_accum_samples = int(remainder.size)
        return audio

    while not stop_event.is_set():
        # Stay disarmed until warm-up (and after soft recoveries that clear the flag).
        if not ollama_ready.is_set():
            _reset_wake_accum()
            time.sleep(0.1)
            continue

        # Yield while TTS / VAD / turn owns the audio queue (half-duplex:
        # mic frames during TTS are discarded by half_duplex_mic_drop).
        if (
            tts_busy.is_set()
            or is_recording.is_set()
            or vad_capture_active.is_set()
            or get_ui_state() != "idle"
        ):
            _reset_wake_accum()
            time.sleep(0.05)
            continue

        if time.monotonic() < cooldown_until:
            time.sleep(0.05)
            continue

        audio = _pull_wake_audio()
        if audio is None:
            continue

        audio_ring.append(audio.copy())
        chunk_rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0

        now = time.monotonic()
        if now >= next_rms_log:
            log_debug("Debug", f"Live Mic RMS: {chunk_rms:.6f}")
            next_rms_log = now + 3.0

        pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
        try:
            prediction = oww.predict(pcm)
        except Exception:
            try:
                prediction = oww.predict(audio)
            except Exception as exc:  # noqa: BLE001
                log("WakeWord", f"WARNING: predict failed: {exc}")
                consecutive_hits = 0
                continue

        pred = prediction if isinstance(prediction, dict) else {}
        best_score = 0.0
        for key, score in pred.items():
            try:
                value = float(score)
            except (TypeError, ValueError):
                continue
            key_l = str(key).lower()
            if wake_token in key_l:
                best_score = max(best_score, value)
            if value > 0.20:
                log_debug("Debug", f"Wake word score: {value:.4f} ({key})")

        score_history.append(best_score)
        hit = wake_score_hit(pred, require_token=wake_token)
        # Sticky high scores on hush never dip; real "Donna" rises from a low baseline.
        recently_low = any(s < WAKE_ONSET_BELOW for s in score_history)
        if hit and recently_low:
            consecutive_hits += 1
        else:
            if hit and not recently_low and now >= next_sticky_reset:
                log(
                    "WakeWord",
                    f"Rejected sticky false wake ({hit}); resetting detector "
                    f"(score never dipped below {WAKE_ONSET_BELOW:.2f})",
                )
                try:
                    oww.reset()
                except Exception:
                    pass
                next_sticky_reset = now + 2.0
            consecutive_hits = 0

        if consecutive_hits < WAKE_MIN_CONSECUTIVE:
            continue

        wake_audio = np.concatenate(list(audio_ring)) if audio_ring else audio
        # Diagnostic only — do not hard-reject on RMS (SteelSeries Sonar chat
        # mics often sit ~0.002–0.003 even on a real "Donna").
        wake_rms = (
            float(np.sqrt(np.mean(np.square(wake_audio)))) if wake_audio.size else 0.0
        )
        log_debug("WakeWord", f"Wake candidate buffer_rms={wake_rms:.5f} hit={hit}")

        if wake_token == "donna" and not wake_phrase_confirmed(wake_audio):
            consecutive_hits = 0
            cooldown_until = time.monotonic() + 1.5
            try:
                oww.reset()
            except Exception:
                pass
            continue

        if not is_engine_engaged():
            # Soft STANDBY — ignore wake hits until Dashboard ENGAGE.
            consecutive_hits = 0
            cooldown_until = time.monotonic() + 1.0
            try:
                oww.reset()
            except Exception:
                pass
            continue

        log("WakeWord", f"Wake word detected ({hit}) -> yield to VAD consumer")
        print(f"[Debug] Wake word HIT ({hit}) on device={AUDIO_INPUT_DEVICE}", flush=True)
        consecutive_hits = 0
        audio_ring.clear()
        score_history.clear()
        _reset_wake_accum()
        # Do NOT flush here — VAD takes the next frames from audio_buffer_queue.
        log_debug("WakeWord", "Consumer yielded; VAD will pull next mic frames")
        is_recording.set()
        cooldown_until = time.monotonic() + WAKE_COOLDOWN_SEC
        try:
            oww.reset()
        except Exception:
            pass

    log("WakeWord", "Stopped.")


# ---------------------------------------------------------------------------
# Model loading - SmolVLM + Whisper
# ---------------------------------------------------------------------------

def load_vlm(local_files_only: bool, device, dtype):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    log(
        "Conversation",
        f"Loading {MODEL_ID} (local_files_only={local_files_only}, dtype={dtype})...",
    )
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        local_files_only=local_files_only,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        local_files_only=local_files_only,
    ).to(device)
    model.eval()
    log("Conversation", f"SmolVLM ready in {time.perf_counter() - t0:.1f}s on {device}.")
    return processor, model


def load_whisper(local_files_only: bool, device):
    # Latency path: Distil-Whisper on cuda:0 FP16 (3B LLM leaves enough VRAM headroom).
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    _ = local_files_only  # Prefer cache first; download only on OSError miss.
    _ = device
    if torch.cuda.is_available():
        whisper_device = torch.device("cuda:0")
        whisper_dtype = torch.float16
    else:
        whisper_device = torch.device("cpu")
        whisper_dtype = torch.float32
    log(
        "Conversation",
        f"Loading {WHISPER_ID} (local_files_only=True first, "
        f"device={whisper_device}, dtype={whisper_dtype})...",
    )
    t0 = time.perf_counter()

    def _load_pair(*, files_only: bool):
        proc = AutoProcessor.from_pretrained(
            WHISPER_ID,
            local_files_only=files_only,
        )
        mdl = AutoModelForSpeechSeq2Seq.from_pretrained(
            WHISPER_ID,
            dtype=whisper_dtype,
            local_files_only=files_only,
        ).to(whisper_device)
        return proc, mdl

    try:
        processor, model = _load_pair(files_only=True)
    except OSError as exc:
        log(
            "Conversation",
            f"{WHISPER_ID} not in local cache ({exc}); "
            "falling back to download (local_files_only=False)...",
        )
        processor, model = _load_pair(files_only=False)

    model.eval()
    # Prefer max_new_tokens-only length control. Whisper configs ship with
    # max_length=448; leaving both set triggers transformers warnings.
    _sanitize_whisper_generation_config(model)
    log(
        "Conversation",
        f"Whisper ready in {time.perf_counter() - t0:.1f}s on {whisper_device} "
        f"(dtype={whisper_dtype}).",
    )
    return processor, model, whisper_dtype, whisper_device


def _sanitize_whisper_generation_config(model: Any) -> None:
    """Drop conflicting length / processor fields from Whisper generation_config."""
    gc = getattr(model, "generation_config", None)
    if gc is None:
        return
    # max_new_tokens alone must control length (do not keep max_length).
    if getattr(gc, "max_length", None) is not None:
        try:
            gc.max_length = None
        except Exception:  # noqa: BLE001
            pass
    # Never keep a stale max_new_tokens on the config — callers pass it per call.
    if hasattr(gc, "max_new_tokens"):
        try:
            gc.max_new_tokens = None
        except Exception:  # noqa: BLE001
            pass


def _whisper_is_english_only(model: Any | None = None) -> bool:
    """True for *.en Distil/Whisper checkpoints that reject language/task kwargs."""
    if model is not None:
        gc = getattr(model, "generation_config", None)
        if gc is not None and getattr(gc, "is_multilingual", None) is False:
            return True
        if getattr(model, "is_multilingual", None) is False:
            return True
    mid = str(WHISPER_ID or "").lower()
    return mid.endswith(".en")


def _whisper_generate_kwargs(
    *,
    max_new_tokens: int,
    language: str,
    task: str,
    model: Any | None = None,
) -> dict[str, Any]:
    """Generation kwargs for Whisper STT — max_new_tokens only, no logits_processor."""
    kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "condition_on_prev_tokens": False,
        # Intentionally omitted: max_length, logits_processor, suppress_tokens,
        # begin_suppress_tokens — transformers builds SuppressTokens* processors
        # from generation_config; passing them again duplicates and warns.
    }
    # English-only Distil-Whisper raises if language/task are set.
    if not _whisper_is_english_only(model):
        kwargs["language"] = language
        kwargs["task"] = task
    return kwargs


def ensure_whisper_bundle(timeout: float = 180.0):
    """Block until background Whisper load finishes; return (proc, model, device, dtype)."""
    global _whisper_load_error
    if not whisper_ready.wait(timeout=timeout):
        raise TimeoutError("Whisper background load timed out")
    with whisper_bundle_lock:
        bundle = whisper_bundle
    if bundle is None:
        raise RuntimeError(
            _whisper_load_error or "Whisper failed to load (no bundle)"
        )
    return bundle


def start_whisper_background_load(local_files_only: bool, device) -> None:
    """Kick off Whisper HF load on a daemon thread; wake-word stays unblocked."""
    global whisper_bundle, _whisper_load_error

    whisper_ready.clear()
    _whisper_load_error = None

    def _load() -> None:
        global whisper_bundle, _whisper_load_error
        try:
            processor, model, whisper_dtype, whisper_device = load_whisper(
                local_files_only, device
            )
            with whisper_bundle_lock:
                whisper_bundle = (
                    processor,
                    model,
                    whisper_device,
                    whisper_dtype,
                )
            if WAKE_PHRASE_VERIFY:
                log("WakeWord", "Whisper phrase-verify gate armed (must hear 'Donna').")
            else:
                log(
                    "WakeWord",
                    "Whisper phrase-verify DISABLED — openWakeWord score onset starts session.",
                )
        except OSError as exc:
            _whisper_load_error = (
                f"Whisper weights unavailable for {WHISPER_ID} "
                f"(cache miss and download failed): {exc}"
            )
            log("Conversation", f"ERROR: {_whisper_load_error}")
            stop_event.set()
        except Exception as exc:  # noqa: BLE001
            _whisper_load_error = f"ERROR loading Whisper: {exc}"
            log("Conversation", _whisper_load_error)
            stop_event.set()
        finally:
            whisper_ready.set()

    threading.Thread(target=_load, name="WhisperLoad", daemon=True).start()
    log(
        "Conversation",
        "Whisper load started in background; wake-word / VAD remain available.",
    )


def resample_to_16k(audio: np.ndarray, src_rate: int) -> np.ndarray:
    """Lightweight linear resample to 16 kHz for VAD / Whisper / OpenWakeWord."""
    return resample_audio(audio, src_rate, SAMPLE_RATE)


def resample_audio(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear-resample a 1-D float audio buffer between sample rates."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if src_rate == dst_rate or audio.size == 0:
        return audio
    dst_len = max(1, int(round(audio.size * dst_rate / float(src_rate))))
    x_old = np.linspace(0.0, 1.0, num=audio.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=dst_len, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


class DcBlocker:
    """Streaming first-order IIR DC blocker / high-pass for mic rumble.

    y[n] = x[n] - x[n-1] + R * y[n-1]
    Removes constant offset and very-low-frequency energy that inflates RMS and
    prevents WebRTC VAD from reaching ``silence_cutoff``.
    """

    __slots__ = ("_r", "_x_prev", "_y_prev")

    def __init__(self, r: float = DC_BLOCKER_R) -> None:
        self._r = float(np.clip(r, 0.9, 0.9999))
        self._x_prev = 0.0
        self._y_prev = 0.0

    def reset(self) -> None:
        self._x_prev = 0.0
        self._y_prev = 0.0

    def apply(self, samples: np.ndarray) -> np.ndarray:
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return x
        y = np.empty_like(x)
        x_prev = self._x_prev
        y_prev = self._y_prev
        r = self._r
        for i in range(x.size):
            xi = float(x[i])
            yi = xi - x_prev + r * y_prev
            y[i] = yi
            x_prev = xi
            y_prev = yi
        self._x_prev = x_prev
        self._y_prev = y_prev
        return y


def remove_dc_offset(audio: np.ndarray, *, r: float = DC_BLOCKER_R) -> np.ndarray:
    """One-shot DC blocker for a finished buffer (Whisper / wake verify)."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return audio
    # Mean subtract first (fast coarse DC kill), then light IIR for rumble.
    centered = audio - float(np.mean(audio))
    return DcBlocker(r=r).apply(centered)


def list_input_devices() -> None:
    """Print a clean table of sounddevice input devices."""
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    n_in = sum(1 for d in devices if int(d.get("max_input_channels", 0)) >= 1)
    log("Audio", f"INPUT devices: {n_in} available (DONNA_DEBUG=1 for full table)")
    log_debug("Audio", "Available INPUT devices:")
    log_debug("Audio", f"{'Index':<7} {'Rate':<8} {'Ch':<4} {'HostAPI':<18} Name")
    log_debug("Audio", "-" * 72)
    for idx, dev in enumerate(devices):
        channels = int(dev.get("max_input_channels", 0))
        if channels < 1:
            continue
        try:
            api = str(hostapis[int(dev["hostapi"])]["name"])
        except Exception:
            api = "?"
        rate = int(round(float(dev.get("default_samplerate", 0))))
        name = str(dev.get("name", ""))
        log_debug("Audio", f"{idx:<7} {rate:<8} {channels:<4} {api:<18} {name}")


def list_output_devices() -> None:
    """Print a clean table of sounddevice output devices (speaker routing check)."""
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    n_out = sum(1 for d in devices if int(d.get("max_output_channels", 0)) >= 1)
    log("Audio", f"OUTPUT devices: {n_out} available (DONNA_DEBUG=1 for full table)")
    log_debug("Audio", "Available OUTPUT devices (Windows speaker / monitor routing):")
    log_debug("Audio", f"{'Index':<7} {'Rate':<8} {'Ch':<4} {'HostAPI':<18} Name")
    log_debug("Audio", "-" * 72)
    for idx, dev in enumerate(devices):
        channels = int(dev.get("max_output_channels", 0))
        if channels < 1:
            continue
        try:
            api = str(hostapis[int(dev["hostapi"])]["name"])
        except Exception:
            api = "?"
        rate = int(round(float(dev.get("default_samplerate", 0))))
        name = str(dev.get("name", ""))
        log_debug("Audio", f"{idx:<7} {rate:<8} {channels:<4} {api:<18} {name}")
    try:
        default_out = sd.default.device[1]
        if default_out is not None and 0 <= int(default_out) < len(devices):
            out_name = devices[int(default_out)].get("name", "?")
            log("Audio", f"Default OUTPUT device: [{default_out}] {out_name}")
        else:
            log_debug("Audio", f"Default OUTPUT device: {default_out}")
    except Exception as exc:  # noqa: BLE001
        log("Audio", f"WARNING: could not resolve default OUTPUT device: {exc}")


def find_steelseries_speaker() -> Optional[tuple[int, str]]:
    """
    Prefer a SteelSeries Sonar playback endpoint that reaches the headset.
    Chat is best for voice; then Media; then Gaming.
    """
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    preferred_apis = ("mme", "wasapi", "directsound")
    channel_rank = ("chat", "media", "gaming")
    matches: list[tuple[int, int, int, int, str]] = []
    # api_rank, channel_rank_i, -channels, idx, name

    for idx, dev in enumerate(devices):
        if int(dev.get("max_output_channels", 0)) < 1:
            continue
        name = str(dev.get("name", ""))
        name_l = name.lower()
        if "steelseries" not in name_l:
            continue
        # Skip mic-monitor / capture-looking outputs.
        if "microphone" in name_l and "chat" not in name_l:
            continue
        try:
            api = str(hostapis[int(dev["hostapi"])]["name"]).lower()
        except Exception:
            api = ""
        if "wdm-ks" in api:
            continue
        api_rank = next(
            (i for i, token in enumerate(preferred_apis) if token in api),
            len(preferred_apis),
        )
        ch_rank = next(
            (i for i, token in enumerate(channel_rank) if token in name_l),
            len(channel_rank),
        )
        channels = int(dev.get("max_output_channels", 0))
        matches.append((api_rank, ch_rank, -channels, idx, name))

    if not matches:
        return None
    matches.sort()
    _a, _c, _ch, idx, name = matches[0]
    return idx, name


def pick_output_device(preferred: Optional[int] = None) -> Optional[int]:
    """Resolve TTS playback device (honors --speaker when provided)."""
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()

    if preferred is not None:
        if preferred < 0 or preferred >= len(devices):
            log("Audio", f"ERROR: --speaker {preferred} is out of range.")
            return None
        dev = devices[preferred]
        if int(dev.get("max_output_channels", 0)) < 1:
            log("Audio", f"ERROR: --speaker {preferred} is not an OUTPUT device.")
            return None
        try:
            api = str(hostapis[int(dev["hostapi"])]["name"])
        except Exception:
            api = "?"
        log(
            "Main",
            f"Selected speaker [{preferred}] {dev.get('name')} ({api}) via --speaker",
        )
        return preferred

    steel = find_steelseries_speaker()
    if steel is not None:
        idx, name = steel
        log("Main", f"Auto-selected SteelSeries speaker: {name} (Index {idx})")
        return idx

    try:
        default_out = sd.default.device[1]
        if default_out is not None:
            name = devices[int(default_out)].get("name", "?")
            log("Audio", f"Using default speaker [{default_out}] {name}")
            return int(default_out)
    except Exception:
        pass
    log("Audio", "WARNING: could not resolve speaker device; using system default.")
    return None


def find_steelseries_mic() -> Optional[tuple[int, int, str]]:
    """
    Prefer a SteelSeries virtual microphone input.
    Returns (index, sample_rate, name) or None.

    Never selects WDM-KS — that host API spams PaErrorCode -9999 / capture-pin
    failures on Sonar devices (especially after TTS playback).
    """
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    # MME is the most reliable for repeated sd.rec / InputStream on Windows Sonar.
    preferred_apis = ("mme", "wasapi", "directsound")
    matches: list[tuple[int, int, int, str, int]] = []
    # rank_api, prefer_mic_name (0 better), idx, name, rate

    for idx, dev in enumerate(devices):
        if int(dev.get("max_input_channels", 0)) < 1:
            continue
        name = str(dev.get("name", ""))
        name_l = name.lower()
        if "steelseries" not in name_l:
            continue
        if "stream" in name_l and "microphone" not in name_l:
            continue
        try:
            api = str(hostapis[int(dev["hostapi"])]["name"]).lower()
        except Exception:
            api = ""
        if "wdm-ks" in api:
            continue
        api_rank = next(
            (i for i, token in enumerate(preferred_apis) if token in api),
            len(preferred_apis),
        )
        # Prefer plain "Microphone" endpoints over chat-capture aliases.
        mic_rank = 0 if name_l.strip().startswith("steelseries sonar - microphone") else 1
        if "microphone" not in name_l:
            mic_rank = 2
        rate = int(round(float(dev.get("default_samplerate", SAMPLE_RATE))))
        matches.append((api_rank, mic_rank, idx, name, rate))

    if not matches:
        return None

    # Among remaining SteelSeries inputs, prefer the endpoint with real signal.
    scored: list[tuple[float, int, int, int, str, int]] = []
    for api_rank, mic_rank, idx, name, rate in matches:
        try:
            rms = probe_mic_rms(idx, rate, seconds=0.25)
        except Exception:
            rms = -1.0
        log("Audio", f"SteelSeries candidate [{idx}] RMS={rms:.6f} {name}")
        scored.append((-rms, api_rank, mic_rank, idx, name, rate))

    scored.sort()
    _neg_rms, _api_rank, _mic_rank, idx, name, rate = scored[0]
    return idx, rate, name


def pick_input_device(preferred: Optional[int] = None) -> tuple[Optional[int], int]:
    """Resolve mic index + native sample rate (honors --mic when provided)."""
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()

    def hostapi_name(dev: dict) -> str:
        try:
            return str(hostapis[int(dev["hostapi"])]["name"]).lower()
        except Exception:
            return ""

    # Explicit --mic always wins; skip SteelSeries auto-search.
    if preferred is not None:
        if preferred < 0 or preferred >= len(devices):
            log("Audio", f"ERROR: --mic {preferred} is out of range.")
            return None, SAMPLE_RATE
        dev = devices[preferred]
        if int(dev.get("max_input_channels", 0)) < 1:
            log("Audio", f"ERROR: --mic {preferred} is not an INPUT device.")
            return None, SAMPLE_RATE
        rate = int(round(float(dev.get("default_samplerate", SAMPLE_RATE))))
        api = hostapi_name(dev)
        log(
            "Audio",
            f"Selected mic device [{preferred}] {dev.get('name')} ({api}, {rate} Hz) via --mic",
        )
        return preferred, rate

    steel = find_steelseries_mic()
    if steel is not None:
        idx, rate, name = steel
        log(
            "Main",
            f"Auto-selected SteelSeries microphone: {name} (Index {idx})",
        )
        return idx, rate

    preferred_substrings = (
        "usb camera",
        "high definition aud",
        "hd audio microphone",
        "microphone (",
    )
    avoid_substrings = ("vb-audio", "cable", "mapper", "primary sound")
    # Never use WDM-KS for capture — it causes PaErrorCode -9999 spam on this machine.
    preferred_apis = ("mme", "wasapi", "directsound")

    candidates: list[tuple[int, int, str, str, int]] = []
    for idx, dev in enumerate(devices):
        if int(dev.get("max_input_channels", 0)) < 1:
            continue
        name = str(dev.get("name", ""))
        name_l = name.lower()
        if any(a in name_l for a in avoid_substrings):
            continue
        api = hostapi_name(dev)
        if "wdm-ks" in api:
            continue
        api_rank = next(
            (i for i, token in enumerate(preferred_apis) if token in api),
            len(preferred_apis),
        )
        rate = int(round(float(dev.get("default_samplerate", SAMPLE_RATE))))
        candidates.append((api_rank, idx, name, api, rate))

    candidates.sort(key=lambda row: (row[0], row[1]))

    for needle in preferred_substrings:
        for _api_rank, idx, name, api, rate in candidates:
            if needle in name.lower():
                log("Audio", f"Selected mic device [{idx}] {name} ({api}, {rate} Hz)")
                return idx, rate

    if candidates:
        _api_rank, idx, name, api, rate = candidates[0]
        log("Audio", f"Selected mic device [{idx}] {name} ({api}, {rate} Hz)")
        return idx, rate

    try:
        default_in = sd.default.device[0]
        name = devices[default_in]["name"] if default_in is not None else "default"
        rate = int(round(float(devices[default_in].get("default_samplerate", SAMPLE_RATE))))
        log("Audio", f"Using default mic device [{default_in}] {name} ({rate} Hz)")
        return (int(default_in) if default_in is not None else None), rate
    except Exception:
        log("Audio", "WARNING: could not resolve mic device; using system default")
        return None, SAMPLE_RATE


def probe_mic_rms(device_idx: Optional[int], rate: int, seconds: float = 0.4) -> float:
    """Capture a short clip and return RMS level (0.0 ~= muted / dead)."""
    frames = max(1, int(round(rate * seconds)))
    last_exc: Optional[Exception] = None
    for channels in (1, 2):
        kwargs: dict[str, Any] = {
            "samplerate": rate,
            "channels": channels,
            "dtype": "float32",
            "blocking": True,
        }
        if device_idx is not None:
            kwargs["device"] = device_idx
        try:
            audio = sd.rec(frames, **kwargs)
            samples = np.asarray(audio, dtype=np.float32)
            if samples.ndim > 1:
                samples = samples[:, 0]
            samples = samples.reshape(-1)
            if samples.size == 0:
                return 0.0
            return float(np.sqrt(np.mean(np.square(samples))))
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            continue
    if last_exc is not None:
        raise last_exc
    return 0.0


def ensure_live_mic(
    device_idx: Optional[int],
    rate: int,
    *,
    min_rms: float = 1e-4,
    allow_fallback: bool = True,
) -> tuple[Optional[int], int]:
    """Warn / optionally auto-fallback when the selected mic looks muted or silent."""
    global _mic_ambient_rms
    try:
        rms = probe_mic_rms(device_idx, rate)
    except Exception as exc:  # noqa: BLE001
        log("Audio", f"WARNING: mic RMS probe failed on [{device_idx}]: {exc}")
        rms = 0.0

    _mic_ambient_rms = float(rms)
    log("Audio", f"Mic RMS probe [{device_idx}]: {rms:.6f}")
    if rms < MIC_AMBIENT_DEAD_RMS:
        # Soft normalization hint for Whisper / VAD on near-silent probes (e.g. 0.000015).
        log(
            "Audio",
            f"WARNING: mic [{device_idx}] ambient RMS is abnormally low ({rms:.6f}); "
            "enabling quiet-mic adaptive VAD floors + Whisper gain. "
            "Speak into the headset to verify the endpoint is live.",
        )
    if rms >= min_rms:
        return device_idx, rate

    # Keep intentional SteelSeries / --mic choices even if ambient RMS is low.
    keep_name = ""
    keep_api = ""
    try:
        if device_idx is not None:
            devices = sd.query_devices()
            hostapis = sd.query_hostapis()
            keep_name = str(devices[device_idx].get("name", ""))
            keep_api = str(hostapis[int(devices[device_idx]["hostapi"])]["name"]).lower()
    except Exception:
        keep_name = ""
    if (not allow_fallback) or ("steelseries" in keep_name.lower()) or ("wdm-ks" in keep_api):
        # If we somehow landed on WDM-KS, force a SteelSeries MME/WASAPI rematch.
        if "wdm-ks" in keep_api:
            steel = find_steelseries_mic()
            if steel is not None:
                idx, new_rate, name = steel
                log(
                    "Audio",
                    f"Replacing unstable WDM-KS mic [{device_idx}] with [{idx}] {name}",
                )
                return idx, new_rate
        log(
            "Audio",
            f"WARNING: mic [{device_idx}] ambient RMS is low ({rms:.6f}); "
            "keeping selected device (speak into the headset to verify).",
        )
        return device_idx, rate

    log(
        "Audio",
        f"WARNING: mic [{device_idx}] looks dead/muted (RMS={rms:.6f}). "
        "Scanning for a live INPUT device...",
    )

    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    best: Optional[tuple[float, int, int]] = None  # rms, idx, rate
    for idx, dev in enumerate(devices):
        if int(dev.get("max_input_channels", 0)) < 1:
            continue
        name_l = str(dev.get("name", "")).lower()
        try:
            api = str(hostapis[int(dev["hostapi"])]["name"]).lower()
        except Exception:
            api = ""
        if "wdm-ks" in api:
            continue
        if any(tok in name_l for tok in ("mapper", "primary sound", "cable", "vb-audio")):
            continue
        if idx == device_idx:
            continue
        cand_rate = int(round(float(dev.get("default_samplerate", SAMPLE_RATE))))
        try:
            cand_rms = probe_mic_rms(idx, cand_rate, seconds=0.35)
        except Exception:
            continue
        log("Audio", f"  candidate [{idx}] RMS={cand_rms:.6f} {dev.get('name')}")
        if best is None or cand_rms > best[0]:
            best = (cand_rms, idx, cand_rate)

    if best is not None and best[0] >= min_rms:
        log(
            "Audio",
            f"Auto-fallback to live mic [{best[1]}] (RMS={best[0]:.6f})",
        )
        return best[1], best[2]

    log("Audio", "WARNING: no live mic found; keeping original selection.")
    return device_idx, rate


def flush_input_buffer(seconds: float = POST_ACK_FLUSH_SEC) -> None:
    """Discard pending mic frames so TTS echo / buffer tail does not enter VAD."""
    # Producer keeps running; just drop queued frames (~seconds worth).
    target = max(1, int(round(float(seconds) * 1000.0 / float(VAD_FRAME_MS))))
    dropped = 0
    for _ in range(target):
        try:
            _ = audio_buffer_queue.get_nowait()
            dropped += 1
        except queue.Empty:
            break
    log_debug(
        "Conversation",
        f"Flushed {dropped} mic frame(s) (~{dropped * VAD_FRAME_MS:.0f} ms) after ack.",
    )


def _run_with_timeout(
    fn: Any,
    *,
    timeout_s: float,
    label: str,
) -> tuple[bool, Any, BaseException | None]:
    """Run ``fn`` on a daemon thread; return (ok, result, error).

    If the call hangs past ``timeout_s``, returns ok=False without blocking the
    caller forever (the worker may still be stuck in PortAudio).
    """
    box: list[Any] = []
    err: list[BaseException] = []

    def _target() -> None:
        try:
            box.append(fn())
        except BaseException as exc:  # noqa: BLE001
            err.append(exc)

    worker = threading.Thread(target=_target, name=f"MicTimed:{label}", daemon=True)
    t0 = time.perf_counter()
    worker.start()
    worker.join(timeout=max(0.05, float(timeout_s)))
    if worker.is_alive():
        log(
            "Audio",
            f"ERROR: {label} hung after {timeout_s:.1f}s "
            f"(elapsed_ms={(time.perf_counter() - t0) * 1000.0:.0f}) — "
            "PortAudio device acquisition/read blocked",
        )
        return False, None, TimeoutError(f"{label} timed out after {timeout_s:.1f}s")
    if err:
        return False, None, err[0]
    return True, (box[0] if box else None), None


def _open_input_stream_with_timeout(
    stream_kwargs: dict[str, Any],
    *,
    timeout_s: float = MIC_STREAM_OPEN_TIMEOUT_S,
    label: str = "InputStream.open",
) -> Any | None:
    """Open+start an InputStream with a hang timeout. Returns stream or None."""

    def _open() -> Any:
        stream = sd.InputStream(**stream_kwargs)
        stream.start()
        return stream

    ok, stream, err = _run_with_timeout(_open, timeout_s=timeout_s, label=label)
    if not ok:
        if err is not None and not isinstance(err, TimeoutError):
            log("Audio", f"ERROR: {label} failed: {type(err).__name__}: {err}")
        return None
    return stream


def _read_input_stream_with_timeout(
    stream: Any,
    frames: int,
    *,
    timeout_s: float = MIC_STREAM_READ_TIMEOUT_S,
    label: str = "InputStream.read",
) -> tuple[np.ndarray | None, bool]:
    """Read frames with a hang timeout. Returns (chunk, overflowed) or (None, False)."""

    def _read() -> tuple[Any, bool]:
        data, overflowed = stream.read(frames)
        return data, bool(overflowed)

    ok, result, err = _run_with_timeout(_read, timeout_s=timeout_s, label=label)
    if not ok or result is None:
        if err is not None and not isinstance(err, TimeoutError):
            log_debug("Audio", f"{label} error: {err}")
        return None, False
    data, overflowed = result
    return np.asarray(data, dtype=np.float32), overflowed


def _close_input_stream(stream: Any, *, label: str = "InputStream") -> None:
    """Best-effort stop+close; never raises."""
    if stream is None:
        return
    try:
        stream.stop()
    except Exception:
        pass
    try:
        stream.close()
    except Exception:
        pass
    log_debug("Audio", f"{label} closed")


def flush_audio_buffer_queue() -> int:
    """Drop all pending mic frames (state transitions / barge-in / standby)."""
    n = 0
    while True:
        try:
            _ = audio_buffer_queue.get_nowait()
            n += 1
        except queue.Empty:
            break
    if n:
        log_debug("MicIngest", f"Flushed {n} stale audio frame(s)")
    return n


def get_mic_frame(*, timeout: float = 0.25) -> Optional[np.ndarray]:
    """Pull one 16 kHz mono float32 VAD frame from the ingest queue."""
    # Stage 7.3 — drop frames while Ghost Typist is typing (no Whisper path).
    try:
        from dana.memory.blackboard import is_typing

        if is_typing():
            flush_audio_buffer_queue()
            return None
    except Exception:  # noqa: BLE001
        pass
    try:
        frame = audio_buffer_queue.get(timeout=max(0.01, float(timeout)))
    except queue.Empty:
        return None
    arr = np.asarray(frame, dtype=np.float32).reshape(-1)
    if arr.size < VAD_FRAME_SAMPLES:
        pad = np.zeros(VAD_FRAME_SAMPLES, dtype=np.float32)
        pad[: arr.size] = arr
        return pad
    if arr.size > VAD_FRAME_SAMPLES:
        return arr[:VAD_FRAME_SAMPLES].copy()
    return arr


def request_mic_ingest_restart() -> None:
    """Ask the producer to reopen InputStream (device change / soft recovery)."""
    mic_ingest_restart.set()
    flush_audio_buffer_queue()


def ensure_mic_ingest_thread() -> None:
    """Start the continuous mic producer once (idempotent)."""
    global _mic_ingest_thread
    t = _mic_ingest_thread
    if t is not None and t.is_alive():
        return
    t = threading.Thread(target=mic_ingest_worker, name="MicIngest", daemon=True)
    _mic_ingest_thread = t
    t.start()
    log("Main", "Started thread: MicIngest")


def input_txt_ingest_worker() -> None:
    """Poll ``execution_jail/input.txt`` with silent empty back-off (0.75s).

    Only logs when non-empty content is successfully read and queued.
    Always ingests into the queue (chat mode no longer silently skips).
    When tasks are queued while in chat mode, escalate to developer and
    drop an empty ``.trigger_ask`` so the conversation loop drains the jail.
    """
    try:
        # Exact stdout marker required for ops / Phase-N verification.
        print("[Ingest] input.txt watcher started (silent when empty)", flush=True)
        log("Ingest", "input.txt watcher started (silent when empty)")
        try:
            ingest.ensure_input_txt()
        except Exception as exc:  # noqa: BLE001
            print(f"[Ingest] CRASH: {exc}", flush=True)
            log("Ingest", f"WARNING: ensure_input_txt failed: {exc}")

        while not stop_event.is_set():
            try:
                n = ingest.ingest_text_to_queue(empty_sleep=0.0)
                if n <= 0:
                    # Silent sleep — prevents continuous CPU polling churn.
                    stop_event.wait(
                        timeout=float(getattr(ingest, "EMPTY_POLL_SLEEP_S", 0.75))
                    )
                    continue
                log("Ingest", f"Queued {n} task(s) from input.txt")
                print(f"[Ingest] Queued {n} task(s) from input.txt", flush=True)
                # Ensure the conversation loop can drain: escalate process mode
                # without stealing the user's durable voice_session_mode.
                try:
                    if get_donna_mode() == "chat":
                        set_donna_mode("developer", as_voice=False)
                        print(
                            "[Ingest] Escalated chat -> developer for queued tasks "
                            "(voice mode preserved)",
                            flush=True,
                        )
                        log(
                            "Ingest",
                            "Escalated chat -> developer (as_voice=False) so task jail can drain",
                        )
                    # Empty trigger wakes Conversation when idle (no mic inject).
                    if get_ui_state() == "idle" and not is_recording.is_set():
                        trigger_path = Path(TRIGGER_FILE)
                        trigger_path.write_text("", encoding="utf-8")
                        print(
                            "[Ingest] Wrote empty .trigger_ask to wake agent",
                            flush=True,
                        )
                except Exception as wake_exc:  # noqa: BLE001
                    print(f"[Ingest] CRASH: {wake_exc}", flush=True)
                    log("Ingest", f"WARNING: post-queue wake failed: {wake_exc}")
            except Exception as exc:  # noqa: BLE001
                print(f"[Ingest] CRASH: {exc}", flush=True)
                log("Ingest", f"WARNING: ingest poll failed: {exc}")
                stop_event.wait(timeout=1.0)
        log("Ingest", "input.txt watcher stopped.")
        print("[Ingest] input.txt watcher stopped.", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[Ingest] CRASH: {exc}", flush=True)
        import traceback

        traceback.print_exc()
        log("Ingest", f"FATAL: input.txt watcher crashed: {exc}")


def mic_ingest_worker() -> None:
    """Continuous producer: open InputStream once, push 16 kHz VAD frames to queue."""
    _nt_hide_console_if_mp_child()
    stream: Any = None
    stream_channels = 1
    next_err_log = 0.0
    log("MicIngest", "Producer starting (single shared InputStream)...")

    def _close() -> None:
        nonlocal stream
        held = mic_lock.acquire(timeout=MIC_STREAM_OPEN_TIMEOUT_S)
        try:
            _close_input_stream(stream, label="MicIngest.InputStream")
            stream = None
        finally:
            if held:
                mic_lock.release()
        mic_ingest_ready.clear()
        wake_mic_released.set()

    def _open() -> bool:
        nonlocal stream, stream_channels
        _close()
        if AUDIO_INPUT_DEVICE is None:
            log("MicIngest", "ERROR: AUDIO_INPUT_DEVICE is None — producer idle")
            return False
        native_frame = max(
            1, int(round(VAD_FRAME_SAMPLES * AUDIO_INPUT_RATE / float(SAMPLE_RATE)))
        )
        last_exc: Optional[BaseException] = None
        for channels in (1, 2):
            kwargs: dict[str, Any] = {
                "device": int(AUDIO_INPUT_DEVICE),
                "samplerate": AUDIO_INPUT_RATE,
                "channels": channels,
                "dtype": "float32",
                "blocksize": native_frame,
            }
            held = mic_lock.acquire(timeout=MIC_STREAM_OPEN_TIMEOUT_S)
            if not held:
                last_exc = TimeoutError("mic_lock timeout")
                continue
            try:
                wake_mic_released.clear()
                candidate = _open_input_stream_with_timeout(
                    kwargs,
                    timeout_s=MIC_STREAM_OPEN_TIMEOUT_S,
                    label="MicIngest.InputStream.open",
                )
                if candidate is None:
                    wake_mic_released.set()
                    last_exc = TimeoutError("InputStream open timed out")
                    continue
                stream = candidate
                stream_channels = channels
                mic_ingest_ready.set()
                wake_mic_released.set()
                flush_audio_buffer_queue()
                log(
                    "MicIngest",
                    f"InputStream open device={AUDIO_INPUT_DEVICE} "
                    f"rate={AUDIO_INPUT_RATE} ch={channels} "
                    f"block={native_frame} (-> {VAD_FRAME_MS}ms @16k)",
                )
                print(
                    f"[Debug] MicIngest InputStream open device={AUDIO_INPUT_DEVICE} "
                    f"rate={AUDIO_INPUT_RATE} ch={channels} block={native_frame}",
                    flush=True,
                )
                return True
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                wake_mic_released.set()
            finally:
                mic_lock.release()
        log("MicIngest", f"ERROR: could not open mic stream ({last_exc})")
        return False

    if not _open():
        # Keep retrying so a late device bind can recover.
        while not stop_event.is_set():
            time.sleep(1.0)
            if _open():
                break

    while not stop_event.is_set():
        if mic_ingest_restart.is_set():
            mic_ingest_restart.clear()
            log("MicIngest", "Restart requested — reopening InputStream")
            if not _open():
                time.sleep(0.5)
            continue
        if stream is None:
            if not _open():
                time.sleep(0.5)
            continue
        native_frame = max(
            1, int(round(VAD_FRAME_SAMPLES * AUDIO_INPUT_RATE / float(SAMPLE_RATE)))
        )
        try:
            chunk, overflowed = _read_input_stream_with_timeout(
                stream,
                native_frame,
                timeout_s=MIC_STREAM_READ_TIMEOUT_S,
                label="MicIngest.InputStream.read",
            )
            if chunk is None:
                raise TimeoutError("MicIngest read timed out")
            if overflowed:
                log_debug("MicIngest", "input overflow")
            arr = np.asarray(chunk, dtype=np.float32)
            if arr.ndim > 1:
                arr = arr[:, 0]
            frame_16k = resample_to_16k(arr.reshape(-1), AUDIO_INPUT_RATE)
            if frame_16k.size < VAD_FRAME_SAMPLES:
                pad = np.zeros(VAD_FRAME_SAMPLES, dtype=np.float32)
                pad[: frame_16k.size] = frame_16k
                frame_16k = pad
            elif frame_16k.size > VAD_FRAME_SAMPLES:
                frame_16k = frame_16k[:VAD_FRAME_SAMPLES]
            # Stage 7.3 — do not enqueue mic frames while Ghost Typist types.
            try:
                from dana.memory.blackboard import is_typing

                if is_typing():
                    continue
            except Exception:  # noqa: BLE001
                pass
            try:
                audio_buffer_queue.put_nowait(frame_16k.copy())
            except queue.Full:
                try:
                    _ = audio_buffer_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    audio_buffer_queue.put_nowait(frame_16k.copy())
                except queue.Full:
                    pass
        except Exception as exc:  # noqa: BLE001
            now = time.monotonic()
            if now >= next_err_log:
                log("MicIngest", f"WARNING: read/reopen cycle ({exc})")
                next_err_log = now + 5.0
            _close()
            time.sleep(0.35)

    _close()
    flush_audio_buffer_queue()
    log("MicIngest", "Producer stopped.")


def adaptive_barge_in_rms() -> float:
    """Dynamic barge-in gate: absolute floor + multiple of ambient (filters TTS bleed)."""
    ambient = float(_mic_ambient_rms or 0.0)
    return max(BARGE_IN_RMS, ambient * BARGE_IN_AMBIENT_MULT, 0.08)


def record_utterance(
    max_seconds: Optional[float] = None,
    *,
    ignore_onset_ms: float = 0.0,
) -> tuple[np.ndarray, float, str, bool]:
    """
    Capture speech with Silero VAD; stop shortly after the user finishes talking.

    ``ignore_onset_ms`` skips early VAD hits (TTS echo / buffer tail after ack).

    Returns (audio_16k, rms_raw, stop_reason, speech_started).
    """
    from dana.audio.vad_consumer import (
        SILERO_SPEECH_THRESHOLD,
        is_speech_frame,
        reset_silero_states,
    )

    silence_needed = max(1, int(round(VAD_SILENCE_MS / VAD_FRAME_MS)))
    min_speech_frames = max(1, int(round(VAD_MIN_SPEECH_MS / VAD_FRAME_MS)))
    limit_s = float(VAD_MAX_SECONDS if max_seconds is None else max_seconds)
    max_frames = max(1, int(limit_s * 1000 / VAD_FRAME_MS))
    ignore_onset_frames = max(0, int(round(float(ignore_onset_ms) / VAD_FRAME_MS)))

    try:
        reset_silero_states()
    except Exception as exc:  # noqa: BLE001
        log("Conversation", f"WARNING: Silero VAD reset failed ({exc})")

    log(
        "Conversation",
        f"VAD recording (Silero queue consumer @16 kHz, frame={VAD_FRAME_MS}ms, "
        f"silence_cut={VAD_SILENCE_MS}ms, max={limit_s:.0f}s, "
        f"speech_prob>{SILERO_SPEECH_THRESHOLD})...",
    )

    collected: list[np.ndarray] = []
    pre_roll: list[np.ndarray] = []
    speech_started = False
    silence_frames = 0
    speech_frames = 0
    stop_reason = "max_timeout"
    t0 = time.perf_counter()
    # Streaming DC / rumble kill — state persists across frames for this utterance.
    dc_blocker = DcBlocker(r=DC_BLOCKER_R)

    def consume_frame(frame_idx: int, samples_16k: np.ndarray) -> bool:
        """Return True when recording should stop."""
        nonlocal speech_started, silence_frames, speech_frames, stop_reason

        # DC-block before Silero so offset cannot inflate probabilities.
        samples_16k = dc_blocker.apply(samples_16k)

        if samples_16k.size < VAD_FRAME_SAMPLES:
            padded = np.zeros(VAD_FRAME_SAMPLES, dtype=np.float32)
            padded[: samples_16k.size] = samples_16k
            samples_16k = padded
        elif samples_16k.size > VAD_FRAME_SAMPLES:
            samples_16k = samples_16k[:VAD_FRAME_SAMPLES]

        try:
            is_speech = bool(is_speech_frame(samples_16k, sample_rate=SAMPLE_RATE))
        except Exception as exc:  # noqa: BLE001
            log("Conversation", f"WARNING: Silero VAD frame error: {exc}")
            is_speech = False

        if not speech_started:
            pre_roll.append(samples_16k.copy())
            if len(pre_roll) > VAD_PRE_ROLL_FRAMES:
                pre_roll.pop(0)

            # Half-duplex: never barge-in from VAD while TTS is playing — speaker
            # echo must not cut Donna. Mic onset is ignored until playback ends.
            if tts_busy.is_set() or frame_idx < ignore_onset_frames:
                return False
            if is_speech:
                speech_started = True
                collected.extend(pre_roll)
                speech_frames = 1
                silence_frames = 0
                log_debug(
                    "Conversation",
                    f"Speech onset at {(frame_idx + 1) * VAD_FRAME_MS} ms",
                )
            return False

        collected.append(samples_16k.copy())
        if is_speech:
            speech_frames += 1
            silence_frames = 0
        else:
            silence_frames += 1
            if speech_frames >= min_speech_frames and silence_frames >= silence_needed:
                stop_reason = "silence_cutoff"
                return True
        return False

    # Producer keeps the InputStream; this consumer takes over queue draining.
    if not mic_ingest_ready.wait(timeout=2.0):
        log(
            "Conversation",
            "ERROR: MicIngest not ready — cannot start VAD consumer",
        )
        audio = np.zeros(VAD_FRAME_SAMPLES, dtype=np.float32)
        return audio, 0.0, "mic_ingest_not_ready", False

    vad_capture_active.set()
    log_debug("Conversation", "VAD consumer attached to audio_buffer_queue")
    try:
        for frame_idx in range(max_frames):
            if stop_event.is_set():
                stop_reason = "shutdown"
                break
            samples_16k = get_mic_frame(timeout=0.35)
            if samples_16k is None:
                if not mic_ingest_ready.is_set():
                    log(
                        "Conversation",
                        "ERROR: MicIngest stalled during VAD — aborting capture",
                    )
                    stop_reason = "ingest_stalled"
                    break
                continue
            if consume_frame(frame_idx, samples_16k):
                break
        else:
            stop_reason = "max_timeout"
    finally:
        vad_capture_active.clear()
        # Drop residual frames so wake-word standby does not see stale speech.
        flush_audio_buffer_queue()

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if not collected:
        audio = (
            np.concatenate(pre_roll).astype(np.float32)
            if pre_roll
            else np.zeros(VAD_FRAME_SAMPLES, dtype=np.float32)
        )
        log(
            "Conversation",
            f"VAD captured no speech onset (elapsed={elapsed_ms:.0f} ms, reason={stop_reason})",
        )
    else:
        audio = np.concatenate(collected).astype(np.float32)
        log(
            "Conversation",
            f"VAD stop reason={stop_reason}; elapsed={elapsed_ms:.0f} ms; "
            f"speech_frames={speech_frames}; silence_tail={silence_frames * VAD_FRAME_MS} ms; "
            f"samples={audio.size} ({audio.size / SAMPLE_RATE:.2f}s)",
        )

    peak = float(np.max(np.abs(audio)) + 1e-9)
    rms_raw = float(np.sqrt(np.mean(np.square(audio))) + 1e-9)
    audio = prepare_audio_for_whisper(audio, rms_raw=rms_raw)
    log(
        "Conversation",
        f"Captured audio peak={peak:.4f}, rms_raw={rms_raw:.4f}",
    )
    return audio, rms_raw, stop_reason, speech_started


def prepare_audio_for_whisper(
    audio: np.ndarray,
    *,
    rms_raw: float | None = None,
) -> np.ndarray:
    """DC-block, gain-normalize quiet mic buffers, then peak-limit for Whisper."""
    audio = remove_dc_offset(np.asarray(audio, dtype=np.float32))
    if audio.size == 0:
        return audio
    # Always measure RMS after DC removal (offset-inflated energy mis-scales gain).
    rms_raw = float(np.sqrt(np.mean(np.square(audio))) + 1e-9)
    if 1e-5 < rms_raw < WHISPER_GAIN_RMS_CEIL:
        if rms_raw < WHISPER_MIN_RMS_FOR_GAIN:
            # Near-silence: do not amplify — Whisper invents YouTube filler.
            log(
                "Conversation",
                f"Whisper gain skipped (rms_raw={rms_raw:.5f} below speech floor)",
            )
        else:
            gain = min(WHISPER_TARGET_RMS / rms_raw, WHISPER_MAX_GAIN)
            audio = audio * float(gain)
            log_debug("Conversation", f"Whisper gain x{gain:.1f} (rms_raw={rms_raw:.5f})")
    peak = float(np.max(np.abs(audio)) + 1e-9)
    if peak > 1e-4:
        audio = np.clip(audio / peak * 0.9, -1.0, 1.0)
    return audio.astype(np.float32, copy=False)


def _whisper_initial_prompt_text() -> str:
    """Deprecated: live STT no longer feeds initial_prompt into the decoder."""
    return ""


def _whisper_prompt_ids(processor, device) -> Optional[Any]:
    """Always ``None`` — do not condition Whisper on prior text / ticket logs."""
    return None


def transcribe_audio(
    audio: np.ndarray,
    whisper_processor,
    whisper_model,
    device,
    whisper_dtype,
) -> str:
    import torch

    raw = np.asarray(audio, dtype=np.float32)
    # Duration from captured samples (before gain) — used for rate hallucination.
    audio_duration_s = float(raw.size) / float(SAMPLE_RATE) if raw.size else 0.0
    audio = prepare_audio_for_whisper(raw)
    inputs = whisper_processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    )
    moved = {}
    for key, value in inputs.items():
        if hasattr(value, "to"):
            if value.is_floating_point():
                moved[key] = value.to(device=device, dtype=whisper_dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value

    # Language from settings.json (default english). task=transcribe keeps
    # speech in the chosen language rather than translating.
    from dana.settings import get_whisper_language

    whisper_lang = get_whisper_language()
    # Fresh VAD capture: never condition on previous text / ticket logs.
    _sanitize_whisper_generation_config(whisper_model)
    gen_kwargs = _whisper_generate_kwargs(
        max_new_tokens=128,
        language=whisper_lang,
        task=WHISPER_TASK,
        model=whisper_model,
    )
    with torch.no_grad():
        generated_ids = whisper_model.generate(
            **moved,
            **gen_kwargs,
        )
    text = whisper_processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    text = correct_known_stt_names(text)
    # Hard discard: physically impossible speaking rate → empty transcript.
    if is_whisper_rate_hallucination(text, audio_duration_s):
        log(
            "Conversation",
            "Dropped physically impossible transcript (rate limit exceeded).",
        )
        return ""
    return text


def execute_tool_call(tc: ToolCall) -> str:
    """Dispatch a validated ToolCall IR; returns an Observation string for ReAct."""
    global active_vision_tool, latest_frame, donna_profile

    broker = get_broker()
    # architect_new_tool: recover empty args from the live utterance before validate.
    if tc.tool_id == "architect_new_tool":
        from dataclasses import replace as _replace

        args = dict(tc.arguments or {})
        if not str(args.get("goal") or args.get("tool_description") or "").strip():
            recovered = str(tc.raw_text or "").strip()
            if recovered:
                args["goal"] = recovered
                tc = _replace(tc, arguments=args)
    # Validate when possible; intent-only vault triggers may lack args.
    try:
        tc = broker.validate_and_correct(tc)
    except ToolValidationError as exc:
        # Last-chance recovery for forge calls with empty structured args.
        if tc.tool_id == "architect_new_tool" and (tc.raw_text or "").strip():
            from dataclasses import replace as _replace

            args = dict(tc.arguments or {})
            args["goal"] = str(tc.raw_text).strip()
            tc = _replace(tc, arguments=args)
        else:
            return f"ERROR: invalid tool call ({exc})"
        try:
            tc = broker.validate_and_correct(tc)
        except ToolValidationError as exc2:
            return f"ERROR: invalid tool call ({exc2})"

    try:
        from dana.telemetry import note_tool_event

        note_tool_event(str(tc.tool_id))
    except Exception:  # noqa: BLE001
        pass

    def _handle_switch_vision(call: ToolCall) -> str:
        global active_vision_tool, latest_frame
        source = str(call.arguments.get("source") or "")
        target: Optional[Union[ScreenAgent, VideoAgent]] = None
        if source == "camera":
            target = camera_tool
        elif source == "screen":
            target = screen_tool
        else:
            return "ERROR: source must be screen or camera"

        with active_vision_lock:
            current = active_vision_tool
            if current is target:
                return f"OK: vision already on {source}"
            if current is camera_tool and target is screen_tool:
                camera_tool.release()
            active_vision_tool = target

        with spatial_memory_lock:
            spatial_memory.clear()
        with latest_dets_lock:
            latest_dets.clear()
        with latest_frame_lock:
            latest_frame = None

        SPATIAL_AGGREGATOR.set_vision_source(source)
        log("Router", f"Vision tool -> {source} via IR {call.tool_id}")
        return f"OK: switched vision to {source}"

    def _handle_analyze_visual(call: ToolCall) -> str:
        """Dispatch to JIT YOLO; observation is already ``[Vision Output] …``."""
        from dana.vision_tools import analyze_visual_context

        source = str(call.arguments.get("source") or "screen").strip().lower()
        if source not in {"screen", "webcam", "camera", "video"}:
            with active_vision_lock:
                source = (
                    "webcam" if active_vision_tool is camera_tool else "screen"
                )
        # Schema enum is screen|webcam; vision_tools also accepts camera.
        if source == "camera":
            source = "webcam"
        return analyze_visual_context(source=source)

    def _handle_ocr_with_region(call: ToolCall) -> str:
        """Florence-2 OCR grounding → text + ROI overlay highlight."""
        from dana.tools.visual_tools import ocr_with_region

        query = str(call.arguments.get("query") or "").strip()
        return ocr_with_region(query=query)

    def _handle_describe_spatial(call: ToolCall) -> str:
        # Prefer live JIT YOLO payload; keep SpatialIR as secondary context.
        from dana.vision_tools import analyze_visual_context

        with active_vision_lock:
            source = "webcam" if active_vision_tool is camera_tool else "screen"
        payload = analyze_visual_context(source=source)
        focus = str(call.arguments.get("focus") or "all")
        block = SPATIAL_AGGREGATOR.synthesize_prompt_block()
        hint = spatial_focus_hint(focus)
        return f"{payload} | SpatialIR={block} | {hint}"

    def _handle_read_vault(call: ToolCall) -> str:
        global donna_profile
        key = str(call.arguments.get("key") or "").strip()
        if not key:
            return "Error: Memory key not found in vault."
        # Pronoun / garbage keys must never crash the loop.
        if key.lower() in {"it", "this", "that", "them", "those", "these", "something"}:
            return "Error: Memory key not found in vault."
        try:
            if not vault_client.session_token:
                return "ERROR: vault session unavailable"
            value = vault_client.read_memory(key)
            donna_profile = dict(vault_client.profile)
            return f"OK: {key}={value!r}"
        except KeyError:
            # Graceful degradation — never raise into the agentic loop.
            return "Error: Memory key not found in vault."
        except Exception as exc:  # noqa: BLE001
            # Some RPC wrappers re-raise KeyError as generic Exception.
            msg = str(exc).lower()
            if "not found" in msg or "keyerror" in msg or "deprecated" in msg:
                return "Error: Memory key not found in vault."
            return f"ERROR: read_vault_memory failed: {exc}"

    def _handle_write_vault(call: ToolCall) -> str:
        global donna_profile
        key = str(call.arguments.get("key") or "").strip()
        value = call.arguments.get("value")
        if not key:
            return "ERROR: missing key"
        if value is None:
            return "ERROR: missing value"
        try:
            if not vault_client.session_token:
                return "ERROR: vault session unavailable"
            vault_client.write_memory(key, value)
            donna_profile = dict(vault_client.profile)
            # Keep settings.json in sync for place/timezone so local clock stays correct.
            nk = key.strip().lower().replace("-", "_").replace(" ", "_")
            if nk in (
                "timezone",
                "time_zone",
                "tz",
                "local_timezone",
                "home_city",
                "city",
                "hometown",
                "location_city",
                "home_region",
                "region",
                "state",
                "province",
                "home_state",
            ):
                try:
                    from dana.settings import update_place_settings

                    kwargs: dict[str, str] = {}
                    if nk in ("timezone", "time_zone", "tz", "local_timezone"):
                        kwargs["timezone"] = str(value)
                    elif nk in ("home_city", "city", "hometown", "location_city"):
                        kwargs["home_city"] = str(value)
                    else:
                        kwargs["home_region"] = str(value)
                    update_place_settings(**kwargs)
                except Exception:
                    pass
            report = vault_client.last_consolidation
            if report.get("skipped") and report.get("pruned_transient"):
                return f"OK: skipped transient key '{key}' (not persisted)"
            overridden = report.get("overridden") or []
            if overridden:
                return f"OK: saved {key}={value!r} (overrode {overridden!r})"
            return f"OK: saved {key}={value!r}"
        except Exception as exc:  # noqa: BLE001
            return f"ERROR: write_vault_memory failed: {exc}"

    def _handle_inject_keystrokes(call: ToolCall) -> str:
        from dana.os_automation import inject_keystrokes

        text = call.arguments.get("text")
        if text is None:
            return "ERROR: missing text"
        result = inject_keystrokes(str(text))
        if not result.get("ok"):
            return f"ERROR: inject_keystrokes blocked/failed: {result.get('error')}"
        mode = "dry_run" if result.get("dry_run") else "typed"
        return (
            f"OK: inject_keystrokes {mode} chars={result.get('chars_typed', 0)} "
            f"stripped={result.get('stripped_controls', 0)}"
        )

    def _handle_read_clipboard(call: ToolCall) -> str:
        from dana.os_automation import read_clipboard_context

        result = read_clipboard_context()
        if not result.get("ok"):
            return f"ERROR: read_clipboard_context failed: {result.get('error')}"
        if result.get("empty"):
            return "OK: clipboard empty or non-text"
        text = result.get("text") or ""
        trunc = " truncated=true" if result.get("truncated") else ""
        return f"OK: clipboard chars={len(text)}{trunc} text={text!r}"

    def _handle_shell_execute(call: ToolCall) -> str:
        from dana.tools.system_repl import shell_execute

        command = call.arguments.get("command")
        if command is None or not str(command).strip():
            return "ERROR: missing command"
        return shell_execute(str(command))

    def _handle_file_editor(call: ToolCall) -> str:
        from dana.tools.system_repl import file_editor

        action = str(call.arguments.get("action") or "").strip()
        filepath = call.arguments.get("filepath")
        if filepath is None or not str(filepath).strip():
            return "ERROR: missing filepath"
        content = call.arguments.get("content")
        content_s = None if content is None else str(content)
        return file_editor(action, str(filepath), content_s)

    def _handle_python_repl(call: ToolCall) -> str:
        from dana.tools.system_repl import python_repl

        code = call.arguments.get("code")
        if code is None or not str(code).strip():
            return "ERROR: missing code"
        return python_repl(str(code))

    def _handle_run_terminal(call: ToolCall) -> str:
        from dana.os_automation import run_terminal_command

        command = call.arguments.get("command")
        if command is None or not str(command).strip():
            return "ERROR: missing command"
        result = run_terminal_command(str(command))
        if str(result).upper().startswith("ERROR"):
            return str(result)
        return f"OK: run_terminal_command output=\n{result}"

    def _handle_flush_memory(call: ToolCall) -> str:
        cleared = flush_conversation_memory(reason="tool_flush_memory")
        return f"OK: Memory flushed successfully. Cleared {cleared} short-term messages."

    def _handle_publish_tool_to_general(call: ToolCall) -> str:
        from dana.tools.promotion import publish_tool_to_general_impl

        tool_name = call.arguments.get("tool_name")
        if tool_name is None or not str(tool_name).strip():
            # Recover from utterance: "promote tool X" / "publish tool_name to general"
            raw = str(call.raw_text or "")
            m = re.search(
                r"(?:promote|publish)\s+(?:tool\s+)?[`'\"]?([A-Za-z_][\w]*)[`'\"]?",
                raw,
                flags=re.I,
            )
            if m:
                tool_name = m.group(1)
            else:
                return "ERROR: missing tool_name for publish_tool_to_general"
        return publish_tool_to_general_impl(str(tool_name).strip())

    def _handle_open_application(call: ToolCall) -> str:
        from dana.os_automation import open_application

        app_name = call.arguments.get("app_name")
        if app_name is None or not str(app_name).strip():
            return "ERROR: Unknown application (empty)."
        return open_application(str(app_name))

    def _handle_read_local_file(call: ToolCall) -> str:
        from dana.os_automation import read_local_file

        filepath = call.arguments.get("filepath")
        if filepath is None or not str(filepath).strip():
            return "ERROR: missing filepath"
        result = read_local_file(str(filepath))
        if str(result).upper().startswith("ERROR"):
            return str(result)
        return f"OK: read_local_file path={str(filepath)!r}\n{result}"

    def _handle_architect(call: ToolCall) -> str:
        """Tool Forge entry — accept goal/tool_description; never crash on empty args."""
        from dana.settings import is_dynamic_tool_synthesis_enabled, synthesis_locked_message
        from dana.tools.broker import reload_broker_registry
        from dana.logging import log_exception

        try:
            if not is_dynamic_tool_synthesis_enabled():
                return (
                    "LOCKED: dynamic_tool_synthesis_disabled | "
                    + synthesis_locked_message(call.source_lang or "en")
                )

            goal = str(
                call.arguments.get("goal")
                or call.arguments.get("tool_description")
                or ""
            ).strip()
            # Empty args from broker/LLM: pull the raw user utterance.
            if not goal:
                goal = str(call.raw_text or "").strip()
            tool_name = str(call.arguments.get("tool_name") or "").strip()
            python_code = call.arguments.get("python_code")

            if not goal and (
                python_code is None or not str(python_code).strip()
            ):
                return (
                    "ERROR: architect_new_tool missing goal — pass the user's "
                    "exact request as goal=..."
                )

            # Prefer Tool Forge (Coder → AST → Security → Hot-Load) when no
            # pre-written source is supplied. Batch utterances forge N tools.
            if python_code is None or not str(python_code).strip():
                from dana.swarm.multi_forge import looks_like_multi_forge, run_batch_tool_forge
                from dana.swarm.tool_forge_graph import route_tool_not_found

                if looks_like_multi_forge(goal):
                    forge = run_batch_tool_forge(goal, missing_tool=tool_name or "")
                    loaded_list = list(forge.get("loaded_tools") or [])
                    if forge.get("status") in ("loaded", "partial") and loaded_list:
                        try:
                            reload_broker_registry()
                        except Exception:  # noqa: BLE001
                            pass
                        try:
                            from dana.telemetry import note_tool_event

                            for name in loaded_list:
                                note_tool_event(f"forge:{name}")
                        except Exception:  # noqa: BLE001
                            pass
                        return (
                            f"OK: Tool Forge batch status={forge.get('status')} "
                            f"loaded={loaded_list}. {forge.get('feedback') or ''}"
                        )
                else:
                    forge = route_tool_not_found(
                        goal,
                        missing_tool=tool_name or "",
                    )
                    if forge.get("status") == "loaded" and forge.get("loaded_tool"):
                        try:
                            reload_broker_registry()
                        except Exception:  # noqa: BLE001
                            pass
                        loaded = forge["loaded_tool"]
                        try:
                            from dana.telemetry import note_tool_event

                            note_tool_event(f"forge:{loaded}")
                        except Exception:  # noqa: BLE001
                            pass
                        return (
                            f"OK: Tool Forge forged and hot-loaded `{loaded}`. "
                            f"{forge.get('feedback') or ''}"
                        )
                # Terminal Failure (AST/security/coder) → Autonomous Bug Tracker.
                err_obs = (
                    f"ERROR: Tool Forge status={forge.get('status')}: "
                    f"{forge.get('feedback') or forge.get('lint_errors')}"
                )
                try:
                    from dana.bug_tracker import log_bug_to_tracker

                    log_bug_to_tracker(
                        err_obs,
                        context=(
                            f"goal={goal}\n"
                            f"missing_tool={tool_name or ''}\n"
                            f"forge_status={forge.get('status')}\n"
                            f"lint={forge.get('lint_errors') or ''}"
                        ),
                        status="PENDING",
                        source="architect_new_tool_terminal_failure",
                        user_query=goal,
                    )
                except Exception:  # noqa: BLE001
                    pass
                return err_obs

            # Legacy path: caller supplied python_code directly.
            from dana_security import architect_new_tool

            if not tool_name:
                tool_name = "forged_tool"
            result = architect_new_tool(tool_name, str(python_code))
            if not result.get("ok"):
                err = f"ERROR: architect_new_tool failed: {result.get('error')}"
                try:
                    from dana.bug_tracker import log_bug_to_tracker

                    log_bug_to_tracker(
                        err,
                        context=f"tool_name={tool_name}\ngoal={goal}",
                        status="PENDING",
                        source="architect_sandbox_failure",
                    )
                except Exception:  # noqa: BLE001
                    pass
                return err
            reload_broker_registry()
            return (
                f"OK: registered tool={result.get('tool_name')} "
                f"test_result={result.get('test_result')!r} path={result.get('path')}"
            )
        except Exception as exc:  # noqa: BLE001
            log_exception(
                "Architect",
                "architect_new_tool execution failed",
                exc=exc,
            )
            try:
                from dana.bug_tracker import log_bug_to_tracker

                log_bug_to_tracker(
                    f"architect_new_tool crashed: {exc}",
                    context=str(call.raw_text or call.arguments or "")[:2000],
                    status="PENDING",
                    source="architect_exception",
                )
            except Exception:  # noqa: BLE001
                pass
            return f"ERROR: architect_new_tool crashed: {exc}"

    def _handle_list_todo_basket(call: ToolCall) -> str:
        from dana.bug_tracker import list_todo_basket

        return list_todo_basket()

    def _handle_capture_and_analyze_screen(call: ToolCall) -> str:
        from dana.tools.os_control import capture_and_analyze_screen

        prompt = str(
            call.arguments.get("prompt")
            or call.arguments.get("query")
            or call.raw_text
            or ""
        ).strip()
        return capture_and_analyze_screen(prompt=prompt)

    def _handle_execute_os_keystrokes(call: ToolCall) -> str:
        from dana.tools.os_control import execute_os_keystrokes

        text = str(call.arguments.get("text") or "").strip()
        hotkey = str(call.arguments.get("hotkey") or "").strip()
        return execute_os_keystrokes(text, hotkey=hotkey)

    def _handle_type_stealth_text(call: ToolCall) -> str:
        from dana.operators.ghost_typist import type_stealth_text

        text = str(call.arguments.get("text") or "")
        hotkey = str(call.arguments.get("hotkey") or "f9")
        wait_raw = call.arguments.get("wait_hotkey", True)
        if isinstance(wait_raw, str):
            wait_hotkey = wait_raw.strip().lower() not in {"0", "false", "no", "off"}
        else:
            wait_hotkey = bool(wait_raw)
        return type_stealth_text(text, wait_hotkey=wait_hotkey, hotkey=hotkey)

    def _handle_evaluate_slide_and_type(call: ToolCall) -> str:
        from dana.tools.slide_review import evaluate_slide_and_type

        rule = str(
            call.arguments.get("rule")
            or call.arguments.get("query")
            or call.raw_text
            or ""
        ).strip()
        delay_raw = call.arguments.get("focus_delay_sec")
        delay: float | None
        try:
            delay = float(delay_raw) if delay_raw is not None else 1.5
        except (TypeError, ValueError):
            delay = 1.5
        return evaluate_slide_and_type(rule=rule, focus_delay_sec=delay)

    def _handle_delegate_to_cursor(call: ToolCall) -> str:
        from dana.tools.cursor_handoff import handle_tool_call

        return handle_tool_call(call)

    def _handle_dispatch_titan_repair(call: ToolCall) -> str:
        from dana.tools.swarm_dispatcher import dispatch_titan_repair

        query = str(
            call.arguments.get("query")
            or call.arguments.get("goal")
            or call.raw_text
            or ""
        ).strip()
        return dispatch_titan_repair(query)

    def _handle_read_architecture(call: ToolCall) -> str:
        from dana.architecture import read_system_architecture

        try:
            payload = read_system_architecture()
        except Exception as exc:  # noqa: BLE001
            return f"ERROR: read_system_architecture failed: {exc}"
        # Compact observation for the LLM (full ARCHITECTURE.md + schema summary).
        arch = payload.get("architecture_md") or ""
        schema = payload.get("tools_schema_summary_text") or ""
        note = payload.get("note") or ""
        return (
            f"OK: architecture_chars={len(arch)} tools={payload.get('tools_schema_summary', {}).get('tool_count')}\n"
            f"NOTE: {note}\n"
            f"--- ARCHITECTURE.md ---\n{arch}\n"
            f"--- TOOLS SCHEMA SUMMARY ---\n{schema}"
        )

    def _handle_web_search(call: ToolCall) -> str:
        from dana.web_search import format_search_observation, web_search

        query = str(call.arguments.get("query") or "").strip()
        if not query:
            return "ERROR: missing query"
        try:
            payload = web_search(query)
        except Exception as exc:  # noqa: BLE001
            return f"ERROR: web_search failed: {exc}"
        return format_search_observation(payload)

    def _handle_dispatch_research_swarm(call: ToolCall) -> str:
        from dana.tools.swarm_dispatcher import dispatch_research_swarm

        topic = call.arguments.get("topic")
        if topic is None or not str(topic).strip():
            topic = call.arguments.get("query")
        if topic is None or not str(topic).strip():
            return "ERROR: missing topic"

        def _speak_when_done(_topic: str, summary: str) -> None:
            # enqueue_speech is thread-safe (queue + lock) in this process.
            enqueue_speech(
                f"My research is complete. Here is what I found: {summary}"
            )

        return dispatch_research_swarm(
            str(topic).strip(),
            on_complete=_speak_when_done,
        )

    def _handle_dispatch_jason_supervisor(call: ToolCall) -> str:
        from dana.swarm.jason_supervisor_graph import dispatch_jason_supervisor_impl

        query = call.arguments.get("query")
        if query is None or not str(query).strip():
            query = call.arguments.get("task")
        if query is None or not str(query).strip():
            query = call.raw_text
        if query is None or not str(query).strip():
            return "ERROR: missing query"
        return dispatch_jason_supervisor_impl(str(query).strip())

    def _handle_dispatch_watchdog(call: ToolCall) -> str:
        from dana.tools.langchain_tools import dispatch_watchdog_impl

        task = call.arguments.get("task")
        if task is None or not str(task).strip():
            task = call.arguments.get("query")
        if task is None or not str(task).strip():
            return "ERROR: missing task"
        return dispatch_watchdog_impl(
            str(task).strip(),
            tts_callback=enqueue_speech,
            vault_client=vault_client if vault_client.session_token else None,
        )

    def _handle_kill_watchdog(call: ToolCall) -> str:
        from dana.tools.langchain_tools import kill_watchdog_impl

        task_id = call.arguments.get("task_id")
        if task_id is None or not str(task_id).strip():
            task_id = call.arguments.get("id")
        if task_id is None or not str(task_id).strip():
            return "ERROR: missing task_id"
        return kill_watchdog_impl(str(task_id).strip())

    def _handle_save_script_to_library(call: ToolCall) -> str:
        from dana.tools.langchain_tools import save_script_to_library_impl

        script_name = call.arguments.get("script_name")
        code = call.arguments.get("code")
        if script_name is None or not str(script_name).strip():
            return "ERROR: missing script_name"
        if code is None or not str(code).strip():
            return "ERROR: missing code"
        return save_script_to_library_impl(str(script_name), str(code))

    def _handle_draft_cursor_prompt(call: ToolCall) -> str:
        """Production path: append PENDING ticket to dana_security/patch_ledger.md."""
        from dana.tools.general.draft_cursor_prompt import draft_cursor_prompt

        return draft_cursor_prompt(
            objective=str(call.arguments.get("objective") or ""),
            context=str(call.arguments.get("context") or ""),
        )

    def _handle_dynamic(call: ToolCall) -> str:
        from dana_security import execute_dynamic_tool

        text = str(call.arguments.get("text") or "")
        sand = execute_dynamic_tool(call.tool_id, text)
        if not sand.ok:
            return f"ERROR: dynamic tool {call.tool_id} failed: {sand.error}"
        return f"OK: {call.tool_id} result={sand.result!r}"

    handlers = {
        "switch_vision_source": _handle_switch_vision,
        "analyze_visual_context": _handle_analyze_visual,
        "ocr_with_region": _handle_ocr_with_region,
        "describe_spatial_scene": _handle_describe_spatial,
        "read_vault_memory": _handle_read_vault,
        "write_vault_memory": _handle_write_vault,
        "inject_keystrokes": _handle_inject_keystrokes,
        "read_clipboard_context": _handle_read_clipboard,
        "run_terminal_command": _handle_run_terminal,
        "shell_execute": _handle_shell_execute,
        "file_editor": _handle_file_editor,
        "python_repl": _handle_python_repl,
        "flush_memory": _handle_flush_memory,
        "publish_tool_to_general": _handle_publish_tool_to_general,
        "open_application": _handle_open_application,
        "read_local_file": _handle_read_local_file,
        "architect_new_tool": _handle_architect,
        "list_todo_basket": _handle_list_todo_basket,
        "capture_and_analyze_screen": _handle_capture_and_analyze_screen,
        "execute_os_keystrokes": _handle_execute_os_keystrokes,
        "type_stealth_text": _handle_type_stealth_text,
        "evaluate_slide_and_type": _handle_evaluate_slide_and_type,
        "delegate_to_cursor": _handle_delegate_to_cursor,
        "dispatch_titan_repair": _handle_dispatch_titan_repair,
        "read_system_architecture": _handle_read_architecture,
        "web_search": _handle_web_search,
        "dispatch_research_swarm": _handle_dispatch_research_swarm,
        "dispatch_jason_supervisor": _handle_dispatch_jason_supervisor,
        "dispatch_watchdog": _handle_dispatch_watchdog,
        "kill_watchdog": _handle_kill_watchdog,
        "save_script_to_library": _handle_save_script_to_library,
        "draft_cursor_prompt": _handle_draft_cursor_prompt,
        "__dynamic__": _handle_dynamic,
    }
    try:
        return str(broker.dispatch(tc, handlers))
    except ToolValidationError as exc:
        return f"ERROR: dispatch failed ({exc})"


def tool_router(whisper_text: str) -> tuple[str, Optional[ToolCall]]:
    """Fast-path bilingual IR router for immediate side effects (vision switch).

    Returns ``(possibly_corrected_text, deferred_tool_or_None)``.
    Bound deferred tools are forced into the agentic loop; unbound tools
    (e.g. describe_spatial_scene) only inject a hard visual-context constraint.
    """
    # STT vocabulary middleware (Notepad phonetic repairs, name fixes, …).
    whisper_text = correct_known_stt_names(whisper_text or "")
    broker = get_broker()
    call: Optional[ToolCall] = None
    try:
        call = broker.parse_utterance(whisper_text)
    except ToolValidationError as exc:
        log("Router", f"Tool IR validation failed ({exc}); ignoring tool intent.")
        return whisper_text, None

    if call is None:
        return whisper_text, None

    if call.tool_id == "switch_vision_source":
        obs = execute_tool_call(call)
        log("Router", f"Fast-path switch: {obs}")
        if obs.startswith("OK: switched"):
            ack = (
                "    ."
                if call.arguments.get("source") == "camera"
                and call.source_lang in ("fa", "mixed")
                else (
                    "     ."
                    if call.arguments.get("source") == "screen"
                    and call.source_lang in ("fa", "mixed")
                    else (
                        "Switching to camera feed."
                        if call.arguments.get("source") == "camera"
                        else "Switching to screen feed."
                    )
                )
            )
            if "already" not in obs:
                enqueue_speech(ack)
                wait_for_speech_idle(timeout=5.0)
        return whisper_text, None

    from dana.tools.langchain_tools import _UNBOUND_TOOL_IDS

    log(
        "Router",
        f"Deferred to agentic loop: {call.tool_id} args={call.arguments} "
        f"(lang={call.source_lang})",
    )
    SPATIAL_AGGREGATOR.update_transcript(user=whisper_text)

    # Do not ack unbound tools — that promised a look the loop cannot perform.
    if call.tool_id not in _UNBOUND_TOOL_IDS:
        try:
            from dana.settings import resolve_reply_lang

            speak_tool_working_ack(call, resolve_reply_lang(whisper_text))
        except Exception as exc:  # noqa: BLE001
            log("Router", f"WARNING: tool working ack failed ({exc})")
    else:
        log(
            "Router",
            f"Skipping working ack for unbound tool `{call.tool_id}` "
            "(agentic loop will answer from visual context).",
        )
    return whisper_text, call


def ask_ollama_messages(
    messages: list[dict[str, str]],
    model: str = OLLAMA_MODEL,
    *,
    num_predict: Optional[int] = None,
) -> str:
    """Isolated Ollama chat call (no conversation_history mutation) for ReAct steps."""
    from dana.agentic import OLLAMA_UNREACHABLE_SPEECH

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        # Hard caps for 8GB VRAM: shorter KV cache + bounded generation.
        "options": {
            "num_ctx": 8192,
            "num_predict": 1024 if num_predict is None else int(num_predict),
        },
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT_SEC)
        resp.raise_for_status()
        data = resp.json()
    except (requests.exceptions.ConnectionError, ConnectionError) as exc:
        raise ConnectionError(OLLAMA_UNREACHABLE_SPEECH) from exc
    except requests.exceptions.Timeout as exc:
        raise TimeoutError(
            f"{OLLAMA_UNREACHABLE_SPEECH} (timed out after {OLLAMA_TIMEOUT_SEC:.0f}s)"
        ) from exc
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(f"Ollama HTTP error: {exc}") from exc

    message = data.get("message") or {}
    content = str(message.get("content", "")).strip()
    if not content:
        raise RuntimeError(f"Ollama returned empty content: {data!r}")
    return content


def ask_ollama(
    system_prompt: str,
    user_text: str,
    model: str = OLLAMA_MODEL,
) -> str:
    """POST to local Ollama /api/chat using global sliding-window history.

    ``conversation_history[0]`` is always the system prompt (fresh YOLO tags).
    Indices 1.. keep at most the last 6 user/assistant messages (3 pairs).
    """
    global conversation_history

    def _rollback_user_turn() -> None:
        if (
            conversation_history
            and conversation_history[-1].get("role") == "user"
        ):
            conversation_history.pop()

    def _pin_system_and_trim(turns_extra: Optional[dict[str, str]] = None) -> list[dict[str, str]]:
        """Keep system at index 0; slide user/assistant window to last 6."""
        if conversation_history and conversation_history[0].get("role") == "system":
            conversation_history[0] = {"role": "system", "content": system_prompt}
        else:
            conversation_history.insert(0, {"role": "system", "content": system_prompt})

        if turns_extra is not None:
            conversation_history.append(turns_extra)

        system_msg = conversation_history[0]
        turns = [
            m
            for m in conversation_history[1:]
            if m.get("role") in ("user", "assistant")
        ]
        if len(turns) > HISTORY_MAX_MESSAGES:
            turns = turns[-HISTORY_MAX_MESSAGES:]
        conversation_history[:] = [system_msg] + turns
        return list(conversation_history)

    with conversation_history_lock:
        messages = _pin_system_and_trim({"role": "user", "content": user_text})

    try:
        content = ask_ollama_messages(messages, model=model)
    except Exception:
        with conversation_history_lock:
            _rollback_user_turn()
        raise

    with conversation_history_lock:
        _pin_system_and_trim({"role": "assistant", "content": content})

    return content


def commit_agentic_turn(system_prompt: str, user_text: str, assistant_text: str) -> None:
    """Pin final ReAct answer into the sliding conversation window (no internal TOOL noise)."""
    global conversation_history
    with conversation_history_lock:
        if conversation_history and conversation_history[0].get("role") == "system":
            conversation_history[0] = {"role": "system", "content": system_prompt}
        else:
            conversation_history.insert(0, {"role": "system", "content": system_prompt})
        conversation_history.append({"role": "user", "content": user_text})
        conversation_history.append({"role": "assistant", "content": assistant_text})
        system_msg = conversation_history[0]
        turns = [
            m
            for m in conversation_history[1:]
            if m.get("role") in ("user", "assistant")
        ]
        if len(turns) > HISTORY_MAX_MESSAGES:
            turns = turns[-HISTORY_MAX_MESSAGES:]
        conversation_history[:] = [system_msg] + turns


def answer_with_vlm(
    frame_bgr: np.ndarray,
    prompt_text: str,
    vlm_processor,
    vlm_model,
    device,
) -> str:
    """Legacy SmolVLM path — unused by the Ollama cascade (kept for optional vision)."""
    import torch

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    prompt = vlm_processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = vlm_processor(text=prompt, images=[image], return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = vlm_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.3,
        )
    prompt_len = inputs["input_ids"].shape[-1]
    new_tokens = generated_ids[:, prompt_len:]
    return vlm_processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


# ---------------------------------------------------------------------------
# Thread 4 - Conversational cascade (wake -> turns -> follow-up loop)
# ---------------------------------------------------------------------------

def conversation_worker(
    local_files_only: bool,
    device,
    dtype,
) -> None:
    _nt_hide_console_if_mp_child()
    # Dual-engine: skip SmolVLM to free VRAM; Ollama is the conversational brain.
    # YOLO is JIT via dana.tracker; Whisper loads in a background thread so
    # openWakeWord can arm while HF weights warm up.
    _ = dtype
    yolo_dev = yolo_device_arg(device)

    def _kick_whisper_after_wakeword() -> None:
        # Let openWakeWord finish constructing before the HF/torch import tax.
        for _ in range(600):  # ~30s
            if stop_event.is_set():
                return
            if _shared_wakeword_model is not None:
                break
            time.sleep(0.05)
        if stop_event.is_set():
            return
        start_whisper_background_load(local_files_only, device)

    threading.Thread(
        target=_kick_whisper_after_wakeword,
        name="WhisperLoadKick",
        daemon=True,
    ).start()
    log(
        "Conversation",
        "Whisper deferred until WakeWord model is ready (or 30s timeout).",
    )

    try:
        probe = requests.get("http://localhost:11434/api/tags", timeout=5.0)
        probe.raise_for_status()
        tags = probe.json()
        names = [m.get("name", "") for m in tags.get("models", [])]
        log("Conversation", f"Ollama reachable. Models: {names or '(none pulled)'}")
        if not any(
            OLLAMA_MODEL in n or n.startswith(OLLAMA_MODEL.split(":")[0]) for n in names
        ):
            log(
                "Conversation",
                f"WARNING: '{OLLAMA_MODEL}' not found locally. "
                f"Run: ollama pull {OLLAMA_MODEL}",
            )
    except Exception as exc:  # noqa: BLE001
        log(
            "Conversation",
            f"WARNING: Ollama not reachable yet ({exc}). "
            "Start Ollama and pull the model before asking questions.",
        )

    log(
        "Conversation",
        f"Donna is ready (brain={OLLAMA_MODEL}). Say 'Donna' to wake — then ask "
        f"follow-ups without the wake word (~{FOLLOWUP_VAD_MAX_SECONDS:.0f}s silence -> Standing by).",
    )

    def _warmup_llm() -> None:
        """Background 1-token ping so llama3.2 weights land in VRAM before first turn.

        Wake-word stays disarmed until ``ollama_ready`` is set (success or fail).
        """
        try:
            ask_ollama_messages(
                [{"role": "user", "content": "hi"}],
                num_predict=1,
            )
            log("Conversation", f"Ollama warm-up complete ({OLLAMA_MODEL}).")
        except Exception as exc:  # noqa: BLE001
            log("Conversation", f"WARNING: Ollama warm-up skipped ({exc})")
        finally:
            ollama_ready.set()
            log("Conversation", "Wake-word arming allowed (ollama_ready=True).")
            maybe_play_boot_ready_audio()

    ollama_ready.clear()
    threading.Thread(target=_warmup_llm, name="OllamaWarmup", daemon=True).start()
    log(
        "Conversation",
        f"Ollama warm-up started in background ({OLLAMA_MODEL}); "
        "wake-word remains disarmed until complete.",
    )

    def end_session_to_idle(message: Optional[str] = None) -> None:
        if message:
            log("Conversation", f'Session end -> "{message}"')
            try:
                emit_live_transcript("Donna", message)
            except Exception:  # noqa: BLE001
                pass
            enqueue_speech(message)
            wait_for_speech_idle(timeout=TTS_IDLE_WAIT_TIMEOUT)
        set_subtitle("")
        set_ui_state("idle")
        # Standby: drop leftover speech so wake-word does not false-trigger.
        flush_audio_buffer_queue()

    def run_brain_turn(
        whisper_text: str,
        t0: float,
        *,
        isolated: bool = False,
    ) -> bool:
        """YOLO eyes + Ollama brain + TTS for one user question.

        ``isolated=True`` skips conversation-history prior messages so batched
        task-queue commands cannot overflow the local LLM context window.
        Chat mode uses a no-tools lightweight Llama path; developer mode uses ReAct.
        """
        global latest_frame

        _tool_working_ack_sent.clear()

        # Clear chat memory fast-path — empties isolated rolling buffer only.
        if parse_clear_chat_memory(whisper_text or ""):
            before = chat_memory_size()
            clear_chat_memory()
            ack = CHAT_MEMORY_CLEARED_ACK
            log(
                "Conversation",
                f"Chat memory cleared ({before} turn(s)); ack={ack!r}",
            )
            log_conversation("User", whisper_text or "")
            log_conversation("Donna", ack)
            emit_live_transcript("User (Whisper)", whisper_text or "")
            emit_live_transcript("Donna", ack)
            enqueue_speech(ack)
            wait_for_speech_idle(timeout=8.0)
            set_subtitle("")
            return True

        # Stage 8.5 — Dictation Loop (keyword "dictate" or GUI Start latch).
        try:
            from dana.management.dictation import (
                DICTATION_ACK,
                handle_dictation,
                should_handle_dictation,
            )

            if should_handle_dictation(whisper_text or ""):
                log_conversation("User", whisper_text or "")
                emit_live_transcript("User (Whisper)", whisper_text or "")
                result = handle_dictation(whisper_text or "")
                sid = str((result.get("session") or {}).get("session_id") or "")
                ack = str(result.get("ack") or DICTATION_ACK)
                log(
                    "Dictation",
                    f"Logged session_id={sid} "
                    f"cmd={result.get('command_text')!r} "
                    f"visual_chars={result.get('visual_chars')}",
                )
                try:
                    emit_trace(
                        "Dictation",
                        "completed",
                        f"Dictation logged ({sid[:8]})",
                        mode=get_donna_mode(),
                    )
                except Exception:  # noqa: BLE001
                    pass
                try:
                    gui_ref = _gui_instance
                    if gui_ref is not None and hasattr(
                        gui_ref, "refresh_dictation_sessions"
                    ):
                        gui_ref.after(0, gui_ref.refresh_dictation_sessions)
                except Exception:  # noqa: BLE001
                    pass
                log_conversation("Donna", ack)
                emit_live_transcript("Donna", ack)
                enqueue_speech(ack, interruptible=False)
                wait_for_speech_idle(timeout=8.0)
                set_subtitle("")
                return True
        except Exception as exc:  # noqa: BLE001
            log("Dictation", f"WARNING: dictation handler failed ({exc})")

        # Mode switch fast-path — no LLM, no YOLO, no tools.
        switched = parse_mode_switch(whisper_text or "")
        if switched is not None:
            active = set_donna_mode(switched)
            ack = mode_switch_spoken_ack(active)
            log("Conversation", f"Mode switch -> {active} (ack={ack!r})")
            try:
                emit_trace(
                    "Mode",
                    "completed",
                    f"Mode switch → {active}",
                    mode=active,
                )
            except Exception:  # noqa: BLE001
                pass
            log_conversation("User", whisper_text or "")
            log_conversation("Donna", ack)
            emit_live_transcript("User (Whisper)", whisper_text or "")
            emit_live_transcript("Donna", ack)
            enqueue_speech(ack, interruptible=False)
            wait_for_speech_idle(timeout=8.0)
            set_subtitle("")
            return True

        use_chat = (not isolated) and get_donna_mode() == "chat"
        tool_force = requires_tool_graph(whisper_text or "")
        if use_chat and tool_force:
            # CRITICAL: never let system/file/code intents stay on lightweight chat.
            use_chat = False
            log(
                "Conversation",
                "Tool-graph escalation: system/file/code intent → ReAct/MoA "
                "(lightweight chat bypassed)",
            )
        routed_tool = None
        if not use_chat:
            whisper_text, routed_tool = tool_router(whisper_text)
        route_tag = (
            "chat"
            if use_chat
            else ("tool" if tool_force else "developer")
        )
        log(
            "Conversation",
            f"User said: \"{whisper_text}\" "
            f"[mode={route_tag}"
            f"{', isolated' if isolated else ''}]",
        )
        log_conversation("User", whisper_text)
        try:
            from dana.telemetry import log_router, log_voice_asr

            log_voice_asr(
                whisper_text or "",
                payload={"mode": route_tag, "isolated": bool(isolated)},
            )
            if tool_force and not use_chat:
                log_router(
                    "tool-graph escalation",
                    current_agent="ReAct_Agent",
                    active_intent="tool_graph",
                    payload={"mode": route_tag},
                )
        except Exception:  # noqa: BLE001
            pass
        SPATIAL_AGGREGATOR.update_transcript(user=whisper_text)

        yolo_labels: list[str] = []
        vision_log = ""
        if use_chat:
            # Stage 4.1 — Chat reads typed objects topic; never runs live YOLO/LLM vision.
            try:
                from dana.memory import read_visual_state

                vision_log = read_visual_state() or ""
            except Exception as exc:  # noqa: BLE001
                log(
                    "Conversation",
                    f"WARNING: read_visual_state failed ({exc})",
                )
                vision_log = ""
            try:
                from dana.middleware.sidekick_supervisor import (
                    format_degraded_chat_hint,
                )

                degraded = format_degraded_chat_hint()
                if degraded:
                    vision_log = (
                        f"{vision_log}\n{degraded}".strip()
                        if vision_log
                        else degraded
                    )
            except Exception:  # noqa: BLE001
                pass
        else:
            # Prefer a fresh frame from the active tool (important after a switch).
            with active_vision_lock:
                tool = active_vision_tool
            frame = None
            try:
                frame = tool.get_frame()
            except Exception as exc:  # noqa: BLE001
                log("Conversation", f"WARNING: active tool get_frame failed ({exc})")
            if frame is not None:
                with latest_frame_lock:
                    latest_frame = frame
            else:
                with latest_frame_lock:
                    frame = None if latest_frame is None else latest_frame.copy()
            if frame is None and not tool_force:
                log("Conversation", "No vision frame available; skipping turn.")
                return False

            if frame is not None:
                with latest_dets_lock:
                    dets = list(latest_dets)
                live_labels = [name for _, name, _ in dets]
                memory_labels = get_spatial_memory_labels()
                yolo_labels = list(dict.fromkeys(live_labels + memory_labels))

                if not yolo_labels:
                    try:
                        from dana.tracker import get_yolo_model

                        yolo = get_yolo_model(YOLO_WEIGHTS)
                        results = yolo.predict(
                            source=frame,
                            conf=YOLO_CONF,
                            device=yolo_dev,
                            verbose=False,
                        )
                        yolo_labels, dets = parse_yolo_results(results)
                        remember_spatial_labels(yolo_labels)
                        with latest_dets_lock:
                            latest_dets[:] = dets
                        SPATIAL_AGGREGATOR.update_from_dets(
                            dets, frame_shape=getattr(frame, "shape", None)
                        )
                    except Exception as exc:  # noqa: BLE001
                        log("Conversation", f"ERROR during YOLO stage: {exc}")
                        return False
                vision_log = format_vision_context_for_llm(yolo_labels)

        if use_chat:
            from dana.settings import resolve_reply_lang

            system_prompt = build_lightweight_chat_system_prompt(
                reply_lang=resolve_reply_lang(whisper_text),
                visual_context=vision_log or None,
            )
        else:
            system_prompt = build_donna_system_prompt(
                yolo_labels, user_text=whisper_text
            )
        log_debug(
            "Conversation",
            vision_log
            if vision_log
            else f"Visual Context: (none) raw=[{format_class_list(yolo_labels)}]",
        )
        prior_turns: list[dict[str, str]] = []
        hist_len = 0
        if use_chat:
            # Isolated chat buffer — never read ReAct conversation_history.
            hist_len = chat_memory_size()
            log_debug(
                "Conversation",
                f"Chat memory window: {hist_len}/{CHAT_MEMORY_WINDOW_K} turns",
            )
        elif isolated:
            log_debug(
                "Conversation",
                f"Memory window: 0/{HISTORY_MAX_MESSAGES} msgs (isolated queue task)",
            )
        else:
            with conversation_history_lock:
                # Count user/assistant turns only (system stays pinned at index 0).
                prior_turns = [
                    {"role": m["role"], "content": m["content"]}
                    for m in conversation_history
                    if m.get("role") in ("user", "assistant") and m.get("content")
                ]
                # Leave room for the new user turn inside the ReAct message window.
                if len(prior_turns) > HISTORY_MAX_MESSAGES:
                    prior_turns = prior_turns[-HISTORY_MAX_MESSAGES:]
                hist_len = len(prior_turns)
            log_debug(
                "Conversation",
                f"Memory window: {hist_len}/{HISTORY_MAX_MESSAGES} msgs",
            )

        set_ui_state("thinking")
        brain_t0 = time.perf_counter()
        try:
            from dana.agentic import (
                OLLAMA_UNREACHABLE_SPEECH,
                is_ollama_connection_error,
                ollama_service_reachable,
            )

            # Health check: fail closed with a spoken diagnosis (never silent).
            if not ollama_service_reachable():
                answer = OLLAMA_UNREACHABLE_SPEECH
                log("Conversation", f'Dānā: "{answer}"')
                log_conversation("Donna", answer)
                emit_live_transcript("Donna", answer)
                enqueue_speech(answer)
                wait_for_speech_idle(timeout=30.0)
                time.sleep(0.15)
                set_subtitle("")
                return True

            if use_chat:
                # Telemetry parity with agentic_react_graph agent/router nodes.
                _chat_trace_t0 = time.perf_counter()
                try:
                    from dana.schema import TraceEvent
                    from dana.ui.trace_bus import TraceEventBus

                    _bus = TraceEventBus.instance()
                    _bus.emit(
                        TraceEvent(
                            event_type="node_enter",
                            node="router",
                            message="Chat turn start",
                            mode="chat",
                            state_keys=("messages",),
                        )
                    )
                    _bus.emit(
                        TraceEvent(
                            event_type="status",
                            node="synthesis",
                            message="LLM synthesis streaming",
                            mode="chat",
                        )
                    )
                except Exception:  # noqa: BLE001
                    pass
                result = run_lightweight_chat(
                    user_text=whisper_text,
                    system_prompt=system_prompt,
                    model=OLLAMA_MODEL,
                    ask_fn=ask_ollama_messages,
                    visual_context=vision_log or None,
                    use_chat_memory=True,
                )
                answer = result.final_text
                try:
                    from dana.schema import TraceEvent
                    from dana.ui.trace_bus import TraceEventBus

                    TraceEventBus.instance().emit(
                        TraceEvent(
                            event_type="node_exit",
                            node="synthesis",
                            message="Chat complete",
                            mode="chat",
                            payload=(answer or "")[:800],
                            latency_ms=(time.perf_counter() - _chat_trace_t0)
                            * 1000.0,
                            state_keys=("final_raw", "messages"),
                        )
                    )
                except Exception:  # noqa: BLE001
                    pass
                log(
                    "Conversation",
                    f"Lightweight chat node (tools/MoA bypassed; "
                    f"chat_memory={chat_memory_size()})",
                )
            else:
                result = run_react_loop(
                    user_text=whisper_text,
                    system_prompt=system_prompt,
                    execute_fn=execute_tool_call,
                    max_iters=REACT_MAX_ITERS,
                    vault_client=vault_client if vault_client.session_token else None,
                    reflect_fn=ask_ollama_messages,
                    prior_messages=prior_turns,
                    on_tool_start=speak_tool_working_ack,
                    # Recency bias: Visual Context lands on the last user message,
                    # not high in the system prompt (8B attention).
                    visual_context=vision_log or None,
                    # LangChain ChatOllama + bind_tools (native tool calling).
                    model=OLLAMA_MODEL,
                    forced_tool=routed_tool,
                    tts_callback=enqueue_speech,
                )
                answer = result.final_text
                # Belt-and-suspenders TTS gate (also runs inside run_react_loop).
                try:
                    from dana.agentic import sanitize_spoken_reply
                    from dana.settings import resolve_reply_lang

                    # Keep the Ollama-down diagnosis intact for TTS.
                    if (answer or "").strip() != OLLAMA_UNREACHABLE_SPEECH:
                        answer = sanitize_spoken_reply(
                            answer or "",
                            reply_lang=resolve_reply_lang(whisper_text),
                            tool_trace=getattr(result, "tool_trace", None),
                        )
                except Exception:  # noqa: BLE001
                    pass
            # ReAct history only — chat turns stay in the isolated chat buffer.
            if not isolated and not use_chat:
                if (answer or "").strip() != OLLAMA_UNREACHABLE_SPEECH:
                    commit_agentic_turn(system_prompt, whisper_text, answer)
            if result.tool_trace:
                # Compact INFO tool ids; full sanitized observations only under DONNA_DEBUG.
                tool_ids = [
                    str(t.get("tool") or "?")
                    for t in result.tool_trace
                    if t.get("tool")
                ]
                log(
                    "Agentic",
                    f"{result.iterations} iter(s) lang={result.reply_lang} "
                    f"tools={tool_ids or '-'}",
                )
                log_debug(
                    "Agentic",
                    f"trace={sanitize_tool_trace(result.tool_trace)}",
                )
            if result.reflection:
                log_debug(
                    "Reflector",
                    f"{result.reflection_ms:.0f} ms "
                    f"rule={result.reflection.get('rule')!r} "
                    f"persisted={result.reflection.get('persisted')}",
                )
        except Exception as exc:  # noqa: BLE001
            log("Conversation", f"ERROR during agentic Ollama loop: {exc}")
            try:
                from dana.agentic import (
                    OLLAMA_UNREACHABLE_SPEECH,
                    is_ollama_connection_error,
                )

                if is_ollama_connection_error(exc):
                    enqueue_speech(OLLAMA_UNREACHABLE_SPEECH)
                    wait_for_speech_idle(timeout=30.0)
                    set_subtitle("")
                    return True
            except Exception:  # noqa: BLE001
                pass
            return False

        brain_ms = (time.perf_counter() - brain_t0) * 1000.0
        latency_ms = (time.perf_counter() - t0) * 1000.0
        log_debug(
            "Conversation",
            f"Ollama {brain_ms:.0f} ms | turn {latency_ms:.0f} ms",
        )
        log("Conversation", f'Dānā: "{answer}"')
        log_conversation("Donna", answer or "", extra=f"{latency_ms:.0f} ms")
        emit_live_transcript("Donna", answer)
        SPATIAL_AGGREGATOR.update_transcript(assistant=answer)

        # Prefer live astream TTS; skip duplicate final enqueue when already spoken.
        if not getattr(result, "tts_streamed", False):
            enqueue_speech(answer if answer else "I'm not sure.")
        elif not (answer or "").strip():
            enqueue_speech("I'm not sure.")
        wait_for_speech_idle(timeout=30.0)
        time.sleep(0.15)
        set_subtitle("")
        return True

    def drain_structured_task_queue() -> int:
        """Dispatch every pending ``task_queue.json`` command as an isolated ReAct turn.

        Returns the number of tasks processed (completed or failed). Never raises
        into the voice loop — broker isolates per-task exceptions.

        System job lane: when Chat has pending jail tasks, escalate process mode
        with ``as_voice=False`` so the user's conversational mode is restored.
        """
        try:
            from dana.tools.broker import dispatch_pending_tasks
            from dana.tools.task_queue import (
                ensure_execution_jail_queue,
                pending_count,
            )

            # Auto-ingest any free-form text dropped into input.txt before drain.
            # Empty files are silent (no log); the InputIngest watcher rate-limits polls.
            try:
                ingest.ingest_text_to_queue(empty_sleep=0.0)
            except Exception as exc:  # noqa: BLE001
                log("TaskQueue", f"WARNING: ingest_text_to_queue failed: {exc}")

            ensure_execution_jail_queue()
            n_pending = pending_count()
            if n_pending <= 0:
                return 0

            log(
                "TaskQueue",
                f"Draining {n_pending} pending task(s) from execution_jail/task_queue.json",
            )

            def _isolated_handler(command: str) -> None:
                preview = command if len(command) <= 160 else command[:157] + "..."
                log("TaskQueue", f'Dispatching isolated ReAct: "{preview}"')
                # Match voice-path logging so queue wakes show User said: …
                log("Conversation", f'User said: "{command}"')
                set_subtitle(f'User (Queue): "{command}"')
                emit_live_transcript("User (TaskQueue)", command)
                if is_standby_command(command):
                    emit_live_transcript("Donna", "Standing by.")
                    enqueue_speech("Standing by.", interruptible=False)
                    wait_for_speech_idle(timeout=8.0)
                    return
                if is_clear_context_command(command):
                    flush_conversation_memory(reason="task_queue")
                    reply = clear_context_spoken_reply(command)
                    emit_live_transcript("Donna", reply)
                    enqueue_speech(reply)
                    wait_for_speech_idle(timeout=8.0)
                    return
                if is_lockdown_command(command):
                    execute_lockdown_shutdown()
                    return
                if is_time_command(command):
                    reply = wall_clock_spoken_reply()
                    emit_live_transcript("Donna", reply)
                    enqueue_speech(reply)
                    wait_for_speech_idle(timeout=8.0)
                    return
                ok = run_brain_turn(command, time.perf_counter(), isolated=True)
                if not ok:
                    raise RuntimeError("isolated ReAct turn failed")

            # System job lane: escalate for ReAct jail without mutating voice mode.
            from dana.agentic import restore_voice_mode

            prior = get_donna_mode()
            escalated = False
            if prior == "chat":
                set_donna_mode("developer", as_voice=False)
                escalated = True
            try:
                results = dispatch_pending_tasks(_isolated_handler)
            finally:
                if escalated:
                    restored = restore_voice_mode()
                    log(
                        "TaskQueue",
                        f"Restored voice mode after jail drain -> {restored}",
                    )
            completed = sum(1 for r in results if r.get("status") == "completed")
            failed = sum(1 for r in results if r.get("status") == "failed")
            log(
                "TaskQueue",
                f"Queue drain finished: {completed} completed, {failed} failed "
                f"(of {len(results)})",
            )
            return len(results)
        except Exception as exc:  # noqa: BLE001
            log("TaskQueue", f"WARNING: queue drain aborted: {exc}")
            return 0

    while not stop_event.is_set():
        triggered = is_recording.wait(timeout=0.1)
        if not triggered:
            continue
        is_recording.clear()
        if not is_engine_engaged():
            # Standby: drop latched recording / file triggers until ENGAGE.
            log("Conversation", "Ignored session latch — engine STANDBY")
            continue

        # ---- Conversation session: initial wake turn + optional follow-ups ----
        follow_up = False
        log("Conversation", "Session started (wake / trigger).")

        while not stop_event.is_set():
            t0 = time.perf_counter()
            set_subtitle("")
            whisper_text: Optional[str] = None

            # Structured task queue (execution_jail/task_queue.json) — replaces
            # legacy flat input.txt so batched commands never share one prompt.
            # CRITICAL: text queue always beats Silero VAD / Whisper mic capture.
            queued_n = drain_structured_task_queue()
            if queued_n > 0:
                log(
                    "Conversation",
                    f"Text queue drained ({queued_n} task(s)); "
                    "bypassing VAD/Whisper audio loop",
                )
                # Do NOT enter follow-up mic listen after a queue-only wake.
                end_session_to_idle()
                break

            # Stage 8.10 — silent Dashboard text may arrive on any turn (incl. follow-up).
            injected, inject_source, inject_logged = pop_injected_question_ex()

            if injected:
                whisper_text = injected
                set_subtitle(f'User: "{whisper_text}"')
                log("Conversation", f'User said: "{whisper_text}"')
                log(
                    "Conversation",
                    f'Injected user question ({inject_source}): "{whisper_text}"',
                )
                if not inject_logged:
                    label = (
                        "User (Text)"
                        if inject_source == "text"
                        else "User (Whisper)"
                    )
                    emit_live_transcript(label, whisper_text)
                if is_standby_command(whisper_text):
                    log("Router", "Fast-path standby triggered")
                    end_session_to_idle("Standing by.")
                    break
                if is_clear_context_command(whisper_text):
                    log("Router", "Fast-path clear-context triggered")
                    log_conversation("User", whisper_text)
                    flush_conversation_memory(reason="voice_command")
                    reply = clear_context_spoken_reply(whisper_text)
                    log_conversation("Donna", reply)
                    emit_live_transcript("Donna", reply)
                    enqueue_speech(reply)
                    wait_for_speech_idle(timeout=8.0)
                    follow_up = True
                    set_ui_state("followup")
                    continue
                if is_lockdown_command(whisper_text):
                    log("Router", "Fast-path lockdown triggered")
                    execute_lockdown_shutdown()
                if is_time_command(whisper_text):
                    reply = wall_clock_spoken_reply()
                    log("Router", f"Fast-path wall-clock -> {reply!r}")
                    log_conversation("User", whisper_text)
                    log_conversation("Donna", reply)
                    end_session_to_idle(reply)
                    break
                if is_whisper_hallucination(whisper_text):
                    log(
                        "Conversation",
                        f"Hallucination/empty transcript (inject): \"{whisper_text}\"",
                    )
                    if is_silent_non_speech_transcript(whisper_text):
                        set_subtitle("")
                        set_ui_state("listening")
                        follow_up = True
                        continue
                    end_session_to_idle("I didn't catch that.")
                    break
            else:
                # Re-check text queue immediately before opening the mic — InputIngest
                # may have raced a .trigger_ask wake ahead of the first drain.
                pre_vad_queued = drain_structured_task_queue()
                if pre_vad_queued > 0:
                    log(
                        "Conversation",
                        f"Text queue drained ({pre_vad_queued} task(s)) "
                        "before VAD; bypassing audio loop",
                    )
                    end_session_to_idle()
                    break

                if follow_up:
                    # Follow-up: no wake word, no ack TTS.
                    log_debug("Conversation", "Follow-up: listening for next question...")
                    set_ui_state("followup")
                    flush_input_buffer(FOLLOWUP_FLUSH_SEC)
                    try:
                        audio, rms_raw, stop_reason, speech_started = record_utterance(
                            max_seconds=FOLLOWUP_VAD_MAX_SECONDS
                        )
                    except Exception as exc:  # noqa: BLE001
                        log("Conversation", f"ERROR recording follow-up audio: {exc}")
                        end_session_to_idle("Standing by.")
                        break

                    # Empty-room timeout: no speech → silent disarm (no "Standing by.").
                    if (not speech_started) and stop_reason in (
                        "max_timeout",
                        "silence_cutoff",
                    ):
                        log(
                            "Conversation",
                            f"Follow-up empty capture (reason={stop_reason}, "
                            f"rms_raw={rms_raw:.5f}) — silent disarm",
                        )
                        end_session_to_idle()
                        break
                    if not speech_started:
                        end_session_to_idle()
                        break
                else:
                    # Initial wake turn: non-blocking "Yes?", then grace so VAD does
                    # not eat speaker bleed (conversation worker only — not UI thread).
                    log("Conversation", 'Acknowledging wake -> "Yes?" (non-blocking TTS)')
                    enqueue_speech("Yes?", interruptible=False)
                    set_ui_state("listening")
                    log(
                        "Conversation",
                        f"Post-ack VAD grace {POST_ACK_VAD_GRACE_SEC:.2f}s "
                        f"(flush echo, max_timeout={VAD_MAX_SECONDS:.0f}s)",
                    )
                    time.sleep(POST_ACK_VAD_GRACE_SEC)
                    # Text may have arrived during the post-ack grace window.
                    grace_queued = drain_structured_task_queue()
                    if grace_queued > 0:
                        log(
                            "Conversation",
                            f"Text queue drained ({grace_queued} task(s)) "
                            "during post-ack grace; bypassing VAD/Whisper",
                        )
                        end_session_to_idle()
                        break
                    flush_input_buffer(POST_ACK_FLUSH_SEC)
                    try:
                        audio, rms_raw, stop_reason, speech_started = record_utterance(
                            max_seconds=VAD_MAX_SECONDS,
                            ignore_onset_ms=POST_ACK_IGNORE_ONSET_MS,
                        )
                    except Exception as exc:  # noqa: BLE001
                        log("Conversation", f"ERROR recording audio: {exc}")
                        end_session_to_idle()
                        break

                    # Empty-room timeout after wake: silent disarm (no TTS announce).
                    if (not speech_started) and stop_reason in (
                        "max_timeout",
                        "silence_cutoff",
                    ):
                        log(
                            "Conversation",
                            f"Wake empty capture (reason={stop_reason}, "
                            f"rms_raw={rms_raw:.5f}) — silent disarm",
                        )
                        end_session_to_idle()
                        break
                    if (not speech_started) or rms_raw < 5e-5:
                        # Quiet-mic miss: keep session open and invite another try.
                        log(
                            "Conversation",
                            "No VAD onset after wake — re-listening once "
                            f"(rms_raw={rms_raw:.5f}, reason={stop_reason})",
                        )
                        enqueue_speech("I'm here.", interruptible=False)
                        wait_for_speech_idle(timeout=8.0)
                        follow_up = True
                        set_ui_state("followup")
                        continue

                set_ui_state("transcribing")
                # Final queue drain immediately before Whisper GPU work.
                late_queued = drain_structured_task_queue()
                if late_queued > 0:
                    log(
                        "Conversation",
                        f"Text queue drained ({late_queued} task(s)) "
                        "before Whisper; discarding mic capture",
                    )
                    end_session_to_idle()
                    break
                try:
                    (
                        whisper_processor,
                        whisper_model,
                        whisper_device,
                        whisper_dtype,
                    ) = ensure_whisper_bundle()
                    whisper_text = transcribe_audio(
                        audio,
                        whisper_processor,
                        whisper_model,
                        whisper_device,
                        whisper_dtype,
                    )
                except Exception as exc:  # noqa: BLE001
                    log("Conversation", f"ERROR during Whisper STT: {exc}")
                    end_session_to_idle("I didn't catch that.")
                    break

                # Root-cause diagnostics for hush/noise transcripts (no hard reject here).
                peak = float(np.max(np.abs(audio))) if audio.size else 0.0
                audio_dur_s = float(audio.size) / float(SAMPLE_RATE) if audio.size else 0.0
                word_count = len((whisper_text or "").split())
                unique_n = len(set((whisper_text or "").split()))
                words_per_sec = (
                    (word_count / audio_dur_s) if audio_dur_s > 1e-3 else float("inf")
                )
                log_debug(
                    "Conversation",
                    f"STT debug: rms_raw={rms_raw:.5f} peak={peak:.5f} "
                    f"speech_started={speech_started} reason={stop_reason} "
                    f"tokens={word_count} unique={unique_n} "
                    f"secs={audio_dur_s:.2f} wps={words_per_sec:.2f}",
                )
                if word_count >= 8 and unique_n <= 2:
                    log(
                        "Conversation",
                        "WARNING: highly repetitive STT transcript — "
                        "likely low-SNR / hush (check mic gain / false wake / VAD).",
                    )
                # Belt-and-suspenders: discard even if transcribe_audio missed the drop.
                if is_whisper_rate_hallucination(whisper_text or "", audio_dur_s):
                    log(
                        "Conversation",
                        "Dropped physically impossible transcript (rate limit exceeded).",
                    )
                    whisper_text = ""
                    set_subtitle("")
                    set_ui_state("listening")
                    follow_up = True
                    continue

                if is_whisper_hallucination(
                    whisper_text, audio_duration_s=audio_dur_s
                ):
                    log(
                        "Conversation",
                        f"Hallucination/empty transcript: \"{whisper_text}\"",
                    )
                    dropped = whisper_text
                    whisper_text = ""
                    # Punctuation / ambient Whisper noise: silent re-listen (no LLM/TTS).
                    if is_silent_non_speech_transcript(dropped):
                        set_subtitle("")
                        set_ui_state("listening")
                        follow_up = True
                        continue
                    # Stay in-session and re-listen once — don't drop to idle
                    # before the user can retry (also restores "Let me check" path).
                    if not follow_up:
                        enqueue_speech("Sorry — say that again.", interruptible=False)
                        wait_for_speech_idle(timeout=8.0)
                        follow_up = True
                        set_ui_state("followup")
                        continue
                    end_session_to_idle("I didn't catch that.")
                    break

                # Empty after hard drop — never reach compiler / input.txt.
                if not (whisper_text or "").strip():
                    set_subtitle("")
                    set_ui_state("listening")
                    follow_up = True
                    continue

                set_subtitle(f'User: "{whisper_text}"')
                emit_live_transcript("User (Whisper)", whisper_text)
                if is_standby_command(whisper_text):
                    log("Router", "Fast-path standby triggered")
                    end_session_to_idle("Standing by.")
                    break
                if is_clear_context_command(whisper_text):
                    log("Router", "Fast-path clear-context triggered")
                    log_conversation("User", whisper_text)
                    flush_conversation_memory(reason="voice_command")
                    reply = clear_context_spoken_reply(whisper_text)
                    log_conversation("Donna", reply)
                    emit_live_transcript("Donna", reply)
                    enqueue_speech(reply)
                    wait_for_speech_idle(timeout=8.0)
                    follow_up = True
                    set_ui_state("followup")
                    continue
                if is_lockdown_command(whisper_text):
                    log("Router", "Fast-path lockdown triggered")
                    execute_lockdown_shutdown()
                if is_time_command(whisper_text):
                    reply = wall_clock_spoken_reply()
                    log("Router", f"Fast-path wall-clock -> {reply!r}")
                    log_conversation("User", whisper_text)
                    log_conversation("Donna", reply)
                    end_session_to_idle(reply)
                    break

                # Chat mode: never feed the task-queue / ReAct jail — lightweight chat only.
                if get_donna_mode() == "chat":
                    if not run_brain_turn(whisper_text, t0):
                        end_session_to_idle("I didn't catch that.")
                        break
                    follow_up = True
                    set_ui_state("followup")
                    continue

                # Developer mode Meta-Planner: compile → input.txt → ingest/drain ReAct.
                compile_and_append_voice_prompt(whisper_text)
                follow_up = True
                set_ui_state("followup")
                continue

            assert whisper_text is not None
            # Injected path: same system-command short-circuit before ReAct.
            if is_standby_command(whisper_text):
                log("Router", "Fast-path standby triggered")
                end_session_to_idle("Standing by.")
                break
            if is_clear_context_command(whisper_text):
                log("Router", "Fast-path clear-context triggered")
                log_conversation("User", whisper_text)
                flush_conversation_memory(reason="voice_command")
                reply = clear_context_spoken_reply(whisper_text)
                log_conversation("Donna", reply)
                emit_live_transcript("Donna", reply)
                enqueue_speech(reply)
                wait_for_speech_idle(timeout=8.0)
                follow_up = True
                set_ui_state("followup")
                continue
            if is_lockdown_command(whisper_text):
                log("Router", "Fast-path lockdown triggered")
                execute_lockdown_shutdown()
            if is_time_command(whisper_text):
                reply = wall_clock_spoken_reply()
                log("Router", f"Fast-path wall-clock -> {reply!r}")
                log_conversation("User", whisper_text)
                log_conversation("Donna", reply)
                end_session_to_idle(reply)
                break

            if not run_brain_turn(whisper_text, t0):
                end_session_to_idle("I didn't catch that.")
                break

            # Successful answer -> stay in session for follow-up (no wake word).
            follow_up = True
            set_ui_state("followup")
            log_debug(
                "Conversation",
                f"Entering follow-up mode "
                f"(silent timeout {FOLLOWUP_VAD_MAX_SECONDS:.0f}s).",
            )

    log("Conversation", "Stopped.")


# ---------------------------------------------------------------------------
# Thread 5 - Audio TTS (piper-tts -> sounddevice)
# ---------------------------------------------------------------------------

def contains_non_latin_script(text: str) -> bool:
    """Legacy helper retained for call-sites; always False (English-only release)."""
    _ = text
    return False


def piper_model_path_for_text(text: str) -> str:
    """Always route Piper to the English voice (public release)."""
    _ = text
    return PIPER_EN_ONNX


def get_piper_voice(model_path: str) -> PiperVoice:
    """Load (and cache) a PiperVoice for the given .onnx path."""
    voice = _piper_voice_cache.get(model_path)
    if voice is not None:
        return voice
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Piper model missing: {model_path}")
    log("Audio", f"Loading Piper voice: {os.path.basename(model_path)}")
    voice = PiperVoice.load(model_path)
    _piper_voice_cache[model_path] = voice
    return voice


def synthesize_to_file(voice: PiperVoice, text: str, path: str) -> bool:
    """Write Piper speech to a WAV path.

    Collects audio from ``voice.synthesize`` first so empty/failed TTS never
    opens a half-initialized ``wave`` writer (``# channels not specified``).

    Returns:
        True when a valid WAV was written; False when TTS produced no audio
        (caller should skip playback).
    """
    # Defaults used when the voice omits format metadata.
    channels = 1
    sampwidth = 2
    framerate = 22050
    try:
        cfg_rate = int(getattr(getattr(voice, "config", None), "sample_rate", 0) or 0)
        if cfg_rate > 0:
            framerate = cfg_rate
    except Exception:  # noqa: BLE001
        pass

    utterance = sanitize_text_for_tts(text or "")
    if not utterance:
        # Empty / markdown-only input — skip without warning spam.
        return False

    chunks: list[Any] = []
    piper_bytes = 0
    t_piper0 = time.perf_counter()
    syn_config = SynthesisConfig(length_scale=float(PIPER_LENGTH_SCALE))
    try:
        for chunk in voice.synthesize(utterance, syn_config=syn_config):
            if chunk is None:
                continue
            try:
                raw = chunk.audio_int16_bytes
            except Exception:  # noqa: BLE001
                raw = b""
            if not raw:
                continue
            piper_bytes += len(raw)
            chunks.append(chunk)
    except Exception as exc:  # noqa: BLE001
        log(
            "Audio",
            f"WARNING: TTS returned empty audio data, skipping synthesis ({exc})",
        )
        return False

    if not chunks:
        log("Audio", "WARNING: TTS returned empty audio data, skipping synthesis")
        return False

    log_debug(
        "Audio",
        f"Piper synthesize chunks={len(chunks)} bytes={piper_bytes} "
        f"chars={len(utterance)} length_scale={PIPER_LENGTH_SCALE} "
        f"dt_ms={(time.perf_counter() - t_piper0) * 1000.0:.1f}",
    )

    first = chunks[0]
    try:
        channels = int(getattr(first, "sample_channels", None) or channels)
        sampwidth = int(getattr(first, "sample_width", None) or sampwidth)
        framerate = int(getattr(first, "sample_rate", None) or framerate)
    except Exception:  # noqa: BLE001
        pass
    if channels < 1:
        channels = 1
    if sampwidth < 1:
        sampwidth = 2
    if framerate < 1:
        framerate = 22050

    try:
        with wave.open(path, "wb") as wav_file:
            # Explicit format BEFORE any frames (required by wave module).
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sampwidth)
            wav_file.setframerate(framerate)
            for chunk in chunks:
                try:
                    frame_bytes = chunk.audio_int16_bytes
                except Exception:  # noqa: BLE001
                    frame_bytes = b""
                if frame_bytes:
                    wav_file.writeframes(frame_bytes)
    except Exception as exc:  # noqa: BLE001
        log(
            "Audio",
            f"WARNING: TTS returned empty audio data, skipping synthesis ({exc})",
        )
        return False

    try:
        if not os.path.isfile(path) or os.path.getsize(path) < 44:
            log("Audio", "WARNING: TTS returned empty audio data, skipping synthesis")
            return False
    except OSError:
        log("Audio", "WARNING: TTS returned empty audio data, skipping synthesis")
        return False
    return True


def _device_output_samplerate(output_device: Optional[int]) -> Optional[int]:
    """Host default output sample rate for ``output_device`` (if queryable)."""
    try:
        info = sd.query_devices(output_device if output_device is not None else None)
        rate = int(float(info.get("default_samplerate") or 0))
        return rate if rate > 0 else None
    except Exception:  # noqa: BLE001
        return None


def _resample_pcm(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear-resample mono float32 PCM when host rate != Piper rate.

    Playing 22050 Hz buffers on a 44100 Hz WASAPI path without resampling
    can make speech sound ~2x too fast (common with virtual mixers like Sonar).
    """
    src = int(src_rate)
    dst = int(dst_rate)
    if src <= 0 or dst <= 0 or src == dst:
        return np.asarray(audio, dtype=np.float32).reshape(-1)
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size < 2:
        return samples
    n_out = max(1, int(round(samples.size * float(dst) / float(src))))
    x_old = np.linspace(0.0, 1.0, num=samples.size, endpoint=False, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False, dtype=np.float64)
    return np.interp(x_new, x_old, samples).astype(np.float32)


def _play_pcm_interruptible(
    audio_data: np.ndarray,
    samplerate: int,
    output_device: Optional[int],
    *,
    interruptible: bool = True,
) -> bool:
    """Stream PCM to the speaker in short chunks; return True if barge-in aborted.

    Holds ``playback_lock`` for the OutputStream lifecycle so a second utterance
    cannot open the device while chunks are still draining. Barge-in registers
    the live stream so ``interrupt_tts()`` can ``abort()`` without waiting on
    this lock (avoids deferred ``sd.stop`` races).

    ``interruptible=False`` plays UI acknowledgments without arming barge-in.
    """
    audio = np.asarray(audio_data, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = audio.reshape(-1)
    if audio.size == 0:
        return False

    play_rate = int(samplerate)
    host_rate = _device_output_samplerate(output_device)
    if host_rate is not None and host_rate != play_rate:
        log_debug(
            "Audio",
            f"resample PCM {play_rate} Hz -> {host_rate} Hz "
            f"(device={output_device}) to prevent sped-up playback",
        )
        audio = _resample_pcm(audio, play_rate, host_rate)
        play_rate = host_rate

    _bind_tts_barge_controller()
    chunk = max(1, int(round(play_rate * (BARGE_IN_CHUNK_MS / 1000.0))))
    stream_kwargs: dict[str, Any] = {
        "samplerate": int(play_rate),
        "channels": 1,
        "dtype": "float32",
        "blocksize": chunk,
    }
    if output_device is not None:
        stream_kwargs["device"] = output_device

    interrupted = False
    used_fallback = False
    t_start = time.perf_counter()
    n_chunks = 0
    bytes_written = 0
    log_debug(
        "Audio",
        f"playback alloc samples={audio.size} sr={play_rate} "
        f"chunk={chunk} ({BARGE_IN_CHUNK_MS:.0f}ms) device={output_device} "
        f"interruptible={interruptible}",
    )
    # Honor the turn-level latch when the spooler already called begin_playback;
    # otherwise (unit tests / direct play) open a local session.
    owns_session = not _tts_barge.is_playback_active()
    if owns_session:
        _tts_barge.begin_playback(interruptible=interruptible)
    try:
        with playback_lock:
            log_debug(
                "Audio",
                f"playback start t={t_start:.3f} samples={audio.size}",
            )
            try:
                with sd.OutputStream(**stream_kwargs) as stream:
                    if interruptible:
                        _tts_barge.register_output_stream(stream)
                    try:
                        for start in range(0, audio.size, chunk):
                            if stop_event.is_set():
                                interrupted = True
                                break
                            if interruptible and _tts_barge.is_set():
                                interrupted = True
                                log_debug(
                                    "Audio",
                                    f"playback interrupt at chunk={n_chunks} "
                                    f"offset={start}/{audio.size}",
                                )
                                try:
                                    stream.abort()
                                except Exception:  # noqa: BLE001
                                    pass
                                break
                            piece = audio[start : start + chunk]
                            if piece.size < chunk:
                                pad = np.zeros(chunk, dtype=np.float32)
                                pad[: piece.size] = piece
                                piece = pad
                            stream.write(piece.reshape(-1, 1))
                            n_chunks += 1
                            bytes_written += int(piece.size) * 4
                    finally:
                        if interruptible:
                            _tts_barge.unregister_output_stream(stream)
            except Exception as exc:  # noqa: BLE001
                if interrupted or (interruptible and _tts_barge.is_set()):
                    interrupted = True
                elif _is_portaudio_error(exc):
                    report_audio_hardware_fault(exc, where="OutputStream")
                    raise
                else:
                    # Fallback: start play under lock, then poll outside so barge-in can stop.
                    log(
                        "Audio",
                        f"WARNING: interruptible OutputStream failed ({exc}); using sd.play",
                    )
                    try:
                        kwargs: dict[str, Any] = {
                            "samplerate": int(play_rate),
                            "blocking": False,
                        }
                        if output_device is not None:
                            kwargs["device"] = output_device
                        sd.play(audio, **kwargs)
                        used_fallback = True
                    except Exception as exc2:  # noqa: BLE001
                        if _is_portaudio_error(exc2):
                            report_audio_hardware_fault(exc2, where="sd.play")
                        else:
                            log_exception("Audio", "TTS Engine Failure", exc=exc2)
                        raise

        if used_fallback and not interrupted:
            duration = audio.size / float(max(1, play_rate))
            deadline = time.perf_counter() + duration + 0.5
            log_debug("Audio", f"playback fallback sd.play duration_s={duration:.2f}")
            while time.perf_counter() < deadline:
                if stop_event.is_set():
                    interrupted = True
                    break
                if interruptible and _tts_barge.is_set():
                    interrupted = True
                    _safe_sd_stop(where="playback_fallback_interrupt", blocking=False)
                    break
                time.sleep(0.03)
            else:
                try:
                    with playback_lock:
                        sd.wait()
                except Exception:
                    pass
    finally:
        if owns_session:
            _tts_barge.end_playback()

    log_debug(
        "Audio",
        f"playback end interrupted={interrupted} chunks={n_chunks} "
        f"bytes≈{bytes_written} fallback={used_fallback} "
        f"dt_ms={(time.perf_counter() - t_start) * 1000.0:.1f}",
    )
    return interrupted


def half_duplex_mic_drop(stop_flag: threading.Event) -> None:
    """Half-duplex: discard mic frames while TTS plays (no barge-in evaluation).

    Speaker echo / room bleed must never call ``TtsWorker.interrupt()``. The
    mic queue is flushed continuously until playback ends; wake-word and VAD
    stand down on the same shared stream for the duration.
    """
    if vad_capture_active.is_set():
        return
    flush_audio_buffer_queue()
    try:
        while (
            not stop_flag.is_set()
            and not stop_event.is_set()
            and tts_busy.is_set()
            and not vad_capture_active.is_set()
        ):
            # Drain then discard — do not run Silero / wake scoring.
            frame = get_mic_frame(timeout=0.05)
            if frame is None:
                flush_audio_buffer_queue()
                time.sleep(0.01)
                continue
            # Drop frame (half-duplex deaf period).
    except Exception as exc:  # noqa: BLE001
        log_debug("BargeIn", f"half-duplex mic drop unavailable ({exc})")
    finally:
        flush_audio_buffer_queue()


# Back-compat alias — former stream-barge watcher is now half-duplex only.
barge_in_watch = half_duplex_mic_drop


def _boost_audio_thread_priority() -> None:
    """Raise OS priority so local LLM inference does not starve the sound buffer."""
    try:
        if os.name == "nt":
            import ctypes

            # THREAD_PRIORITY_ABOVE_NORMAL = 1
            handle = ctypes.windll.kernel32.GetCurrentThread()
            ctypes.windll.kernel32.SetThreadPriority(handle, 1)
        else:
            # Best-effort: nicer values are lower; negative raises priority when permitted.
            os.nice(-5)
    except Exception:  # noqa: BLE001
        pass


def _normalize_canned_ux_key(text: str) -> str:
    """Lowercase + strip common punctuation so cache hits ignore trailing marks."""
    key = sanitize_text_for_tts(text or "")
    key = key.lower()
    key = re.sub(r"[.,?!;:\"'`…]+", "", key)
    key = re.sub(r"\s+", " ", key).strip()
    return key


# Fuzzy lookup: normalized phrase → WAV filename (built once from canon map).
_CANNED_UX_FUZZY_WAV: dict[str, str] = {
    _normalize_canned_ux_key(phrase): filename
    for phrase, filename in _CANNED_UX_WAV_FILES.items()
}


def canned_ux_cache_path(text: str) -> Optional[Path]:
    """Return cache WAV path when ``text`` fuzzy-matches a canned UX acknowledgment."""
    key = _normalize_canned_ux_key(text or "")
    if not key:
        return None
    filename = _CANNED_UX_FUZZY_WAV.get(key)
    if not filename:
        return None
    return AUDIO_CACHE_DIR / filename


def ensure_canned_ux_audio_cache() -> None:
    """Pre-synthesize standard UX WAV files under ``dana/assets/audio_cache/``."""
    AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for phrase, filename in _CANNED_UX_WAV_FILES.items():
        path = AUDIO_CACHE_DIR / filename
        if path.is_file() and path.stat().st_size > 44:
            continue
        try:
            model_path = piper_model_path_for_text(phrase)
            voice = get_piper_voice(model_path)
            if synthesize_to_file(voice, phrase, str(path)):
                log("TTS", f"Cached UX audio: {filename}")
            else:
                log("TTS", f"WARNING: failed to cache UX audio for {phrase!r}")
        except Exception as exc:  # noqa: BLE001
            log("TTS", f"WARNING: UX audio cache skip ({filename}): {exc}")


def _play_ready_chime(output_device: Optional[int]) -> bool:
    """Short mechanical ready tone — no Piper during peak startup CPU."""
    sr = 22050
    duration_s = 0.16
    n = max(1, int(sr * duration_s))
    t = np.linspace(0.0, duration_s, n, endpoint=False, dtype=np.float32)
    tone = (0.22 * np.sin(2.0 * np.pi * 880.0 * t)).astype(np.float32)
    # Soft attack/release so the chime does not click.
    fade = min(n // 8, int(0.02 * sr))
    if fade > 0:
        tone[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
        tone[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return _play_pcm_interruptible(tone, sr, output_device)


def _play_cached_wav(
    path: Path,
    output_device: Optional[int],
    *,
    interruptible: bool = True,
) -> bool:
    """Play a pre-rendered WAV via the interruptible PCM path (no Piper)."""
    audio_data, samplerate = sf.read(str(path), dtype="float32")
    if getattr(audio_data, "size", 0) == 0:
        log("TTS", f"WARNING: empty cached WAV, falling back to Piper: {path.name}")
        raise ValueError("empty cached wav")
    frames = int(np.asarray(audio_data).reshape(-1).shape[0])
    log_debug(
        "TTS",
        f"cache hit {path.name} frames={frames} sr={samplerate} "
        f"interruptible={interruptible}",
    )
    interrupted = _play_pcm_interruptible(
        np.asarray(audio_data, dtype=np.float32),
        int(samplerate),
        output_device,
        interruptible=interruptible,
    )
    if interrupted:
        _safe_sd_stop(where="tts_cache_interrupted", blocking=False)
        log("TTS", "playback interrupted (barge-in)")
    else:
        time.sleep(0.08)
        log_debug(
            "TTS",
            f"Playback finished ({frames / float(samplerate):.2f}s) [cache]",
        )
    return interrupted


def _synthesize_and_play(
    text: str,
    output_device: Optional[int],
    *,
    interruptible: bool = True,
    agent_id: str | None = None,
) -> bool:
    """Worker-only: synth + interruptible playback under ``playback_lock``.

    Returns True if barge-in aborted playback. Producers must use ``enqueue_speech``.
    Canned UX strings play from ``dana/assets/audio_cache/`` when available.
    Stage 8.8 — specialized ``agent_id`` voices bypass the receptionist Piper path.
    """
    text = sanitize_text_for_tts(text or "")
    if not text:
        return False

    try:
        from dana.audio.multi_voice_tts import (
            set_active_tts_agent,
            synthesize_speech,
            uses_receptionist_piper,
        )

        set_active_tts_agent(agent_id or "broker")
        use_piper = uses_receptionist_piper(agent_id)
    except Exception:  # noqa: BLE001
        use_piper = True

    # Specialized personas: multi-voice WAV → same interruptible PCM path.
    if not use_piper:
        tmp_path: str | None = None
        try:
            wav = synthesize_speech(text, agent_id=agent_id)
            tmp_path = str(wav)
            return _play_cached_wav(
                Path(tmp_path), output_device, interruptible=interruptible
            )
        except Exception as exc:  # noqa: BLE001
            log_debug("TTS", f"multi-voice play failed ({exc}); Piper fallback")
        finally:
            if tmp_path:
                try:
                    if os.path.isfile(tmp_path) and "donna_" in os.path.basename(tmp_path):
                        os.remove(tmp_path)
                except OSError:
                    pass

    cache_path = canned_ux_cache_path(text)
    if cache_path is not None and cache_path.is_file() and cache_path.stat().st_size > 44:
        try:
            return _play_cached_wav(
                cache_path, output_device, interruptible=interruptible
            )
        except Exception as exc:  # noqa: BLE001
            log_debug("TTS", f"cache play failed ({exc}); live Piper fallback")

    model_path = piper_model_path_for_text(text)
    lang = "en"
    t_synth0 = time.perf_counter()
    log_debug(
        "TTS",
        f"Piper route -> {lang} ({os.path.basename(model_path)}) chars={len(text)} "
        f"interruptible={interruptible} agent={agent_id or 'broker'}",
    )

    voice = get_piper_voice(model_path)
    out_path = PIPER_TEMP_WAV
    interrupted = False
    try:
        try:
            if not synthesize_to_file(voice, text, out_path):
                log("TTS", "WARNING: skipping playback — Piper produced no audio")
                return False
            # Persist newly rendered canned UX into the cache for next time.
            if cache_path is not None:
                try:
                    AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    import shutil

                    shutil.copyfile(out_path, str(cache_path))
                except Exception as exc:  # noqa: BLE001
                    log_debug("TTS", f"cache write skipped: {exc}")
            audio_data, samplerate = sf.read(out_path, dtype="float32")
            if getattr(audio_data, "size", 0) == 0:
                log("TTS", "WARNING: TTS returned empty audio data, skipping synthesis")
                return False
            frames = int(np.asarray(audio_data).reshape(-1).shape[0])
            log_debug(
                "TTS",
                f"Piper buffer ready frames={frames} sr={samplerate} "
                f"synth_ms={(time.perf_counter() - t_synth0) * 1000.0:.1f}",
            )

            interrupted = _play_pcm_interruptible(
                np.asarray(audio_data, dtype=np.float32),
                int(samplerate),
                output_device,
                interruptible=interruptible,
            )
            if interrupted:
                _safe_sd_stop(where="tts_play_interrupted", blocking=False)
                log("TTS", "playback interrupted (barge-in)")
            else:
                time.sleep(0.08)
                log_debug(
                    "TTS",
                    f"Playback finished ({frames / float(samplerate):.2f}s)",
                )
        except Exception as exc:  # noqa: BLE001
            if _is_portaudio_error(exc):
                if not audio_hardware_fault.is_set():
                    report_audio_hardware_fault(exc, where="tts_worker/Piper")
            else:
                log_exception("TTS", "TTS Engine Failure", exc=exc)
            raise
    finally:
        try:
            if os.path.isfile(out_path):
                os.remove(out_path)
        except OSError:
            pass
    return interrupted


def speak_text(text: str, output_device: Optional[int] = None) -> bool:
    """Legacy name — producers should prefer ``enqueue_speech`` (non-blocking).

    When called off the TTS worker, this enqueues and returns False (not interrupted).
    The TTS worker calls ``_synthesize_and_play`` directly.
    """
    if threading.current_thread() is _tts_worker_thread:
        return _synthesize_and_play(text, output_device if output_device is not None else AUDIO_OUTPUT_DEVICE)
    enqueue_speech(text)
    return False


def _speak_with_timeout(
    text: str,
    output_device: Optional[int],
    *,
    max_seconds: float = TTS_UTTERANCE_MAX_SECONDS,
    interruptible: bool = True,
    agent_id: str | None = None,
) -> bool:
    """Play on a watchdog-guarded helper thread; abort if it exceeds ``max_seconds``."""
    result: list[bool] = [False]
    error: list[BaseException | None] = [None]

    def _run() -> None:
        _boost_audio_thread_priority()
        try:
            result[0] = bool(
                _synthesize_and_play(
                    text,
                    output_device,
                    interruptible=interruptible,
                    agent_id=agent_id,
                )
            )
        except BaseException as exc:  # noqa: BLE001
            error[0] = exc

    worker = threading.Thread(target=_run, name="TTSUtterance", daemon=True)
    worker.start()
    worker.join(timeout=max_seconds)
    if worker.is_alive():
        log(
            "TTS",
            f"WARNING: utterance exceeded {max_seconds:.0f}s — "
            "aborting playback and releasing audio device",
        )
        # Hard timeout always wins — even UX acks must not hang forever.
        interrupt_tts(reason="utterance_timeout", force=True)
        worker.join(timeout=2.0)
        if worker.is_alive():
            log(
                "TTS",
                "WARNING: utterance thread still alive after abort — forcing state reset",
            )
            reset_tts_audio_state(
                "hung TTSUtterance thread",
                ui_state="listening",
                flush_queue=False,
            )
        return True
    if error[0] is not None:
        raise error[0]
    return result[0]


def maybe_play_boot_ready_audio() -> None:
    """Play ``Donna is ready.`` only after Ollama + Piper + wake-word are all ready.

    Safe to call from multiple boot threads; fires at most once per process.
    """
    global _boot_ready_audio_played
    if _boot_ready_audio_played:
        return
    if not (
        ollama_ready.is_set()
        and piper_voices_ready.is_set()
        and wakeword_armed.is_set()
    ):
        return
    with _boot_ready_audio_lock:
        if _boot_ready_audio_played:
            return
        _boot_ready_audio_played = True
    log(
        "TTS",
        "Boot complete (ollama_ready + Piper + wake armed) — playing ready signal",
    )
    try:
        enqueue_speech("Donna is ready.")
    except Exception as exc:  # noqa: BLE001
        log("TTS", f"WARNING: boot ready enqueue failed ({exc})")


def tts_worker() -> None:
    """TTS consumer: block on ``tts_queue``, honor VAD hold, then play under lock."""
    global _tts_worker_thread
    _tts_worker_thread = threading.current_thread()
    _nt_hide_console_if_mp_child()
    _boost_audio_thread_priority()
    log("TTS", "Initializing offline Piper TTS spooler...")
    try:
        download_piper_models()
    except Exception as exc:  # noqa: BLE001
        log("TTS", f"ERROR downloading Piper models: {exc}")
        stop_event.set()
        return

    if AUDIO_OUTPUT_DEVICE is not None:
        try:
            out_name = sd.query_devices()[AUDIO_OUTPUT_DEVICE].get("name", "?")
        except Exception:
            out_name = "?"
        log("TTS", f"playback device [{AUDIO_OUTPUT_DEVICE}] {out_name}")
    else:
        log("TTS", "playback device: system default")

    try:
        get_piper_voice(PIPER_EN_ONNX)
        log("TTS", "Piper voice ready (en_US-hfc_female).")
    except Exception as exc:  # noqa: BLE001
        log("TTS", f"ERROR loading Piper voices: {exc}")
        stop_event.set()
        return

    try:
        ensure_canned_ux_audio_cache()
    except Exception as exc:  # noqa: BLE001
        log("TTS", f"WARNING: UX audio cache warm-up failed: {exc}")

    # Defer ready audio until Ollama warm-up + wake-word arming also complete.
    piper_voices_ready.set()
    if tts_queue.empty() and not tts_busy.is_set():
        speech_idle.set()
    maybe_play_boot_ready_audio()

    log("TTS", "spooler ready; waiting for messages (barge-in armed).")
    _bind_tts_barge_controller()
    while not stop_event.is_set():
        try:
            try:
                raw_item = tts_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if raw_item is None:
                speech_idle.set()
                break

            text, item_interruptible, item_agent = _parse_tts_spool_item(raw_item)

            # Drop orphaned spool items after a barge-in latch.
            if _tts_barge.is_set():
                flush_tts_queue()
                _tts_barge.clear()
                if tts_queue.empty() and not tts_busy.is_set():
                    speech_idle.set()
                continue

            # Hold while the user is speaking — do not overlap mic capture.
            if not _wait_tts_clear_of_user_speech(text):
                if tts_queue.empty() and not tts_busy.is_set():
                    speech_idle.set()
                continue

            _tts_barge.clear()
            tts_busy.set()
            speech_idle.clear()
            # Latch exemption BEFORE any mic/VAD path can see tts_busy (self-barge race).
            _tts_barge.begin_playback(interruptible=item_interruptible)
            prev_ui = get_ui_state()
            set_ui_state("speaking")
            watcher_stop = threading.Event()
            # Half-duplex: always discard mic frames during TTS (acks + LLM speech).
            watcher = threading.Thread(
                target=half_duplex_mic_drop,
                args=(watcher_stop,),
                name="HalfDuplexMicDrop",
                daemon=True,
            )
            watcher.start()
            interrupted = False
            turn_t0 = time.perf_counter()
            time.sleep(0.05)
            try:
                log_debug(
                    "TTS",
                    f'play start t={turn_t0:.3f} chars={len(text)} '
                    f'interruptible={item_interruptible} agent={item_agent} '
                    f'pending={tts_queue.qsize()} preview="{text[:80]}"',
                )
                interrupted = bool(
                    _speak_with_timeout(
                        text,
                        AUDIO_OUTPUT_DEVICE,
                        interruptible=item_interruptible,
                        agent_id=item_agent,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                if _is_portaudio_error(exc):
                    if not audio_hardware_fault.is_set():
                        report_audio_hardware_fault(exc, where="tts_worker")
                else:
                    log_exception("TTS", "TTS Engine Failure", exc=exc)
                interrupted = True
            finally:
                watcher_stop.set()
                if watcher is not None:
                    try:
                        watcher.join(timeout=1.0)
                    except Exception:
                        pass
                _safe_sd_stop(where="tts_worker_turn_end", blocking=False)
                barged = bool(
                    item_interruptible
                    and (interrupted or tts_interrupt_event.is_set())
                )
                tts_busy.clear()
                _tts_barge.end_playback()
                if barged:
                    # Instantly dump any pending system messages after cut-off.
                    dropped = flush_tts_queue()
                    flush_audio_buffer_queue()
                    tts_interrupt_event.clear()
                    set_ui_state("listening")
                    speech_idle.set()
                    log(
                        "BargeIn",
                        f"Flushed {dropped} TTS spool item(s); state -> listening "
                        f"dt_ms={(time.perf_counter() - turn_t0) * 1000.0:.1f}",
                    )
                else:
                    if prev_ui in ("thinking", "transcribing", "followup", "listening"):
                        set_ui_state(prev_ui)
                    elif prev_ui == "speaking":
                        set_ui_state("listening")
                    else:
                        # Boot ready / standby TTS starts from idle — must return
                        # to idle or WakeWord never consumes mic frames.
                        set_ui_state("idle")
                    if tts_queue.empty():
                        speech_idle.set()
                    else:
                        speech_idle.clear()
                    log_debug(
                        "TTS",
                        f"play end ok dt_ms="
                        f"{(time.perf_counter() - turn_t0) * 1000.0:.1f} "
                        f"pending={tts_queue.qsize()} ui={get_ui_state()}",
                    )
        except Exception as exc:  # noqa: BLE001
            log_exception("TTS", "TTS Engine Failure (spooler)", exc=exc)
            if _is_portaudio_error(exc):
                report_audio_hardware_fault(exc, where="TTS spooler")
            reset_tts_audio_state(
                f"TTS spooler error: {exc}",
                ui_state="listening",
            )

    log("TTS", "spooler stopped.")


def audio_worker() -> None:
    """Backward-compatible entrypoint — runs the TTS output spooler consumer."""
    tts_worker()


# ---------------------------------------------------------------------------
# System tray + CustomTkinter settings GUI
# ---------------------------------------------------------------------------

_UI_STATE_LABELS = {
    "idle": "Idle",
    "listening": "Listening",
    "speaking": "Speaking",
    "followup": "Listening (follow-up)",
    "transcribing": "Processing",
    "thinking": "Processing",
}

# Soft microphone glyph — blue when idle/busy, green while VAD is listening.
_TRAY_FILL_IDLE = (37, 99, 235, 255)  # blue
_TRAY_FILL_LISTENING = (22, 163, 74, 255)  # green
_TRAY_GLYPH = (226, 232, 240, 255)
_TRAY_LISTENING_STATES = frozenset({"listening", "followup"})


def create_tray_image(mode: str = "idle") -> Image.Image:
    """Branded tray icon; prefers keyed RGBA logo, procedural fallback."""
    size = 64
    try:
        from dana.ui.startup_tray import build_tray_image

        logo = build_tray_image(mode=mode, size=size)
    except Exception:  # noqa: BLE001
        logo = None
    if logo is not None:
        return logo.convert("RGBA") if hasattr(logo, "convert") else logo
    fill = _TRAY_FILL_LISTENING if mode == "listening" else _TRAY_FILL_IDLE
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(
        (4, 4, size - 5, size - 5),
        radius=14,
        fill=fill,
    )
    draw.ellipse((18, 16, 46, 44), fill=_TRAY_GLYPH)
    draw.rectangle((28, 40, 36, 52), fill=_TRAY_GLYPH)
    # Extra bright status pip when listening (glances faster in the tray).
    if mode == "listening":
        draw.ellipse((42, 8, 56, 22), fill=(250, 250, 250, 255))
        draw.ellipse((45, 11, 53, 19), fill=(34, 197, 94, 255))
    return img


def update_tray_icon_for_state(state: str) -> None:
    """Swap tray icon / tooltip when entering or leaving the listening states."""
    icon = _tray_icon
    if icon is None:
        return
    listening = state in _TRAY_LISTENING_STATES
    mode = "listening" if listening else "idle"
    title = "Dānā — Listening" if listening else "Dānā · Cybernetic Control Plane"
    try:
        icon.icon = create_tray_image(mode)
        icon.title = title
    except Exception as exc:  # noqa: BLE001
        log_debug("UI", f"Tray icon update skipped ({exc})")


def _device_menu_label(index: int, name: str) -> str:
    return f"[{index}] {name}"


def _parse_device_menu_label(label: str) -> Optional[int]:
    if not label.startswith("[") or "]" not in label:
        return None
    try:
        return int(label[1 : label.index("]")])
    except ValueError:
        return None


def request_donna_quit(icon: Optional[pystray.Icon] = None, _item: Any = None) -> None:
    """Tray Quit / cleanup — stop agent threads and close the GUI."""
    log("Main", "Quit requested (system tray).")
    try:
        from dana.telemetry import set_system_status, stop_dashboard_thread

        set_system_status("Restarting")
        stop_dashboard_thread()
    except Exception:
        pass
    try:
        from dana.tools.registry import cleanup_ephemeral_tools

        cleaned = cleanup_ephemeral_tools(archive=True)
        if cleaned:
            log("Main", f"Ephemeral tool GC archived {len(cleaned)} tool(s): {cleaned}")
    except Exception as exc:  # noqa: BLE001
        log("Main", f"WARNING: ephemeral tool GC failed: {exc}")
    stop_event.set()
    reset_tts_audio_state("application quit", flush_queue=False)
    try:
        speech_queue.put_nowait(None)
    except queue.Full:
        pass
    if icon is not None:
        try:
            icon.stop()
        except Exception:
            pass
    global _tray_icon
    _tray_icon = None
    gui = _gui_instance
    if gui is not None:
        try:
            gui.after(0, gui.destroy)
        except Exception:
            try:
                gui.destroy()
            except Exception:
                pass


def _shutdown_agent_threads(*, join_timeout: float = 8.0) -> None:
    """Signal workers to stop and wait for AgentLoop (which joins Tracker/Wake/Audio/Conv)."""
    try:
        from dana.tools.registry import cleanup_ephemeral_tools

        cleaned = cleanup_ephemeral_tools(archive=True)
        if cleaned:
            log("Main", f"Ephemeral tool GC archived {len(cleaned)} tool(s): {cleaned}")
    except Exception as exc:  # noqa: BLE001
        log("Main", f"WARNING: ephemeral tool GC failed: {exc}")
    try:
        from dana.telemetry import set_system_status, stop_dashboard_thread, write_dashboard

        set_system_status("Restarting")
        write_dashboard()
        stop_dashboard_thread()
    except Exception:
        pass
    stop_event.set()
    try:
        speech_queue.put_nowait(None)
    except queue.Full:
        pass
    try:
        tts_interrupt_event.set()
        speech_idle.set()
    except Exception:
        pass
    thread = _agent_loop_thread
    if thread is not None and thread.is_alive():
        thread.join(timeout=join_timeout)
        if thread.is_alive():
            log("Main", "WARNING: AgentLoop did not exit within join timeout.")


def _install_signal_handlers(gui: "DonnaGUI") -> None:
    """Ctrl+C / SIGTERM → destroy GUI on the Tk thread (avoids hanging workers)."""

    def _handler(signum: int, _frame: Any) -> None:
        log("Main", f"Signal {signum} received — shutting down.")
        try:
            from dana.tools.registry import cleanup_ephemeral_tools

            cleaned = cleanup_ephemeral_tools(archive=True)
            if cleaned:
                log(
                    "Main",
                    f"Ephemeral tool GC archived {len(cleaned)} tool(s): {cleaned}",
                )
        except Exception as exc:  # noqa: BLE001
            log("Main", f"WARNING: ephemeral tool GC failed: {exc}")
        stop_event.set()
        # Instantly dump TTS spool + stream sentence buffer (no shutdown spam).
        try:
            from dana.agentic import reset_stream_sentence_tts

            reset_stream_sentence_tts()
        except Exception:
            pass
        try:
            dropped = flush_tts_queue()
            if dropped:
                log("TTS", f"Shutdown flushed {dropped} pending spool item(s)")
        except Exception:
            pass
        try:
            tts_interrupt_event.set()
        except Exception:
            pass
        try:
            speech_queue.put_nowait(None)
        except queue.Full:
            pass
        try:
            gui.after(0, gui.destroy)
        except Exception:
            pass

    for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
        if sig is None:
            continue
        try:
            signal.signal(sig, _handler)
        except Exception:
            pass

def run_system_tray(gui: "DonnaGUI") -> None:
    """Blocking pystray loop (run in a daemon thread)."""
    global _tray_icon

    def open_settings(icon: pystray.Icon, _item: Any = None) -> None:
        gui.after(0, gui.show_window)

    from dana.ui.startup_tray import (
        check_startup_registry_status,
        toggle_run_on_startup,
    )
    from dana.ui.watchdog import (
        check_shell_watchdog_status,
        get_shared_watchdog,
        toggle_shell_watchdog,
    )

    # Ensure shared watchdog is constructed (wires toast/planner when enabled).
    try:
        get_shared_watchdog()
    except Exception:  # noqa: BLE001
        pass

    menu = pystray.Menu(
        pystray.MenuItem("Open Settings", open_settings, default=True),
        pystray.MenuItem(
            "Run on Startup",
            toggle_run_on_startup,
            checked=lambda item: check_startup_registry_status(item),
        ),
        pystray.MenuItem(
            "Enable Shell Watchdog",
            toggle_shell_watchdog,
            checked=lambda item: check_shell_watchdog_status(item),
        ),
        pystray.MenuItem("Quit", request_donna_quit),
    )
    icon = pystray.Icon(
        "Dana",
        create_tray_image("idle"),
        "Dānā · Cybernetic Control Plane",
        menu,
    )
    _tray_icon = icon
    log("Main", "System tray icon ready (bottom-right notification area).")
    icon.run()


class TraceCell(ctk.CTkFrame):
    """One pipeline stage row in the Live Trace scroll area."""

    def __init__(self, master: Any, stage: str, message: str, status: str = "active") -> None:
        super().__init__(
            master,
            corner_radius=8,
            border_width=2,
            border_color=_TRACE_IDLE_COLOR,
            fg_color=("gray92", "gray17"),
        )
        self.stage = stage
        self.current_status = "active"
        self.icon_label = ctk.CTkLabel(
            self,
            text=_TRACE_STATUS_ICONS["active"],
            width=28,
            font=ctk.CTkFont(size=16),
        )
        self.icon_label.pack(side="left", padx=(10, 6), pady=8)
        self.msg_label = ctk.CTkLabel(
            self,
            text=message or stage,
            anchor="w",
            justify="left",
            font=ctk.CTkFont(size=13),
        )
        self.msg_label.pack(side="left", fill="x", expand=True, padx=(0, 12), pady=8)
        self.update_status(status, message=message)

    def update_status(
        self,
        status: str,
        message: str | None = None,
        *,
        accent: str | None = None,
    ) -> None:
        normalized = (status or "active").strip().lower()
        if normalized not in _TRACE_STATUS_ICONS:
            normalized = "active"
        self.current_status = normalized
        self.icon_label.configure(text=_TRACE_STATUS_ICONS[normalized])
        if message is not None:
            self.msg_label.configure(text=message or self.stage)
        color = accent or _TRACE_IDLE_COLOR
        if normalized == "active":
            border = color if color != _TRACE_IDLE_COLOR else _UI_ACCENT
            text_color = border
        elif normalized == "completed":
            border = color if color != _TRACE_IDLE_COLOR else "#10B981"
            text_color = ("gray20", "gray85")
        else:  # bypassed
            border = _TRACE_IDLE_COLOR
            text_color = _TRACE_IDLE_COLOR
        try:
            self.configure(border_color=border)
            self.msg_label.configure(text_color=text_color)
        except Exception:  # noqa: BLE001
            pass


class DonnaGUI(ctk.CTk):
    """Live Trace window with settings tabs; retreats to the tray on close."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Donna — Control Dashboard")
        self.geometry("880x780")
        self.minsize(650, 750)
        self._dictation_active = False
        self._behavior_locked = False
        # Stage 8.9.7 — soft LangGraph engine ignition (False = STANDBY).
        self.engine_active = False
        self._behavior_sliders: dict[str, ctk.CTkSlider] = {}
        self._behavior_labels: dict[str, ctk.CTkLabel] = {}
        self._behavior_last_write: dict[str, float] = {}
        # Stage 8.5.1 — static settings that must not change while system is hot.
        self._static_behavior_widgets: list[Any] = []
        self._behavior_lock_hint: ctk.CTkLabel | None = None
        self._behavior_reload_btn: ctk.CTkButton | None = None
        self.dictation_status: ctk.CTkLabel | None = None
        self._engine_status_lbl: ctk.CTkLabel | None = None
        self._engine_warn_lbl: ctk.CTkLabel | None = None
        self._engage_btn: ctk.CTkButton | None = None
        self._standby_btn: ctk.CTkButton | None = None
        self._engine_warn_job: str | None = None
        # Stage 9.3 — Settings auto-updater chrome.
        self._update_status_lbl: ctk.CTkLabel | None = None
        self._update_check_btn: ctk.CTkButton | None = None
        self._update_apply_btn: ctk.CTkButton | None = None
        self._update_busy = False
        # Stage 8.10 — Dashboard silent text chat.
        self.chat_entry: ctk.CTkEntry | None = None
        self._chat_send_btn: ctk.CTkButton | None = None
        self.transcript_box = None

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        try:
            self.configure(fg_color=_UI_CANVAS)
        except Exception:  # noqa: BLE001
            pass

        self._mic_labels: list[str] = []
        self._speaker_labels: list[str] = []
        self._mic_by_label: dict[str, int] = {}
        self._speaker_by_label: dict[str, int] = {}
        self._trace_cells: dict[str, TraceCell] = {}
        self._pulse_on = False
        self._header_mode = "chat"
        self.assistive_orb: Any | None = None

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close_to_tray)
        self.withdraw()
        try:
            from dana.ui.logo import apply_window_icon

            apply_window_icon(self)
        except Exception:  # noqa: BLE001
            pass
        # Stage 8.7 — floating AssistiveTouch orb (supplements system tray).
        try:
            self.after(200, self._start_assistive_orb)
        except Exception:  # noqa: BLE001
            pass
        self.after(400, self._refresh_stats)
        self.after(100, self.process_telemetry)
        self.after(500, self._pulse_active_cells)

    def _mode_accent(self, mode: str | None = None) -> str:
        key = (mode or self._header_mode or "chat").strip().lower()
        if self._dictation_active and key in {"chat", "developer", "vision", "research"}:
            # Dictation latch overrides the glowing status badge.
            return _TRACE_MODE_COLORS.get("dictation", "#9C27B0")
        return _TRACE_MODE_COLORS.get(key, _TRACE_IDLE_COLOR)

    def _set_mode_indicator(self, mode: str | None) -> None:
        key = (mode or "chat").strip().lower()
        if key not in _TRACE_MODE_COLORS:
            key = "chat"
        self._header_mode = key
        display_key = "dictation" if self._dictation_active else key
        color = _TRACE_MODE_COLORS.get(display_key, self._mode_accent(key))
        label = "Dictation" if self._dictation_active else key.title()
        try:
            # Stage 8.9.8 — pill only (no plain-text Mode / Dictation label).
            if hasattr(self, "mode_badge") and self.mode_badge is not None:
                self.mode_badge.configure(
                    text=f"  ●  {label.upper()}  ",
                    text_color=color,
                    fg_color=_UI_GHOST,
                )
        except Exception:  # noqa: BLE001
            pass
        try:
            if self.assistive_orb is not None:
                self.assistive_orb.refresh_controls()
        except Exception:  # noqa: BLE001
            pass

    def _make_card(
        self,
        parent: Any,
        *,
        title: str,
        padx: int = 12,
        pady: tuple[int, int] = (10, 10),
        expand: bool = True,
    ) -> ctk.CTkFrame:
        """Floating dark card container with padded header."""
        card = ctk.CTkFrame(
            parent,
            fg_color=_UI_CARD,
            corner_radius=16,
            border_width=1,
            border_color=_UI_CARD_BORDER,
        )
        card.pack(fill="both" if expand else "x", expand=expand, padx=padx, pady=pady)
        ctk.CTkLabel(
            card,
            text=title,
            anchor="w",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#F3F4F6",
        ).pack(fill="x", padx=14, pady=(12, 6))
        body = ctk.CTkFrame(card, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        return body

    def _build_ui(self) -> None:
        # Top navigation / status bar
        header = ctk.CTkFrame(
            self,
            fg_color=_UI_CARD,
            corner_radius=0,
            border_width=0,
            height=56,
        )
        header.pack(fill="x", padx=0, pady=0)
        # Stage 8.9.8 / 8.9.9 — OS title owns brand text; header shows logo + status.
        self.mode_dot = None
        self.mode_label = None
        self._header_logo_img = None
        try:
            from dana.ui.logo import load_premium_logo

            self._header_logo_img = load_premium_logo((36, 36))
        except Exception:  # noqa: BLE001
            self._header_logo_img = None
        if self._header_logo_img is not None:
            try:
                self._header_logo_lbl = ctk.CTkLabel(
                    header,
                    text="",
                    image=self._header_logo_img,
                    fg_color="transparent",
                )
                self._header_logo_lbl.pack(side="left", padx=(16, 8), pady=10)
            except Exception:  # noqa: BLE001 — Tk pyimage race across roots
                self._header_logo_img = None
                self._header_logo_lbl = None
        else:
            self._header_logo_lbl = None
        self.mode_badge = ctk.CTkLabel(
            header,
            text="  ●  CHAT  ",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=self._mode_accent("chat"),
            fg_color=_UI_GHOST,
            corner_radius=999,
            padx=8,
            pady=4,
        )
        self.mode_badge.pack(side="left", padx=(8, 12), pady=12)
        # Stage 8.9.2 — emergency kill switch (runs stop_dana.vbs / stop_dana.bat).
        self.stop_donna_btn = ctk.CTkButton(
            header,
            text="STOP DONNA",
            width=128,
            height=32,
            corner_radius=8,
            fg_color="#d32f2f",
            hover_color="#b71c1c",
            text_color="#FFFFFF",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._on_stop_donna_clicked,
        )
        self.stop_donna_btn.pack(side="right", padx=(8, 16), pady=12)
        ctk.CTkLabel(
            header,
            text="local · offline-first",
            font=ctk.CTkFont(size=11),
            text_color="#6B7280",
            anchor="e",
        ).pack(side="right", padx=(8, 4), pady=14)

        # Stage 8.9.4 — optimized tab order (widget refs, not notebook indices).
        tabs = ctk.CTkTabview(
            self,
            fg_color=_UI_CANVAS,
            segmented_button_fg_color=_UI_CARD,
            segmented_button_selected_color=_UI_ACCENT,
            segmented_button_selected_hover_color=_UI_ACCENT_HOVER,
            segmented_button_unselected_color=_UI_GHOST,
            segmented_button_unselected_hover_color="#34344A",
            text_color="#E5E7EB",
            corner_radius=14,
            border_width=0,
        )
        tabs.pack(fill="both", expand=True, padx=14, pady=(10, 14))
        self._tabs = tabs
        try:
            tabs._segmented_button.configure(corner_radius=999, height=34)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass

        tab_dashboard = tabs.add("Dashboard")
        tab_dictation = tabs.add("Dictation")
        tab_behavior = tabs.add("Behavior")
        tab_trace = tabs.add("Live Trace")
        tab_settings = tabs.add("Settings")

        self._build_dashboard_tab(tab_dashboard)
        self._build_dictation_tab(tab_dictation)
        self._build_behavior_tab(tab_behavior)

        # LangGraph Live Trace panel (queue drain via self.after — never worker threads).
        try:
            from dana.ui.trace_window import LiveTracePanel

            self.live_trace = LiveTracePanel(tab_trace, poll_ms=50)
            self.live_trace.pack(fill="both", expand=True, padx=2, pady=2)
            self.trace_scroll = self.live_trace.timeline
        except Exception:  # noqa: BLE001
            self.live_trace = None
            fallback = self._make_card(tab_trace, title="Pipeline stages")
            self.trace_scroll = ctk.CTkScrollableFrame(fallback, fg_color="transparent")
            self.trace_scroll.pack(fill="both", expand=True)

        self._build_settings_tab(tab_settings)

        try:
            tabs.set("Dashboard")
        except Exception:  # noqa: BLE001
            pass

        self._reload_device_menus()
        try:
            self.after(300, self.refresh_dictation_sessions)
            self.after(350, self._reload_behavior_sliders)
        except Exception:  # noqa: BLE001
            pass

    def _select_tab(self, name: str) -> None:
        """Switch notebook by tab name (stable across reorder)."""
        tabs = getattr(self, "_tabs", None)
        if tabs is None:
            return
        try:
            tabs.set(str(name))
        except Exception:  # noqa: BLE001
            pass

    def _build_dashboard_tab(self, tab) -> None:  # noqa: ANN001
        """Stage 8.9.4 — welcome / quick actions + Live Transcript + bottom chat bar.

        Pack order matters: bottom input is reserved first so the transcript is the
        only region that expands/shrinks on resize (input stays visible).
        """
        try:
            tab.configure(fg_color=_UI_CANVAS)
        except Exception:  # noqa: BLE001
            pass

        # --- Bottom input (pack first, side=bottom) ---
        chat_bar = ctk.CTkFrame(tab, fg_color="transparent")
        chat_bar.pack(side="bottom", fill="x", padx=14, pady=(6, 14))
        chat_bar.grid_columnconfigure(0, weight=1)
        self.chat_entry = ctk.CTkEntry(
            chat_bar,
            placeholder_text="Type below or say Donna, then speak.",
            height=36,
            corner_radius=10,
            fg_color=_UI_GHOST,
            border_width=1,
            border_color=_UI_CARD_BORDER,
            text_color="#F9FAFB",
            placeholder_text_color=_UI_MUTED,
            font=ctk.CTkFont(size=13),
        )
        self.chat_entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.chat_entry.bind("<Return>", self.submit_text_command)
        self._chat_send_btn = ctk.CTkButton(
            chat_bar,
            text="Send",
            width=92,
            height=36,
            corner_radius=999,
            fg_color=_UI_ACCENT,
            hover_color=_UI_ACCENT_HOVER,
            text_color="#FFFFFF",
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self.submit_text_command,
        )
        self._chat_send_btn.grid(row=0, column=1, sticky="e")

        # --- Top welcome / ignition (fixed height, no expand) ---
        welcome = self._make_card(
            tab, title="System Ready", pady=(10, 4), expand=False
        )
        center = ctk.CTkFrame(welcome, fg_color="transparent")
        center.pack(fill="x", pady=(4, 4))
        self._dash_logo_img = None
        try:
            from dana.ui.logo import load_premium_logo

            self._dash_logo_img = load_premium_logo((72, 72))
        except Exception:  # noqa: BLE001
            self._dash_logo_img = None
        if self._dash_logo_img is not None:
            try:
                ctk.CTkLabel(
                    center,
                    text="",
                    image=self._dash_logo_img,
                    fg_color="transparent",
                ).pack(pady=(2, 4))
            except Exception:  # noqa: BLE001 — Tk pyimage race across roots
                self._dash_logo_img = None
        ctk.CTkLabel(
            center,
            text="Donna is online",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color="#F9FAFB",
        ).pack(pady=(2, 2))
        self.status_value = ctk.CTkLabel(
            center,
            text="Idle",
            font=ctk.CTkFont(size=13),
            text_color=_UI_MUTED,
        )
        self.status_value.pack(pady=(0, 2))
        self.wake_value = ctk.CTkLabel(
            center,
            text="Wake: Donna",
            font=ctk.CTkFont(size=12),
            text_color="#00E676",
        )
        self.wake_value.pack(pady=(0, 6))

        engine_row = ctk.CTkFrame(center, fg_color="transparent")
        engine_row.pack(pady=(2, 2))
        try:
            _engage_font = ctk.CTkFont(
                family="Segoe UI Historic", size=13, weight="bold"
            )
        except Exception:  # noqa: BLE001
            _engage_font = ctk.CTkFont(size=13, weight="bold")
        self._engage_btn = ctk.CTkButton(
            engine_row,
            text="ENGAGE \U000103ad",
            width=160,
            height=36,
            corner_radius=999,
            fg_color="#2E7D32",
            hover_color="#1B5E20",
            text_color="#FFFFFF",
            font=_engage_font,
            command=self.engage_engine,
        )
        self._engage_btn.pack(side="left", padx=6)
        self._standby_btn = ctk.CTkButton(
            engine_row,
            text="STANDBY",
            width=130,
            height=36,
            corner_radius=999,
            fg_color="#6B6560",
            hover_color="#5A5550",
            text_color="#E8E0D5",
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self.standby_engine,
        )
        self._standby_btn.pack(side="left", padx=6)
        self._engine_status_lbl = ctk.CTkLabel(
            center,
            text="Status: STANDBY (Variables Unlocked)",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#A1887F",
        )
        self._engine_status_lbl.pack(pady=(4, 2))
        self._engine_warn_lbl = ctk.CTkLabel(
            center,
            text="",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#F87171",
        )
        self._engine_warn_lbl.pack(pady=(0, 4))

        actions = ctk.CTkFrame(center, fg_color="transparent")
        actions.pack(pady=(2, 4))
        for label, cmd in (
            ("Start Chat", self._dashboard_start_chat),
            ("Trigger Dictation", self._dashboard_trigger_dictation),
            ("Open Live Trace", self._dashboard_open_trace),
        ):
            ctk.CTkButton(
                actions,
                text=label,
                width=140,
                height=32,
                corner_radius=999,
                fg_color=_UI_GHOST,
                hover_color="#34344A",
                border_width=1,
                border_color=_UI_CARD_BORDER,
                text_color="#E5E7EB",
                font=ctk.CTkFont(size=12),
                command=cmd,
            ).pack(side="left", padx=6)
        try:
            self._refresh_engine_ui()
        except Exception:  # noqa: BLE001
            pass

        # --- Middle transcript (only expanding region) ---
        transcript_card = self._make_card(
            tab, title="Live Transcript", pady=(4, 4), expand=True
        )
        ctk.CTkLabel(
            transcript_card,
            text="Persona-colored STT / agent replies",
            anchor="w",
            text_color=_UI_MUTED,
        ).pack(fill="x", pady=(0, 6))
        self.transcript_box = ctk.CTkTextbox(
            transcript_card,
            wrap="word",
            font=ctk.CTkFont(family="Segoe UI", size=14),
            fg_color=_UI_CANVAS,
            corner_radius=12,
            border_width=0,
        )
        self.transcript_box.pack(fill="both", expand=True)
        self._init_persona_transcript_tags()
        # One-time welcome banner (never re-logged from Start Chat / idle loops).
        self.transcript_box.insert(
            "1.0",
            "[Donna] Type below or say Donna, then speak.\n\n",
        )
        self.transcript_box.configure(state="disabled")

    def submit_text_command(self, event=None):  # noqa: ANN001
        """Stage 8.10 — inject typed text as the next user utterance (no STT)."""
        if not self._require_engine():
            return "break" if event is not None else None
        entry = self.chat_entry
        if entry is None:
            return "break" if event is not None else None
        try:
            text = str(entry.get() or "").strip()
        except Exception:  # noqa: BLE001
            text = ""
        if not text:
            return "break" if event is not None else None
        try:
            entry.delete(0, "end")
        except Exception:  # noqa: BLE001
            pass
        # Instant distinct echo — conversation path will not re-log.
        try:
            self.log_transcript("User (Text)", text)
        except Exception:  # noqa: BLE001
            pass
        try:
            set_injected_question(text, source="text", already_logged=True)
            is_recording.set()
        except Exception as exc:  # noqa: BLE001
            log("UI", f"WARNING: silent text inject failed ({exc})")
            self._flash_engine_warning("Could not dispatch text command.")
            return "break" if event is not None else None
        preview = text if len(text) <= 120 else text[:117] + "..."
        log("UI", f'Silent text → LangGraph: "{preview}"')
        try:
            set_subtitle(f'User (Text): "{text}"')
        except Exception:  # noqa: BLE001
            pass
        return "break" if event is not None else None

    def _build_settings_tab(self, tab) -> None:  # noqa: ANN001
        """Stage 8.9.4 — merged Stats + Audio into one Settings surface."""
        try:
            tab.configure(fg_color=_UI_CANVAS)
        except Exception:  # noqa: BLE001
            pass

        row = ctk.CTkFrame(tab, fg_color="transparent")
        row.pack(fill="both", expand=True, padx=8, pady=8)
        row.grid_columnconfigure(0, weight=1)
        row.grid_columnconfigure(1, weight=1)
        row.grid_rowconfigure(0, weight=1)

        left_host = ctk.CTkFrame(row, fg_color="transparent")
        left_host.grid(row=0, column=0, sticky="nsew", padx=(4, 6), pady=4)
        right_host = ctk.CTkFrame(row, fg_color="transparent")
        right_host.grid(row=0, column=1, sticky="nsew", padx=(6, 4), pady=4)

        stats_card = self._make_card(left_host, title="Runtime", padx=4, pady=(0, 0))
        ctk.CTkLabel(
            stats_card,
            text="Wake word",
            anchor="w",
            text_color=_UI_MUTED,
        ).pack(fill="x", pady=(0, 4))
        self._settings_wake_lbl = ctk.CTkLabel(
            stats_card,
            text="Active wake word: Donna",
            anchor="w",
            justify="left",
            wraplength=320,
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#00E676",
        )
        self._settings_wake_lbl.pack(fill="x", pady=(0, 12))
        try:
            from dana.settings import is_open_window_on_startup

            open_on_start = bool(is_open_window_on_startup())
        except Exception:  # noqa: BLE001
            open_on_start = True
        self._open_window_var = ctk.BooleanVar(value=open_on_start)
        self._open_window_chk = ctk.CTkCheckBox(
            stats_card,
            text="Open window on startup",
            variable=self._open_window_var,
            command=self._on_open_window_startup_toggle,
            fg_color=_UI_ACCENT,
            hover_color=_UI_ACCENT_HOVER,
            text_color=_UI_MUTED,
        )
        self._open_window_chk.pack(anchor="w", pady=(0, 12))
        ctk.CTkLabel(
            stats_card,
            text=(
                "Audio device changes apply after Save & Apply. "
                "Pipeline mode is shown in the header and Assistive Orb."
            ),
            anchor="w",
            justify="left",
            wraplength=320,
            text_color=_UI_MUTED,
        ).pack(fill="x", pady=(0, 4))

        audio_card = self._make_card(right_host, title="Audio Devices", padx=4, pady=(0, 0))
        ctk.CTkLabel(
            audio_card, text="Microphone", anchor="w", text_color=_UI_MUTED
        ).pack(fill="x", pady=(0, 4))
        self.mic_menu = ctk.CTkOptionMenu(
            audio_card,
            values=["(none)"],
            fg_color=_UI_GHOST,
            button_color=_UI_ACCENT,
            button_hover_color=_UI_ACCENT_HOVER,
            corner_radius=10,
        )
        self.mic_menu.pack(fill="x", pady=(0, 12))
        ctk.CTkLabel(
            audio_card, text="Speaker", anchor="w", text_color=_UI_MUTED
        ).pack(fill="x", pady=(0, 4))
        self.speaker_menu = ctk.CTkOptionMenu(
            audio_card,
            values=["(none)"],
            fg_color=_UI_GHOST,
            button_color=_UI_ACCENT,
            button_hover_color=_UI_ACCENT_HOVER,
            corner_radius=10,
        )
        self.speaker_menu.pack(fill="x", pady=(0, 16))
        self.save_btn = ctk.CTkButton(
            audio_card,
            text="Save & Apply",
            command=self._save_and_apply_audio,
            corner_radius=999,
            fg_color=_UI_ACCENT,
            hover_color=_UI_ACCENT_HOVER,
            height=32,
        )
        self.save_btn.pack(anchor="w", pady=(4, 8))
        self.apply_note = ctk.CTkLabel(
            audio_card,
            text="",
            text_color=_UI_MUTED,
            anchor="w",
            wraplength=320,
            justify="left",
        )
        self.apply_note.pack(fill="x", pady=(4, 0))

        # Stage 9.3 — System Updates (git fetch / pull + pip + restart).
        updates = self._make_card(
            tab, title="System Updates", padx=8, pady=(10, 8), expand=False
        )
        ctk.CTkLabel(
            updates,
            text=(
                "Fetch from GitHub, compare revisions, then update dependencies "
                "and restart Dānā in one click."
            ),
            anchor="w",
            justify="left",
            wraplength=720,
            text_color=_UI_MUTED,
        ).pack(fill="x", pady=(0, 8))
        self._update_status_lbl = ctk.CTkLabel(
            updates,
            text="Status: idle",
            anchor="w",
            text_color=_UI_MUTED,
            font=ctk.CTkFont(size=12),
        )
        self._update_status_lbl.pack(fill="x", pady=(0, 8))
        btn_row = ctk.CTkFrame(updates, fg_color="transparent")
        btn_row.pack(fill="x")
        self._update_check_btn = ctk.CTkButton(
            btn_row,
            text="Check for Updates",
            width=160,
            height=32,
            corner_radius=999,
            fg_color=_UI_GHOST,
            hover_color="#34344A",
            border_width=1,
            border_color=_UI_CARD_BORDER,
            text_color="#E5E7EB",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._on_check_for_updates,
        )
        self._update_check_btn.pack(side="left", padx=(0, 8))
        self._update_apply_btn = ctk.CTkButton(
            btn_row,
            text="Update & Restart Dānā",
            width=190,
            height=32,
            corner_radius=999,
            fg_color="#FB8C00",
            hover_color="#EF6C00",
            text_color="#FFFFFF",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._on_apply_update_and_restart,
        )
        # Hidden until check_for_updates() reports True.
        # (pack later via _set_update_available)

    def _set_update_status(self, text: str, *, color: str | None = None) -> None:
        lbl = self._update_status_lbl
        if lbl is None:
            return
        try:
            kwargs: dict[str, Any] = {"text": str(text)}
            if color:
                kwargs["text_color"] = color
            lbl.configure(**kwargs)
        except Exception:  # noqa: BLE001
            pass

    def _set_update_available(self, available: bool) -> None:
        btn = self._update_apply_btn
        if btn is None:
            return
        try:
            if available:
                if not btn.winfo_ismapped():
                    btn.pack(side="left", padx=(0, 8))
            else:
                btn.pack_forget()
        except Exception:  # noqa: BLE001
            pass

    def _on_check_for_updates(self) -> None:
        """Stage 9.3 — background git fetch + rev compare (non-blocking UI)."""
        if self._update_busy:
            return
        self._update_busy = True
        try:
            if self._update_check_btn is not None:
                self._update_check_btn.configure(state="disabled")
        except Exception:  # noqa: BLE001
            pass
        self._set_update_status("Checking GitHub…", color=_UI_ACCENT)
        self._set_update_available(False)

        def _worker() -> None:
            available = False
            err = ""
            try:
                from dana.utils.updater import check_for_updates

                available = bool(check_for_updates())
            except Exception as exc:  # noqa: BLE001
                err = f"{type(exc).__name__}: {exc}"
                log("Updater", f"check_for_updates raised: {err}")

            def _ui() -> None:
                self._update_busy = False
                try:
                    if self._update_check_btn is not None:
                        self._update_check_btn.configure(state="normal")
                except Exception:  # noqa: BLE001
                    pass
                if err:
                    self._set_update_status(
                        f"Update check failed: {err}",
                        color="#F87171",
                    )
                    self._set_update_available(False)
                    return
                if available:
                    self._set_update_status(
                        "Update available — review then Update & Restart.",
                        color="#FB8C00",
                    )
                    self._set_update_available(True)
                else:
                    self._set_update_status(
                        "System is up to date.",
                        color="#66BB6A",
                    )
                    self._set_update_available(False)

            try:
                self.after(0, _ui)
            except Exception:  # noqa: BLE001
                _ui()

        threading.Thread(target=_worker, name="UpdateCheck", daemon=True).start()

    def _on_apply_update_and_restart(self) -> None:
        """Stage 9.3 — git pull + pip install + relaunch (background)."""
        if self._update_busy:
            return
        self._update_busy = True
        try:
            if self._update_check_btn is not None:
                self._update_check_btn.configure(state="disabled")
            if self._update_apply_btn is not None:
                self._update_apply_btn.configure(state="disabled")
        except Exception:  # noqa: BLE001
            pass
        self._set_update_status("Updating from GitHub…", color="#FB8C00")

        def _worker() -> None:
            from dana.utils.updater import apply_update_and_restart

            result = apply_update_and_restart(restart=True)

            # Only reached on failure (success calls sys.exit).
            def _ui() -> None:
                self._update_busy = False
                try:
                    if self._update_check_btn is not None:
                        self._update_check_btn.configure(state="normal")
                    if self._update_apply_btn is not None:
                        self._update_apply_btn.configure(state="normal")
                except Exception:  # noqa: BLE001
                    pass
                msg = result.message or "Update Failed."
                self._set_update_status(msg, color="#F87171")
                if result.stderr:
                    log("Updater", f"stderr:\n{result.stderr[:2000]}")

            try:
                self.after(0, _ui)
            except Exception:  # noqa: BLE001
                _ui()

        threading.Thread(target=_worker, name="UpdateApply", daemon=True).start()

    def _flash_engine_warning(self, message: str = "Please Engage Engine First.") -> None:
        """Brief Dashboard toast when a task is attempted in STANDBY."""
        lbl = self._engine_warn_lbl
        if lbl is None:
            return
        if self._engine_warn_job is not None:
            try:
                self.after_cancel(self._engine_warn_job)
            except Exception:  # noqa: BLE001
                pass
            self._engine_warn_job = None
        try:
            lbl.configure(text=str(message))
        except Exception:  # noqa: BLE001
            pass

        def _clear() -> None:
            self._engine_warn_job = None
            try:
                if self._engine_warn_lbl is not None:
                    self._engine_warn_lbl.configure(text="")
            except Exception:  # noqa: BLE001
                pass

        try:
            self._engine_warn_job = self.after(2800, _clear)
        except Exception:  # noqa: BLE001
            pass

    def _require_engine(self) -> bool:
        """Return True when engine is ACTIVE; else flash warning and return False."""
        if self.engine_active and is_engine_engaged():
            return True
        self._select_tab("Dashboard")
        self._flash_engine_warning("Please Engage Engine First.")
        return False

    def _apply_behavior_mixer_payload(self) -> dict[str, int]:
        """Push current Behavior sliders into Blackboard persona_mixer."""
        values: dict[str, int] = {}
        for key, slider in self._behavior_sliders.items():
            try:
                values[key] = int(round(float(slider.get())))
            except Exception:  # noqa: BLE001
                continue
        if not values:
            try:
                from dana.memory.blackboard import get_persona_mixer

                return dict(get_persona_mixer())
            except Exception:  # noqa: BLE001
                return {}
        try:
            from dana.memory.blackboard import set_persona_mixer

            return dict(set_persona_mixer(values))
        except Exception as exc:  # noqa: BLE001
            log("UI", f"WARNING: engage mixer apply failed ({exc})")
            return values

    def _refresh_engine_ui(self) -> None:
        """Sync ENGAGE/STANDBY chrome with ``engine_active``."""
        active = bool(self.engine_active)
        status = self._engine_status_lbl
        if status is not None:
            try:
                if active:
                    status.configure(
                        text="Status: ACTIVE (Variables Locked)",
                        text_color="#66BB6A",
                    )
                else:
                    status.configure(
                        text="Status: STANDBY (Variables Unlocked)",
                        text_color="#A1887F",
                    )
            except Exception:  # noqa: BLE001
                pass
        try:
            if self._engage_btn is not None:
                self._engage_btn.configure(state="disabled" if active else "normal")
            if self._standby_btn is not None:
                self._standby_btn.configure(state="normal" if active else "disabled")
        except Exception:  # noqa: BLE001
            pass

    def engage_engine(self) -> None:
        """Stage 8.9.7 — arm engine, apply mixer, lock Behavior sliders."""
        applied = self._apply_behavior_mixer_payload()
        self.engine_active = True
        set_engine_engaged(True)
        self._set_behavior_controls_locked(True)
        self._refresh_engine_ui()
        try:
            if self._engine_warn_lbl is not None:
                self._engine_warn_lbl.configure(text="")
        except Exception:  # noqa: BLE001
            pass
        log(
            "UI",
            f"ENGAGE engine — behavior locked mixer={list(applied.keys())}",
        )
        try:
            self.log_transcript(
                "Donna",
                "Engine ENGAGED — Behavior variables locked. Ready for chat.",
                agent_id="broker",
            )
        except Exception:  # noqa: BLE001
            pass
        try:
            if self.assistive_orb is not None:
                self.assistive_orb.refresh_controls()
        except Exception:  # noqa: BLE001
            pass
        # Lazy warm: LangGraph import only — Florence/YOLO stay JIT.
        try:
            threading.Thread(
                target=_warm_heavy_runtime_assets,
                name="HeavyWarm",
                daemon=True,
            ).start()
        except Exception:  # noqa: BLE001
            pass

    def standby_engine(self) -> None:
        """Stage 8.9.7 — soft pause loops, unlock Behavior (GUI stays alive)."""
        # Drop dictation latch if hot — mixer must be editable in STANDBY.
        if self._dictation_active:
            try:
                from dana.management.dictation import toggle_dictation_mode

                toggle_dictation_mode(False)
            except Exception:  # noqa: BLE001
                pass
            self._dictation_active = False
            try:
                self.dictation_btn.configure(
                    text="  ●  OFF  ",
                    fg_color=_UI_GHOST,
                    hover_color="#34344A",
                    border_color=_UI_CARD_BORDER,
                    text_color="#F87171",
                )
            except Exception:  # noqa: BLE001
                pass
        self.engine_active = False
        set_engine_engaged(False)
        # Soft-pause: clear pending mic latch; do NOT touch stop_event / STOP DONNA.
        try:
            is_recording.clear()
        except Exception:  # noqa: BLE001
            pass
        try:
            reset_tts_audio_state("standby_engine", ui_state="idle")
        except Exception:  # noqa: BLE001
            pass
        self._set_behavior_controls_locked(False)
        self._refresh_engine_ui()
        log("UI", "STANDBY engine — behavior unlocked (soft pause)")
        try:
            self.log_transcript(
                "Donna",
                "Engine STANDBY — Behavior variables unlocked.",
                agent_id="broker",
            )
        except Exception:  # noqa: BLE001
            pass
        try:
            if self.assistive_orb is not None:
                self.assistive_orb.refresh_controls()
        except Exception:  # noqa: BLE001
            pass

    def _dashboard_start_chat(self) -> None:
        """Quick action — focus Dashboard silent chat entry (no transcript spam)."""
        if not self._require_engine():
            return
        self._select_tab("Dashboard")
        try:
            self.lift()
            self.focus_force()
        except Exception:  # noqa: BLE001
            pass
        try:
            if self.chat_entry is not None:
                self.chat_entry.focus_set()
        except Exception:  # noqa: BLE001
            pass

    def _dashboard_trigger_dictation(self) -> None:
        """Quick action — open Dictation tab and arm the latch if cold."""
        if not self._require_engine():
            return
        self._select_tab("Dictation")
        if not self._dictation_active:
            try:
                self._toggle_dictation_mode()
            except Exception:  # noqa: BLE001
                pass

    def _dashboard_open_trace(self) -> None:
        self._select_tab("Live Trace")

    def _transcript_tk(self):
        """Return the raw ``tk.Text`` inside CTkTextbox (for tag_configure)."""
        box = getattr(self, "transcript_box", None)
        if box is None:
            return None
        return getattr(box, "_textbox", None) or getattr(box, "textbox", None)

    def _init_persona_transcript_tags(self) -> None:
        """Stage 8.5.2 — register persona styles via Tk ``tag_configure``."""
        tk_text = self._transcript_tk()
        if tk_text is None:
            return
        try:
            tk_text.tag_configure(
                "jason",
                foreground="#9c27b0",
                font=("Segoe UI", 14, "bold"),
            )
            tk_text.tag_configure(
                "llama",
                foreground="#0288d1",
                font=("Segoe UI", 14),
            )
            tk_text.tag_configure(
                "deepseek",
                foreground="#d32f2f",
                font=("Courier New", 10),
            )
            tk_text.tag_configure(
                "vision",
                foreground="#388e3c",
                font=("Segoe UI", 14),
            )
            tk_text.tag_configure(
                "typist",
                foreground="#f57c00",
                font=("Segoe UI", 14, "italic"),
            )
            # Stage 8.10 — silent Dashboard text (distinct from Whisper).
            tk_text.tag_configure(
                "user_text",
                foreground=_UI_ACCENT,
                font=("Segoe UI", 14, "italic"),
            )
            # Theme-safe default when no agent_id is provided.
            tk_text.tag_configure(
                "default",
                foreground="#E5E7EB",
                font=("Segoe UI", 14),
            )
        except Exception as exc:  # noqa: BLE001
            log("UI", f"WARNING: persona tag_configure failed ({exc})")

    @staticmethod
    def _persona_tag_for_agent(
        agent_id: str | None,
        *,
        speaker: str = "",
    ) -> str:
        """Map agent_id / speaker heuristics → transcript tag name."""
        key = (agent_id or "").strip().lower()
        aliases = {
            "jason": "jason",
            "jason_cto": "jason",
            "cto": "jason",
            "llama": "llama",
            "llama3": "llama",
            "broker": "llama",
            "chat": "llama",
            "chat_node": "llama",
            "receptionist": "llama",
            "dana": "llama",
            "donna": "llama",
            "deepseek": "deepseek",
            "moa": "deepseek",
            "moa_reasoner": "deepseek",
            "reasoner": "deepseek",
            "vision": "vision",
            "yolo": "vision",
            "florence": "vision",
            "ocr": "vision",
            "vision_agent": "vision",
            "typist": "typist",
            "ghost": "typist",
            "ghost_typist": "typist",
            "keystroke": "typist",
            "nav": "typist",
            "navigation": "typist",
        }
        if key in aliases:
            return aliases[key]
        sp = (speaker or "").strip().lower()
        if "jason" in sp:
            return "jason"
        if "deepseek" in sp or "moa" in sp:
            return "deepseek"
        if any(x in sp for x in ("vision", "yolo", "florence", "ocr")):
            return "vision"
        if any(x in sp for x in ("typist", "ghost", "keystroke", "nav")):
            return "typist"
        if sp.startswith(("dana", "donna")) or "ollama" in sp or "llama" in sp:
            return "llama"
        if sp.startswith("user") and "text" in sp:
            return "user_text"
        if sp.startswith("user"):
            return "default"
        return "default"

    def _build_dictation_tab(self, tab) -> None:  # noqa: ANN001
        """Stage 8.5 — pill toggle dictation + recent sessions card."""
        try:
            tab.configure(fg_color=_UI_CANVAS)
        except Exception:  # noqa: BLE001
            pass

        control_card = self._make_card(
            tab, title="Dictation Latch", pady=(10, 8), expand=False
        )
        ctk.CTkLabel(
            control_card,
            text=(
                "When ON, every utterance is logged with Florence OCR visual state. "
                "You can also say “dictate …” while OFF."
            ),
            anchor="w",
            text_color=_UI_MUTED,
            wraplength=640,
            justify="left",
        ).pack(fill="x", pady=(0, 12))

        controls = ctk.CTkFrame(control_card, fg_color="transparent")
        controls.pack(fill="x", pady=(0, 4))
        # Unified pill toggle — OFF: dim gray + red pill; ON: accent glow + DICTATING.
        self.dictation_btn = ctk.CTkButton(
            controls,
            text="  ●  OFF  ",
            width=168,
            height=40,
            corner_radius=999,
            fg_color=_UI_GHOST,
            hover_color="#34344A",
            border_width=1,
            border_color=_UI_CARD_BORDER,
            text_color="#F87171",
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._toggle_dictation_mode,
        )
        self.dictation_btn.pack(side="left")
        # Compatibility stub — pill on dictation_btn is the sole latch chrome
        # (Stage 8.9.4 removed the duplicate Status: label beside the toggle).
        self.dictation_status = ctk.CTkLabel(
            controls,
            text="● OFF",
            anchor="w",
            text_color="#F87171",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=_UI_CANVAS,
            corner_radius=999,
            padx=10,
            pady=4,
        )
        ctk.CTkButton(
            controls,
            text="Refresh",
            width=88,
            height=30,
            corner_radius=999,
            fg_color=_UI_CANVAS,
            hover_color="#34344A",
            border_width=1,
            border_color=_UI_CARD_BORDER,
            text_color="#D1D5DB",
            font=ctk.CTkFont(size=11),
            command=self.refresh_dictation_sessions,
        ).pack(side="right")

        sessions_card = self._make_card(tab, title="Recent sessions", pady=(4, 10))
        self.dictation_list = ctk.CTkTextbox(
            sessions_card,
            wrap="word",
            font=ctk.CTkFont(family="Consolas", size=12),
            height=280,
            fg_color=_UI_CANVAS,
            corner_radius=12,
            border_width=0,
        )
        self.dictation_list.pack(fill="both", expand=True)
        self.dictation_list.insert("1.0", "(no dictation sessions yet)\n")
        self.dictation_list.configure(state="disabled")
        # Sync latch from Blackboard (may already be on).
        try:
            from dana.memory.blackboard import is_dictation_mode

            self._set_dictation_ui(bool(is_dictation_mode()))
        except Exception:  # noqa: BLE001
            self._set_dictation_ui(False)

    def _build_behavior_tab(self, tab) -> None:  # noqa: ANN001
        """Stage 8.5 — Behavior Mixer sliders → Blackboard persona_mixer."""
        try:
            tab.configure(fg_color=_UI_CANVAS)
        except Exception:  # noqa: BLE001
            pass

        card = self._make_card(tab, title="Behavior Mixer")
        ctk.CTkLabel(
            card,
            text=(
                "Static weights for Jason / Receptionist prompts "
                "(blackboard persona_mixer). Locked while engine is ACTIVE."
            ),
            anchor="w",
            text_color=_UI_MUTED,
            wraplength=640,
            justify="left",
        ).pack(fill="x", pady=(0, 4))
        self._behavior_lock_hint = ctk.CTkLabel(
            card,
            text="Control plane: unlocked (engine STANDBY)",
            anchor="w",
            text_color="#888888",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self._behavior_lock_hint.pack(fill="x", pady=(0, 12))

        specs = (
            ("Autonomy Level", "autonomy"),
            ("Verbosity", "verbosity"),
            ("Creativity", "creativity"),
            ("Tech Depth", "technical_depth"),
        )
        try:
            from dana.memory.blackboard import (
                PERSONA_MIXER_DEFAULTS,
                get_persona_mixer,
            )

            state = get_persona_mixer()
        except Exception:  # noqa: BLE001
            PERSONA_MIXER_DEFAULTS = {
                "autonomy": 40,
                "verbosity": 50,
                "creativity": 50,
                "technical_depth": 80,
            }
            state = dict(PERSONA_MIXER_DEFAULTS)

        self._static_behavior_widgets = []
        for label, key in specs:
            frame = ctk.CTkFrame(card, fg_color="transparent")
            frame.pack(fill="x", pady=6)
            row = ctk.CTkFrame(frame, fg_color="transparent")
            row.pack(fill="x")
            ctk.CTkLabel(
                row, text=label, width=120, anchor="w", text_color="#E5E7EB"
            ).pack(side="left")
            val = int(state.get(key, PERSONA_MIXER_DEFAULTS.get(key, 50)))
            val_lbl = ctk.CTkLabel(
                row, text=str(val), width=36, text_color="#F9FAFB"
            )
            val_lbl.pack(side="right")
            self._behavior_labels[key] = val_lbl
            slider = ctk.CTkSlider(
                frame,
                from_=0,
                to=100,
                number_of_steps=100,
                progress_color=_UI_ACCENT,
                button_color="#E5E7EB",
                button_hover_color="#F9FAFB",
                fg_color=_UI_GHOST,
                command=lambda v, k=key: self._on_behavior_drag(k, v),
            )
            slider.set(float(val))
            slider.pack(fill="x", pady=(4, 0))
            slider.bind(
                "<ButtonRelease-1>",
                lambda _e, k=key: self._commit_behavior(k, force=True),
            )
            self._behavior_sliders[key] = slider
            # Stage 8.5.1 — flag as static (restart/session-sensitive) control.
            self._static_behavior_widgets.append(slider)

        self._behavior_reload_btn = ctk.CTkButton(
            card,
            text="Reload from DB",
            width=140,
            height=30,
            corner_radius=999,
            fg_color=_UI_CANVAS,
            hover_color="#34344A",
            border_width=1,
            border_color=_UI_CARD_BORDER,
            text_color="#D1D5DB",
            font=ctk.CTkFont(size=11),
            command=self._reload_behavior_sliders,
        )
        self._behavior_reload_btn.pack(pady=(16, 4), anchor="w")
        self._static_behavior_widgets.append(self._behavior_reload_btn)

    def _set_behavior_controls_locked(self, locked: bool) -> None:
        """Stage 8.5.1 — grey out static Behavior Mixer widgets when system is hot."""
        self._behavior_locked = bool(locked)
        state = "disabled" if self._behavior_locked else "normal"
        for widget in list(self._static_behavior_widgets):
            try:
                widget.configure(state=state)
            except Exception:  # noqa: BLE001
                pass
        hint = self._behavior_lock_hint
        if hint is not None:
            try:
                if self._behavior_locked:
                    hint.configure(
                        text="Control plane: LOCKED (engine ACTIVE — STANDBY to edit)",
                        text_color=("#DC2626", "#F87171"),
                    )
                else:
                    hint.configure(
                        text="Control plane: unlocked (engine STANDBY)",
                        text_color="#888888",
                    )
            except Exception:  # noqa: BLE001
                pass

    def _set_dictation_ui(self, active: bool) -> None:
        self._dictation_active = bool(active)
        try:
            if self._dictation_active:
                self.dictation_btn.configure(
                    text="  ●  DICTATING  ",
                    fg_color="#388e3c",
                    hover_color="#2E7D32",
                    border_color="#66BB6A",
                    text_color="#E8F5E9",
                )
                if self.dictation_status is not None:
                    self.dictation_status.configure(
                        text="● DICTATING",
                        text_color="#66BB6A",
                    )
            else:
                self.dictation_btn.configure(
                    text="  ●  OFF  ",
                    fg_color=_UI_GHOST,
                    hover_color="#34344A",
                    border_color=_UI_CARD_BORDER,
                    text_color="#F87171",
                )
                if self.dictation_status is not None:
                    self.dictation_status.configure(
                        text="● OFF",
                        text_color="#F87171",
                    )
        except Exception:  # noqa: BLE001
            pass
        # Stage 8.9.7 — Behavior lock follows engine ignition (not dictation alone).
        self._set_behavior_controls_locked(bool(self.engine_active))
        # Keep top status bar glowing badge in sync with latch.
        try:
            self._set_mode_indicator(self._header_mode)
        except Exception:  # noqa: BLE001
            pass
        try:
            if self.assistive_orb is not None:
                self.assistive_orb.refresh_controls()
        except Exception:  # noqa: BLE001
            pass

    def _toggle_dictation_mode(self) -> None:
        """Non-blocking GUI latch → Blackboard dictation_mode + mixer lock."""
        # Turning ON requires ENGAGE; turning OFF is always allowed.
        if (not self._dictation_active) and (not self._require_engine()):
            return
        try:
            from dana.management.dictation import toggle_dictation_mode

            active = toggle_dictation_mode(not self._dictation_active)
            self._set_dictation_ui(active)
            log(
                "UI",
                f"Dictation mode -> {'on' if active else 'off'} "
                f"(behavior_locked={self._behavior_locked})",
            )
        except Exception as exc:  # noqa: BLE001
            log("UI", f"WARNING: dictation toggle failed ({exc})")

    def refresh_dictation_sessions(self) -> None:
        """Query Blackboard dictation_sessions on the Tk thread."""
        if not self.winfo_exists():
            return
        try:
            from dana.memory.blackboard import list_dictation_sessions

            rows = list_dictation_sessions(limit=40)
        except Exception as exc:  # noqa: BLE001
            rows = []
            log("UI", f"WARNING: list_dictation_sessions failed ({exc})")
        lines: list[str] = []
        if not rows:
            lines.append("(no dictation sessions yet)\n")
        else:
            for r in rows:
                ts = str(r.get("timestamp") or "")[:19].replace("T", " ")
                cmd = str(r.get("command_text") or "").replace("\n", " ")
                if len(cmd) > 90:
                    cmd = cmd[:87] + "..."
                vis_n = len(str(r.get("visual_state_reference") or ""))
                sid = str(r.get("session_id") or "")[:8]
                st = str(r.get("status") or "recorded")
                lines.append(f"[{ts}] {sid}  {st}\n  {cmd}\n  visual_chars={vis_n}\n\n")
        try:
            self.dictation_list.configure(state="normal")
            self.dictation_list.delete("1.0", "end")
            self.dictation_list.insert("1.0", "".join(lines))
            self.dictation_list.configure(state="disabled")
        except Exception:  # noqa: BLE001
            pass

    def _on_behavior_drag(self, trait: str, value: float) -> None:
        if self._behavior_locked:
            return
        n = int(round(float(value)))
        lbl = self._behavior_labels.get(trait)
        if lbl is not None:
            try:
                lbl.configure(text=str(n))
            except Exception:  # noqa: BLE001
                pass
        now = time.monotonic()
        if now - float(self._behavior_last_write.get(trait, 0.0)) >= 0.15:
            self._commit_behavior(trait, force=False)

    def _commit_behavior(self, trait: str, *, force: bool) -> None:
        if self._behavior_locked:
            return
        slider = self._behavior_sliders.get(trait)
        if slider is None:
            return
        n = int(round(float(slider.get())))
        now = time.monotonic()
        if not force and (now - float(self._behavior_last_write.get(trait, 0.0))) < 0.15:
            return
        try:
            from dana.memory.blackboard import set_persona_trait

            set_persona_trait(trait, n)
            self._behavior_last_write[trait] = now
        except Exception as exc:  # noqa: BLE001
            log("UI", f"WARNING: behavior slider write failed ({exc})")

    def _reload_behavior_sliders(self) -> None:
        if not self.winfo_exists():
            return
        if self._behavior_locked:
            return
        try:
            from dana.memory.blackboard import (
                PERSONA_MIXER_DEFAULTS,
                get_persona_mixer,
            )

            state = get_persona_mixer()
        except Exception:  # noqa: BLE001
            return
        for key, slider in self._behavior_sliders.items():
            n = int(state.get(key, PERSONA_MIXER_DEFAULTS.get(key, 50)))
            try:
                slider.set(float(n))
                lbl = self._behavior_labels.get(key)
                if lbl is not None:
                    lbl.configure(text=str(n))
            except Exception:  # noqa: BLE001
                pass

    def process_telemetry(self) -> None:
        """Drain legacy ``gui_telemetry_queue`` on the Tk main thread (~10 Hz).

        Primary Live Trace rendering is owned by ``LiveTracePanel`` (50ms bus poll).
        This path keeps header mode / fallback TraceCells in sync.
        """
        if not self.winfo_exists():
            return
        try:
            while True:
                try:
                    event = gui_telemetry_queue.get_nowait()
                except queue.Empty:
                    break
                if not isinstance(event, dict):
                    continue
                stage = str(event.get("stage") or "stage")
                status = str(event.get("status") or "active")
                message = str(event.get("message") or stage)
                mode = event.get("mode")
                if mode:
                    self._set_mode_indicator(str(mode))
                # When LiveTracePanel is mounted, skip duplicate TraceCell rows.
                if getattr(self, "live_trace", None) is not None:
                    continue
                accent = self._mode_accent(
                    str(mode) if mode else self._header_mode
                )
                cell = self._trace_cells.get(stage)
                if cell is None:
                    cell = TraceCell(
                        self.trace_scroll,
                        stage=stage,
                        message=message,
                        status=status,
                    )
                    cell.pack(fill="x", padx=4, pady=4)
                    self._trace_cells[stage] = cell
                cell.update_status(status, message=message, accent=accent)
                try:
                    self.trace_scroll._parent_canvas.yview_moveto(1.0)  # type: ignore[attr-defined]
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass
        try:
            self.after(100, self.process_telemetry)
        except Exception:  # noqa: BLE001
            pass

    def _pulse_active_cells(self) -> None:
        if not self.winfo_exists():
            return
        self._pulse_on = not self._pulse_on
        accent = self._mode_accent()
        dim = "#4B5563"
        for cell in self._trace_cells.values():
            if cell.current_status != "active":
                continue
            try:
                cell.configure(
                    border_color=accent if self._pulse_on else dim
                )
            except Exception:  # noqa: BLE001
                pass
        try:
            self.after(500, self._pulse_active_cells)
        except Exception:  # noqa: BLE001
            pass

    def log_transcript(
        self,
        speaker: str,
        text: str,
        *,
        agent_id: str | None = None,
    ) -> None:
        """Append a speaker line to Live Transcript (Tk thread; persona tags)."""
        line = f"[{speaker}] {text}\n\n"
        tag = self._persona_tag_for_agent(agent_id, speaker=speaker)

        def _append() -> None:
            try:
                if not self.winfo_exists():
                    return
                self.transcript_box.configure(state="normal")
                # Prefer tagged insert on underlying Text; fall back to CTk API.
                tk_text = self._transcript_tk()
                if tk_text is not None:
                    tk_text.insert("end", line, (tag,))
                else:
                    try:
                        self.transcript_box.insert("end", line, tag)
                    except TypeError:
                        self.transcript_box.insert("end", line)
                self.transcript_box.see("end")
                self.transcript_box.configure(state="disabled")
            except Exception:
                pass

        try:
            self.after(0, _append)
        except Exception:
            pass

    def _reload_device_menus(self) -> None:
        devices = sd.query_devices()
        self._mic_labels = []
        self._speaker_labels = []
        self._mic_by_label = {}
        self._speaker_by_label = {}

        for idx, dev in enumerate(devices):
            name = str(dev.get("name", f"Device {idx}"))
            label = _device_menu_label(idx, name)
            if int(dev.get("max_input_channels", 0)) >= 1:
                self._mic_labels.append(label)
                self._mic_by_label[label] = idx
            if int(dev.get("max_output_channels", 0)) >= 1:
                self._speaker_labels.append(label)
                self._speaker_by_label[label] = idx

        if not self._mic_labels:
            self._mic_labels = ["(no microphones found)"]
        if not self._speaker_labels:
            self._speaker_labels = ["(no speakers found)"]

        self.mic_menu.configure(values=self._mic_labels)
        self.speaker_menu.configure(values=self._speaker_labels)

        mic_id = AUDIO_INPUT_DEVICE
        speaker_id = AUDIO_OUTPUT_DEVICE
        if mic_id is None or speaker_id is None:
            try:
                with open(SETTINGS_FILE, "r", encoding="utf-8") as fh:
                    cfg = json.load(fh)
                mic_id = int(cfg.get("mic_id", mic_id if mic_id is not None else -1))
                speaker_id = int(
                    cfg.get("speaker_id", speaker_id if speaker_id is not None else -1)
                )
            except Exception:
                pass

        mic_label = next(
            (lbl for lbl, i in self._mic_by_label.items() if i == mic_id),
            self._mic_labels[0],
        )
        speaker_label = next(
            (lbl for lbl, i in self._speaker_by_label.items() if i == speaker_id),
            self._speaker_labels[0],
        )
        self.mic_menu.set(mic_label)
        self.speaker_menu.set(speaker_label)

    def _refresh_stats(self) -> None:
        if not self.winfo_exists():
            return
        raw = get_ui_state()
        label = _UI_STATE_LABELS.get(raw, raw.title())
        try:
            self.status_value.configure(text=label)
        except Exception:  # noqa: BLE001
            pass
        wake = ", ".join(WAKEWORD_MODELS) if WAKEWORD_MODELS else "—"
        wake_disp = wake.title() if wake != "—" else wake
        try:
            self.wake_value.configure(text=f"Wake: {wake_disp}")
        except Exception:  # noqa: BLE001
            pass
        settings_wake = getattr(self, "_settings_wake_lbl", None)
        if settings_wake is not None:
            try:
                settings_wake.configure(text=f"Active wake word: {wake_disp}")
            except Exception:  # noqa: BLE001
                pass
        try:
            self._set_mode_indicator(get_donna_mode())
        except Exception:  # noqa: BLE001
            pass
        self.after(500, self._refresh_stats)

    def _save_and_apply_audio(self) -> None:
        global AUDIO_INPUT_DEVICE, AUDIO_INPUT_RATE, AUDIO_OUTPUT_DEVICE

        mic_label = self.mic_menu.get()
        speaker_label = self.speaker_menu.get()
        mic_id = self._mic_by_label.get(mic_label)
        speaker_id = self._speaker_by_label.get(speaker_label)
        if mic_id is None:
            mic_id = _parse_device_menu_label(mic_label)
        if speaker_id is None:
            speaker_id = _parse_device_menu_label(speaker_label)

        if mic_id is None or not _validate_mic_id(mic_id):
            self.apply_note.configure(text="Invalid microphone selection.")
            return
        if speaker_id is None or not _validate_speaker_id(speaker_id):
            self.apply_note.configure(text="Invalid speaker selection.")
            return

        try:
            save_audio_settings(mic_id, speaker_id)
        except OSError as exc:
            self.apply_note.configure(text=f"Could not write settings.json: {exc}")
            return

        # Speaker is read per TTS utterance — apply immediately.
        AUDIO_OUTPUT_DEVICE = speaker_id
        # Mic producer rebinds via MicIngest restart (single shared InputStream).
        AUDIO_INPUT_DEVICE = mic_id
        AUDIO_INPUT_RATE = _device_rate(mic_id)
        request_mic_ingest_restart()
        ensure_mic_ingest_thread()
        self.apply_note.configure(
            text=(
                "Saved settings.json. Speaker applied now. "
                "Microphone ingest stream is rebinding."
            )
        )
        log(
            "Audio",
            f"GUI Save & Apply -> mic={mic_id}, speaker={speaker_id}",
        )

    def show_window(self) -> None:
        self._reload_device_menus()
        self.deiconify()
        self.lift()
        self.focus_force()
        try:
            self.attributes("-topmost", True)
            self.after(200, lambda: self.attributes("-topmost", False))
        except Exception:
            pass

    def _on_open_window_startup_toggle(self) -> None:
        """Persist Settings → Open window on startup immediately."""
        try:
            from dana.settings import set_open_window_on_startup

            enabled = bool(self._open_window_var.get())
            set_open_window_on_startup(enabled)
            log(
                "UI",
                f"open_window_on_startup={'True' if enabled else 'False'} (saved)",
            )
        except Exception as exc:  # noqa: BLE001
            log("UI", f"WARNING: could not save open_window_on_startup ({exc})")

    def kill_donna_processes(self) -> dict[str, Any]:
        """Stage 8.9.2 — launch ``stop_dana.vbs`` / ``stop_dana.bat`` (non-blocking).

        Prefers the VBS silent runner (no console flash). Uses a detached
        ``subprocess.Popen`` so teardown can finish after this GUI process dies.
        """
        try:
            from dana.paths import PROJECT_ROOT

            root = Path(PROJECT_ROOT)
        except Exception:  # noqa: BLE001
            root = Path(__file__).resolve().parents[1]
        vbs = root / "stop_dana.vbs"
        bat = root / "stop_dana.bat"
        runner = vbs if vbs.is_file() else bat
        if not runner.is_file():
            msg = f"stop_dana.vbs / stop_dana.bat not found under {root}"
            log("UI", f"WARNING: {msg}")
            return {"ok": False, "error": "FileNotFoundError", "message": msg}
        try:
            creationflags = 0
            startupinfo = None
            if sys.platform == "win32":
                # Hide wscript/cmd host for stop_dana (no flashing console).
                try:
                    from dana.vault_service import windows_no_window_creationflags

                    creationflags |= windows_no_window_creationflags()
                except Exception:  # noqa: BLE001
                    creationflags |= int(
                        getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
                    )
                if hasattr(subprocess, "DETACHED_PROCESS"):
                    creationflags |= int(subprocess.DETACHED_PROCESS)
                if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
                    creationflags |= int(subprocess.CREATE_NEW_PROCESS_GROUP)
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = int(getattr(subprocess, "SW_HIDE", 0))
            # shell=True + absolute path — matches Windows .vbs/.bat launch semantics.
            proc = subprocess.Popen(  # noqa: S603
                f'"{runner}"',
                cwd=str(root),
                shell=True,
                creationflags=creationflags,
                startupinfo=startupinfo,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )
            log("UI", f"STOP DONNA — launched {runner.name} pid={proc.pid}")
            return {"ok": True, "pid": int(proc.pid), "path": str(runner)}
        except FileNotFoundError as exc:
            msg = f"Failed to launch {runner.name}: {exc}"
            log("UI", f"WARNING: {msg}")
            return {"ok": False, "error": "FileNotFoundError", "message": msg}
        except OSError as exc:
            msg = f"Failed to launch {runner.name}: {exc}"
            log("UI", f"WARNING: {msg}")
            return {"ok": False, "error": type(exc).__name__, "message": msg}

    def _on_stop_donna_clicked(self) -> None:
        """Show TERMINATING… then fire the kill switch (Tk main thread)."""
        btn = getattr(self, "stop_donna_btn", None)
        try:
            if btn is not None:
                btn.configure(
                    text="TERMINATING...",
                    state="disabled",
                    fg_color="#b71c1c",
                )
        except Exception:  # noqa: BLE001
            pass
        try:
            self.update_idletasks()
        except Exception:  # noqa: BLE001
            pass

        def _launch() -> None:
            result = self.kill_donna_processes()
            if not result.get("ok"):
                try:
                    if btn is not None:
                        btn.configure(
                            text="STOP DONNA",
                            state="normal",
                            fg_color="#d32f2f",
                        )
                except Exception:  # noqa: BLE001
                    pass
                try:
                    log(
                        "UI",
                        f"STOP DONNA aborted: {result.get('message') or 'unknown'}",
                    )
                except Exception:  # noqa: BLE001
                    pass

        try:
            # Brief paint of TERMINATING… before the batch tears us down.
            self.after(80, _launch)
        except Exception:  # noqa: BLE001
            _launch()

    def _start_assistive_orb(self) -> None:
        """Stage 8.7 — spawn frameless topmost orb on the Tk main thread."""
        if self.assistive_orb is not None:
            return
        try:
            from dana.ui.assistive_orb import AssistiveTouchOrb

            def _active_agent() -> str:
                try:
                    from dana.audio.multi_voice_tts import get_active_tts_agent

                    return str(get_active_tts_agent() or "broker")
                except Exception:  # noqa: BLE001
                    return "broker"

            self.assistive_orb = AssistiveTouchOrb(
                self,
                on_toggle_dictation=self._toggle_dictation_mode,
                on_open_dashboard=self.show_window,
                on_approve_ticket=self._orb_approve_ticket,
                on_deny_ticket=self._orb_deny_ticket,
                dictation_getter=lambda: bool(self._dictation_active),
                mode_getter=lambda: str(self._header_mode or "chat"),
                accent_getter=lambda: self._mode_accent(),
                agent_getter=_active_agent,
            )
        except Exception as exc:  # noqa: BLE001
            log("UI", f"WARNING: AssistiveTouch orb failed to start ({exc})")
            self.assistive_orb = None

    def open_github_issue(
        self,
        ticket_content: dict[str, Any] | str | None = None,
        jason_critique: str = "",
    ) -> str:
        """Stage 8.9.3 — open a pre-filled GitHub issue in the default browser."""
        try:
            from dana.middleware.hitl_ticket import get_pending
            from dana.ui.github_escalation import open_github_issue as _open

            pending = ticket_content if ticket_content is not None else get_pending()
            return _open(pending, jason_critique or str((pending or {}).get("jason_critique") or ""))
        except Exception as exc:  # noqa: BLE001
            log("UI", f"WARNING: open_github_issue failed ({exc})")
            return ""

    def _orb_approve_ticket(self) -> None:
        try:
            from dana.middleware.hitl_ticket import submit_decision

            submit_decision(True, action="approve")
        except Exception as exc:  # noqa: BLE001
            log("UI", f"WARNING: orb approve failed ({exc})")
        try:
            if getattr(self, "live_trace", None) is not None:
                self.live_trace._set_hitl_visible(False)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass

    def _orb_deny_ticket(self) -> None:
        try:
            from dana.middleware.hitl_ticket import submit_decision

            submit_decision(False, action="deny")
        except Exception as exc:  # noqa: BLE001
            log("UI", f"WARNING: orb deny failed ({exc})")
        try:
            if getattr(self, "live_trace", None) is not None:
                self.live_trace._set_hitl_visible(False)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass

    def _on_close_to_tray(self) -> None:
        # Dashboard hides; AssistiveTouch orb stays visible as the always-on control.
        self.withdraw()


# ---------------------------------------------------------------------------
# Main - agent loop (background) + GUI main thread
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline Donna voice agent (YOLO eyes + Ollama brain + Whisper + OpenWakeWord).",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help=(
            "Allow Hugging Face / OpenWakeWord to download/cache model weights (online). "
            "Omit for offline HF loads where supported. Distil-Whisper STT still "
            "auto-falls back to download on a local cache miss."
        ),
    )
    parser.add_argument(
        "--reset-audio",
        action="store_true",
        help="Delete settings.json and re-run the interactive mic/speaker setup.",
    )
    parser.add_argument(
        "--reset-vault",
        action="store_true",
        help="Delete donna_memory.enc and exit so the next run creates a fresh vault.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Headless mode: skip CustomTkinter Live Trace / tray UI.",
    )
    return parser.parse_args()


def agent_loop(args: Optional[argparse.Namespace] = None) -> int:
    global AUDIO_INPUT_DEVICE, AUDIO_INPUT_RATE, AUDIO_OUTPUT_DEVICE

    if args is None:
        args = parse_args()
    local_files_only = not args.download

    log("Main", "=== CAMGRASPER Donna voice agent ===")
    import torch

    try:
        torch.backends.mkldnn.enabled = True
    except Exception:
        pass
    log("Main", f"MKLDNN enabled: {torch.backends.mkldnn.enabled}")
    if local_files_only:
        log("Main", "Mode: OFFLINE HF loads (local_files_only=True)")
    else:
        log(
            "Main",
            "Mode: DOWNLOAD - will fetch Whisper/OWW weights if missing.",
        )

    # Unlock encrypted long-term memory before loading models / audio threads.
    try:
        unlock_donna_memory()
    except SystemExit:
        stop_event.set()
        return 1

    if args.reset_audio:
        try:
            os.remove(SETTINGS_FILE)
            print("[Audio] Deleted settings.json — interactive setup will run.", flush=True)
            log("Audio", "Removed settings.json (--reset-audio).")
        except FileNotFoundError:
            print("[Audio] settings.json not found — setup will run anyway.", flush=True)
            log("Audio", "settings.json already absent (--reset-audio).")
        except OSError as exc:
            log("Audio", f"WARNING: could not remove settings.json: {exc}")
            print(f"[Audio] WARNING: could not remove settings.json: {exc}", flush=True)

    list_input_devices()
    list_output_devices()
    mic_id, speaker_id, mic_rate = load_audio_settings()
    log(
        "Audio",
        f"Audio pipeline: mic={mic_id} speaker={speaker_id} rate={mic_rate}",
    )
    AUDIO_INPUT_DEVICE = mic_id
    AUDIO_OUTPUT_DEVICE = speaker_id
    AUDIO_INPUT_RATE = mic_rate

    # Verify the configured mic is live; do not silently switch to another device.
    AUDIO_INPUT_DEVICE, AUDIO_INPUT_RATE = ensure_live_mic(
        AUDIO_INPUT_DEVICE,
        AUDIO_INPUT_RATE,
        allow_fallback=False,
    )
    if AUDIO_INPUT_DEVICE is None:
        log("Main", "Aborting: configured microphone is not usable.")
        stop_event.set()
        return 2
    if not _validate_speaker_id(int(AUDIO_OUTPUT_DEVICE)):
        log("Main", "Aborting: configured speaker is not usable.")
        stop_event.set()
        return 2

    device = select_device()
    dtype = select_dtype(device)

    # Single PortAudio InputStream producer before wake/VAD consumers start.
    ensure_mic_ingest_thread()
    if not mic_ingest_ready.wait(timeout=8.0):
        log(
            "Main",
            "WARNING: MicIngest not ready after 8s — wake/VAD will wait on the queue",
        )

    threads = [
        threading.Thread(
            target=tracker_worker,
            name="Tracker",
            args=(device,),
            daemon=True,
        ),
        threading.Thread(target=wakeword_worker, name="WakeWord", daemon=True),
        threading.Thread(
            target=conversation_worker,
            name="Conversation",
            args=(local_files_only, device, dtype),
            daemon=True,
        ),
        threading.Thread(
            target=input_txt_ingest_worker,
            name="InputIngest",
            daemon=True,
        ),
        threading.Thread(target=tts_worker, name="TTSWorker", daemon=True),
    ]
    for t in threads:
        t.start()
        log("Main", f"Started thread: {t.name}")

    # Stage 7.2 — hardware panic button (F12 / DONNA_KILL_HOTKEY).
    try:
        from dana.middleware.kill_switch import start_kill_switch_listener

        if start_kill_switch_listener():
            log("Main", "Kill switch listener armed (default hotkey F12)")
        else:
            log("Main", "Kill switch listener not started")
    except Exception as exc:  # noqa: BLE001
        log("Main", f"WARNING: kill switch listener failed: {exc}")

    # Persist conversational mode so system jobs cannot steal it later.
    try:
        set_donna_mode(get_donna_mode(), as_voice=True)
        log("Main", f"Voice session mode seeded: {get_donna_mode()}")
    except Exception as exc:  # noqa: BLE001
        log("Main", f"WARNING: voice session mode seed failed: {exc}")

    # Sidekick supervisor — eyes (vision_poller) + hands (actuator_executor).
    try:
        from dana.middleware.sidekick_supervisor import start_sidekick_supervisor

        if start_sidekick_supervisor(as_thread=True):
            log("Main", "Sidekick supervisor started (vision_poller + actuator_executor)")
        else:
            log("Main", "Sidekick supervisor not started")
    except Exception as exc:  # noqa: BLE001
        log("Main", f"WARNING: sidekick supervisor failed: {exc}")

    log(
        "Main",
        "Donna is ready. Say 'Donna' to wake. | Tray Quit / Ctrl+C=quit",
    )
    try:
        from dana.telemetry import start_dashboard_thread

        start_dashboard_thread()
        log("Main", "Live telemetry dashboard started (CAMGRASPER/dashboard.md)")
    except Exception as exc:  # noqa: BLE001
        log("Main", f"WARNING: dashboard thread failed: {exc}")
    try:
        from dana.settings import (
            get_assistant_language,
            get_whisper_language,
            is_dynamic_tool_synthesis_enabled,
            load_donna_settings,
        )

        load_donna_settings(force_reload=True)
        log(
            "Main",
            f"Language lock: assistant={get_assistant_language()} "
            f"whisper={get_whisper_language()} (English-only release)",
        )
        log(
            "Main",
            f"Tool Forge: enable_dynamic_tool_synthesis="
            f"{is_dynamic_tool_synthesis_enabled()}",
        )
    except Exception as exc:  # noqa: BLE001
        log("Main", f"WARNING: language settings unread ({exc})")

    try:
        while not stop_event.is_set():
            # File trigger for automation / remote ask.
            # Empty file => start mic listening.
            # Non-empty file => inject that text as the user transcript (skip mic).
            if os.path.isfile(TRIGGER_FILE):
                try:
                    with open(TRIGGER_FILE, "r", encoding="utf-8-sig") as fh:
                        injected = fh.read().strip()
                except OSError:
                    injected = ""
                if get_ui_state() == "idle" and not is_recording.is_set():
                    if not is_engine_engaged():
                        # Soft STANDBY — leave trigger file for retry after ENGAGE.
                        pass
                    else:
                        try:
                            os.remove(TRIGGER_FILE)
                        except OSError:
                            pass
                        if injected:
                            log("Main", f"File trigger inject -> \"{injected}\"")
                            set_injected_question(injected)
                            is_recording.set()
                        else:
                            log("Main", "File trigger -> start listening")
                            clear_injected_question()
                            is_recording.set()
                # else: leave file in place until idle so automation retries cleanly

            # PortAudio / PaErrorCode from Audio thread → soft restart before freeze.
            if audio_hardware_fault.is_set():
                detail = consume_audio_hardware_fault()
                soft_recover_audio_hardware(detail)

            time.sleep(0.1)
    except KeyboardInterrupt:
        log("Main", "Quit requested (Ctrl+C).")
    finally:
        stop_event.set()
        try:
            camera_tool.release()
            screen_tool.release()
        except Exception:
            pass
        try:
            speech_queue.put_nowait(None)
        except queue.Full:
            pass
        for t in threads:
            t.join(timeout=5.0)
        log("Main", "Shutdown complete.")

    return 0


def main() -> int:
    """GUI owns the main thread; agent_loop + tray run in background daemons."""
    global _gui_instance, _agent_loop_thread

    try:
        from dana.stdio_boot import ensure_stdio

        ensure_stdio()
    except Exception:
        pass

    # Cwd-independent asset paths (onnx, logs, yolov8n.pt, settings.json, …).
    chdir_project_root()

    # Workspace dirs: logs/tracker/execution_jail/custom_tools + legacy migrate.
    try:
        from dana.workspace import ensure_donna_workspace

        ensure_donna_workspace(migrate=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[Workspace] WARNING: ensure_donna_workspace failed: {exc}")

    # Startup: sweep RESOLVED/FAILED tickets into patch_ledger_archive.md.
    try:
        from dana.tools.archive_ledger import archive_completed_tickets

        archive_msg = archive_completed_tickets()
        print(f"[Ledger] {archive_msg}")
    except Exception as exc:  # noqa: BLE001
        print(f"[Ledger] WARNING: archive_completed_tickets failed: {exc}")

    # Best-effort UTF-8 stdout so non-ASCII logs do not crash worker threads.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    log_path = enable_runtime_file_logging()
    log("Main", f"PROJECT_ROOT={PROJECT_ROOT}")
    try:
        from dana.paths import DONNA_WORKSPACE

        log("Main", f"DONNA_WORKSPACE={DONNA_WORKSPACE}")
    except Exception:
        pass

    # Dual-registry boot: Git-tracked general + Desktop custom (ephemeral).
    try:
        from dana.tools.registry import (
            load_custom_tools_from_disk,
            load_general_tools_from_disk,
        )

        loaded_general = load_general_tools_from_disk()
        if loaded_general:
            log("Main", f"Loaded general tools from disk: {loaded_general!r}")
        loaded_custom = load_custom_tools_from_disk()
        if loaded_custom:
            log(
                "Main",
                f"Loaded custom/ephemeral tools from disk: {loaded_custom!r}",
            )
    except Exception as exc:  # noqa: BLE001
        log("Main", f"WARNING: load_general/custom tools from disk failed: {exc}")

    enforce_singleton()
    args = parse_args()

    if args.reset_vault:
        reset_donna_vault()
        return 0

    log("Main", f"Runtime log -> {log_path}")
    log("Main", f"Conversation log (latest) -> {CONVERSATION_LOG_PATH}")

    # Headless: no CustomTkinter / tray — agent loop owns the process.
    if getattr(args, "no_gui", False):
        try:
            from dana.ui.trace_bus import get_trace_bus

            get_trace_bus().set_enabled(False)
        except Exception:  # noqa: BLE001
            pass
        log("Main", "Headless mode (--no-gui): Live Trace UI disabled.")
        return agent_loop(args)

    # Stage 8.9.7 — GUI paints in STANDBY first; heavy agent loop deferred.
    gui = DonnaGUI()
    _gui_instance = gui
    # Boot visible by default; honor open_window_on_startup (tray-only when False).
    try:
        from dana.settings import is_open_window_on_startup

        _show_on_boot = bool(is_open_window_on_startup())
    except Exception:  # noqa: BLE001
        _show_on_boot = True
    if _show_on_boot:
        try:
            gui.after(150, gui.show_window)
        except Exception:  # noqa: BLE001
            try:
                gui.show_window()
            except Exception:  # noqa: BLE001
                pass
    else:
        log(
            "Main",
            "open_window_on_startup=False — tray/orb only (dashboard hidden).",
        )
    try:
        emit_trace(
            "Boot",
            "completed",
            "Live Trace UI online (STANDBY — engine unlocked)",
            mode=get_donna_mode(),
        )
        emit_trace("STT", "idle", "STT: deferred until AgentLoop warm")
        emit_trace("Router", "idle", "Router: waiting for ENGAGE")
    except Exception:  # noqa: BLE001
        pass

    # Keep tray-close UX: X hides the window; Quit lives on the system tray.
    try:
        gui.protocol("WM_DELETE_WINDOW", gui._on_close_to_tray)
    except Exception:
        pass

    _install_signal_handlers(gui)

    def _boot_agent_loop() -> None:
        """Deferred background start so Standby chrome paints instantly."""
        global _agent_loop_thread
        if _agent_loop_thread is not None:
            return
        _agent_loop_thread = threading.Thread(
            target=agent_loop,
            name="AgentLoop",
            kwargs={"args": args},
            daemon=True,
        )
        _agent_loop_thread.start()
        threading.Thread(
            target=run_system_tray,
            name="SystemTray",
            args=(gui,),
            daemon=True,
        ).start()
        try:
            emit_trace("STT", "active", "STT: Whisper pipeline arming")
            emit_trace("Router", "active", "Router: waiting for turn")
        except Exception:  # noqa: BLE001
            pass

    try:
        # ~350ms lets Tk draw ENGAGE/STANDBY before Whisper/tracker threads.
        gui.after(350, _boot_agent_loop)
    except Exception:  # noqa: BLE001
        _boot_agent_loop()

    try:
        gui.mainloop()
    except KeyboardInterrupt:
        log("Main", "Interrupted — shutting down.")
        stop_event.set()
        try:
            speech_queue.put_nowait(None)
        except queue.Full:
            pass
    finally:
        _shutdown_agent_threads(join_timeout=8.0)
        icon = _tray_icon
        if icon is not None:
            try:
                icon.stop()
            except Exception:
                pass
        _gui_instance = None
        _agent_loop_thread = None
        log("Main", "GUI closed.")

    return 0


if __name__ == "__main__":
    try:
        from dana.stdio_boot import ensure_stdio

        ensure_stdio()
    except Exception:
        pass
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        try:
            _shutdown_agent_threads(join_timeout=5.0)
        except Exception:
            pass
        sys.exit(130)
