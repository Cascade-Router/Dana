"""Cross-thread/cross-module shared state, extracted verbatim from ``dana.core_agent``.

Phase 1 of the core_agent.py decomposition (see the approved plan). This
module holds the same objects under the same names that used to live in
``dana.core_agent``'s "# Shared state" block — locks, events, queues, and a
handful of plain values — read/written across all five background threads
(vision tracker, wake-word, conversation/STT, TTS, and the GUI/CLI).

Two access patterns, deliberately kept distinct:

- **Mutate-in-place objects** (``threading.Event``/``Lock``, ``queue.Queue``,
  a ``dict``/``list`` only ever changed via ``d[k]=v``/``.append()``/``.clear()``)
  are safe to import by bare name (``from dana.core.shared_state import
  is_recording``) — a consumer's ``global is_recording`` + in-place mutation
  never rebinds the name, so it stays the same object as this module's.
- **Reassigned values** (anything set via a bare ``name = new_value``
  somewhere, e.g. ``ui_state = "listening"`` or ``latest_frame = frame``)
  are NOT safe as a bare import: reassigning a name inside another module
  only rebinds that module's own copy, silently diverging from this
  module's attribute. Every such name is commented ``# reassigned`` below;
  callers MUST go through ``import dana.core.shared_state as state`` and
  read/write ``state.name``, never a bare imported name.

This phase intentionally does not change behavior, locking, or eagerness of
initialization — ``screen_tool``/``camera_tool``/``vault_client`` are
instantiated at import time here exactly as they were in core_agent.py.
Converting them to lazy singletons is deferred to the phase that moves their
respective bootstrap logic (vision tracker / vault bootstrap), where the
change can be reviewed alongside the code that actually initializes them.
"""

from __future__ import annotations

import queue
import re
import threading
from typing import Any, Callable, Optional, Union

import numpy as np
from spatial_context import SPATIAL_AGGREGATOR

from dana.audio.audio_pipeline import AudioRouter
from dana.logging import log_debug
from dana.audio.noise_floor import (  # noqa: F401  (re-exported for callers)
    ABSOLUTE_MIN_SPEECH_FLOOR,
    NOISE_FLOOR_MULTIPLIER,
    compute_ambient_baseline,
    compute_dynamic_speech_floor,
)
from dana.audio.tts_manager import get_tts_manager as _get_tts_manager
from dana.audio.tts_worker import get_tts_worker as _get_tts_worker
from dana.audio.wake_poller import WakePoller
from dana.paths import SETTINGS_PATH as _SETTINGS_PATH, TRIGGER_ASK_PATH
from dana.secure_memory import SecureMemory, default_vault_path
from dana.vault_service import VaultClient
from dana.vision_tools import ScreenAgent, VideoAgent

# ---------------------------------------------------------------------------
# Vision (tracker thread <-> tool-dispatch <-> GUI)
# ---------------------------------------------------------------------------

latest_frame_lock = threading.Lock()
latest_frame: Optional[np.ndarray] = None  # BGR, 640x480  # reassigned

latest_dets_lock = threading.Lock()
latest_dets: list[tuple[np.ndarray, str, float]] = []  # reassigned

# Dynamic vision tool calling (ScreenAgent / VideoAgent).
screen_tool = ScreenAgent()
camera_tool = VideoAgent()
active_vision_lock = threading.Lock()
active_vision_tool: Union[ScreenAgent, VideoAgent] = screen_tool  # reassigned

# Sliding-window chat memory for Ollama (last 6 messages = 3 user + 3 assistant).
conversation_history: list[dict[str, str]] = []
conversation_history_lock = threading.Lock()
HISTORY_MAX_MESSAGES = 6

# Decrypted long-term profile (AES vault); injected into Ollama system prompt.
dana_profile: dict[str, Any] = {}  # reassigned
dana_vault: Optional["SecureMemory"] = None  # reassigned
# High-frequency identity keys prefetched post-unlock (skip ReAct vault tools).
VAULT_HOT_CACHE: dict[str, str] = {}  # reassigned

# Optional Arabic-script detection (unused for English-only TTS routing).
ARABIC_SCRIPT_RE = re.compile(r"[؀-ۿ]")

# Short-term spatial memory so flickering detections still answer "where is X?"
spatial_memory_lock = threading.Lock()
spatial_memory: dict[str, float] = {}  # label -> last_seen monotonic time
SPATIAL_MEMORY_SEC = 2.5

# ---------------------------------------------------------------------------
# Mic / VAD
# ---------------------------------------------------------------------------

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
_mic_ingest_thread: Optional[threading.Thread] = None  # reassigned

# Resolved device selection (startup-resolved, touched by audio + GUI settings + CLI).
AUDIO_INPUT_DEVICE: Optional[int] = None  # reassigned
AUDIO_INPUT_RATE: int = 16000  # reassigned  # mirrors core_agent.SAMPLE_RATE
AUDIO_OUTPUT_DEVICE: Optional[int] = None  # reassigned

# ---------------------------------------------------------------------------
# Whisper STT
# ---------------------------------------------------------------------------

# Shared Whisper bundle for wake-phrase verification (set by conversation_worker).
whisper_bundle_lock = threading.Lock()
whisper_bundle: Optional[tuple[Any, Any, Any, Any]] = None  # reassigned
# Set when background Whisper load finishes (success or failure).
whisper_ready = threading.Event()
_whisper_load_error: Optional[str] = None  # reassigned

# ---------------------------------------------------------------------------
# Conversation / UI telemetry
# ---------------------------------------------------------------------------

# Conversation phase: idle | listening | followup | transcribing | thinking
ui_state_lock = threading.Lock()
ui_state = "idle"  # reassigned

# Latest Whisper transcript (logged for headless debugging).
subtitle_lock = threading.Lock()
subtitle_text = ""  # reassigned

# Optional injected question from .trigger_ask file contents (automation / tests).
injected_question_lock = threading.Lock()
injected_question: Optional[str] = None  # reassigned
_injected_source: str = "inject"  # reassigned
_injected_already_logged: bool = False  # reassigned

# ---------------------------------------------------------------------------
# TTS Output Spooler — producers push (text, interruptible); consumer owns PortAudio.
# ---------------------------------------------------------------------------

# ``interruptible=False`` = UI ack exemption (no self-barge-in on speaker bleed).
# Stage 8.8 — spool items are (text, interruptible, agent_id).
# Canonical owner: ``dana.audio.tts_manager.TTSManager`` (shared speech_queue).
_tts_manager = _get_tts_manager()
tts_queue: queue.Queue[Optional[tuple[str, bool, str]]] = _tts_manager.speech_queue
speech_queue = tts_queue  # backward-compatible alias / TTSManager.speech_queue
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
_tts_barge = _get_tts_worker(barge_in_event=tts_interrupt_event)
_tts_worker_thread: Optional[threading.Thread] = None  # reassigned

# ---------------------------------------------------------------------------
# VAD / engine lifecycle
# ---------------------------------------------------------------------------

# True while ``record_utterance`` owns the microphone (barge-in watcher must stand down).
vad_capture_active = threading.Event()
# Set by text/chat ingest to abort active Silero VAD without waiting for max_timeout.
vad_abort_event = threading.Event()
# Set when startup mic probe is below DEAD_MIC_RMS_FLOOR (Text-Only / Quiet Mic).
quiet_mic_mode = threading.Event()
# Cleared until conversation_worker's Ollama warm-up finishes (gates wake-word arming).
ollama_ready = threading.Event()
# Soft-drop audit: last chat mid-task prompt awaiting completion (VAD timeout).
_active_mid_task_prompt: str | None = None  # reassigned
_active_mid_task_lock = threading.Lock()
# Boot coordination: ready audio plays only when all three are set.
piper_voices_ready = threading.Event()
wakeword_armed = threading.Event()
# Stage 8.9.7 — soft engine ignition (clear = STANDBY; set = ACTIVE).
# Distinct from stop_event / STOP DANA (hard exit).
engine_engaged = threading.Event()
_boot_ready_audio_lock = threading.Lock()
_boot_ready_audio_played = False  # reassigned
# Shared OpenWakeWord model for stream-barge during TTS (set by wakeword_worker).
_shared_wakeword_model: Any = None  # reassigned
_shared_wakeword_token: str = "dana"  # reassigned
_dual_wake_router: Optional[AudioRouter] = None  # reassigned
_dual_wake_poller: Optional[WakePoller] = None  # reassigned
# Set when the TTS spooler is drained and nothing is playing.
speech_idle = threading.Event()
speech_idle.set()
# One "Let me check" per conversational turn (router + ReAct share this).
_tool_working_ack_sent = threading.Event()
stop_event = threading.Event()
# PortAudio / hardware fault signal: Audio thread -> Main (soft recovery).
audio_hardware_fault = threading.Event()
_audio_hardware_fault_lock = threading.Lock()
_audio_hardware_fault_detail: str = ""  # reassigned

# ---------------------------------------------------------------------------
# Paths / vault
# ---------------------------------------------------------------------------

TRIGGER_FILE = str(TRIGGER_ASK_PATH)
SETTINGS_FILE = str(_SETTINGS_PATH)
MEMORY_FILE = default_vault_path()
MEMORY_SALT = b"dana_secure_salt"
PBKDF2_ITERATIONS = 390_000
vault_client = VaultClient()  # reassigned (unlock flow replaces this with a fresh instance)

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

_CODE_FENCE_TTS_RE = re.compile(r"```[\w+-]*\n?[\s\S]*?```", re.MULTILINE)
_CODE_FENCE_TTS_UNCLOSED_RE = re.compile(r"```[\w+-]*\n?[\s\S]*$", re.MULTILINE)
_TTS_MD_MARKERS_RE = re.compile(r"`+|\*{1,3}|_{2,}")
_PUNCT_OR_SPACE_ONLY_RE = re.compile(r"^[\s\W_]+$", re.UNICODE)

# ---------------------------------------------------------------------------
# UI-state / transcript event hooks
# ---------------------------------------------------------------------------
#
# Audio, agent-loop, and vision code all need to announce state changes (the
# GUI dashboard, the tray icon, live transcript panes) — but must never import
# DanaGUI or tray functions directly, since those still live in core_agent.py
# (and later move to dana/ui/*). Emitters call notify_*() below; whoever owns
# the actual GUI/tray widget registers a listener at its own init time. This
# decouples "something changed state" from "here is how the GUI shows it",
# so neither side needs to import the other.

_ui_state_listeners: list[Callable[[str], None]] = []
_ui_state_listeners_lock = threading.Lock()

_transcript_listeners: list[Callable[[str, str, Optional[str]], None]] = []
_transcript_listeners_lock = threading.Lock()


def register_ui_state_listener(fn: Callable[[str], None]) -> None:
    """Called by the GUI/tray owner (currently core_agent.py) at init time."""
    with _ui_state_listeners_lock:
        if fn not in _ui_state_listeners:
            _ui_state_listeners.append(fn)


def unregister_ui_state_listener(fn: Callable[[str], None]) -> None:
    with _ui_state_listeners_lock:
        try:
            _ui_state_listeners.remove(fn)
        except ValueError:
            pass


def _notify_ui_state_listeners(state: str) -> None:
    with _ui_state_listeners_lock:
        listeners = list(_ui_state_listeners)
    for fn in listeners:
        try:
            fn(state)
        except Exception:  # noqa: BLE001
            pass


def register_transcript_listener(fn: Callable[[str, str, Optional[str]], None]) -> None:
    """Called by the GUI dashboard owner at init time (unregister on teardown)."""
    with _transcript_listeners_lock:
        if fn not in _transcript_listeners:
            _transcript_listeners.append(fn)


def unregister_transcript_listener(fn: Callable[[str, str, Optional[str]], None]) -> None:
    with _transcript_listeners_lock:
        try:
            _transcript_listeners.remove(fn)
        except ValueError:
            pass


def _notify_transcript_listeners(speaker: str, text: str, agent_id: Optional[str]) -> None:
    with _transcript_listeners_lock:
        listeners = list(_transcript_listeners)
    for fn in listeners:
        try:
            fn(speaker, text, agent_id)
        except Exception:  # noqa: BLE001
            pass


def emit_live_transcript(
    speaker: str,
    text: str,
    *,
    agent_id: str | None = None,
) -> None:
    """Thread-safe bridge from audio/LLM workers into the Dashboard transcript."""
    _notify_transcript_listeners(speaker, text, agent_id)


def set_ui_state(state: str) -> None:
    """Conversation phase: idle | listening | followup | transcribing | thinking."""
    global ui_state
    with ui_state_lock:
        ui_state = state
    SPATIAL_AGGREGATOR.set_ui_state(state)
    log_debug("UI", f"State -> {state}")
    # Visual cue: tray icon turns green while VAD is actively listening.
    _notify_ui_state_listeners(state)
    # Dashboard STATE_CHANGE (mic / system status); headless-safe.
    try:
        from dana.ui.status_bus import emit_state_change

        if state in ("listening", "followup"):
            emit_state_change("listening")
        elif state in ("transcribing", "thinking"):
            emit_state_change("processing")
        elif state == "idle":
            emit_state_change("idle")
    except Exception:  # noqa: BLE001
        pass


def get_ui_state() -> str:
    with ui_state_lock:
        return ui_state


def set_subtitle(text: str) -> None:
    """Latest Whisper transcript (logged for headless debugging)."""
    global subtitle_text
    with subtitle_lock:
        subtitle_text = text
    if text:
        log_debug("UI", f"Subtitle -> {text}")
