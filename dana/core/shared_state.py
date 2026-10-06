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
initialization — ``vault_client`` is instantiated at import time here exactly
as it was in core_agent.py.
"""

from __future__ import annotations

import queue
import re
import threading
from typing import Any, Callable, Optional


from dana.audio.speech_state import (  # noqa: F401 — re-exported: legacy readers use state.<name>
    ARABIC_SCRIPT_RE,
    WHISPER_AMBIENT_SILENT,
    WHISPER_HALLUCINATIONS,
    _CODE_FENCE_TTS_RE,
    _CODE_FENCE_TTS_UNCLOSED_RE,
    _PUNCT_OR_SPACE_ONLY_RE,
    _TTS_MD_MARKERS_RE,
    stop_event,
    whisper_bundle_lock,
    whisper_ready,
)
from dana.logging import log_debug
from dana.audio.tts_manager import get_tts_manager as _get_tts_manager
from dana.audio.tts_worker import get_tts_worker as _get_tts_worker
from dana.paths import SETTINGS_PATH as _SETTINGS_PATH, TRIGGER_ASK_PATH
from dana.secure_memory import default_vault_path
from dana.vault_service import VaultClient

# ---------------------------------------------------------------------------
# Vision (tracker thread <-> tool-dispatch <-> GUI)
# ---------------------------------------------------------------------------

latest_frame_lock = threading.Lock()

latest_dets_lock = threading.Lock()

active_vision_lock = threading.Lock()

conversation_history_lock = threading.Lock()


# Short-term spatial memory so flickering detections still answer "where is X?"
spatial_memory_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Mic / VAD
# ---------------------------------------------------------------------------

# Wake word / .trigger_ask starts a conversational turn.
is_recording = threading.Event()
# Legacy name kept for call sites: producer-ready / stream healthy.
wake_mic_released = threading.Event()
wake_mic_released.set()
mic_ingest_ready = threading.Event()


# ---------------------------------------------------------------------------
# Conversation / UI telemetry
# ---------------------------------------------------------------------------


# Optional injected question from .trigger_ask file contents (automation / tests).
injected_question_lock = threading.Lock()

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
# Max phrases allowed to pile up while a stream already owns the speaker.
_SPEECH_MAX_PENDING_WHILE_BUSY = 3
# Set while tts_worker is actively rendering/playing TTS (mic must stay idle).
tts_busy = threading.Event()
# Barge-in: set by VAD when user speaks over TTS; checked in the playback chunk loop.
tts_interrupt_event = threading.Event()
# Process-wide barge-in controller (shares ``tts_interrupt_event``).
_tts_barge = _get_tts_worker(barge_in_event=tts_interrupt_event)

# ---------------------------------------------------------------------------
# VAD / engine lifecycle
# ---------------------------------------------------------------------------

# True while ``record_utterance`` owns the microphone (barge-in watcher must stand down).
vad_capture_active = threading.Event()
# Cleared until conversation_worker's Ollama warm-up finishes (gates wake-word arming).
ollama_ready = threading.Event()
_active_mid_task_lock = threading.Lock()
_boot_ready_audio_lock = threading.Lock()
# Set when the TTS spooler is drained and nothing is playing.
speech_idle = threading.Event()
speech_idle.set()
# One "Let me check" per conversational turn (router + ReAct share this).
_tool_working_ack_sent = threading.Event()
_audio_hardware_fault_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Paths / vault
# ---------------------------------------------------------------------------

TRIGGER_FILE = str(TRIGGER_ASK_PATH)
MEMORY_FILE = default_vault_path()
vault_client = VaultClient()  # reassigned (unlock flow replaces this with a fresh instance)


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

_ui_state_listeners_lock = threading.Lock()

_transcript_listeners_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Vault unlock prompt (background AgentLoop thread <-> GUI main thread)
# ---------------------------------------------------------------------------
#
# unlock_dana_memory() (core_agent.py) runs on the background "AgentLoop"
# thread. When no env var / OS keyring credential unlocks the vault, it must
# not just SystemExit the thread silently while a Dashboard is attached — the
# GUI owner registers a listener here, shows its own modal (on the Tk main
# thread, via its own after()/thread-safe scheduling), and calls
# supply_vault_unlock_response() once the user submits or cancels. This mirrors
# the ui_state/transcript listener pattern above: shared_state never imports
# GUI code, it just brokers the request/response handoff.

_vault_prompt_listeners_lock = threading.Lock()


# ---------------------------------------------------------------------------
# One-off GUI actions from agent-loop code (HITL approval card, dictation
# session refresh) — same decoupling as ui_state/transcript above: the
# agent-loop bucket must not import DanaGUI / touch ``_gui_instance``
# directly, and the listener (registered by whoever owns the GUI) is
# responsible for its own thread-safe ``.after()`` hand-off to Tk.
# ---------------------------------------------------------------------------

_spec_approval_listeners_lock = threading.Lock()

_dictation_sessions_listeners_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Feature/plugin toggles (dana.features.feature_manager -> GUI owner) — same
# decoupling as above: this module never imports feature_manager or GUI code,
# it only brokers the fire-and-forget notification; the listener is
# responsible for its own thread-safe ``.after()`` hand-off to Tk.
# ---------------------------------------------------------------------------

_feature_flags_listeners_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Live Trace telemetry (background threads -> Tk main thread via Queue only)
# ---------------------------------------------------------------------------
# Producer (emit_trace, any thread) and consumer (dana.ui.app_gui's
# DanaGUI/TraceCell, Tk main thread) both live in this module's callers now;
# emit_trace itself moved here in Phase 7 since its only dependencies
# (the queue and the icon table below) already lived here.

