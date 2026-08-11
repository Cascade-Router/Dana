"""
CAMGRASPER - Offline Voice-to-Voice Dana assistant.

Pipeline (4 threads + agent keep-alive loop + GUI main thread):
  1. Tracker   - YOLOv8n rolling buffer (~2s) from active_vision_tool + live ROI overlay
  2. WakeWord  - OpenWakeWord custom "dana.onnx" on mic @ 16 kHz
  3. Conversation - VAD -> Whisper STT -> tool_router -> YOLO + Ollama LLM -> TTS
  4. Audio     - offline TTS via piper-tts (en_US-hfc_female-medium)

UI:
  - Windows system tray icon (Open Settings / Quit)
  - CustomTkinter Live Trace window (header mode + pipeline TraceCells; Stats/Audio tabs)

Dual-engine cascade:
  - Eyes: YOLO spatial labels from ScreenAgent / VideoAgent
  - Brain: local Ollama chat API (qwen2.5-coder:7b)

Audio devices are configured via settings.json (interactive first-run setup or GUI).
Long-term user profile is stored in an AES-256 encrypted vault (dana_memory.enc).

Triggers:
  - Say 'Dana' to wake (then multi-turn follow-up without wake word)
  - Create .trigger_ask (empty = listen; non-empty = inject transcript)
  - Enqueue tasks in CAMGRASPER/execution_jail/task_queue.json then wake — bypasses Whisper
  - Tray Quit / Ctrl+C to quit

Setup:
  python -m dana.core_agent --download   # one-time Whisper/OWW cache
  python scripts/diagnostics/audio_diagnostics.py  # verify mic/speaker/TTS
  python -m dana.core_agent
"""

from __future__ import annotations

import argparse
import asyncio
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
            "dana.assistant.desktop.v1"
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

import tkinter as tk
import customtkinter as ctk
import numpy as np
import pystray
import requests
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from PIL import Image, ImageDraw
from dana.vision_tools import ScreenAgent, VideoAgent
from dana.paths import (
    ENV_PATH,
    PROJECT_ROOT as _PATHS_PROJECT_ROOT,
    TEXT_INJECTION_PATH,
    WAKEWORD_ONNX,
    chdir_project_root,
    ensure_project_root_on_syspath,
    resolve_wakeword_onnx,
)
# TEXT_INJECTION_PATH kept for legacy migrate; ingestion uses task_queue.json.

# Keep bootstrap string and dana.paths.PROJECT_ROOT in sync.
PROJECT_ROOT = os.path.abspath(str(_PATHS_PROJECT_ROOT))
ensure_project_root_on_syspath()
import ingest  # noqa: E402 — scripts/ingest.py → task_queue.json converter
from dana.secure_memory import SecureMemory
from dana.vault_service import VaultClient
from dana.tools import ToolCall, ToolValidationError, get_broker
from dana.agentic import (
    CHAT_MEMORY_CLEARED_ACK,
    CHAT_MEMORY_WINDOW_K,
    REACT_MAX_ITERS,
    build_lightweight_chat_system_prompt,
    chat_memory_size,
    clear_chat_memory,
    get_dana_mode,
    mode_switch_spoken_ack,
    parse_clear_chat_memory,
    parse_mode_switch,
    requires_tool_graph,
    run_lightweight_chat,
    run_react_loop,
    set_dana_mode,
)
from dana.logging import (
    CONVERSATION_LOG_PATH,
    enable_runtime_file_logging,
    log,
    log_conversation,
    log_debug,
    log_exception,
)
from dana.audio.audio_pipeline import AudioRouter
from dana.audio.wake_poller import WakePoller
from dana.core.telemetry import AsyncRingBuffer, NeuralStreamEmitter
from dana.utils.adaptive_poller import AdaptivePoller
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
_gui_instance: Optional["DanaGUI"] = None
_agent_loop_thread: Optional[threading.Thread] = None

# Live Trace telemetry (background threads → Tk main thread via Queue only).
gui_telemetry_queue: queue.Queue = queue.Queue()
# Shared dark-slate tokens (dana.ui.theme) — keep local aliases for call sites.
try:
    from dana.ui import theme as _UI_THEME

    _UI_CANVAS = _UI_THEME.BG
    _UI_CARD = _UI_THEME.CARD
    _UI_CARD_BORDER = _UI_THEME.BORDER
    _UI_GHOST = _UI_THEME.GHOST
    _UI_MUTED = _UI_THEME.MUTED
    _UI_TEXT = _UI_THEME.TEXT
    _UI_ACCENT = _UI_THEME.ACCENT
    _UI_ACCENT_HOVER = _UI_THEME.ACCENT_HOVER
    _UI_EMERALD = _UI_THEME.EMERALD
    _UI_EMERALD_HOVER = _UI_THEME.EMERALD_HOVER
    _UI_ROSE = _UI_THEME.ROSE
    _UI_ROSE_HOVER = _UI_THEME.ROSE_HOVER
    _UI_AMBER = _UI_THEME.AMBER
except Exception:  # noqa: BLE001
    _UI_CANVAS = "#0a0e17"
    _UI_CARD = "#131b2e"
    _UI_CARD_BORDER = "#1e293b"
    _UI_GHOST = "#1e293b"
    _UI_MUTED = "#94A3B8"
    _UI_TEXT = "#F8FAFC"
    _UI_ACCENT = "#10b981"
    _UI_ACCENT_HOVER = "#059669"
    _UI_EMERALD = "#10B981"
    _UI_EMERALD_HOVER = "#059669"
    _UI_ROSE = "#F43F5E"
    _UI_ROSE_HOVER = "#E11D48"
    _UI_AMBER = "#F59E0B"

# Master telemetry dispatcher cadences (seconds) — see _master_telemetry_tick.
# One AdaptivePoller-driven heartbeat replaces five independent self.after()
# loops; each consumer below still only fires at its own interval, tracked
# via monotonic elapsed time so a backed-off (idle) heartbeat never drifts.
_TELEMETRY_CADENCES_S: dict[str, float] = {
    "live_trace": 0.08,
    "process_telemetry": 0.10,
    "state_changes": 0.10,
    "dag_monitor": 0.25,
    "task_tracker": 0.40,
}

_TRACE_MODE_COLORS: dict[str, str] = {
    "chat": _UI_EMERALD,
    "developer": _UI_AMBER,
    "vision": _UI_ACCENT,
    "research": _UI_AMBER,
    "dictation": "#A855F7",
}
_TRACE_IDLE_COLOR = _UI_MUTED
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
    """Bind a local TCP port so only one Dana process can run."""
    global _singleton_socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # Do not set SO_REUSEADDR — that would defeat the singleton check.
        sock.bind(("127.0.0.1", _SINGLETON_PORT))
        sock.listen(1)
    except OSError:
        print("[System] Dana is already running.", flush=True)
        try:
            sock.close()
        except OSError:
            pass
        sys.exit(0)
    _singleton_socket = sock


# ---------------------------------------------------------------------------
# Piper TTS model download (offline voices)
# ---------------------------------------------------------------------------

DANA_WAKEWORD_ONNX = str(WAKEWORD_ONNX)




from openwakeword.model import Model as OpenWakeWordModel


# Phase-3-prep: shared, cross-bucket app-wide constants now live in
# dana.core.constants (moved out ahead of the audio/agent-loop split so
# neither bucket has to import config from the other). Re-imported in full
# so every existing bare-name reference below keeps working unchanged.
from dana.core.constants import (  # noqa: E402,F401
    BARGE_IN_AMBIENT_MULT,
    BARGE_IN_CHUNK_MS,
    BARGE_IN_MIN_SPEECH_MS,
    BARGE_IN_PLAYBACK_GRACE_MS,
    BARGE_IN_RMS,
    BARGE_IN_SETTLE_MS,
    BARGE_IN_SILERO_CONSEC_FRAMES,
    BARGE_IN_SILERO_THRESHOLD,
    CAMERA_KEYWORDS,
    DC_BLOCKER_R,
    DEAD_MIC_RMS_FLOOR,
    FOLLOWUP_FLUSH_SEC,
    FOLLOWUP_VAD_MAX_SECONDS,
    FRAME_SIZE,
    MIC_AMBIENT_DEAD_RMS,
    MIN_SPEECH_RMS,
    MODEL_ID,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT_SEC,
    OLLAMA_URL,
    POST_ACK_FLUSH_SEC,
    POST_ACK_IGNORE_ONSET_MS,
    POST_ACK_SETTLE_SEC,
    POST_ACK_VAD_GRACE_SEC,
    SAMPLE_RATE,
    SCREEN_KEYWORDS,
    SENDGRID_MAIL_URL,
    STREAM_BARGE_RMS,
    TRACKER_BUFFER_INTERVAL_S,
    TRACKER_SLEEP_SEC,
    TTS_CHUNK_MAX_CHARS,
    TTS_IDLE_WAIT_TIMEOUT,
    TTS_UTTERANCE_MAX_SECONDS,
    VAD_FRAME_MS,
    VAD_FRAME_SAMPLES,
    VAD_MAX_SECONDS,
    VAD_MIN_SPEECH_MS,
    VAD_PRE_ROLL_FRAMES,
    VAD_SILENCE_MS,
    WAKEWORD_MODELS,
    WAKE_CHUNK,
    WAKE_COOLDOWN_SEC,
    WAKE_MIN_CONSECUTIVE,
    WAKE_ONSET_BELOW,
    WAKE_ONSET_LOOKBACK,
    WAKE_PHRASE_ALIASES,
    WAKE_PHRASE_REJECT,
    WAKE_PHRASE_TOKENS,
    WAKE_PHRASE_VERIFY,
    WAKE_PHRASE_WINDOW_CHUNKS,
    WAKE_THRESHOLD,
    WHISPER_GAIN_RMS_CEIL,
    WHISPER_ID,
    WHISPER_INITIAL_PROMPT,
    WHISPER_LANGUAGE,
    WHISPER_MAX_GAIN,
    WHISPER_MAX_WORDS_PER_SEC,
    WHISPER_MIN_RMS_FOR_GAIN,
    WHISPER_TARGET_RMS,
    WHISPER_TASK,
    YOLO_CONF,
    YOLO_WEIGHTS,
    _STT_NAME_FIXES,
)

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

# Phase 1 of the core_agent.py decomposition (see the approved decomposition
# plan): this block now lives in dana.core.shared_state, imported back here
# in full so every existing ``global X`` / bare-name read-write site below
# keeps working unchanged. Names commented "reassigned" there are NOT yet
# safe to read via ``dana.core.shared_state.X`` from outside this file --
# that only becomes true once the functions that reassign them move out in
# a later phase.

from dana.core.shared_state import (  # noqa: E402,F401
    ARABIC_SCRIPT_RE,
    HISTORY_MAX_MESSAGES,
    MEMORY_FILE,
    MEMORY_SALT,
    PBKDF2_ITERATIONS,
    SETTINGS_FILE,
    SPATIAL_MEMORY_SEC,
    TRIGGER_FILE,
    WHISPER_AMBIENT_SILENT,
    WHISPER_HALLUCINATIONS,
    _audio_hardware_fault_lock,
    _boot_ready_audio_lock,
    _CODE_FENCE_TTS_RE,
    _CODE_FENCE_TTS_UNCLOSED_RE,
    _dual_wake_poller,
    _dual_wake_router,
    _injected_already_logged,
    _injected_source,
    _PUNCT_OR_SPACE_ONLY_RE,
    _shared_wakeword_token,
    _SPEECH_MAX_PENDING_WHILE_BUSY,
    _speech_enqueue_lock,
    _tool_working_ack_sent,
    _TTS_HOLD_FOR_VAD_MAX_S,
    _TTS_MD_MARKERS_RE,
    _tts_barge,
    _tts_enqueue_lock,
    _tts_manager,
    active_vision_lock,
    audio_buffer_queue,
    audio_hardware_fault,
    camera_tool,
    conversation_history,
    conversation_history_lock,
    dana_vault,
    engine_engaged,
    injected_question,
    injected_question_lock,
    is_recording,
    latest_dets_lock,
    latest_frame_lock,
    mic_ingest_ready,
    mic_ingest_restart,
    mic_lock,
    ollama_ready,
    piper_voices_ready,
    playback_lock,
    quiet_mic_mode,
    screen_tool,
    spatial_memory,
    spatial_memory_lock,
    speech_idle,
    speech_queue,
    stop_event,
    tts_busy,
    tts_interrupt_event,
    tts_queue,
    vad_abort_event,
    vad_capture_active,
    wake_mic_released,
    wakeword_armed,
    whisper_bundle_lock,
    whisper_ready,
)
# Reassigned cross-module values (whisper_bundle, AUDIO_INPUT_DEVICE/RATE,
# AUDIO_OUTPUT_DEVICE, VAULT_HOT_CACHE, _active_mid_task_prompt,
# active_vision_tool, latest_frame, latest_dets, dana_profile, vault_client,
# _shared_wakeword_model, ...) are NOT bare-imported above -- their
# readers/writers now span multiple modules, so every read/write goes
# through this module reference instead of a stale import-time snapshot.
import dana.core.shared_state as state

# Phase 3 of the core_agent.py decomposition (see the plan): the audio bucket
# (DcBlocker, noise floor, Whisper STT, mic ingestion, TTS text/queue
# pipeline, Piper synthesis + playback) now lives in dana.audio.*, imported
# back here in full so every existing bare-name call site below keeps
# working unchanged.
from dana.audio.dc_blocker import DcBlocker, remove_dc_offset
from dana.audio.noise_floor import (  # noqa: F401
    audio_buffer_rms,
    calibrate_noise_floor,
    get_dynamic_speech_floor,
    should_skip_wake_predict,
)
from dana.audio.stt import (  # noqa: F401
    _sanitize_whisper_generation_config,
    _whisper_generate_kwargs,
    _whisper_initial_prompt_text,
    _whisper_is_english_only,
    _whisper_prompt_ids,
    correct_known_stt_names,
    ensure_whisper_bundle,
    is_punctuation_or_whitespace_only,
    is_silent_non_speech_transcript,
    is_whisper_hallucination,
    is_whisper_prompt_echo,
    is_whisper_rate_hallucination,
    load_whisper,
    prepare_audio_for_whisper,
    resample_audio,
    resample_to_16k,
    start_whisper_background_load,
    transcribe_audio,
)
from dana.audio.mic_input import (  # noqa: F401
    _close_input_stream,
    _device_rate,
    _open_input_stream_with_timeout,
    _parse_settings_device_id,
    _read_input_stream_with_timeout,
    _run_with_timeout,
    _validate_mic_id,
    _validate_speaker_id,
    adaptive_barge_in_rms,
    ensure_live_mic,
    ensure_mic_ingest_thread,
    find_steelseries_mic,
    find_steelseries_speaker,
    flush_audio_buffer_queue,
    flush_input_buffer,
    get_mic_frame,
    interactive_audio_setup,
    list_input_devices,
    list_output_devices,
    load_audio_settings,
    mic_ingest_worker,
    pick_input_device,
    pick_output_device,
    probe_mic_rms,
    record_utterance,
    request_mic_ingest_restart,
    save_audio_settings,
)
from dana.audio.tts_manager import (  # noqa: F401
    _parse_tts_spool_item,
    chunk_text_for_tts,
    enqueue_speech_impl as enqueue_speech,
    flush_speech_queue,
    flush_tts_queue,
    sanitize_text_for_tts,
    strip_code_blocks_for_tts,
)
from dana.audio.tts_worker import (  # noqa: F401
    AUDIO_CACHE_DIR,
    DEFAULT_PIPER_ONNX,
    PIPER_EN_JSON,
    PIPER_EN_ONNX,
    PIPER_LENGTH_SCALE,
    PIPER_MODEL_URLS,
    PIPER_TEMP_WAV,
    PIPER_VOICE_ID,
    TTS_MODELS_DIR,
    _boost_audio_thread_priority,
    _bind_tts_barge_controller,
    _CANNED_UX_WAV_FILES,
    _device_output_samplerate,
    _download_file,
    _is_portaudio_error,
    _normalize_canned_ux_key,
    _PIPER_HF_BASE,
    _PIPER_LEGACY_JSON,
    _PIPER_LEGACY_ONNX,
    _PIPER_LEGACY_VOICE_ID,
    _PIPER_VOICE_RELPATHS,
    _piper_file_ready,
    _piper_hf_urls,
    _piper_voice_cache,
    _play_cached_wav,
    _play_pcm_interruptible,
    _play_ready_chime,
    _resample_pcm,
    _safe_sd_stop,
    _speak_with_timeout,
    _synthesize_and_play,
    audio_worker,
    barge_in_watch,
    canned_ux_cache_path,
    consume_audio_hardware_fault,
    download_piper_models,
    ensure_canned_ux_audio_cache,
    get_piper_voice,
    half_duplex_mic_drop,
    interrupt_tts,
    maybe_play_boot_ready_audio,
    piper_model_path_for_text,
    report_audio_hardware_fault,
    reset_tts_audio_state,
    soft_recover_audio_hardware,
    speak_text,
    synthesize_to_file,
    tts_worker,
    wait_for_speech_idle,
    _wait_tts_clear_of_user_speech,
)


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


# emit_live_transcript / set_ui_state / get_ui_state / set_subtitle now live in
# dana.core.shared_state, which dispatches UI-state and transcript changes to
# registered listeners instead of touching _gui_instance / update_tray_icon_for_state
# directly (decouples emitters from whoever owns the GUI/tray widget — see
# shared_state.register_ui_state_listener / register_transcript_listener, and
# the registration calls near _gui_instance assignment / run_system_tray below).
from dana.core.shared_state import (  # noqa: E402,F401
    emit_live_transcript,
    get_ui_state,
    has_vault_prompt_listener,
    notify_dictation_sessions_changed,
    notify_spec_approval_requested,
    notify_vault_unlocked,
    register_dictation_sessions_listener,
    register_spec_approval_listener,
    register_transcript_listener,
    register_ui_state_listener,
    register_vault_prompt_listener,
    request_vault_unlock,
    set_subtitle,
    set_ui_state,
    supply_vault_unlock_response,
    unregister_dictation_sessions_listener,
    unregister_spec_approval_listener,
    unregister_transcript_listener,
    unregister_ui_state_listener,
    unregister_vault_prompt_listener,
)




def abort_vad_listening(*, reason: str = "text_override") -> None:
    """Abort active Silero VAD so text/chat can run without the 10s audio timeout."""
    vad_abort_event.set()
    if vad_capture_active.is_set():
        log("Audio", f"VAD abort requested ({reason}) — resetting voice to standby")
        try:
            from dana.db_core import log_vad_state

            log_vad_state("abort", detail=str(reason or ""), route="standby")
        except Exception:  # noqa: BLE001
            pass


def prioritize_text_input(*, reason: str = "text") -> None:
    """Text-chat override: abort VAD listening immediately (queue/inject still processed)."""
    abort_vad_listening(reason=reason)


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
    # Text always preempts an in-flight mic listen (no 10s VAD wait).
    prioritize_text_input(reason=f"inject:{source or 'inject'}")
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
        # Allow wake-prefixed forms: "Dana, clear context"
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
    client = client if client is not None else state.vault_client
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
    state.VAULT_HOT_CACHE = {
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
    global dana_vault

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
                state.AUDIO_OUTPUT_DEVICE,
            )
        except Exception:
            pass

    try:
        state.vault_client.lock_vault()
        print(
            "[Security] Vault session purged. Password will be required next boot.",
            flush=True,
        )
        log("Security", "Vault daemon RAM key + sessions purged.")
    except Exception as exc:  # noqa: BLE001
        print(f"[Security] Failed to contact daemon for purge: {exc}", flush=True)
        log("Security", f"ERROR: lockdown purge failed ({exc})")

    try:
        if dana_vault is not None:
            dana_vault.lock()
    except Exception:  # noqa: BLE001
        pass
    dana_vault = None
    state.dana_profile = {}
    state.VAULT_HOT_CACHE = {}

    # Force immediate termination of all threads and GUI.
    os._exit(0)




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
            "execute_powershell": "  .",
            "write_to_file": "  .",
            "execute_command": "  .",
            "execute_python_script": "  .",
            "get_sandbox_job_status": "  .",
            "fetch_webpage": "  .",
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
            "execute_powershell": "Running that in PowerShell.",
            "write_to_file": "Okay — writing that file.",
            "execute_command": "Okay — running that command.",
            "execute_python_script": "Okay — running that Python script in the sandbox.",
            "get_sandbox_job_status": "Let me check that sandbox job.",
            "fetch_webpage": "Let me open that page.",
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
                    "Your Dana Backup Recovery Key is below.\n"
                    "Store it offline. Anyone with this key can unlock Dana's memory vault.\n\n"
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


def unlock_dana_memory() -> SecureMemory:
    """Unlock via in-RAM vault daemon (Option B). Password only if resume fails.

    Daemon handshake (try_resume_session) ALWAYS runs before any password prompt.
    """
    global dana_vault
    state.vault_client = VaultClient()
    try:
        state.vault_client.ensure_ready()
    except Exception as exc:  # noqa: BLE001
        print(f"[Memory Error] Vault daemon unavailable: {exc}", flush=True)
        log("Memory", f"ERROR vault daemon: {exc}")
        raise SystemExit(1) from exc

    # --- HARD GATE: resume first; password ONLY in the else branch ---
    resumed = False
    try:
        resumed = bool(state.vault_client.try_resume_session())
    except Exception as exc:  # noqa: BLE001
        log("Memory", f"try_resume_session failed ({exc})")
        resumed = False

    if resumed:
        state.dana_profile = dict(state.vault_client.profile)
        vault = SecureMemory(path=MEMORY_FILE)
        try:
            from dana.vault_service import _rpc
            import base64 as _b64

            resp = _rpc(
                {
                    "op": "export_data_key",
                    "session_token": state.vault_client.session_token,
                },
                timeout=5.0,
            )
            if resp.get("ok"):
                key = _b64.urlsafe_b64decode(resp["data_key_b64"].encode("ascii"))
                vault.unlock_with_data_key(key)
                vault.profile = dict(state.dana_profile)
        except Exception as exc:  # noqa: BLE001
            log("Memory", f"WARNING: local vault hydrate skipped ({exc})")
        dana_vault = vault
        populate_vault_hot_cache(state.vault_client)
        log(
            "Memory",
            f"Vault unlocked via daemon session "
            f"(keys={len(state.dana_profile)}; token cached in RAM daemon).",
        )
        print(
            "[Memory] Vault unlocked via daemon session (keys cached in RAM).",
            flush=True,
        )
        notify_vault_unlocked()
        return vault

    # else: daemon locked → resolve credential (env → keyring → TTY prompt →
    # GUI modal, when a Dashboard has registered to handle vault prompts).
    from dana.tools.vault import VaultCredentialsMissing, _get_master_key

    prompt = "Enter Master Password (or pasted Recovery Key) to unlock Dānā: "
    vault_exists = os.path.isfile(MEMORY_FILE)

    def _resolve_password(gui_reason: str) -> str:
        try:
            return _get_master_key(prompt=prompt)
        except VaultCredentialsMissing:
            if not has_vault_prompt_listener():
                raise
            gui_password = request_vault_unlock(gui_reason)
            if not gui_password:
                raise VaultCredentialsMissing(
                    "Vault unlock cancelled from the Dashboard prompt."
                ) from None
            return gui_password

    try:
        password = _resolve_password(
            "Dana's memory vault needs your Master Password (or Recovery Key) "
            "to unlock — no credential found in the environment or OS keyring."
        )
    except VaultCredentialsMissing as exc:
        print(f"[Memory Error] {exc}", flush=True)
        log("Memory", f"ERROR: {exc}")
        raise SystemExit(1) from exc

    if not vault_exists:
        recovery_key = secrets.token_urlsafe(32)
        try:
            state.dana_profile = state.vault_client.unlock(
                password, create=True, recovery_key=recovery_key
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[Memory Error] Could not create vault: {exc}", flush=True)
            log("Memory", f"ERROR creating vault: {exc}")
            raise SystemExit(1) from exc
        email_recovery_key(recovery_key)
    else:
        # Wrong password from the GUI modal gets a few retries (a mistyped
        # passcode should not tear down the whole AgentLoop thread); CLI/TTY
        # callers with no listener registered keep the original fail-fast.
        max_attempts = 3
        attempt = 0
        while True:
            attempt += 1
            try:
                state.dana_profile = state.vault_client.unlock(password, create=False)
                break
            except Exception as exc:  # noqa: BLE001
                print(f"[Memory Error] {exc}", flush=True)
                log("Memory", f"ERROR: {exc} (attempt {attempt}/{max_attempts})")
                if not has_vault_prompt_listener() or attempt >= max_attempts:
                    raise SystemExit(1) from exc
                password = request_vault_unlock(
                    f"Wrong Master Password / Recovery Key ({exc}). "
                    "Try again to unlock Dana's memory vault."
                )
                if not password:
                    raise SystemExit(1) from exc

    vault = SecureMemory(path=MEMORY_FILE)
    try:
        from dana.vault_service import _rpc
        import base64 as _b64

        resp = _rpc(
            {
                "op": "export_data_key",
                "session_token": state.vault_client.session_token,
            },
            timeout=5.0,
        )
        if resp.get("ok"):
            key = _b64.urlsafe_b64decode(resp["data_key_b64"].encode("ascii"))
            vault.unlock_with_data_key(key)
            vault.profile = dict(state.dana_profile)
    except Exception as exc:  # noqa: BLE001
        log("Memory", f"WARNING: local vault hydrate failed ({exc})")

    dana_vault = vault
    populate_vault_hot_cache(state.vault_client)
    log(
        "Memory",
        f"Vault unlocked via daemon session "
        f"(keys={len(state.dana_profile)}; token cached in RAM daemon).",
    )
    notify_vault_unlocked()
    return vault


def reset_dana_vault() -> None:
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



# ---------------------------------------------------------------------------
# Agent-loop / conversational FSM bucket (Phase 4 of the core_agent.py
# decomposition) now lives in dana.core.agent_loop, imported back here in
# full so every existing bare-name call site below keeps working unchanged.
# ---------------------------------------------------------------------------
from dana.core.agent_loop import (  # noqa: F401
    _ask_ollama_messages_unlocked,
    _clear_mid_task_prompt,
    _drop_mid_task_on_vad_timeout,
    _mark_mid_task_prompt,
    _register_chat_soft_drop,
    ask_ollama_messages,
    build_dana_system_prompt,
    commit_agentic_turn,
    conversation_worker,
    execute_tool_call,
    tool_router,
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
        f"active_vision_tool.get_frame() (maxlen=60 thumbnails).",
    )

    frames = 0
    while not stop_event.is_set():
        # Wait out the ~1s cadence before grabbing (avoids busy mss polling).
        if not should_push_frame(interval_s=buf_interval):
            rem = buf_interval - seconds_since_last_push()
            if rem == float("inf"):
                rem = 0.0
            time.sleep(max(0.05, min(TRACKER_SLEEP_SEC, max(0.0, rem))))
            continue

        with active_vision_lock:
            tool = state.active_vision_tool
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
            state.latest_frame = frame

        try:
            mode = get_dana_mode()
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
                state.latest_dets = dets

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

        # Phase 3 — paced screen_history extraction (~12s; OCR, not every frame).
        if tool_name == "screen":
            try:
                from dana.tools.vision import maybe_extract_screen_history

                maybe_extract_screen_history()
            except Exception:  # noqa: BLE001
                pass

        frames += 1
        if frames % 15 == 0:
            from dana.tracker import buffer_len

            log_debug(
                "Tracker",
                f"Alive - {frames} samples via {tool_name}; "
                f"buffer={buffer_len()}/60; last=[{format_class_list(labels)}]",
            )

        time.sleep(TRACKER_SLEEP_SEC)

    log("Tracker", "Stopped.")
    try:
        from dana.vision.overlay import clear_roi

        clear_roi()
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Thread 3 - Wake word (OpenWakeWord — Dana only)
# ---------------------------------------------------------------------------

def wake_score_hit(
    prediction: dict[str, Any],
    *,
    require_token: str = "dana",
    threshold: float = WAKE_THRESHOLD,
) -> Optional[str]:
    """Return matched wake-word key if score crosses threshold for require_token."""
    token = (require_token or "dana").lower()
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


def _wake_text_matches_dana(normalized: str) -> bool:
    """True if Whisper text is Dana or a known Dana mishearing."""
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
    """Second gate: Whisper must hear Dana / Hey Dana in the wake buffer.

    When WAKE_PHRASE_VERIFY is False, openWakeWord score+onset alone starts the session.
    """
    if not WAKE_PHRASE_VERIFY:
        return True
    if audio_16k.size < SAMPLE_RATE // 4:
        return False

    with whisper_bundle_lock:
        bundle = state.whisper_bundle
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
    if _wake_text_matches_dana(normalized):
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


def _trigger_dual_wake_event() -> None:
    global _dual_wake_router, _dual_wake_poller
    if _dual_wake_router is not None:
        try:
            _dual_wake_router.flush()
        except Exception:
            pass
    if _dual_wake_poller is not None:
        try:
            _dual_wake_poller.router.whisper_queue = asyncio.Queue()
            _dual_wake_poller.router.standard_queue = asyncio.Queue()
        except Exception:
            pass
    if not is_engine_engaged():
        return
    if not wakeword_armed.is_set():
        return
    log("WakeWord", "Dual-threshold wake trigger fired")
    is_recording.set()
    cooldown_until = time.monotonic() + WAKE_COOLDOWN_SEC
    try:
        if state._shared_wakeword_model is not None:
            state._shared_wakeword_model.reset()
    except Exception:
        pass


def wakeword_worker() -> None:
    _nt_hide_console_if_mp_child()

    # Prefer custom Dana wake model (dana.onnx), with legacy dana.onnx fallback.
    wake_token = "dana"
    model_paths: list[str]
    onnx_path = str(resolve_wakeword_onnx())
    if os.path.isfile(onnx_path):
        model_paths = [onnx_path]
        log("WakeWord", f"Loading OpenWakeWord model: {onnx_path}")
        try:
            from openwakeword.utils import download_models

            # Feature extractors (melspec/embedding) live in the package resources.
            download_models()
        except Exception as exc:  # noqa: BLE001
            log("WakeWord", f"WARNING: could not refresh OWW feature models ({exc})")
    else:
        print(
            "[Warning] dana.onnx / dana.onnx not found! Temporary Alexa wake-word "
            "enabled for mic debugging. Say 'Alexa' (not Dana). Place dana.onnx "
            "(or legacy dana.onnx) in assets/models/ to switch back.",
            flush=True,
        )
        log(
            "WakeWord",
            "WARNING: wake-word ONNX missing — temporary Alexa debug wake-word active.",
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
    global _shared_wakeword_token
    state._shared_wakeword_model = oww
    _shared_wakeword_token = wake_token
    if wake_token == "dana":
        print("Say 'Dana' to wake.", flush=True)
        listen_msg = "Dana"
    else:
        print("DEBUG: Say 'Alexa' to wake (temporary until dana.onnx is added).", flush=True)
        listen_msg = "Alexa (debug)"
    log(
        "WakeWord",
        f"Listening for {listen_msg} on mic [{state.AUDIO_INPUT_DEVICE}] @ {state.AUDIO_INPUT_RATE} Hz "
        "(or .trigger_ask)...",
    )
    print(
        f"[Debug] WakeWord using device={state.AUDIO_INPUT_DEVICE} "
        f"rate={state.AUDIO_INPUT_RATE} threshold={WAKE_THRESHOLD} "
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
    if quiet_mic_mode.is_set():
        log(
            "WakeWord",
            "Quiet Mic / Text-Only mode — wake-word polling disarmed "
            "(awaiting physical mic energy or text trigger)",
        )
        wakeword_armed.clear()
    else:
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
        chunk_rms = audio_buffer_rms(audio)
        skip_wake = should_skip_wake_predict(chunk_rms)

        now = time.monotonic()
        if now >= next_rms_log:
            log_debug("Debug", f"Live Mic RMS: {chunk_rms:.6f}")
            # TEMP WIRETAP: wake-path diagnosis
            try:
                _dev_idx = state.AUDIO_INPUT_DEVICE
                if _dev_idx is not None:
                    _dev_name = str(sd.query_devices(int(_dev_idx)).get("name", "?"))
                else:
                    _dev_name = str(sd.query_devices(None).get("name", "?"))
            except Exception:
                _dev_idx = state.AUDIO_INPUT_DEVICE
                _dev_name = "?"
            print(
                f"!!! [WIRETAP] Mic: [{_dev_idx}] {_dev_name} "
                f"| Stream Rate: {state.AUDIO_INPUT_RATE} "
                f"| Array Shape: {audio.shape} "
                f"| RMS: {chunk_rms:.6f} "
                f"| Skipping: {skip_wake}",
                flush=True,
            )
            next_rms_log = now + 2.0

        # Dead / virtual mic silence: skip OpenWakeWord to prevent phantom wakes.
        if skip_wake:
            consecutive_hits = 0
            continue

        # Quiet-mic fallback: stay disarmed until physical energy (or text) arrives.
        if quiet_mic_mode.is_set():
            quiet_mic_mode.clear()
            wakeword_armed.set()
            log(
                "WakeWord",
                f"Physical mic energy detected (rms={chunk_rms:.6f}) — "
                "re-arming wake-word polling",
            )
            # Do NOT replay boot-ready TTS here — that caused random
            # "Dana is ready" mid-session when quiet-mic mode cleared.

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
            if value >= 0.50:
                # Near-miss / hit visibility for live threshold tuning.
                log(
                    "WakeWord",
                    f"score={value:.3f} key={key} "
                    f"(threshold={WAKE_THRESHOLD:.2f})",
                )
            elif value > 0.20:
                log_debug("Debug", f"Wake word score: {value:.4f} ({key})")

        score_history.append(best_score)
        hit = wake_score_hit(pred, require_token=wake_token)
        # Sticky high scores on hush never dip; real "Dana" rises from a low baseline.
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
        # mics often sit ~0.002–0.003 even on a real "Dana").
        wake_rms = (
            float(np.sqrt(np.mean(np.square(wake_audio)))) if wake_audio.size else 0.0
        )
        log_debug("WakeWord", f"Wake candidate buffer_rms={wake_rms:.5f} hit={hit}")

        if wake_token == "dana" and not wake_phrase_confirmed(wake_audio):
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
        print(f"[Debug] Wake word HIT ({hit}) on device={state.AUDIO_INPUT_DEVICE}", flush=True)
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
# Model loading - Whisper
# ---------------------------------------------------------------------------



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
                # Abort any in-flight VAD listen so text drains without a 10s wait.
                prioritize_text_input(reason="input_txt")
                # Ensure the conversation loop can drain: escalate process mode
                # without stealing the user's durable voice_session_mode.
                try:
                    if get_dana_mode() == "chat":
                        set_dana_mode("developer", as_voice=False)
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





# ---------------------------------------------------------------------------
# Thread 5 - Audio TTS (piper-tts -> sounddevice)
# ---------------------------------------------------------------------------



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


# Safe to register unconditionally — update_tray_icon_for_state no-ops until
# _tray_icon exists (see the None-guard above).
register_ui_state_listener(update_tray_icon_for_state)


def _device_menu_label(index: int, name: str) -> str:
    return f"[{index}] {name}"


def _parse_device_menu_label(label: str) -> Optional[int]:
    from dana.audio.devices import SYSTEM_DEFAULT_LABEL

    if not label or label == SYSTEM_DEFAULT_LABEL:
        return None
    if label.strip().lower() in {"system default (auto)", "system default", "(none)"}:
        return None
    if not label.startswith("[") or "]" not in label:
        return None
    try:
        return int(label[1 : label.index("]")])
    except ValueError:
        return None


def request_dana_quit(icon: Optional[pystray.Icon] = None, _item: Any = None) -> None:
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


def _install_signal_handlers(gui: "DanaGUI") -> None:
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

def run_system_tray(gui: "DanaGUI") -> None:
    """Blocking pystray loop — must only run in a daemon thread (never on CTk)."""
    global _tray_icon

    try:
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
            pystray.MenuItem("Quit", request_dana_quit),
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
    except Exception as exc:  # noqa: BLE001
        log("Main", f"WARNING: system tray exited ({type(exc).__name__}: {exc})")
        _tray_icon = None


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


class DanaGUI(ctk.CTk):
    """Live Trace window with settings tabs; retreats to the tray on close."""

    def __init__(self) -> None:
        # Theme must load before CTk root/widgets (not after super()).
        try:
            from dana.ui.theme import apply_dana_ctk_theme

            apply_dana_ctk_theme()
        except Exception:  # noqa: BLE001
            pass
        super().__init__()
        global _gui_instance
        _gui_instance = self
        register_transcript_listener(self._on_transcript_event)
        register_vault_prompt_listener(self._on_vault_unlock_request)
        register_spec_approval_listener(self._on_spec_approval_requested)
        register_dictation_sessions_listener(self._on_dictation_sessions_changed)
        self.title("Dana — Control Dashboard")
        self.geometry("1440x900")
        self.minsize(1280, 800)
        self._dictation_active = False
        self._behavior_locked = False
        # Stage 8.9.7 — soft LangGraph engine ignition (False = STANDBY).
        self.engine_active = False
        self._engine_stopped = False
        self._behavior_sliders: dict[str, ctk.CTkSlider] = {}
        self._behavior_labels: dict[str, ctk.CTkLabel] = {}
        self._behavior_last_write: dict[str, float] = {}
        # Stage 8.5.1 — static settings that must not change while system is hot.
        self._static_behavior_widgets: list[Any] = []
        self._behavior_lock_hint: ctk.CTkLabel | None = None
        self._behavior_lock_overlay: ctk.CTkFrame | None = None
        self._behavior_mixer_host: ctk.CTkFrame | None = None
        self._behavior_reload_btn: ctk.CTkButton | None = None
        self.dictation_status: ctk.CTkLabel | None = None
        self._engine_status_lbl: ctk.CTkLabel | None = None
        self._engine_warn_lbl: ctk.CTkLabel | None = None
        self._vault_warn_lbl: ctk.CTkLabel | None = None
        self._vault_unlock_win: ctk.CTkToplevel | None = None
        self._engage_btn: ctk.CTkButton | None = None
        self._standby_btn: ctk.CTkButton | None = None  # legacy; merged into toggle
        self._engage_toggle_btn: ctk.CTkButton | None = None
        self._tasks_toggle_btn: ctk.CTkButton | None = None
        self._dag_toggle_btn: ctk.CTkButton | None = None
        self._diag_header_btn: ctk.CTkButton | None = None
        self._header_status_lbl: ctk.CTkLabel | None = None
        self._header_seg: Any | None = None
        self._assistant_main: Any | None = None
        self._assistant_side: Any | None = None
        self.task_tracker_frame: Any | None = None
        self._tasks_drawer_visible = False
        self.dag_monitor_frame: Any | None = None
        self.dag_monitor_view: Any | None = None
        self._dag_drawer_visible = False
        self._dag_status_lbl: Any | None = None
        self._dag_stream_thread: Any | None = None
        self._engine_warn_job: str | None = None
        self._diag_overlay: Any | None = None
        self._spec_approval_host: Any | None = None
        self._spec_approval_card: Any | None = None
        self._spec_approval_visible = False
        self._pending_spec_payload: dict[str, Any] | None = None
        # Stage 9.3 — Settings auto-updater chrome.
        self._update_status_lbl: ctk.CTkLabel | None = None
        self._update_check_btn: ctk.CTkButton | None = None
        self._update_apply_btn: ctk.CTkButton | None = None
        self._update_busy = False
        # Phase 1 OTA — Auto-Update Mode + Hot Apply pill.
        self._ota_mode_var: Any = None
        self._ota_mode_menu: ctk.CTkOptionMenu | None = None
        self._ota_pill_lbl: ctk.CTkLabel | None = None
        self._ota_slot_lbl: ctk.CTkLabel | None = None
        self._ota_staging_lbl: ctk.CTkLabel | None = None
        self._ota_hot_apply_btn: ctk.CTkButton | None = None
        self._ota_manager: Any = None
        # Stage 8.10 — Dashboard silent text chat.
        self.chat_entry: ctk.CTkEntry | None = None
        self._chat_send_btn: ctk.CTkButton | None = None
        self.bottom_input_frame: Any | None = None
        self._tasks_empty_lbl: Any | None = None
        self.transcript_box = None
        self._chat_view = None
        # VAD / supervisor STATE_CHANGE indicators (above chat input).
        self._vad_mic_lbl: ctk.CTkLabel | None = None
        self._system_status_lbl: ctk.CTkLabel | None = None
        self._vad_listening = False
        self._vad_pulse_on = False

        try:
            self.configure(fg_color=_UI_CANVAS)
        except Exception:  # noqa: BLE001
            pass

        self._mic_labels: list[str] = []
        self._speaker_labels: list[str] = []
        self._mic_by_label: dict[str, int] = {}
        self._speaker_by_label: dict[str, int] = {}
        self.mic_menu = None
        self.speaker_menu = None
        self.save_btn = None
        self.apply_note = None
        self._theme_menu = None
        self._theme_var = None
        self._trace_cells: dict[str, TraceCell] = {}
        self._pulse_on = False
        self._header_mode = "chat"
        self.assistive_orb: Any | None = None
        self._perception_feed_job: str | None = None
        self._perception_feed_img = None
        self._perception_feed_lbl = None
        self._perception_feed_busy = False

        # Unified Agent Canvas — 60/40 split (chat | workspace inspector).
        self._canvas_frame: Any | None = None
        self._workspace_inspector: Any | None = None
        self._neural_stream_text: Any | None = None
        self.artifact_viewer: Any | None = None
        self._neural_rendered = 0
        self._telemetry_buffer = AsyncRingBuffer(capacity=500)
        self._telemetry_emitter = NeuralStreamEmitter(self._telemetry_buffer)
        # Master telemetry dispatcher — see _master_telemetry_tick. Tracks the
        # last-fired monotonic timestamp per consumer so one adaptive
        # heartbeat can gate all five original polling cadences.
        self._telemetry_last: dict[str, float] = {}
        self._adaptive_poller = AdaptivePoller(
            self._telemetry_had_activity, t_min=0.05, t_max=0.5, gamma=1.5
        )

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close_to_tray)
        self.withdraw()
        # Post-init icon lifecycle: uniquely named runtime PNG/ICO + PhotoImage keepalive.
        self._icon_keepalive = None
        try:
            from dana.ui.logo import apply_window_icon, schedule_window_icon

            apply_window_icon(self)
            schedule_window_icon(self, delay_ms=100)
        except Exception:  # noqa: BLE001
            pass
        # Stage 8.7 — floating AssistiveTouch orb (DISABLED for now).
        # try:
        #     self.after(200, self._start_assistive_orb)
        # except Exception:  # noqa: BLE001
        #     pass
        self.after(400, self._refresh_stats)
        self.after(500, self._pulse_active_cells)
        # process_telemetry / _poll_state_changes no longer self-schedule —
        # _master_telemetry_tick dispatches both (plus LiveTracePanel /
        # DagMonitorView / TaskTrackerView) on one shared heartbeat whose
        # delay adapts via self._adaptive_poller.note_activity() each tick.
        # AdaptivePoller.start() is intentionally NOT used — see its
        # docstring: a background thread cannot safely touch Tk, so this
        # chain stays a normal main-thread self.after() loop throughout.
        self.after(int(self._adaptive_poller.t_min * 1000), self._master_telemetry_tick)
        # Phase 2A — optional IPC attach (no-op / degrade when daemon down).
        try:
            self.after(250, self._init_daemon_client)
        except Exception:  # noqa: BLE001
            pass
        try:
            self.after(700, self._schedule_perception_feed)
        except Exception:  # noqa: BLE001
            pass

    def _init_daemon_client(self) -> None:
        """Attach Control Dashboard to Agent Engine sidecar (graceful if absent)."""
        try:
            from dana.ui.daemon_client import DaemonClient, daemon_ipc_enabled
        except Exception:  # noqa: BLE001
            return
        if not daemon_ipc_enabled():
            return
        if getattr(self, "_daemon_client", None) is not None:
            return

        def _on_state(state: str, badge: str) -> None:
            try:
                self.after(
                    0,
                    lambda b=badge, s=state: self._set_daemon_badge(
                        b if s == "reconnecting" else ""
                    ),
                )
            except Exception:  # noqa: BLE001
                pass

        try:
            client = DaemonClient(on_state_change=_on_state)
            self._daemon_client = client
            client.connect(retries=1)
            client.start_auto_reconnect()
        except Exception:  # noqa: BLE001
            self._daemon_client = None

    def _set_daemon_badge(self, text: str) -> None:
        lbl = getattr(self, "daemon_badge", None)
        if lbl is None:
            return
        try:
            lbl.configure(text=text or "")
        except Exception:  # noqa: BLE001
            pass

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
            # Stage 8.9.8 — header live status (CHAT badge removed).
            if hasattr(self, "mode_badge") and self.mode_badge is not None:
                self.mode_badge.configure(
                    text=f"  ●  {label.upper()}  ",
                    text_color=color,
                    fg_color=_UI_GHOST,
                )
            hdr = getattr(self, "_header_status_lbl", None)
            if hdr is not None and not bool(getattr(self, "engine_active", False)):
                # When engaged, _refresh_engine_ui owns the ACTIVE/STANDBY text.
                hdr.configure(text=f"• {label.upper()}", text_color=color)
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
            text_color=_UI_TEXT,
        ).pack(fill="x", padx=14, pady=(12, 6))
        body = ctk.CTkFrame(card, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        return body

    def _build_ui(self) -> None:
        # Single HUD row — grid keeps brand | tabs | controls from overlapping.
        header = ctk.CTkFrame(
            self,
            fg_color=_UI_CARD,
            corner_radius=0,
            border_width=0,
            height=54,
        )
        header.pack(fill="x", padx=0, pady=0)
        try:
            header.grid_columnconfigure(0, weight=0, minsize=200)
            header.grid_columnconfigure(1, weight=1, minsize=420)
            header.grid_columnconfigure(2, weight=0, minsize=360)
            header.grid_propagate(False)
            header.configure(height=54)
        except Exception:  # noqa: BLE001
            pass
        self.mode_dot = None
        self.mode_label = None
        self._header_logo_img = None
        self._header_logo_lbl = None
        self.mode_badge = None  # removed redundant CHAT badge

        left = ctk.CTkFrame(header, fg_color="transparent")
        left.grid(row=0, column=0, sticky="w", padx=(10, 4), pady=6)
        try:
            from dana.ui.logo import invalidate_logo_cache, load_premium_logo

            invalidate_logo_cache()
            self._header_logo_img = load_premium_logo((24, 24))
            if self._header_logo_img is not None:
                self._header_logo_lbl = ctk.CTkLabel(
                    left,
                    text="",
                    image=self._header_logo_img,
                    width=24,
                    height=24,
                )
                self._header_logo_lbl.pack(side="left", padx=(0, 6))
        except Exception:  # noqa: BLE001
            self._header_logo_img = None
            self._header_logo_lbl = None
        ctk.CTkLabel(
            left,
            text="Dānā",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=_UI_TEXT,
            anchor="w",
        ).pack(side="left", padx=(0, 8))
        self._header_status_lbl = ctk.CTkLabel(
            left,
            text="• STANDBY",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=_UI_MUTED,
            anchor="w",
        )
        self._header_status_lbl.pack(side="left")
        # Mic / VAD pipeline indicator (Idle | Listening | Processing).
        self._vad_mic_lbl = ctk.CTkLabel(
            left,
            text="● Idle",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=_UI_MUTED,
            anchor="w",
        )
        self._vad_mic_lbl.pack(side="left", padx=(8, 0))
        self._system_status_lbl = ctk.CTkLabel(
            left,
            text="Idle",
            font=ctk.CTkFont(size=10),
            text_color=_UI_MUTED,
            anchor="w",
        )
        self._system_status_lbl.pack(side="left", padx=(6, 0))
        self._vad_listening = False
        self._vad_processing = False
        # Phase 2A — engine sidecar reconnect badge (hidden until IPC drop).
        self.daemon_badge = ctk.CTkLabel(
            left,
            text="",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#FBBF24",
            fg_color="transparent",
        )
        self.daemon_badge.pack(side="left", padx=(6, 0))
        self._daemon_client = None

        # Center tab switcher (mirrors CTkTabview; built-in segment hidden below).
        center = ctk.CTkFrame(header, fg_color="transparent")
        center.grid(row=0, column=1, sticky="ew", padx=4, pady=6)
        self._header_seg = ctk.CTkSegmentedButton(
            center,
            values=["Assistant & Tasks", "Perception", "Memory & Settings"],
            command=self._on_header_tab,
            height=28,
            corner_radius=8,
            font=ctk.CTkFont(size=10, weight="bold"),
        )
        self._header_seg.pack(fill="x", expand=True, padx=2)
        try:
            self._header_seg.set("Assistant & Tasks")
        except Exception:  # noqa: BLE001
            pass

        right = ctk.CTkFrame(header, fg_color="transparent")
        right.grid(row=0, column=2, sticky="e", padx=(4, 10), pady=6)

        self.stop_dana_btn = ctk.CTkButton(
            right,
            text="STOP DANA",
            width=96,
            height=28,
            corner_radius=8,
            fg_color=_UI_ROSE,
            hover_color=_UI_ROSE_HOVER,
            text_color="#FFFFFF",
            font=ctk.CTkFont(size=10, weight="bold"),
            command=self._on_stop_dana_clicked,
        )
        self.stop_dana_btn.pack(side="right", padx=(4, 0))

        self._diag_header_btn = ctk.CTkButton(
            right,
            text="Diagnostics",
            width=88,
            height=28,
            corner_radius=8,
            fg_color=_UI_GHOST,
            hover_color="#475569",
            border_width=1,
            border_color=_UI_CARD_BORDER,
            text_color=_UI_TEXT,
            font=ctk.CTkFont(size=10),
            command=self._dashboard_open_trace,
        )
        self._diag_header_btn.pack(side="right", padx=(4, 0))

        try:
            _engage_font = ctk.CTkFont(
                family="Segoe UI Historic", size=10, weight="bold"
            )
        except Exception:  # noqa: BLE001
            _engage_font = ctk.CTkFont(size=10, weight="bold")
        self._engage_toggle_btn = ctk.CTkButton(
            right,
            text="Engaged",
            width=88,
            height=28,
            corner_radius=8,
            font=_engage_font,
            fg_color=_UI_EMERALD,
            hover_color="#059669",
            text_color="#ECFDF5",
            command=self.toggle_engine_engage,
        )
        self._engage_toggle_btn.pack(side="right", padx=(4, 0))
        self._engage_btn = self._engage_toggle_btn
        self._standby_btn = None

        self._tasks_toggle_btn = ctk.CTkButton(
            right,
            text="Tasks ▸",
            width=64,
            height=28,
            corner_radius=8,
            fg_color=_UI_GHOST,
            hover_color="#475569",
            border_width=1,
            border_color=_UI_CARD_BORDER,
            text_color=_UI_TEXT,
            font=ctk.CTkFont(size=10, weight="bold"),
            command=self._toggle_tasks_drawer,
        )
        self._tasks_toggle_btn.pack(side="right", padx=(4, 0))

        self._dag_toggle_btn = ctk.CTkButton(
            right,
            text="DAG ▸",
            width=64,
            height=28,
            corner_radius=8,
            fg_color=_UI_GHOST,
            hover_color="#475569",
            border_width=1,
            border_color=_UI_CARD_BORDER,
            text_color=_UI_TEXT,
            font=ctk.CTkFont(size=10, weight="bold"),
            command=self._toggle_dag_drawer,
        )
        self._dag_toggle_btn.pack(side="right", padx=(4, 0))

        try:
            from dana.ui.tooltips import attach_tooltip

            attach_tooltip(
                self._engage_toggle_btn,
                "When Engaged, Dānā listens for the wake word and responds to "
                "voice/text. Click to put in Standby.",
            )
            attach_tooltip(
                self._diag_header_btn,
                "Open system logs, audio RMS meters, and component health status.",
            )
            attach_tooltip(
                self._tasks_toggle_btn,
                "Shows real-time status of background Python sandbox scripts "
                "and research swarms.",
            )
            attach_tooltip(
                self._dag_toggle_btn,
                "Live LangGraph DAG execution tree and tool micro-log. "
                "Hotkey: Ctrl+Shift+D.",
            )
            attach_tooltip(
                self.stop_dana_btn,
                "Emergency kill-switch. Instantly halts all background processes "
                "and shuts down the engine.",
            )
        except Exception:  # noqa: BLE001
            pass

        # Three polished surfaces: Assistant, Perception, Memory & Settings.
        tabs = ctk.CTkTabview(
            self,
            fg_color=_UI_CANVAS,
            text_color=_UI_TEXT,
            corner_radius=14,
            border_width=0,
        )
        tabs.pack(fill="both", expand=True, padx=14, pady=(8, 14))
        self._tabs = tabs

        tab_assistant = tabs.add("Assistant & Tasks")
        tab_perception = tabs.add("Perception")
        tab_memory = tabs.add("Memory & Settings")

        def _hide_builtin_tab_strip() -> None:
            """CTkTabview re-grids ``_segmented_button`` on every ``add`` — hide it."""
            try:
                seg = tabs._segmented_button  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                return
            for forget in (
                getattr(seg, "grid_forget", None),
                getattr(seg, "pack_forget", None),
                getattr(seg, "place_forget", None),
            ):
                if forget is None:
                    continue
                try:
                    forget()
                except Exception:  # noqa: BLE001
                    pass
            try:
                seg.configure(height=0, width=0)
            except Exception:  # noqa: BLE001
                pass

        _hide_builtin_tab_strip()
        try:
            # Stop future layout passes from restoring the duplicate tab row.
            tabs._set_grid_segmented_button = (  # type: ignore[attr-defined]
                lambda *a, **k: None
            )
        except Exception:  # noqa: BLE001
            pass
        try:
            self.after(50, _hide_builtin_tab_strip)
            self.after(250, _hide_builtin_tab_strip)
        except Exception:  # noqa: BLE001
            pass

        try:
            # Unified Agent Canvas owns tab_assistant's whole surface (pack-based);
            # Perception / Memory & Settings keep their own grid.
            tab_perception.grid_columnconfigure(0, weight=1)
            tab_perception.grid_rowconfigure(0, weight=1)
            tab_memory.grid_columnconfigure(0, weight=1)
            tab_memory.grid_rowconfigure(0, weight=1)
        except Exception:  # noqa: BLE001
            pass

        self._build_unified_canvas(tab_assistant)
        self._build_spec_approval_host(tab_assistant)
        self._build_perception_tab(tab_perception)
        try:
            from dana.ui.tooltips import attach_tooltip

            # Perception tab tip via header segment (best-effort).
            attach_tooltip(
                self._header_seg,
                "Perception tab: view live screen captures, camera feeds, and "
                "spatial tracking buffers. Other segments open Assistant or Settings.",
            )
        except Exception:  # noqa: BLE001
            pass

        # Memory & Settings: configuration only (telemetry lives in Diagnostics overlay).
        mem_scroll = ctk.CTkScrollableFrame(tab_memory, fg_color=_UI_CANVAS)
        mem_scroll.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        try:
            mem_scroll.grid_columnconfigure(0, weight=1)
        except Exception:  # noqa: BLE001
            pass
        # Row 1 — Engine Runtime · Row 2 — Behavior Mixer ·
        # Row 3 — Memory + Appearance · Row 4 — Updates + Dictation.
        self._build_settings_tab(mem_scroll)
        self._build_behavior_tab(mem_scroll)
        self._build_memory_appearance_row(mem_scroll)
        self._build_updates_dictation_row(mem_scroll)
        self._build_developer_diagnostics(mem_scroll)

        try:
            tabs.set("Assistant & Tasks")
        except Exception:  # noqa: BLE001
            pass
        _hide_builtin_tab_strip()

        try:
            self.after(300, self.refresh_dictation_sessions)
            self.after(350, self._reload_behavior_sliders)
        except Exception:  # noqa: BLE001
            pass

    def _on_header_tab(self, name: str) -> None:
        self._select_tab(str(name))

    def _assistant_tab(self) -> Any | None:
        tabs = getattr(self, "_tabs", None)
        if tabs is None:
            return None
        try:
            return tabs.tab("Assistant & Tasks")
        except Exception:  # noqa: BLE001
            return None

    def _toggle_tasks_drawer(self) -> None:
        """Expand / collapse the Task Tracker overlay over the Workspace Inspector."""
        if bool(getattr(self, "_tasks_drawer_visible", False)):
            self._collapse_tasks_drawer()
        else:
            self._expand_tasks_drawer()

    def _collapse_tasks_drawer(self) -> None:
        frame = getattr(self, "task_tracker_frame", None) or getattr(
            self, "_assistant_side", None
        )
        if frame is None:
            self._tasks_drawer_visible = False
            return
        self._tasks_drawer_visible = False
        try:
            frame.place_forget()
        except Exception:  # noqa: BLE001
            pass
        try:
            if self._tasks_toggle_btn is not None:
                self._tasks_toggle_btn.configure(text="Tasks ▸")
        except Exception:  # noqa: BLE001
            pass

    def _expand_tasks_drawer(self) -> None:
        frame = getattr(self, "task_tracker_frame", None) or getattr(
            self, "_assistant_side", None
        )
        if frame is None:
            self._tasks_drawer_visible = True
            return
        self._tasks_drawer_visible = True
        try:
            frame.place(relx=0, rely=0, relwidth=1.0, relheight=1.0)
            frame.lift()
        except Exception:  # noqa: BLE001
            pass
        try:
            if self._tasks_toggle_btn is not None:
                self._tasks_toggle_btn.configure(text="Tasks")
        except Exception:  # noqa: BLE001
            pass

    def _restore_assistant_layout(self) -> None:
        """Re-apply the Unified Canvas + drawer overlays after tab switches remount the page."""
        tab = self._assistant_tab()
        if tab is None:
            return
        frame = getattr(self, "_canvas_frame", None)
        if frame is not None:
            try:
                frame.pack(fill="both", expand=True, padx=14, pady=(10, 4))
            except Exception:  # noqa: BLE001
                pass
        if bool(getattr(self, "_tasks_drawer_visible", False)):
            self._expand_tasks_drawer()
        if bool(getattr(self, "_dag_drawer_visible", False)):
            self._expand_dag_drawer()
        if bool(getattr(self, "_spec_approval_visible", False)):
            host = getattr(self, "_spec_approval_host", None)
            if host is not None:
                try:
                    host.place(relx=0, rely=1.0, anchor="sw", relwidth=1.0)
                    host.lift()
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
        seg = getattr(self, "_header_seg", None)
        if seg is not None:
            try:
                seg.set(str(name))
            except Exception:  # noqa: BLE001
                pass
        # CTkTabview remounts pages — re-lock the 65/35 Assistant layout.
        if str(name) == "Assistant & Tasks":
            try:
                self.after(10, self._restore_assistant_layout)
            except Exception:  # noqa: BLE001
                self._restore_assistant_layout()

    def _build_unified_canvas(self, tab) -> None:  # noqa: ANN001
        """Unified Agent Canvas — single 60/40 split-pane dashboard.

        Left (60%): Conversation + input. Right (40%): Neural Stream telemetry
        (top) + Artifact Viewer (bottom). Task Tracker and the DAG monitor are
        preserved as toggleable overlay drawers (place/place_forget) so the
        header's existing Tasks / DAG controls keep working without a
        permanent third column competing for width.
        """
        try:
            tab.configure(fg_color=_UI_CANVAS)
        except Exception:  # noqa: BLE001
            pass

        app_frame = ctk.CTkFrame(tab, fg_color="transparent")
        app_frame.pack(fill="both", expand=True, padx=14, pady=(10, 4))
        self._canvas_frame = app_frame
        try:
            app_frame.grid_columnconfigure(0, weight=6)
            app_frame.grid_columnconfigure(1, weight=4)
            app_frame.grid_rowconfigure(0, weight=1)
        except Exception:  # noqa: BLE001
            pass

        # ---- Left Pane (60%) — Chat & Interaction --------------------------
        left = ctk.CTkFrame(app_frame, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self._assistant_main = left
        try:
            left.grid_columnconfigure(0, weight=1)
            left.grid_rowconfigure(0, weight=1)
            left.grid_rowconfigure(1, weight=0)
        except Exception:  # noqa: BLE001
            pass

        # Status / wake chrome live in the header HUD (mic + system labels).
        self.status_value = None
        self.wake_value = None
        self._engine_status_lbl = None
        self._engine_warn_lbl = None
        self._vault_warn_lbl = None
        # Keep header-created mic / system labels — do not null them here.

        # Ensure monitor bus exists so background graph streams can publish.
        try:
            from dana.graph.monitor_bus import get_monitor_bus

            get_monitor_bus(create=True)
        except Exception:  # noqa: BLE001
            pass

        # _make_card packs its shell into parent; re-grid the shell so siblings
        # on ``left`` can use grid (Tk forbids mixing pack+grid on one parent).
        chat_card = self._make_card(
            left, title="Conversation", padx=0, pady=(0, 0), expand=True
        )
        chat_shell = getattr(chat_card, "master", None)
        try:
            if chat_shell is not None:
                chat_shell.pack_forget()
                chat_shell.grid(row=0, column=0, sticky="nsew")
        except Exception:  # noqa: BLE001
            try:
                chat_card.grid(row=0, column=0, sticky="nsew")
            except Exception:  # noqa: BLE001
                pass

        self._chat_view = None
        try:
            from dana.ui.chat_view import ChatBubbleView

            self._chat_view = ChatBubbleView(chat_card, wraplength=480)
            self._chat_view.pack(fill="both", expand=True)
            self.transcript_box = self._chat_view.transcript_box
            try:
                left.bind(
                    "<Configure>",
                    lambda e: self._on_chat_host_configure(e),
                    add="+",
                )
            except Exception:  # noqa: BLE001
                pass
        except Exception as exc:  # noqa: BLE001
            log("UI", f"WARNING: ChatBubbleView unavailable ({exc})")
            self.transcript_box = ctk.CTkTextbox(
                chat_card,
                wrap="word",
                font=ctk.CTkFont(family="Segoe UI", size=14),
                fg_color=_UI_CANVAS,
                corner_radius=12,
                border_width=0,
            )
            self.transcript_box.pack(fill="both", expand=True)
        self._init_persona_transcript_tags()
        welcome = "Type below or say Dana, then speak."
        try:
            self.transcript_box.configure(state="normal")
            self.transcript_box.insert("1.0", f"[Dana] {welcome}\n\n")
            self.transcript_box.configure(state="disabled")
        except Exception:  # noqa: BLE001
            pass
        if self._chat_view is not None:
            try:
                self._chat_view.append_bubble("Dana", welcome, agent_id="broker")
            except Exception:  # noqa: BLE001
                pass

        # Input row — entry + send button, bottom of the left pane's stack.
        input_row = ctk.CTkFrame(left, fg_color="transparent")
        input_row.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.bottom_input_frame = input_row
        try:
            input_row.grid_columnconfigure(0, weight=1)
            input_row.grid_columnconfigure(1, weight=0)
        except Exception:  # noqa: BLE001
            pass
        self.chat_entry = ctk.CTkEntry(
            input_row,
            placeholder_text="Type below or say Dana, then speak.",
            height=36,
            corner_radius=10,
            fg_color=_UI_GHOST,
            border_width=1,
            border_color=_UI_CARD_BORDER,
            text_color=_UI_TEXT,
            placeholder_text_color=_UI_MUTED,
            font=ctk.CTkFont(size=13),
        )
        self.chat_entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.chat_entry.bind("<Return>", self.submit_text_command)
        self._chat_send_btn = ctk.CTkButton(
            input_row,
            text="Send",
            width=92,
            height=36,
            corner_radius=999,
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self.submit_text_command,
        )
        self._chat_send_btn.grid(row=0, column=1, sticky="e")
        # Brief STANDBY toast ("Please Engage Engine First.") — see
        # _flash_engine_warning; empty text reserves the row so it doesn't
        # reflow the input row when a warning flashes/clears.
        self._engine_warn_lbl = ctk.CTkLabel(
            input_row,
            text="",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#F59E0B",
            anchor="w",
        )
        self._engine_warn_lbl.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))
        # Persistent banner (not a toast — stays until unlocked) so a locked
        # vault at startup is visible instead of the wake-word thread just
        # never starting with no explanation.
        self._vault_warn_lbl = ctk.CTkLabel(
            input_row,
            text="",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#EF4444",
            anchor="w",
        )
        self._vault_warn_lbl.grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))

        # DAG monitor — collapsible overlay drawer above the input row.
        self._build_dag_monitor_section(left)
        try:
            self.bind("<Control-Shift-D>", lambda _e: self._toggle_dag_drawer())
            self.bind("<Control-Shift-d>", lambda _e: self._toggle_dag_drawer())
        except Exception:  # noqa: BLE001
            pass

        # ---- Right Pane (40%) — Workspace Inspector -------------------------
        right = ctk.CTkFrame(app_frame, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self._workspace_inspector = right
        try:
            right.grid_columnconfigure(0, weight=1)
            right.grid_rowconfigure(0, weight=1)
            right.grid_rowconfigure(1, weight=1)
        except Exception:  # noqa: BLE001
            pass

        self._build_neural_stream_pane(right)
        self._build_artifact_viewer_pane(right)

        # Task Tracker — collapsible overlay drawer covering the inspector pane.
        self._build_task_tracker_section(right)

        try:
            self.engage_engine()
        except Exception:  # noqa: BLE001
            try:
                self._refresh_engine_ui()
            except Exception:  # noqa: BLE001
                pass

    def _build_neural_stream_pane(self, parent) -> None:  # noqa: ANN001
        """Top half of the Workspace Inspector — live color-coded telemetry."""
        card = ctk.CTkFrame(
            parent,
            fg_color=_UI_CARD,
            corner_radius=14,
            border_width=1,
            border_color=_UI_CARD_BORDER,
        )
        card.grid(row=0, column=0, sticky="nsew", pady=(0, 6))
        try:
            card.grid_columnconfigure(0, weight=1)
            card.grid_rowconfigure(1, weight=1)
        except Exception:  # noqa: BLE001
            pass
        ctk.CTkLabel(
            card,
            text="Neural Stream",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=_UI_TEXT,
            anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(10, 4))

        body = tk.Frame(card, bg=_UI_CANVAS, highlightthickness=0)
        body.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        try:
            body.grid_columnconfigure(0, weight=1)
            body.grid_rowconfigure(0, weight=1)
        except Exception:  # noqa: BLE001
            pass

        text = tk.Text(
            body,
            wrap="word",
            height=10,
            bg=_UI_CANVAS,
            fg=_UI_TEXT,
            insertbackground=_UI_TEXT,
            borderwidth=0,
            highlightthickness=0,
            font=("Consolas", 10),
        )
        text.grid(row=0, column=0, sticky="nsew")
        scroll = tk.Scrollbar(body, orient="vertical", command=text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        text.configure(yscrollcommand=scroll.set)
        text.tag_configure("error", foreground="#ff4444")
        text.tag_configure("tool", foreground="#00cc66")
        text.tag_configure("thought", foreground="#3399ff")
        text.configure(state="disabled")
        self._neural_stream_text = text

    def _build_artifact_viewer_pane(self, parent) -> None:  # noqa: ANN001
        """Bottom half of the Workspace Inspector — workspace code / file preview."""
        card = ctk.CTkFrame(
            parent,
            fg_color=_UI_CARD,
            corner_radius=14,
            border_width=1,
            border_color=_UI_CARD_BORDER,
        )
        card.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        try:
            card.grid_columnconfigure(0, weight=1)
            card.grid_rowconfigure(1, weight=1)
        except Exception:  # noqa: BLE001
            pass
        ctk.CTkLabel(
            card,
            text="Artifact Viewer",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=_UI_TEXT,
            anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(10, 4))
        box = ctk.CTkTextbox(
            card,
            wrap="none",
            font=ctk.CTkFont(family="Consolas", size=12),
            fg_color=_UI_CANVAS,
            corner_radius=8,
            border_width=0,
        )
        box.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        box.insert("1.0", "// No file selected.\n")
        box.configure(state="disabled")
        self.artifact_viewer = box

    def show_artifact(self, title: str, content: str) -> None:
        """Preview a file/code snippet in the Artifact Viewer pane."""
        box = getattr(self, "artifact_viewer", None)
        if box is None:
            return
        try:
            box.configure(state="normal")
            box.delete("1.0", "end")
            box.insert("1.0", f"# {title}\n\n{content}")
            box.configure(state="disabled")
        except Exception:  # noqa: BLE001
            pass

    def _build_dag_monitor_section(self, main) -> None:  # noqa: ANN001
        """Collapsible live DAG drawer — overlays the conversation pane when expanded."""
        shell = ctk.CTkFrame(
            main,
            fg_color=_UI_CARD,
            corner_radius=14,
            border_width=1,
            border_color=_UI_CARD_BORDER,
            height=220,
        )
        self.dag_monitor_frame = shell
        self._dag_drawer_visible = False
        try:
            shell.grid_propagate(False)
            shell.grid_columnconfigure(0, weight=1)
            shell.grid_rowconfigure(0, weight=0)
            shell.grid_rowconfigure(1, weight=1)
        except Exception:  # noqa: BLE001
            pass

        hdr = ctk.CTkFrame(shell, fg_color="transparent")
        hdr.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 2))
        try:
            hdr.grid_columnconfigure(0, weight=1)
            hdr.grid_columnconfigure(1, weight=0)
        except Exception:  # noqa: BLE001
            pass
        ctk.CTkLabel(
            hdr,
            text="DAG Execution",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=_UI_TEXT,
            anchor="w",
        ).grid(row=0, column=0, sticky="w")
        right_hdr = ctk.CTkFrame(hdr, fg_color="transparent")
        right_hdr.grid(row=0, column=1, sticky="e")
        try:
            from dana.graph.cloud_planner import planner_mode_label

            _planner_mode = planner_mode_label()
        except Exception:  # noqa: BLE001
            _planner_mode = "LOCAL"
        self._dag_planner_mode_lbl = ctk.CTkLabel(
            right_hdr,
            text=f"Planner Mode: [{_planner_mode}]",
            font=ctk.CTkFont(size=10),
            text_color=_UI_MUTED,
            anchor="e",
        )
        self._dag_planner_mode_lbl.pack(side="top", anchor="e")
        self._dag_status_lbl = ctk.CTkLabel(
            right_hdr,
            text="Idle",
            font=ctk.CTkFont(size=11),
            text_color=_UI_MUTED,
            anchor="e",
        )
        self._dag_status_lbl.pack(side="top", anchor="e")

        try:
            from dana.ui.dag_monitor_view import DagMonitorView
            from dana.ui.tooltips import attach_tooltip

            self.dag_monitor_view = DagMonitorView(
                shell,
                poll_ms=250,
                show_header=False,
                status_label=self._dag_status_lbl,
                on_multistep=self._on_dag_multistep,
                on_complete=self._on_dag_complete,
                external_tick=True,
            )
            self.dag_monitor_view.grid(
                row=1, column=0, sticky="nsew", padx=4, pady=(0, 8)
            )
            attach_tooltip(
                self.dag_monitor_view,
                "Live supervisor plan, worker status, and tool staging events.",
            )
        except Exception as exc:  # noqa: BLE001
            self.dag_monitor_view = None
            log("UI", f"WARNING: DagMonitorView unavailable ({exc})")
            ctk.CTkLabel(
                shell,
                text="DAG monitor unavailable",
                text_color=_UI_MUTED,
            ).grid(row=1, column=0, sticky="nw", padx=12, pady=12)

        # Start collapsed (unplaced); multi-step plans auto-expand via place().

    def _on_dag_multistep(self) -> None:
        """Auto-expand the DAG drawer when a multi-step plan arrives."""
        try:
            self.after(0, self._expand_dag_drawer)
        except Exception:  # noqa: BLE001
            self._expand_dag_drawer()

    def _on_dag_complete(self, summary: str) -> None:
        try:
            log("UI", f"DAG complete: {summary}")
        except Exception:  # noqa: BLE001
            pass

    def _toggle_dag_drawer(self) -> None:
        if bool(getattr(self, "_dag_drawer_visible", False)):
            self._collapse_dag_drawer()
        else:
            self._expand_dag_drawer()

    def _collapse_dag_drawer(self) -> None:
        frame = getattr(self, "dag_monitor_frame", None)
        if frame is None:
            self._dag_drawer_visible = False
            return
        self._dag_drawer_visible = False
        try:
            frame.place_forget()
        except Exception:  # noqa: BLE001
            pass
        try:
            if self._dag_toggle_btn is not None:
                self._dag_toggle_btn.configure(text="DAG ▸")
        except Exception:  # noqa: BLE001
            pass

    def _expand_dag_drawer(self) -> None:
        frame = getattr(self, "dag_monitor_frame", None)
        if frame is None:
            self._dag_drawer_visible = True
            return
        self._dag_drawer_visible = True
        try:
            frame.place(relx=0, rely=1.0, anchor="sw", relwidth=1.0, height=260)
        except Exception:  # noqa: BLE001
            pass
        try:
            if self._dag_toggle_btn is not None:
                self._dag_toggle_btn.configure(text="DAG")
        except Exception:  # noqa: BLE001
            pass
        view = getattr(self, "dag_monitor_view", None)
        if view is not None:
            try:
                view.refresh()
            except Exception:  # noqa: BLE001
                pass

    def start_dag_monitor_stream(
        self,
        user_prompt: str,
        *,
        tool_fn: Any | None = None,
        planner: Any | None = None,
    ) -> None:
        """Run ``stream_dag_supervisor`` on a daemon thread (UI stays responsive)."""
        import threading

        def _run() -> None:
            try:
                from dana.graph.builder import stream_dag_supervisor
                from dana.graph.monitor_bus import get_monitor_bus

                get_monitor_bus(create=True)
                for _chunk in stream_dag_supervisor(
                    user_prompt,
                    planner=planner,
                    tool_fn=tool_fn,
                    monitor=True,
                ):
                    pass
            except Exception as exc:  # noqa: BLE001
                try:
                    log("UI", f"DAG stream error: {exc}")
                except Exception:  # noqa: BLE001
                    pass

        t = threading.Thread(target=_run, name="dana-dag-monitor", daemon=True)
        self._dag_stream_thread = t
        t.start()

    def _build_spec_approval_host(self, tab) -> None:  # noqa: ANN001
        """HITL Spec Approval Card host (row 1) — hidden until a draft is ready."""
        host = ctk.CTkFrame(tab, fg_color="transparent")
        self._spec_approval_host = host
        self._spec_approval_visible = False
        self._spec_approval_card = None
        self._pending_spec_payload: dict[str, Any] | None = None
        try:
            from dana.ui.spec_approval_view import SpecApprovalCard

            card = SpecApprovalCard(
                host,
                on_approve=self._on_spec_approve,
                on_edit=self._on_spec_edit,
                on_cancel=self._on_spec_cancel,
            )
            card.pack(fill="x", expand=False)
            self._spec_approval_card = card
        except Exception as exc:  # noqa: BLE001
            log("UI", f"WARNING: SpecApprovalCard unavailable ({exc})")
            self._spec_approval_card = None
        # Start collapsed (unplaced) — shown as a bottom overlay on demand.

    def show_spec_approval(self, payload: dict[str, Any]) -> None:
        """Present a compiled ``/broker`` draft and wait for Approve / Edit / Cancel."""
        self._pending_spec_payload = dict(payload or {})
        card = getattr(self, "_spec_approval_card", None)
        host = getattr(self, "_spec_approval_host", None)
        if card is None or host is None:
            return
        try:
            self._select_tab("Assistant & Tasks")
        except Exception:  # noqa: BLE001
            pass
        try:
            card.present(self._pending_spec_payload)
        except Exception:  # noqa: BLE001
            pass
        self._spec_approval_visible = True
        try:
            host.place(relx=0, rely=1.0, anchor="sw", relwidth=1.0)
            host.lift()
        except Exception:  # noqa: BLE001
            pass
        try:
            emit_live_transcript(
                "Dana",
                "Spec compiled — review the Approval Card, then Approve & Run.",
            )
        except Exception:  # noqa: BLE001
            pass

    def hide_spec_approval(self) -> None:
        host = getattr(self, "_spec_approval_host", None)
        self._spec_approval_visible = False
        self._pending_spec_payload = None
        if host is None:
            return
        try:
            host.place_forget()
        except Exception:  # noqa: BLE001
            pass

    def _on_spec_approve(self, compiled_spec: str) -> None:
        """Dispatch the approved macro to Meta-Broker (approved=True bypasses HITL)."""
        macro = str(compiled_spec or "").strip()
        self.hide_spec_approval()
        if not macro:
            return
        try:
            emit_live_transcript("Dana", "Approved — dispatching Meta-Broker…")
        except Exception:  # noqa: BLE001
            pass

        def _run() -> None:
            try:
                obs = execute_tool_call(
                    ToolCall(
                        tool_id="meta_broker",
                        arguments={"prompt": macro, "approved": True},
                        raw_text=macro,
                        confidence=0.99,
                    )
                )
                spoken = str(obs or "")
                if spoken.startswith("OK:"):
                    lines = [
                        ln.strip()
                        for ln in spoken.splitlines()
                        if ln.strip() and not ln.startswith("epic_log:")
                    ]
                    spoken = lines[1] if len(lines) > 1 else lines[0]
                if len(spoken) > 420:
                    spoken = spoken[:417] + "..."
                try:
                    log_conversation("Dana", spoken)
                    emit_live_transcript("Dana", spoken)
                except Exception:  # noqa: BLE001
                    pass
            except Exception as exc:  # noqa: BLE001
                try:
                    emit_live_transcript(
                        "Dana", f"Meta-Broker dispatch failed: {exc}"
                    )
                except Exception:  # noqa: BLE001
                    pass

        try:
            import threading

            threading.Thread(
                target=_run, name="dana-spec-approve", daemon=True
            ).start()
        except Exception:  # noqa: BLE001
            _run()

    def _on_spec_edit(self, compiled_spec: str) -> None:
        """Copy the compiled macro into the chat input for manual tweaking."""
        macro = str(compiled_spec or "").strip()
        self.hide_spec_approval()
        entry = getattr(self, "chat_entry", None)
        if entry is not None and macro:
            try:
                entry.delete(0, "end")
                entry.insert(0, macro)
                entry.focus_set()
            except Exception:  # noqa: BLE001
                pass
        try:
            emit_live_transcript(
                "Dana",
                "Macro copied to input — edit freely, then send to re-submit.",
            )
        except Exception:  # noqa: BLE001
            pass

    def _on_spec_cancel(self) -> None:
        self.hide_spec_approval()
        try:
            emit_live_transcript("Dana", "Spec approval cancelled — Meta-Broker not started.")
        except Exception:  # noqa: BLE001
            pass

    def _on_chat_host_configure(self, event: Any) -> None:
        """Keep bubble wraplength proportional to Conversation width."""
        view = getattr(self, "_chat_view", None)
        if view is None or event is None:
            return
        try:
            width = int(getattr(event, "width", 0) or 0)
        except Exception:  # noqa: BLE001
            return
        if width < 240:
            return
        # Leave room for bubble chrome / scrollbar; cap so text stays inside pane.
        wrap = max(220, min(560, width - 96))
        try:
            setter = getattr(view, "set_wraplength", None)
            if callable(setter):
                setter(wrap)
            else:
                view._wraplength = wrap  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass

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

    def _build_task_tracker_section(self, tab) -> None:  # noqa: ANN001
        """Task Tracker overlay drawer — covers the Workspace Inspector when expanded."""
        side = ctk.CTkFrame(
            tab,
            fg_color=_UI_CARD,
            corner_radius=14,
            border_width=1,
            border_color=_UI_CARD_BORDER,
        )
        self._assistant_side = side
        self.task_tracker_frame = side
        try:
            side.grid_columnconfigure(0, weight=1)
            side.grid_rowconfigure(0, weight=0)
            side.grid_rowconfigure(1, weight=1)
        except Exception:  # noqa: BLE001
            pass
        # Start collapsed (unplaced); Tasks header toggle brings it forward.

        # Permanent chrome — never pack_forget / grid_forget these inners.
        hdr = ctk.CTkFrame(side, fg_color="transparent")
        hdr.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))
        try:
            hdr.grid_columnconfigure(0, weight=1)
            hdr.grid_columnconfigure(1, weight=0)
        except Exception:  # noqa: BLE001
            pass
        ctk.CTkLabel(
            hdr,
            text="Task Tracker",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=_UI_TEXT,
            anchor="w",
        ).grid(row=0, column=0, sticky="w")
        self._tasks_empty_lbl = ctk.CTkLabel(
            hdr,
            text="No active tasks",
            font=ctk.CTkFont(size=11),
            text_color=_UI_MUTED,
            anchor="e",
        )
        self._tasks_empty_lbl.grid(row=0, column=1, sticky="e")

        try:
            from dana.ui.task_tracker_view import TaskTrackerView
            from dana.ui.tooltips import attach_tooltip

            self.task_tracker_view = TaskTrackerView(
                side,
                poll_ms=400,
                show_header=False,
                status_label=self._tasks_empty_lbl,
                external_tick=True,
            )
            self.task_tracker_view.grid(
                row=1, column=0, sticky="nsew", padx=4, pady=(0, 8)
            )
            attach_tooltip(
                self.task_tracker_view,
                "Shows real-time status of background Python sandbox scripts "
                "and research swarms.",
            )
        except Exception as exc:  # noqa: BLE001
            self.task_tracker_view = None
            log("UI", f"WARNING: TaskTrackerView unavailable ({exc})")
            ctk.CTkLabel(
                side,
                text="Task Tracker unavailable",
                text_color=_UI_MUTED,
            ).grid(row=1, column=0, sticky="nw", padx=12, pady=12)

    def _build_developer_diagnostics(self, tab) -> None:  # noqa: ANN001
        """Collapsible Live Trace / LangGraph diagnostics (Settings tab)."""
        self._diag_expanded = False
        self.live_trace = None
        self.trace_scroll = None
        self._diag_expander = None
        shell = ctk.CTkFrame(tab, fg_color="transparent")
        shell.pack(fill="x", expand=False, padx=8, pady=(4, 8))
        self._diag_shell = shell
        self._diag_btn = ctk.CTkButton(
            shell,
            text="▸ Developer Diagnostics",
            anchor="w",
            height=28,
            corner_radius=8,
            fg_color=_UI_GHOST,
            hover_color="#475569",
            border_width=1,
            border_color=_UI_CARD_BORDER,
            text_color=_UI_MUTED,
            font=ctk.CTkFont(size=12),
            command=self._toggle_developer_diagnostics,
        )
        self._diag_btn.pack(fill="x")
        self._diag_body = ctk.CTkFrame(shell, fg_color=_UI_CANVAS, height=220)
        # Hidden until toggled.
        try:
            from dana.ui.trace_window import LiveTracePanel

            self.live_trace = LiveTracePanel(
                self._diag_body, poll_ms=80, external_tick=True
            )
            self.live_trace.pack(fill="both", expand=True, padx=2, pady=2)
            self.trace_scroll = self.live_trace.timeline
        except Exception:  # noqa: BLE001
            self.live_trace = None
            fallback = self._make_card(self._diag_body, title="Pipeline stages")
            self.trace_scroll = ctk.CTkScrollableFrame(fallback, fg_color="transparent")
            self.trace_scroll.pack(fill="both", expand=True)
        self._diag_expander = self

    def _toggle_developer_diagnostics(self) -> None:
        body = getattr(self, "_diag_body", None)
        btn = getattr(self, "_diag_btn", None)
        if body is None:
            return
        self._diag_expanded = not bool(getattr(self, "_diag_expanded", False))
        try:
            if self._diag_expanded:
                body.pack(fill="both", expand=True, pady=(6, 0))
                if btn is not None:
                    btn.configure(text="▾ Developer Diagnostics")
            else:
                body.pack_forget()
                if btn is not None:
                    btn.configure(text="▸ Developer Diagnostics")
        except Exception:  # noqa: BLE001
            pass

    def _build_perception_tab(self, tab) -> None:  # noqa: ANN001
        """Compact Perception idle; preview expands only when vision is active."""
        try:
            tab.configure(fg_color=_UI_CANVAS)
        except Exception:  # noqa: BLE001
            pass

        grid = ctk.CTkFrame(tab, fg_color="transparent")
        grid.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        grid.grid_columnconfigure(0, weight=1)
        grid.grid_rowconfigure(1, weight=1)
        self._perception_grid = grid
        self._perception_preview_active = False

        # Compact idle banner (always visible).
        standby_card = ctk.CTkFrame(
            grid,
            fg_color=_UI_CARD,
            corner_radius=14,
            border_width=1,
            border_color=_UI_CARD_BORDER,
        )
        standby_card.grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 8))
        self._vision_standby_lbl = ctk.CTkLabel(
            standby_card,
            text="Vision Standby — Ready for screen OCR / YOLO",
            anchor="w",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=_UI_MUTED,
            wraplength=640,
            justify="left",
        )
        self._vision_standby_lbl.pack(fill="x", padx=14, pady=(12, 4))
        self._vision_status_lbl = ctk.CTkLabel(
            standby_card,
            text="Hybrid grounding: idle · ROI overlay: standby",
            anchor="w",
            text_color=_UI_MUTED,
            font=ctk.CTkFont(size=11),
            wraplength=640,
            justify="left",
        )
        self._vision_status_lbl.pack(fill="x", padx=14, pady=(0, 8))
        btn_row = ctk.CTkFrame(standby_card, fg_color="transparent")
        btn_row.pack(fill="x", padx=14, pady=(0, 12))
        ctk.CTkButton(
            btn_row,
            text="Refresh status",
            width=120,
            height=28,
            corner_radius=999,
            fg_color=_UI_GHOST,
            hover_color="#475569",
            command=self._refresh_perception_status,
        ).pack(side="left", padx=(0, 8))
        ctk.CTkButton(
            btn_row,
            text="Inspect UIA tree",
            width=130,
            height=28,
            corner_radius=999,
            command=self._inspect_uia_tree,
        ).pack(side="left")

        # Expandable workspace — hidden until a visual task is active.
        workspace = ctk.CTkFrame(grid, fg_color="transparent")
        self._perception_workspace = workspace
        workspace.grid_columnconfigure(0, weight=1, uniform="perc")
        workspace.grid_columnconfigure(1, weight=1, uniform="perc")
        workspace.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(workspace, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(4, 6), pady=4)
        right = ctk.CTkFrame(workspace, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 4), pady=4)

        roi_card = self._make_card(left, title="Live Perception feed", padx=4, pady=(0, 0))
        self._roi_preview_lbl = ctk.CTkLabel(
            roi_card,
            text="Live screen feed idle — open this tab to stream (~8 FPS).",
            anchor="w",
            text_color=_UI_MUTED,
            wraplength=360,
            justify="left",
        )
        self._roi_preview_lbl.pack(fill="x", pady=(0, 8))
        self._roi_overlay_lbl = ctk.CTkLabel(
            roi_card,
            text="Grounding overlays: standby",
            anchor="w",
            text_color=_UI_MUTED,
            font=ctk.CTkFont(size=11),
            wraplength=360,
            justify="left",
        )
        self._roi_overlay_lbl.pack(fill="x", pady=(0, 8))
        self._roi_canvas = ctk.CTkFrame(
            roi_card,
            fg_color=_UI_CANVAS,
            corner_radius=12,
            border_width=1,
            border_color=_UI_CARD_BORDER,
            height=220,
        )
        self._roi_canvas.pack(fill="both", expand=True, pady=(0, 4))
        self._perception_feed_lbl = ctk.CTkLabel(
            self._roi_canvas,
            text="Screen / ROI preview",
            text_color=_UI_MUTED,
            font=ctk.CTkFont(size=12),
        )
        self._perception_feed_lbl.pack(expand=True, fill="both", pady=8, padx=8)
        self._perception_feed_img = None

        tree_card = self._make_card(right, title="Win32 UIA tree", padx=4, pady=(0, 0))
        self._uia_tree_box = ctk.CTkTextbox(
            tree_card,
            wrap="word",
            font=ctk.CTkFont(family="Consolas", size=11),
            fg_color=_UI_CANVAS,
            text_color=_UI_TEXT,
            corner_radius=12,
            border_width=0,
        )
        self._uia_tree_box.pack(fill="both", expand=True)
        self._uia_tree_box.insert(
            "1.0",
            "Click “Inspect UIA tree” to dump the foreground window hierarchy "
            "(best-effort; requires Windows UIA backends).\n",
        )
        self._uia_tree_box.configure(state="disabled")
        try:
            self.after(600, self._refresh_perception_status)
        except Exception:  # noqa: BLE001
            pass

    def _set_perception_preview_visible(self, active: bool) -> None:
        """Show ROI/UIA workspace only while a visual task is live."""
        workspace = getattr(self, "_perception_workspace", None)
        if workspace is None:
            return
        want = bool(active)
        prev = bool(getattr(self, "_perception_preview_active", False))
        self._perception_preview_active = want
        if want != prev:
            try:
                if want:
                    workspace.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)
                else:
                    workspace.grid_remove()
            except Exception:  # noqa: BLE001
                pass
        standby = getattr(self, "_vision_standby_lbl", None)
        if standby is not None:
            try:
                if want:
                    standby.configure(
                        text="Vision Active — screen OCR / YOLO",
                        text_color=_UI_EMERALD,
                    )
                else:
                    standby.configure(
                        text="Vision Standby — Ready for screen OCR / YOLO",
                        text_color=_UI_MUTED,
                    )
            except Exception:  # noqa: BLE001
                pass

    def _refresh_perception_status(self) -> None:
        bits: list[str] = []
        overlay_txt = "Grounding overlays: standby"
        vision_active = False
        try:
            from dana.graph.nodes.vision import get_hybrid_grounding

            grounder = get_hybrid_grounding()
            bits.append(f"Hybrid grounding: {type(grounder).__name__}")
        except Exception as exc:  # noqa: BLE001
            bits.append(f"Hybrid grounding: unavailable ({type(exc).__name__})")
        try:
            from dana.vision.overlay import get_overlay

            overlay = get_overlay()
            visible = bool(getattr(overlay, "_visible", False))
            current = getattr(overlay, "_current", None)
            vision_active = bool(visible or current)
            bits.append(
                f"ROI overlay: {'visible' if visible else 'standby'}"
                + (f" @ {current}" if current else "")
            )
            overlay_txt = (
                f"Grounding overlays: {'visible' if visible else 'standby'}"
                + (f" @ {current}" if current else "")
            )
            roi_lbl = getattr(self, "_roi_preview_lbl", None)
            if roi_lbl is not None:
                if current:
                    label = str(getattr(overlay, "_label", "") or "")
                    roi_lbl.configure(
                        text=f"Last ROI {current}"
                        + (f" — {label}" if label else "")
                    )
                else:
                    roi_lbl.configure(
                        text="No ROI captured yet. Grounding hits appear here."
                    )
        except Exception as exc:  # noqa: BLE001
            bits.append(f"ROI overlay: unavailable ({type(exc).__name__})")
            overlay_txt = f"Grounding overlays: unavailable ({type(exc).__name__})"
        lbl = getattr(self, "_vision_status_lbl", None)
        if lbl is not None:
            try:
                lbl.configure(text=" · ".join(bits) if bits else "Vision idle")
            except Exception:  # noqa: BLE001
                pass
        ol = getattr(self, "_roi_overlay_lbl", None)
        if ol is not None:
            try:
                ol.configure(text=overlay_txt)
            except Exception:  # noqa: BLE001
                pass
        try:
            self._set_perception_preview_visible(vision_active)
        except Exception:  # noqa: BLE001
            pass

    def _format_uia_tree_text(self, raw: Any) -> str:
        """Indent UIA dump lines for readability."""
        if isinstance(raw, (list, tuple)):
            lines: list[str] = []
            for item in list(raw)[:120]:
                s = str(item)
                # Heuristic depth from leading spaces / pipe / depth keys.
                if isinstance(item, dict):
                    depth = int(item.get("depth") or item.get("level") or 0)
                    name = (
                        item.get("name")
                        or item.get("control_type")
                        or item.get("AutomationId")
                        or s
                    )
                    lines.append(("  " * max(0, depth)) + str(name))
                else:
                    stripped = s.lstrip()
                    lead = len(s) - len(stripped)
                    depth = lead // 2 if lead else 0
                    # Nested markers like "└─" / "|-" keep as-is; else indent.
                    if stripped.startswith(("└", "├", "|", "+", "-")):
                        lines.append(s)
                    else:
                        lines.append(("  " * depth) + stripped)
            return "\n".join(lines) or "(no controls)"
        text = str(raw)
        out: list[str] = []
        for line in text.splitlines():
            stripped = line.lstrip(" \t")
            lead = len(line) - len(stripped)
            indent = "  " * (lead // 2) if lead else ""
            out.append(indent + stripped if lead else line)
        return "\n".join(out) if out else text

    def _inspect_uia_tree(self) -> None:
        box = getattr(self, "_uia_tree_box", None)
        if box is None:
            return

        def _worker() -> None:
            text = "(empty)"
            try:
                from dana.vision.uia_provider import Win32UIAProvider

                provider = Win32UIAProvider()
                dump = getattr(provider, "dump_tree", None) or getattr(
                    provider, "list_controls", None
                )
                if callable(dump):
                    result = dump()
                    text = self._format_uia_tree_text(result)
                else:
                    # Best-effort: probe find_element path existence.
                    text = (
                        "UIA provider loaded. No dump_tree API — "
                        f"provider={type(provider).__name__}. "
                        "Hybrid grounding still uses UIA hit-testing at runtime."
                    )
            except Exception as exc:  # noqa: BLE001
                text = f"UIA inspect failed: {type(exc).__name__}: {exc}"

            def _ui() -> None:
                try:
                    box.configure(state="normal")
                    box.delete("1.0", "end")
                    box.insert("1.0", text + "\n")
                    box.configure(state="disabled")
                except Exception:  # noqa: BLE001
                    pass
                try:
                    self._refresh_perception_status()
                except Exception:  # noqa: BLE001
                    pass
                # Keep workspace open after inspect even if ROI is idle.
                try:
                    self._perception_preview_active = False
                    self._set_perception_preview_visible(True)
                except Exception:  # noqa: BLE001
                    pass

            try:
                self.after(0, _ui)
            except Exception:  # noqa: BLE001
                _ui()

        try:
            self._perception_preview_active = False
            self._set_perception_preview_visible(True)
        except Exception:  # noqa: BLE001
            pass
        threading.Thread(target=_worker, name="UIAInspect", daemon=True).start()

    def _perception_tab_visible(self) -> bool:
        """True when the Perception tab is the active notebook page."""
        tabs = getattr(self, "_tabs", None)
        if tabs is None:
            return False
        try:
            return str(tabs.get()) == "Perception"
        except Exception:  # noqa: BLE001
            return False

    def _schedule_perception_feed(self) -> None:
        """Lightweight ~8 FPS mss feed while Perception tab is visible."""
        if not self.winfo_exists():
            return
        try:
            if self._perception_tab_visible():
                self._set_perception_preview_visible(True)
                if not bool(getattr(self, "_perception_feed_busy", False)):
                    self._perception_feed_busy = True
                    threading.Thread(
                        target=self._capture_perception_frame,
                        name="PerceptionFeed",
                        daemon=True,
                    ).start()
            else:
                # Idle when tab hidden — keep UI responsive.
                lbl = getattr(self, "_roi_preview_lbl", None)
                if lbl is not None:
                    try:
                        lbl.configure(
                            text="Live screen feed idle — open this tab to stream (~8 FPS)."
                        )
                    except Exception:  # noqa: BLE001
                        pass
        except Exception:  # noqa: BLE001
            pass
        try:
            self._perception_feed_job = self.after(125, self._schedule_perception_feed)
        except Exception:  # noqa: BLE001
            self._perception_feed_job = None

    def _capture_perception_frame(self) -> None:
        """Background: grab primary monitor via mss, downscale, push to UI."""
        pil_img = None
        err = ""
        try:
            import mss
            from PIL import Image

            factory = getattr(mss, "mss", None) or getattr(mss, "MSS", None)
            with factory() as sct:
                mon = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
                raw = sct.grab(mon)
                img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
                # Downscale for ~5–10 FPS UI (non-blocking).
                resample = getattr(
                    getattr(Image, "Resampling", Image), "BILINEAR", Image.BILINEAR
                )
                img.thumbnail((480, 270), resample)
                pil_img = img
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"

        def _apply() -> None:
            self._perception_feed_busy = False
            feed = getattr(self, "_perception_feed_lbl", None)
            if feed is None or not self.winfo_exists():
                return
            if pil_img is None:
                try:
                    feed.configure(image=None, text=err or "Capture unavailable")
                    self._perception_feed_img = None
                except Exception:  # noqa: BLE001
                    pass
                return
            try:
                ctk_img = ctk.CTkImage(
                    light_image=pil_img,
                    dark_image=pil_img,
                    size=pil_img.size,
                )
                self._perception_feed_img = ctk_img
                feed.configure(image=ctk_img, text="")
                status = getattr(self, "_roi_preview_lbl", None)
                if status is not None:
                    status.configure(text=f"Live feed {pil_img.size[0]}×{pil_img.size[1]}")
            except Exception as exc:  # noqa: BLE001
                try:
                    feed.configure(text=f"Feed error: {type(exc).__name__}")
                except Exception:  # noqa: BLE001
                    pass

        try:
            self.after(0, _apply)
        except Exception:  # noqa: BLE001
            self._perception_feed_busy = False

    def _build_memory_appearance_row(self, tab) -> None:  # noqa: ANN001
        """Row 3 — Episodic Memory search + Appearance theme."""
        grid = ctk.CTkFrame(tab, fg_color="transparent")
        grid.pack(fill="x", expand=False, padx=4, pady=(4, 4))
        try:
            grid.grid_columnconfigure(0, weight=1, uniform="mem3")
            grid.grid_columnconfigure(1, weight=1, uniform="mem3")
        except Exception:  # noqa: BLE001
            pass
        left = ctk.CTkFrame(grid, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(4, 6), pady=4)
        right = ctk.CTkFrame(grid, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 4), pady=4)
        self._build_episodic_memory_section(left)
        self._build_appearance_card(right)

    def _build_updates_dictation_row(self, tab) -> None:  # noqa: ANN001
        """Row 4 — System Updates + Dictation Latch (no live session telemetry)."""
        grid = ctk.CTkFrame(tab, fg_color="transparent")
        grid.pack(fill="x", expand=False, padx=4, pady=(4, 8))
        try:
            grid.grid_columnconfigure(0, weight=1, uniform="mem4")
            grid.grid_columnconfigure(1, weight=1, uniform="mem4")
        except Exception:  # noqa: BLE001
            pass
        left = ctk.CTkFrame(grid, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(4, 6), pady=4)
        right = ctk.CTkFrame(grid, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 4), pady=4)
        self._build_system_updates_card(left)
        self._build_dictation_tab(right)

    def _build_memory_settings_grid(self, tab) -> None:  # noqa: ANN001
        """Legacy entrypoint — delegates to the reordered Memory & Settings rows."""
        self._build_memory_appearance_row(tab)
        self._build_updates_dictation_row(tab)

    def _build_system_updates_card(self, tab) -> None:  # noqa: ANN001
        """Stage 9.3 — System Updates card (left column) + Phase 1 OTA chrome."""
        updates = self._make_card(
            tab, title="System Updates", padx=4, pady=(0, 8), expand=False
        )
        ctk.CTkLabel(
            updates,
            text=(
                "Fetch from GitHub, compare revisions, then update dependencies "
                "and restart Dānā in one click."
            ),
            anchor="w",
            justify="left",
            wraplength=360,
            text_color=_UI_MUTED,
        ).pack(fill="x", pady=(0, 8))

        # Auto-Update Mode (Silent / Manual) — wired to OTAManifestManager.
        mode_row = ctk.CTkFrame(updates, fg_color="transparent")
        mode_row.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(
            mode_row,
            text="Auto-Update Mode",
            anchor="w",
            text_color=_UI_MUTED,
            font=ctk.CTkFont(size=12),
        ).pack(side="left")
        initial_mode = "Manual"
        try:
            from dana.updater.manifest import get_ota_manager

            self._ota_manager = get_ota_manager()
            initial_mode = (
                "Silent"
                if self._ota_manager.auto_update_mode == "silent"
                else "Manual"
            )
        except Exception:  # noqa: BLE001
            self._ota_manager = None
        self._ota_mode_var = ctk.StringVar(value=initial_mode)
        self._ota_mode_menu = ctk.CTkOptionMenu(
            mode_row,
            values=["Silent", "Manual"],
            variable=self._ota_mode_var,
            width=110,
            height=28,
            corner_radius=8,
            text_color=_UI_TEXT,
            command=self._on_ota_mode_changed,
        )
        self._ota_mode_menu.pack(side="right")

        self._ota_pill_lbl = ctk.CTkLabel(
            updates,
            text="[UP TO DATE]",
            anchor="w",
            text_color=_UI_MUTED,
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self._ota_pill_lbl.pack(fill="x", pady=(0, 4))

        # Phase 2B — Active slot + staging health pills.
        _slot_active_color = _UI_EMERALD
        try:
            _slot_active_color = getattr(_UI_THEME, "STATUS_SLOT_ACTIVE", _UI_EMERALD)
        except Exception:  # noqa: BLE001
            _slot_active_color = _UI_EMERALD
        self._ota_slot_lbl = ctk.CTkLabel(
            updates,
            text="Active: Slot A (v0.0.0)",
            anchor="w",
            text_color=_slot_active_color,
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self._ota_slot_lbl.pack(fill="x", pady=(0, 2))
        self._ota_staging_lbl = ctk.CTkLabel(
            updates,
            text="Staging: idle",
            anchor="w",
            text_color=_UI_MUTED,
            font=ctk.CTkFont(size=12),
        )
        self._ota_staging_lbl.pack(fill="x", pady=(0, 4))

        self._update_status_lbl = ctk.CTkLabel(
            updates,
            text="Status: idle",
            anchor="w",
            text_color=_UI_MUTED,
            font=ctk.CTkFont(size=12),
        )
        self._update_status_lbl.pack(fill="x", pady=(0, 8))
        btn_row = ctk.CTkFrame(updates, fg_color="transparent")
        btn_row.pack(fill="x", side="bottom")
        self._update_check_btn = ctk.CTkButton(
            btn_row,
            text="Check for Updates",
            width=150,
            height=30,
            corner_radius=999,
            fg_color=_UI_GHOST,
            hover_color="#475569",
            border_width=1,
            border_color=_UI_CARD_BORDER,
            text_color=_UI_TEXT,
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._on_check_for_updates,
        )
        self._update_check_btn.pack(side="right", padx=(8, 0))
        self._update_apply_btn = ctk.CTkButton(
            btn_row,
            text="Update & Restart",
            width=150,
            height=30,
            corner_radius=999,
            fg_color=_UI_AMBER,
            hover_color="#D97706",
            text_color="#FFFFFF",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._on_apply_update_and_restart,
        )
        # Hidden until check_for_updates() reports True.
        self._ota_hot_apply_btn = ctk.CTkButton(
            btn_row,
            text="Hot Apply",
            width=110,
            height=30,
            corner_radius=999,
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._on_ota_hot_apply,
        )
        # Hidden until OTAManifestManager reports a staged patch.
        try:
            self._refresh_ota_ui()
        except Exception:  # noqa: BLE001
            pass

    def _build_episodic_memory_section(self, tab) -> None:  # noqa: ANN001
        """Episodic memory keyword search (SQLite store)."""
        card = self._make_card(tab, title="Episodic Memory", padx=4, pady=(0, 8), expand=False)
        self._memory_query = ctk.CTkEntry(
            card,
            placeholder_text="Search preferences & facts…",
            height=32,
            corner_radius=8,
            fg_color=_UI_GHOST,
            border_width=1,
            border_color=_UI_CARD_BORDER,
            text_color=_UI_TEXT,
        )
        self._memory_query.pack(fill="x", pady=(0, 8))
        self._memory_query.bind("<Return>", lambda _e: self._run_episodic_search())
        self._memory_results = ctk.CTkTextbox(
            card,
            wrap="word",
            height=140,
            font=ctk.CTkFont(family="Consolas", size=11),
            fg_color=_UI_CANVAS,
            text_color=_UI_TEXT,
            corner_radius=10,
        )
        self._memory_results.pack(fill="x", pady=(0, 8))
        self._memory_results.insert("1.0", "(enter a query to search episodic memory)\n")
        self._memory_results.configure(state="disabled")
        actions = ctk.CTkFrame(card, fg_color="transparent")
        actions.pack(fill="x")
        ctk.CTkButton(
            actions,
            text="Search",
            width=88,
            height=30,
            corner_radius=999,
            command=self._run_episodic_search,
        ).pack(side="right")

    def _run_episodic_search(self) -> None:
        box = getattr(self, "_memory_results", None)
        entry = getattr(self, "_memory_query", None)
        if box is None:
            return
        try:
            query = str(entry.get() if entry is not None else "").strip()
        except Exception:  # noqa: BLE001
            query = ""
        try:
            from dana.memory.store import get_episodic_store

            facts = get_episodic_store().search_facts(query, limit=24)
            lines = []
            for fact in facts:
                key = fact.get("key") or ""
                val = fact.get("value") or ""
                cat = fact.get("category") or ""
                lines.append(f"[{cat}] {key} = {val}")
            text = "\n".join(lines) if lines else "(no matching facts)\n"
        except Exception as exc:  # noqa: BLE001
            text = f"Memory search failed: {type(exc).__name__}: {exc}\n"
        try:
            box.configure(state="normal")
            box.delete("1.0", "end")
            box.insert("1.0", text if text.endswith("\n") else text + "\n")
            box.configure(state="disabled")
        except Exception:  # noqa: BLE001
            pass

    def _build_settings_tab(self, tab) -> None:  # noqa: ANN001
        """Row 1 — Engine Runtime (Hybrid Broker, Wake Word, Startup)."""
        try:
            tab.configure(fg_color=_UI_CANVAS)
        except Exception:  # noqa: BLE001
            pass

        stats_card = self._make_card(
            tab, title="Engine Runtime", padx=8, pady=(8, 4), expand=False
        )
        ctk.CTkLabel(
            stats_card,
            text="Wake word",
            anchor="w",
            text_color=_UI_MUTED,
        ).pack(fill="x", pady=(0, 4))
        self._settings_wake_lbl = ctk.CTkLabel(
            stats_card,
            text="Active wake word: Dana",
            anchor="w",
            justify="left",
            wraplength=640,
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=_UI_EMERALD,
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
            text_color=_UI_MUTED,
        )
        self._open_window_chk.pack(anchor="w", pady=(0, 8))
        try:
            from dana.settings import is_hybrid_planner_enabled

            hybrid_on = bool(is_hybrid_planner_enabled())
        except Exception:  # noqa: BLE001
            hybrid_on = False
        self._hybrid_planner_var = ctk.BooleanVar(value=hybrid_on)
        self._hybrid_planner_chk = ctk.CTkCheckBox(
            stats_card,
            text="Hybrid Broker (Cloud Planner)",
            variable=self._hybrid_planner_var,
            command=self._on_hybrid_planner_toggle,
            text_color=_UI_MUTED,
        )
        self._hybrid_planner_chk.pack(anchor="w", pady=(0, 4))
        ctk.CTkLabel(
            stats_card,
            text=(
                "Off by default (fully local). When on, Meta-Broker epic split "
                "and Supervisor DAG JSON may use Gemini if GEMINI_API_KEY is set "
                "in .env. Workers and the runtime harness always stay local."
            ),
            anchor="w",
            justify="left",
            wraplength=640,
            text_color=_UI_MUTED,
        ).pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(
            stats_card,
            text=(
                "Mic and speaker follow the OS System Default automatically. "
                "Pipeline mode is shown in the header and Assistive Orb."
            ),
            anchor="w",
            justify="left",
            wraplength=640,
            text_color=_UI_MUTED,
        ).pack(fill="x", pady=(0, 4))
        # Compatibility stubs (menus removed — autonomous System Default).
        self.mic_menu = None
        self.speaker_menu = None
        self.save_btn = None

        remote_card = self._make_card(
            tab, title="Remote Access", padx=8, pady=(0, 4), expand=False
        )
        ctk.CTkLabel(
            remote_card,
            text=(
                "Pushover push notifications and a personal Telegram bot let "
                "Dana reach you (or take commands from you) while running "
                "unattended in the background."
            ),
            anchor="w",
            justify="left",
            wraplength=640,
            text_color=_UI_MUTED,
        ).pack(fill="x", pady=(0, 8))
        self._integrations_setup_btn = ctk.CTkButton(
            remote_card,
            text="Integrations Setup (Pushover & Telegram)",
            height=32,
            corner_radius=8,
            command=self._show_integrations_setup_guide,
        )
        self._integrations_setup_btn.pack(anchor="w")

    def _show_integrations_setup_guide(self) -> None:  # noqa: ANN001
        """Popup with the Pushover/Telegram setup guide (Settings tab)."""
        try:
            from dana.ui.settings import get_integrations_setup_text

            guide_text = get_integrations_setup_text()
        except Exception:  # noqa: BLE001
            guide_text = "Integrations setup guide is unavailable right now."

        try:
            win = ctk.CTkToplevel(self)
            win.title("Integrations Setup")
            win.geometry("640x520")
            box = ctk.CTkTextbox(
                win,
                wrap="word",
                font=ctk.CTkFont(family="Segoe UI", size=13),
                fg_color=_UI_CANVAS,
                corner_radius=8,
                border_width=0,
            )
            box.pack(fill="both", expand=True, padx=12, pady=12)
            box.insert("1.0", guide_text)
            box.configure(state="disabled")
            win.lift()
            win.focus_force()
        except Exception:  # noqa: BLE001
            pass

    def _build_appearance_card(self, tab) -> None:  # noqa: ANN001
        """Appearance / theme picker (Memory & Settings row 3)."""
        appear_card = self._make_card(
            tab, title="Appearance", padx=4, pady=(0, 8), expand=False
        )
        ctk.CTkLabel(
            appear_card, text="UI Theme", anchor="w", text_color=_UI_MUTED
        ).pack(fill="x", pady=(0, 4))
        try:
            from dana.ui.theme import THEME_NAMES, active_theme_name

            theme_values = list(THEME_NAMES)
            initial_theme = active_theme_name()
        except Exception:  # noqa: BLE001
            theme_values = ["Obsidian Mint", "Cyber Amber", "Ghost Light"]
            initial_theme = "Obsidian Mint"
        self._theme_var = ctk.StringVar(value=initial_theme)
        self._theme_menu = ctk.CTkOptionMenu(
            appear_card,
            values=theme_values,
            variable=self._theme_var,
            corner_radius=10,
            command=self._on_ui_theme_changed,
        )
        self._theme_menu.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(
            appear_card,
            text="Obsidian Mint · Cyber Amber · Ghost Light — switches instantly.",
            anchor="w",
            justify="left",
            wraplength=320,
            text_color=_UI_MUTED,
            font=ctk.CTkFont(size=11),
        ).pack(fill="x", pady=(0, 4))
        self.apply_note = ctk.CTkLabel(
            appear_card,
            text="Audio: System Default (Auto)",
            text_color=_UI_MUTED,
            anchor="w",
            wraplength=320,
            justify="left",
            font=ctk.CTkFont(size=11),
        )
        self.apply_note.pack(fill="x", pady=(4, 0))

    def _sync_ui_theme_aliases(self) -> None:
        """Refresh module-level ``_UI_*`` aliases from ``dana.ui.theme``."""
        global _UI_CANVAS, _UI_CARD, _UI_CARD_BORDER, _UI_GHOST, _UI_MUTED
        global _UI_TEXT, _UI_ACCENT, _UI_ACCENT_HOVER, _UI_EMERALD, _UI_EMERALD_HOVER
        global _UI_ROSE, _UI_ROSE_HOVER, _UI_AMBER
        try:
            from dana.ui import theme as T

            _UI_CANVAS = T.BG
            _UI_CARD = T.CARD
            _UI_CARD_BORDER = T.BORDER
            _UI_GHOST = T.GHOST
            _UI_MUTED = T.MUTED
            _UI_TEXT = T.TEXT
            _UI_ACCENT = T.ACCENT
            _UI_ACCENT_HOVER = T.ACCENT_HOVER
            _UI_EMERALD = T.EMERALD
            _UI_EMERALD_HOVER = T.EMERALD_HOVER
            _UI_ROSE = T.ROSE
            _UI_ROSE_HOVER = T.ROSE_HOVER
            _UI_AMBER = T.AMBER
        except Exception:  # noqa: BLE001
            pass

    def _on_ui_theme_changed(self, choice: str) -> None:
        """Runtime theme switch — recolor dashboard tree."""
        try:
            from dana.ui.theme import set_theme

            set_theme(str(choice), root=self)
        except Exception as exc:  # noqa: BLE001
            log("UI", f"WARNING: theme switch failed ({exc})")
            return
        self._sync_ui_theme_aliases()
        try:
            self.configure(fg_color=_UI_CANVAS)
        except Exception:  # noqa: BLE001
            pass
        note = getattr(self, "apply_note", None)
        if note is not None:
            try:
                note.configure(text=f"Theme: {choice}")
            except Exception:  # noqa: BLE001
                pass
        log("UI", f"UI Theme → {choice}")

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

    def _ota_mgr(self) -> Any:
        mgr = getattr(self, "_ota_manager", None)
        if mgr is not None:
            return mgr
        try:
            from dana.updater.manifest import get_ota_manager

            self._ota_manager = get_ota_manager()
            return self._ota_manager
        except Exception:  # noqa: BLE001
            return None

    def _on_ota_mode_changed(self, choice: str) -> None:
        mgr = self._ota_mgr()
        if mgr is None:
            return
        try:
            mgr.set_auto_update_mode("silent" if str(choice).lower() == "silent" else "manual")
        except Exception as exc:  # noqa: BLE001
            log("Updater", f"auto_update_mode change failed: {exc}")
        self._refresh_ota_ui()

    def _refresh_ota_ui(self) -> None:
        """Sync OTA status pill + Hot Apply visibility (headless-safe)."""
        mgr = self._ota_mgr()
        pill = getattr(self, "_ota_pill_lbl", None)
        slot_lbl = getattr(self, "_ota_slot_lbl", None)
        staging_lbl = getattr(self, "_ota_staging_lbl", None)
        btn = getattr(self, "_ota_hot_apply_btn", None)

        def _token(name: str, fallback: str) -> str:
            try:
                return str(getattr(_UI_THEME, name, fallback))
            except Exception:  # noqa: BLE001
                return fallback

        if mgr is None:
            if pill is not None:
                try:
                    pill.configure(
                        text="[UP TO DATE]",
                        text_color=_token("STATUS_IDLE", _UI_MUTED),
                    )
                except Exception:  # noqa: BLE001
                    pass
            return
        try:
            st = mgr.state()
        except Exception:  # noqa: BLE001
            return
        pill_text = st.status_pill()
        health = str(getattr(st, "staging_health", "idle") or "idle")
        if pill is not None:
            try:
                if health == "checking":
                    color = _token("STATUS_STAGING", _UI_AMBER)
                elif health == "failed":
                    color = _token("STATUS_FAILED", _UI_ROSE)
                elif health == "healthy":
                    color = _token("STATUS_HEALTHY", _UI_EMERALD)
                elif st.staged_version:
                    color = _token("STATUS_UPDATE_READY", _UI_EMERALD)
                elif st.update_available:
                    color = _token("STATUS_UPDATE_AVAILABLE", _UI_ACCENT)
                else:
                    color = _token("STATUS_IDLE", _UI_MUTED)
                pill.configure(text=pill_text, text_color=color)
            except Exception:  # noqa: BLE001
                pass
        if slot_lbl is not None:
            try:
                label = str(getattr(st, "active_slot_label", "") or "").strip()
                if not label:
                    label = f"Slot A (v{st.local_version.lstrip('vV')})"
                slot_lbl.configure(
                    text=f"Active: {label}",
                    text_color=_token("STATUS_SLOT_ACTIVE", _UI_EMERALD),
                )
            except Exception:  # noqa: BLE001
                pass
        if staging_lbl is not None:
            try:
                staging_colors = {
                    "idle": _token("STATUS_IDLE", _UI_MUTED),
                    "checking": _token("STATUS_STAGING", _UI_AMBER),
                    "healthy": _token("STATUS_HEALTHY", _UI_EMERALD),
                    "failed": _token("STATUS_FAILED", _UI_ROSE),
                }
                staging_lbl.configure(
                    text=f"Staging: {health}",
                    text_color=staging_colors.get(health, _UI_MUTED),
                )
            except Exception:  # noqa: BLE001
                pass
        if btn is None:
            return
        try:
            if st.staged_version:
                if not btn.winfo_ismapped():
                    check = self._update_check_btn
                    if check is not None and check.winfo_ismapped():
                        btn.pack(side="right", padx=(0, 8), before=check)
                    else:
                        btn.pack(side="right", padx=(0, 8))
            else:
                btn.pack_forget()
        except Exception:  # noqa: BLE001
            pass

    def _on_ota_hot_apply(self) -> None:
        """Promote staged OTA via blue-green health gate + tool reload."""
        if self._update_busy:
            return
        mgr = self._ota_mgr()
        if mgr is None:
            self._set_update_status("OTA manager unavailable.", color="#F87171")
            return
        # Attach sidecar IPC for hot_restart when available.
        try:
            client = getattr(self, "_daemon_client", None)
            if client is not None and getattr(mgr, "_ipc_client", None) is None:
                mgr._ipc_client = client
        except Exception:  # noqa: BLE001
            pass
        self._update_busy = True
        try:
            if self._ota_hot_apply_btn is not None:
                self._ota_hot_apply_btn.configure(state="disabled")
        except Exception:  # noqa: BLE001
            pass
        self._set_update_status("Blue-green promote + health check…", color=_UI_AMBER)

        def _worker() -> None:
            err = ""
            version = ""
            active_label = ""
            try:
                result = mgr.hot_apply()
                version = str((result or {}).get("version") or "")
                active_label = str((result or {}).get("active_label") or "")
            except Exception as exc:  # noqa: BLE001
                err = f"{type(exc).__name__}: {exc}"
                log("Updater", f"hot_apply failed: {err}")

            def _ui() -> None:
                self._update_busy = False
                try:
                    if self._ota_hot_apply_btn is not None:
                        self._ota_hot_apply_btn.configure(state="normal")
                except Exception:  # noqa: BLE001
                    pass
                if err:
                    self._set_update_status(f"Hot Apply failed: {err}", color="#F87171")
                else:
                    suffix = f" — Active: {active_label}" if active_label else ""
                    self._set_update_status(
                        f"Hot Apply complete — v{version or '?'}{suffix}",
                        color="#66BB6A",
                    )
                self._refresh_ota_ui()

            try:
                self.after(0, _ui)
            except Exception:  # noqa: BLE001
                _ui()

        threading.Thread(target=_worker, name="OTAHotApply", daemon=True).start()

    def _set_update_available(self, available: bool) -> None:
        btn = self._update_apply_btn
        if btn is None:
            return
        try:
            if available:
                if not btn.winfo_ismapped():
                    check = self._update_check_btn
                    if check is not None and check.winfo_ismapped():
                        btn.pack(side="right", padx=(0, 8), before=check)
                    else:
                        btn.pack(side="right", padx=(0, 8))
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
        self._select_tab("Assistant & Tasks")
        self._flash_engine_warning("Please Engage Engine First.")
        return False

    def _set_vault_status(self, message: str) -> None:
        """Persistent banner (empty clears it) — see ``_vault_warn_lbl``."""
        lbl = self._vault_warn_lbl
        if lbl is None:
            return
        try:
            lbl.configure(text=str(message))
        except Exception:  # noqa: BLE001
            pass

    def _on_vault_unlock_request(self, reason: str) -> None:
        """shared_state listener callback — runs on the AgentLoop thread.

        Hands off to the Tk main thread via ``after()`` since CTk widgets may
        only be created/touched there. ``reason == ""`` means "unlocked" —
        clear the banner / close any open prompt instead of showing one.
        """
        try:
            if reason:
                self.after(0, lambda: self._show_vault_unlock_dialog(reason))
            else:
                self.after(0, self._clear_vault_unlock_prompt)
        except Exception:  # noqa: BLE001
            pass

    def _on_spec_approval_requested(self, payload: dict) -> None:
        """shared_state listener — runs on the AgentLoop thread; hand off to Tk."""
        try:
            if hasattr(self, "show_spec_approval"):
                self.after(0, lambda p=payload: self.show_spec_approval(p))
        except Exception:  # noqa: BLE001
            pass

    def _on_dictation_sessions_changed(self) -> None:
        """shared_state listener — runs on the AgentLoop thread; hand off to Tk."""
        try:
            if hasattr(self, "refresh_dictation_sessions"):
                self.after(0, self.refresh_dictation_sessions)
        except Exception:  # noqa: BLE001
            pass

    def _clear_vault_unlock_prompt(self) -> None:
        self._set_vault_status("")
        win = self._vault_unlock_win
        self._vault_unlock_win = None
        if win is not None:
            try:
                win.destroy()
            except Exception:  # noqa: BLE001
                pass

    def _show_vault_unlock_dialog(self, reason: str) -> None:
        """Modal passcode prompt; unblocks ``unlock_dana_memory()`` on submit/cancel.

        Runs on the Tk main thread (scheduled by ``_on_vault_unlock_request``).
        """
        self._set_vault_status("Vault Locked — Enter Passcode to Unlock")
        try:
            self.show_window()
        except Exception:  # noqa: BLE001
            pass

        existing = self._vault_unlock_win
        if existing is not None:
            try:
                existing.destroy()
            except Exception:  # noqa: BLE001
                pass
            self._vault_unlock_win = None

        def _submit(password: Optional[str]) -> None:
            win = self._vault_unlock_win
            self._vault_unlock_win = None
            if win is not None:
                try:
                    win.destroy()
                except Exception:  # noqa: BLE001
                    pass
            if password:
                self._set_vault_status("Vault Locked — Unlocking...")
            supply_vault_unlock_response(password)

        try:
            win = ctk.CTkToplevel(self)
            self._vault_unlock_win = win
            win.title("Vault Locked")
            win.geometry("460x220")
            win.resizable(False, False)
            ctk.CTkLabel(
                win,
                text="Vault Locked — Enter Passcode to Unlock",
                font=ctk.CTkFont(size=15, weight="bold"),
                text_color="#EF4444",
            ).pack(padx=16, pady=(16, 4), anchor="w")
            ctk.CTkLabel(
                win,
                text=str(reason),
                wraplength=420,
                justify="left",
                anchor="w",
                text_color=_UI_MUTED,
            ).pack(padx=16, pady=(0, 10), anchor="w", fill="x")
            entry = ctk.CTkEntry(
                win,
                show="*",
                placeholder_text="Master Password / Recovery Key",
                width=420,
            )
            entry.pack(padx=16, pady=(0, 12))
            entry.focus_set()

            btn_row = ctk.CTkFrame(win, fg_color="transparent")
            btn_row.pack(padx=16, pady=(0, 12), fill="x")
            ctk.CTkButton(
                btn_row,
                text="Cancel",
                fg_color="transparent",
                border_width=1,
                command=lambda: _submit(None),
            ).pack(side="left")
            ctk.CTkButton(
                btn_row,
                text="Unlock",
                command=lambda: _submit(entry.get().strip() or None),
            ).pack(side="right")
            entry.bind("<Return>", lambda _e: _submit(entry.get().strip() or None))
            win.protocol("WM_DELETE_WINDOW", lambda: _submit(None))
            win.transient(self)
            win.grab_set()
            win.lift()
            win.focus_force()
        except Exception as exc:  # noqa: BLE001
            log("UI", f"WARNING: vault unlock dialog failed to open ({exc})")
            self._vault_unlock_win = None
            supply_vault_unlock_response(None)

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
        """Sync Engage/Standby toggle chrome with ``engine_active``."""
        active = bool(self.engine_active)
        stopped = bool(getattr(self, "_engine_stopped", False)) and not active
        status = self._engine_status_lbl
        if status is not None:
            try:
                if active:
                    status.configure(
                        text="  ● ACTIVE | Local Engine  ",
                        text_color=_UI_EMERALD,
                        fg_color=_UI_GHOST,
                    )
                elif stopped:
                    status.configure(
                        text="  ● STOPPED | Local Engine  ",
                        text_color=_UI_ROSE,
                        fg_color=_UI_GHOST,
                    )
                else:
                    status.configure(
                        text="  ● STANDBY | Local Engine  ",
                        text_color=_UI_MUTED,
                        fg_color=_UI_GHOST,
                    )
            except Exception:  # noqa: BLE001
                pass
        header_status = getattr(self, "_header_status_lbl", None)
        if header_status is not None:
            try:
                if active:
                    header_status.configure(text="• ACTIVE", text_color=_UI_EMERALD)
                elif stopped:
                    header_status.configure(text="• STOPPED", text_color=_UI_ROSE)
                else:
                    header_status.configure(text="• STANDBY", text_color=_UI_MUTED)
            except Exception:  # noqa: BLE001
                pass
        toggle = getattr(self, "_engage_toggle_btn", None) or self._engage_btn
        try:
            if toggle is not None:
                if active:
                    toggle.configure(
                        text="Engaged",
                        state="normal",
                        fg_color=_UI_EMERALD,
                        hover_color="#059669",
                        text_color="#ECFDF5",
                    )
                else:
                    toggle.configure(
                        text="Standby",
                        state="normal",
                        fg_color=_UI_GHOST,
                        hover_color="#475569",
                        text_color=_UI_TEXT,
                    )
        except Exception:  # noqa: BLE001
            pass

    def toggle_engine_engage(self) -> None:
        """Single HUD control: Engaged ↔ Standby."""
        if bool(self.engine_active) and is_engine_engaged():
            self.standby_engine()
        else:
            self.engage_engine()

    def engage_engine(self) -> None:
        """Stage 8.9.7 — arm engine, apply mixer, lock Behavior sliders."""
        applied = self._apply_behavior_mixer_payload()
        self.engine_active = True
        self._engine_stopped = False
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
                "Dana",
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
        # Soft-pause: clear pending mic latch; do NOT touch stop_event / STOP DANA.
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
                "Dana",
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
        """Quick action — focus Assistant silent chat entry (no transcript spam)."""
        if not self._require_engine():
            return
        self._select_tab("Assistant & Tasks")
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
        """Quick action — open Memory & Settings and arm the latch if cold."""
        if not self._require_engine():
            return
        self._select_tab("Memory & Settings")
        if not self._dictation_active:
            try:
                self._toggle_dictation_mode()
            except Exception:  # noqa: BLE001
                pass

    def _dashboard_open_trace(self) -> None:
        """Open a dedicated Diagnostics / Live Trace overlay (not Memory & Settings)."""
        try:
            existing = getattr(self, "_diag_overlay", None)
            if existing is not None and bool(existing.winfo_exists()):
                try:
                    existing.deiconify()
                    existing.lift()
                    existing.focus_force()
                    return
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass
        try:
            from dana.ui.trace_window import LiveTraceWindow

            win = LiveTraceWindow(self)
            self._diag_overlay = win
            try:
                win.title("Dānā — Diagnostics / Live Trace")
            except Exception:  # noqa: BLE001
                pass
            try:
                win.lift()
                win.focus_force()
            except Exception:  # noqa: BLE001
                pass
            return
        except Exception as exc:  # noqa: BLE001
            log("UI", f"WARNING: Diagnostics overlay unavailable ({exc})")
        # Last resort: expand embedded Developer Diagnostics without stealing
        # the Memory & Settings tab identity from the header segment.
        try:
            self._select_tab("Memory & Settings")
            if not bool(getattr(self, "_diag_expanded", False)):
                self._toggle_developer_diagnostics()
        except Exception:  # noqa: BLE001
            pass

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
            # Readable slate palette — no electric cyan / neon green.
            tk_text.tag_configure(
                "jason",
                foreground="#C084FC",
                font=("Segoe UI", 14, "bold"),
            )
            tk_text.tag_configure(
                "llama",
                foreground=_UI_TEXT,
                font=("Segoe UI", 14),
            )
            tk_text.tag_configure(
                "deepseek",
                foreground=_UI_ROSE,
                font=("Courier New", 10),
            )
            tk_text.tag_configure(
                "vision",
                foreground=_UI_EMERALD,
                font=("Segoe UI", 14),
            )
            tk_text.tag_configure(
                "typist",
                foreground=_UI_AMBER,
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
                foreground=_UI_TEXT,
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
            "dana": "llama",
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
        if sp.startswith(("dana", "dana")) or "ollama" in sp or "llama" in sp:
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
            tab, title="Dictation Latch", padx=4, pady=(0, 8), expand=False
        )
        ctk.CTkLabel(
            control_card,
            text=(
                "When ON, every utterance is logged with Florence OCR visual state. "
                "You can also say “dictate …” while OFF."
            ),
            anchor="w",
            text_color=_UI_MUTED,
            wraplength=360,
            justify="left",
        ).pack(fill="x", pady=(0, 12))

        controls = ctk.CTkFrame(control_card, fg_color="transparent")
        controls.pack(fill="x", pady=(0, 4))
        # Unified pill toggle — OFF: dim gray + rose pill; ON: accent glow + DICTATING.
        self.dictation_btn = ctk.CTkButton(
            controls,
            text="  ●  OFF  ",
            width=140,
            height=36,
            corner_radius=999,
            fg_color=_UI_GHOST,
            hover_color="#475569",
            border_width=1,
            border_color=_UI_CARD_BORDER,
            text_color=_UI_ROSE,
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
            text_color=_UI_ROSE,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=_UI_CANVAS,
            corner_radius=999,
            padx=10,
            pady=4,
        )
        actions = ctk.CTkFrame(control_card, fg_color="transparent")
        actions.pack(fill="x", pady=(8, 0))
        ctk.CTkButton(
            actions,
            text="Refresh status",
            width=110,
            height=30,
            corner_radius=999,
            fg_color=_UI_GHOST,
            hover_color="#475569",
            border_width=1,
            border_color=_UI_CARD_BORDER,
            text_color=_UI_TEXT,
            font=ctk.CTkFont(size=11),
            command=self.refresh_dictation_sessions,
        ).pack(side="right")
        # Recent Sessions telemetry lives in the Diagnostics overlay only.
        # Keep attribute stub so refresh_dictation_sessions stays safe.
        if getattr(self, "dictation_list", None) is None:
            self.dictation_list = None
        # Sync latch from Blackboard (may already be on).
        try:
            from dana.memory.blackboard import is_dictation_mode

            self._set_dictation_ui(bool(is_dictation_mode()))
        except Exception:  # noqa: BLE001
            self._set_dictation_ui(False)

    def _build_behavior_tab(self, tab) -> None:  # noqa: ANN001
        """Stage 8.5 — Behavior Mixer 2×2 grid → Blackboard persona_mixer."""
        try:
            tab.configure(fg_color=_UI_CANVAS)
        except Exception:  # noqa: BLE001
            pass

        card = self._make_card(tab, title="Behavior Mixer")
        ctk.CTkLabel(
            card,
            text="Autonomy · Verbosity · Creativity · Tech Depth",
            anchor="w",
            text_color=_UI_MUTED,
            wraplength=640,
            justify="left",
        ).pack(fill="x", pady=(0, 4))
        # Neutral hint (no jarring lock warning — overlay handles engage lock).
        self._behavior_lock_hint = ctk.CTkLabel(
            card,
            text="Adjust traits while engine is on standby, then Apply & Save",
            anchor="w",
            text_color="#888888",
            font=ctk.CTkFont(size=12),
        )
        self._behavior_lock_hint.pack(fill="x", pady=(0, 8))

        specs = (
            ("Autonomy", "autonomy"),
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

        mixer_host = ctk.CTkFrame(card, fg_color="transparent")
        mixer_host.pack(fill="both", expand=True, pady=(0, 4))
        self._behavior_mixer_host = mixer_host
        mixer_host.grid_columnconfigure(0, weight=1)
        mixer_host.grid_columnconfigure(1, weight=1)
        mixer_host.grid_rowconfigure(0, weight=1)
        mixer_host.grid_rowconfigure(1, weight=1)

        self._static_behavior_widgets = []
        for idx, (label, key) in enumerate(specs):
            r, c = divmod(idx, 2)
            cell = ctk.CTkFrame(mixer_host, fg_color="transparent")
            cell.grid(row=r, column=c, sticky="nsew", padx=8, pady=8)
            row = ctk.CTkFrame(cell, fg_color="transparent")
            row.pack(fill="x")
            ctk.CTkLabel(
                row, text=label, anchor="w", text_color="#E5E7EB"
            ).pack(side="left")
            val = int(state.get(key, PERSONA_MIXER_DEFAULTS.get(key, 50)))
            val_lbl = ctk.CTkLabel(
                row, text=str(val), width=36, text_color="#F9FAFB"
            )
            val_lbl.pack(side="right")
            self._behavior_labels[key] = val_lbl
            slider = ctk.CTkSlider(
                cell,
                from_=0,
                to=100,
                number_of_steps=100,
                command=lambda v, k=key: self._on_behavior_drag(k, v),
            )
            slider.set(float(val))
            slider.pack(fill="x", pady=(6, 0))
            slider.bind(
                "<ButtonRelease-1>",
                lambda _e, k=key: self._commit_behavior(k, force=True),
            )
            self._behavior_sliders[key] = slider
            self._static_behavior_widgets.append(slider)

        self._behavior_reload_btn = ctk.CTkButton(
            card,
            text="Apply & Save Traits",
            width=160,
            height=30,
            corner_radius=8,
            fg_color=_UI_EMERALD,
            hover_color=_UI_EMERALD_HOVER,
            text_color="#ECFDF5",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self.update_behavior_traits,
        )
        self._behavior_reload_btn.pack(pady=(8, 4), anchor="w")
        self._static_behavior_widgets.append(self._behavior_reload_btn)

        # Dim overlay when engine is engaged (no red warning copy).
        self._behavior_lock_overlay = ctk.CTkFrame(
            mixer_host,
            fg_color=("#0a0e17", "#0a0e17"),
            corner_radius=12,
            border_width=0,
        )
        try:
            self._behavior_lock_overlay.configure(cursor="arrow")
        except Exception:  # noqa: BLE001
            pass
        overlay_lbl = ctk.CTkLabel(
            self._behavior_lock_overlay,
            text="Mixer locked",
            font=ctk.CTkFont(size=12),
            text_color=_UI_MUTED,
        )
        overlay_lbl.place(relx=0.5, rely=0.5, anchor="center")

    def _set_behavior_controls_locked(self, locked: bool) -> None:
        """Grey out Behavior Mixer + place dim overlay when engine is hot."""
        self._behavior_locked = bool(locked)
        state = "disabled" if self._behavior_locked else "normal"
        for widget in list(self._static_behavior_widgets):
            try:
                widget.configure(state=state)
            except Exception:  # noqa: BLE001
                pass
        # Hard-disable sliders again (some CTk builds ignore batch configure).
        for slider in list(self._behavior_sliders.values()):
            try:
                slider.configure(state=state)
            except Exception:  # noqa: BLE001
                pass
        overlay = getattr(self, "_behavior_lock_overlay", None)
        if overlay is not None:
            try:
                if self._behavior_locked:
                    overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
                    try:
                        overlay.lift()
                    except Exception:  # noqa: BLE001
                        pass
                    # Soft dim — CTk has no true alpha; use near-canvas fill.
                    overlay.configure(fg_color=("#131b2e", "#131b2e"))
                else:
                    overlay.place_forget()
            except Exception:  # noqa: BLE001
                pass
        hint = self._behavior_lock_hint
        if hint is not None:
            try:
                if self._behavior_locked:
                    hint.configure(
                        text="Mixer locked — standby / idle required to edit traits",
                        text_color=_UI_AMBER,
                    )
                else:
                    hint.configure(
                        text="Adjust traits while engine is on standby, then Apply & Save",
                        text_color="#888888",
                    )
            except Exception:  # noqa: BLE001
                pass

    def _sync_behavior_lock_from_engine_state(self) -> None:
        """Lock mixer when Engaged/Active or while the pipeline is processing."""
        engaged = bool(getattr(self, "engine_active", False))
        processing = bool(getattr(self, "_vad_processing", False))
        self._set_behavior_controls_locked(engaged or processing)

    def update_behavior_traits(self) -> None:
        """Apply & Save Traits — write current slider values to Blackboard."""
        if self._behavior_locked:
            hint = self._behavior_lock_hint
            if hint is not None:
                try:
                    hint.configure(
                        text="Cannot save — put engine in Standby first",
                        text_color=_UI_ROSE,
                    )
                except Exception:  # noqa: BLE001
                    pass
            return
        applied = self._apply_behavior_mixer_payload()
        try:
            self._reload_behavior_sliders()
        except Exception:  # noqa: BLE001
            pass
        hint = self._behavior_lock_hint
        if hint is not None:
            try:
                keys = ", ".join(f"{k}={v}" for k, v in sorted(applied.items()))
                hint.configure(
                    text=f"Traits saved — {keys}" if keys else "Traits saved",
                    text_color=_UI_EMERALD,
                )
            except Exception:  # noqa: BLE001
                pass
        log("UI", f"Apply & Save Traits → {applied}")

    def _set_dictation_ui(self, active: bool) -> None:
        self._dictation_active = bool(active)
        try:
            if self._dictation_active:
                self.dictation_btn.configure(
                    text="  ●  DICTATING  ",
                    fg_color=_UI_EMERALD,
                    hover_color="#059669",
                    border_color=_UI_EMERALD,
                    text_color="#ECFDF5",
                )
                if self.dictation_status is not None:
                    self.dictation_status.configure(
                        text="● DICTATING",
                        text_color=_UI_EMERALD,
                    )
            else:
                self.dictation_btn.configure(
                    text="  ●  OFF  ",
                    fg_color=_UI_GHOST,
                    hover_color="#475569",
                    border_color=_UI_CARD_BORDER,
                    text_color=_UI_ROSE,
                )
                if self.dictation_status is not None:
                    self.dictation_status.configure(
                        text="● OFF",
                        text_color=_UI_ROSE,
                    )
        except Exception:  # noqa: BLE001
            pass
        # Stage 8.9.7 — Behavior lock follows engine ignition (not dictation alone).
        self._sync_behavior_lock_from_engine_state()
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
        box = getattr(self, "dictation_list", None)
        if box is None:
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
            box.configure(state="normal")
            box.delete("1.0", "end")
            box.insert("1.0", "".join(lines))
            box.configure(state="disabled")
        except Exception:  # noqa: BLE001
            pass

    def _refresh_system_log(self) -> None:
        box = getattr(self, "_system_log_box", None)
        if box is None:
            return
        text = "(log unavailable)\n"
        try:
            from dana.logging import RUNTIME_LOG_PATH

            path = RUNTIME_LOG_PATH
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    lines = fh.readlines()
                text = "".join(lines[-40:]) or "(log empty)\n"
            else:
                legacy = os.path.join(os.path.dirname(path), "dana_runtime.log")
                if os.path.isfile(legacy):
                    with open(legacy, "r", encoding="utf-8", errors="replace") as fh:
                        lines = fh.readlines()
                    text = "".join(lines[-40:]) or "(log empty)\n"
                else:
                    text = f"(no log yet at {path})\n"
        except Exception as exc:  # noqa: BLE001
            text = f"Could not read log: {exc}\n"
        try:
            box.configure(state="normal")
            box.delete("1.0", "end")
            box.insert("1.0", text)
            box.configure(state="disabled")
            box.see("end")
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

    def _telemetry_had_activity(self) -> bool:
        """Cheap, side-effect-free peek: is there new telemetry to render?

        Called synchronously from ``_master_telemetry_tick`` — main thread
        only. Touches only thread-safe sources (``queue.Queue``, the
        lock-protected ``AsyncRingBuffer``, and ``MonitorBus.pending()``)
        without draining anything itself; the dispatch below does the real
        draining. Feeds ``AdaptivePoller.note_activity()`` so the shared
        heartbeat speeds back up under load and rests while idle.

        NOTE: this does *not* run on ``AdaptivePoller``'s background thread.
        An earlier version of this dispatcher tried exactly that (per the
        textbook "marshal via self.after(0, ...)" pattern) and it does not
        work: registering a Tk callback is itself a Tcl/Tk call, and
        CPython's Tkinter (3.12+) raises ``RuntimeError: main thread is not
        in main loop`` — or simply stalls the poller thread — the instant a
        non-main thread calls ``widget.after()``, ``self.after(0, ...)``
        included. See ``AdaptivePoller``'s docstring. Everything telemetry
        related in ``DanaGUI`` therefore stays on the Tk main thread; only
        the backoff *interval math* is delegated to ``AdaptivePoller``.
        """
        had_activity = False
        try:
            had_activity = not gui_telemetry_queue.empty()
        except Exception:  # noqa: BLE001
            pass
        if not had_activity:
            try:
                had_activity = len(self._telemetry_buffer.snapshot()) != self._neural_rendered
            except Exception:  # noqa: BLE001
                pass
        if not had_activity:
            try:
                from dana.graph.monitor_bus import get_monitor_bus

                bus = get_monitor_bus(create=False)
                had_activity = bool(bus is not None and bus.pending() > 0)
            except Exception:  # noqa: BLE001
                pass
        return had_activity

    def _master_telemetry_tick(self) -> None:
        """Single Tk-main-thread dispatcher for every telemetry consumer.

        A conventional self-rescheduling ``self.after()`` chain — like every
        other poller in this file — except the *delay* it re-arms itself
        with adapts each tick via ``self._adaptive_poller.note_activity()``
        (50ms while busy, backing off toward 500ms while idle) instead of a
        fixed interval. Replaces what used to be five independent
        ``self.after()`` loops (``LiveTracePanel`` 80ms, ``process_telemetry``
        100ms, ``_poll_state_changes`` 100ms, ``DagMonitorView`` 250ms,
        ``TaskTrackerView`` 400ms) with one shared heartbeat. Each consumer
        still only fires at its own cadence — tracked via monotonic elapsed
        time in ``self._telemetry_last`` — so a backed-off (idle) heartbeat
        never causes drift or double-fires; a slow heartbeat just means a
        consumer's next check waits longer.
        """
        if not self.winfo_exists():
            return
        try:
            had_activity = self._telemetry_had_activity()
        except Exception:  # noqa: BLE001
            had_activity = False
        now = time.monotonic()
        last = self._telemetry_last
        cadences = _TELEMETRY_CADENCES_S

        live_trace = getattr(self, "live_trace", None)
        if live_trace is not None and now - last.get("live_trace", 0.0) >= cadences["live_trace"]:
            last["live_trace"] = now
            try:
                live_trace.tick()
            except Exception:  # noqa: BLE001
                pass

        if now - last.get("process_telemetry", 0.0) >= cadences["process_telemetry"]:
            last["process_telemetry"] = now
            try:
                self.process_telemetry()
            except Exception:  # noqa: BLE001
                pass

        if now - last.get("state_changes", 0.0) >= cadences["state_changes"]:
            last["state_changes"] = now
            try:
                self._poll_state_changes()
            except Exception:  # noqa: BLE001
                pass

        dag_view = getattr(self, "dag_monitor_view", None)
        if dag_view is not None and now - last.get("dag_monitor", 0.0) >= cadences["dag_monitor"]:
            last["dag_monitor"] = now
            try:
                dag_view.refresh()
            except Exception:  # noqa: BLE001
                pass

        task_view = getattr(self, "task_tracker_view", None)
        if task_view is not None and now - last.get("task_tracker", 0.0) >= cadences["task_tracker"]:
            last["task_tracker"] = now
            try:
                task_view.tick()
            except Exception:  # noqa: BLE001
                pass

        try:
            next_s = self._adaptive_poller.note_activity(had_activity)
        except Exception:  # noqa: BLE001
            next_s = self._adaptive_poller.t_min
        try:
            self.after(max(1, int(next_s * 1000)), self._master_telemetry_tick)
        except Exception:  # noqa: BLE001
            pass

    def process_telemetry(self) -> None:
        """Drain legacy ``gui_telemetry_queue`` on the Tk main thread.

        Called from ``_master_telemetry_tick`` (see its cadence in
        ``_TELEMETRY_CADENCES_S["process_telemetry"]``) — does not
        reschedule itself; the master dispatcher owns cadence for all five
        telemetry consumers now. Keeps header mode / fallback TraceCells in
        sync, and mirrors every event into the Neural Stream ring buffer for
        the Unified Canvas.
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
                try:
                    self._telemetry_emitter.emit(
                        stage, {"message": f"[{stage}] {message}", "status": status}
                    )
                except Exception:  # noqa: BLE001
                    pass
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
            self._render_neural_stream()
        except Exception:  # noqa: BLE001
            pass

    def _render_neural_stream(self) -> None:
        """Flush new ``AsyncRingBuffer`` events into the Neural Stream Text widget.

        Applies keyword color tags and a tail-drop limiter (cap 500 lines) so
        a busy session can never grow the widget large enough to lag Tk.
        """
        text = getattr(self, "_neural_stream_text", None)
        buffer = getattr(self, "_telemetry_buffer", None)
        if text is None or buffer is None or not self.winfo_exists():
            return
        events = buffer.snapshot()
        rendered = getattr(self, "_neural_rendered", 0)
        new_events = events[rendered:]
        if not new_events:
            return
        self._neural_rendered = len(events)
        try:
            text.configure(state="normal")
            for event in new_events:
                payload = event.get("payload") or {} if isinstance(event, dict) else {}
                message = str(payload.get("message") or event.get("type") or "").strip()
                if not message:
                    continue
                upper = message.upper()
                if "EXECUTION ERROR" in upper or str(payload.get("status") or "") == "error":
                    tag = "error"
                elif "THOUGHT:" in upper:
                    tag = "thought"
                elif "TOOL" in upper:
                    tag = "tool"
                else:
                    tag = None
                line = f"{message}\n"
                if tag:
                    text.insert("end", line, (tag,))
                else:
                    text.insert("end", line)
            # Tail-Drop Limiter — keep at most 500 lines so Tk never lags.
            line_count = int(text.index("end-1c").split(".")[0])
            if line_count > 500:
                text.delete("1.0", f"{line_count - 500}.0")
            text.see("end")
            text.configure(state="disabled")
        except Exception:  # noqa: BLE001
            pass

    def _poll_state_changes(self) -> None:
        """Drain STATE_CHANGE bus → VAD mic + System Status line.

        Called from ``_master_telemetry_tick`` — does not reschedule itself.
        """
        if not self.winfo_exists():
            return
        try:
            from dana.ui.status_bus import drain_state_changes

            # Drain until empty (capped) so bursts between ticks never stall.
            events: list = []
            for _ in range(8):
                batch = drain_state_changes(max_items=64)
                if not batch:
                    break
                events.extend(batch)
                if len(events) >= 256:
                    break
            if events:
                # Latest wins for widgets; full drain guarantees no dropped tip.
                self._apply_state_change(events[-1])
        except Exception:  # noqa: BLE001
            pass

    def _apply_state_change(self, event: dict) -> None:
        """Update VAD mic pip + System Status line from a STATE_CHANGE payload."""
        try:
            from dana.ui.status_bus import format_system_status_line
        except Exception:  # noqa: BLE001
            return
        status = str(event.get("status") or "idle").strip().lower()
        tool = str(event.get("tool") or "")
        message = str(event.get("message") or "")
        self._vad_listening = status == "listening"
        self._vad_processing = status in {"processing", "routing", "executing"}
        # Instantly lock / unlock Behavior Mixer with pipeline activity.
        try:
            self._sync_behavior_lock_from_engine_state()
        except Exception:  # noqa: BLE001
            pass
        line = format_system_status_line(
            status, tool=tool, message=message
        )
        mic_text = "● Idle"
        mic_color = _UI_MUTED
        if status == "listening":
            mic_text = "● Listening"
            mic_color = _UI_EMERALD
        elif status == "processing":
            mic_text = "● Processing"
            mic_color = _UI_AMBER
        elif status == "routing":
            mic_text = "● Processing"
            mic_color = _UI_AMBER
        elif status == "executing":
            mic_text = "● Processing"
            mic_color = _UI_ACCENT
        lbl = getattr(self, "_system_status_lbl", None)
        if lbl is not None:
            try:
                color = _UI_MUTED
                if status == "routing":
                    color = _UI_AMBER
                elif status == "executing":
                    color = _UI_ACCENT
                elif status == "listening":
                    color = _UI_EMERALD
                elif status == "processing":
                    color = _UI_AMBER
                elif status == "idle":
                    color = _UI_MUTED
                if tool == "proactive_briefing":
                    color = _UI_AMBER
                lbl.configure(text=line or "Idle", text_color=color)
            except Exception:  # noqa: BLE001
                pass
        if tool == "proactive_briefing":
            hdr = getattr(self, "_header_status_lbl", None)
            if hdr is not None:
                try:
                    hdr.configure(text="• BRIEFING", text_color=_UI_AMBER)
                except Exception:  # noqa: BLE001
                    pass
            badge = getattr(self, "daemon_badge", None)
            if badge is not None and message:
                try:
                    badge.configure(text="● UPDATE")
                except Exception:  # noqa: BLE001
                    pass
        mic = getattr(self, "_vad_mic_lbl", None)
        if mic is not None:
            try:
                mic.configure(text=mic_text, text_color=mic_color)
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
        # Pulsating VAD mic pip while listening (theme emerald ↔ muted).
        mic = getattr(self, "_vad_mic_lbl", None)
        if mic is not None and getattr(self, "_vad_listening", False):
            self._vad_pulse_on = not getattr(self, "_vad_pulse_on", False)
            try:
                mic.configure(
                    text="● Listening",
                    text_color=_UI_EMERALD
                    if self._vad_pulse_on
                    else _UI_ACCENT,
                )
            except Exception:  # noqa: BLE001
                pass
        elif mic is not None and getattr(self, "_vad_processing", False):
            self._vad_pulse_on = not getattr(self, "_vad_pulse_on", False)
            try:
                mic.configure(
                    text="● Processing",
                    text_color=_UI_AMBER if self._vad_pulse_on else _UI_ACCENT,
                )
            except Exception:  # noqa: BLE001
                pass
        try:
            self.after(500, self._pulse_active_cells)
        except Exception:  # noqa: BLE001
            pass

    def _on_transcript_event(
        self, speaker: str, text: str, agent_id: str | None
    ) -> None:
        """dana.core.shared_state transcript-listener adapter for log_transcript."""
        try:
            self.log_transcript(speaker, text, agent_id=agent_id)
        except Exception as exc:  # noqa: BLE001
            log("UI", f"WARNING: live transcript update failed ({exc})")

    def log_transcript(
        self,
        speaker: str,
        text: str,
        *,
        agent_id: str | None = None,
    ) -> None:
        """Append a speaker line to Conversation (bubbles + mirror textbox)."""
        line = f"[{speaker}] {text}\n\n"
        tag = self._persona_tag_for_agent(agent_id, speaker=speaker)
        try:
            self._telemetry_emitter.emit("transcript", {"message": f"[{speaker}] {text}"})
        except Exception:  # noqa: BLE001
            pass

        def _append() -> None:
            try:
                if not self.winfo_exists():
                    return
                box = getattr(self, "transcript_box", None)
                if box is not None:
                    box.configure(state="normal")
                    # Prefer tagged insert on underlying Text; fall back to CTk API.
                    tk_text = self._transcript_tk()
                    if tk_text is not None:
                        tk_text.insert("end", line, (tag,))
                    else:
                        try:
                            box.insert("end", line, tag)
                        except TypeError:
                            box.insert("end", line)
                    try:
                        box.see("end")
                    except Exception:  # noqa: BLE001
                        pass
                    box.configure(state="disabled")
                chat = getattr(self, "_chat_view", None)
                if chat is not None:
                    try:
                        from dana.ui.chat_view import _classify_role

                        role = _classify_role(speaker, agent_id)
                        if tag == "vision":
                            role = "system"
                        chat.append_bubble(
                            speaker, text, agent_id=agent_id, role=role
                        )
                        try:
                            chat._scroll_to_latest()
                        except Exception:  # noqa: BLE001
                            pass
                    except Exception:  # noqa: BLE001
                        pass
            except Exception:
                pass

        try:
            self.after(0, _append)
        except Exception:
            pass

    def _reload_device_menus(self) -> None:
        """No-op — Mic/Speaker menus removed; streams always use System Default."""
        try:
            from dana.audio.devices import SYSTEM_DEFAULT_LABEL
        except Exception:  # noqa: BLE001
            SYSTEM_DEFAULT_LABEL = "System Default (Auto)"
        self._mic_labels = [SYSTEM_DEFAULT_LABEL]
        self._speaker_labels = [SYSTEM_DEFAULT_LABEL]
        self._mic_by_label = {SYSTEM_DEFAULT_LABEL: None}
        self._speaker_by_label = {SYSTEM_DEFAULT_LABEL: None}

    def _refresh_stats(self) -> None:
        if not self.winfo_exists():
            return
        raw = get_ui_state()
        label = _UI_STATE_LABELS.get(raw, raw.title())
        try:
            if self.status_value is not None:
                self.status_value.configure(text=label)
        except Exception:  # noqa: BLE001
            pass
        wake = ", ".join(WAKEWORD_MODELS) if WAKEWORD_MODELS else "—"
        _wake_display = {"dana": "Dana", "alexa": "Alexa"}
        if wake != "—":
            parts = [w.strip() for w in wake.split(",") if w.strip()]
            wake_disp = ", ".join(_wake_display.get(p.lower(), p.title()) for p in parts)
        else:
            wake_disp = wake
        try:
            if self.wake_value is not None:
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
            self._set_mode_indicator(get_dana_mode())
        except Exception:  # noqa: BLE001
            pass
        self.after(500, self._refresh_stats)

    def _save_and_apply_audio(self) -> None:
        """Compatibility stub — audio always binds System Default (device=None)."""
        state.AUDIO_INPUT_DEVICE = None
        state.AUDIO_OUTPUT_DEVICE = None
        state.AUDIO_INPUT_RATE = _device_rate(None)
        try:
            save_audio_settings(None, None)
        except Exception:  # noqa: BLE001
            pass
        request_mic_ingest_restart()
        ensure_mic_ingest_thread()
        note = getattr(self, "apply_note", None)
        if note is not None:
            try:
                note.configure(text="Audio: System Default (Auto)")
            except Exception:  # noqa: BLE001
                pass
        log("Audio", "GUI audio → System Default (autonomous; menus removed)")

    def show_window(self) -> None:
        self.deiconify()
        self.lift()
        self.focus_force()
        try:
            from dana.ui.logo import schedule_window_icon

            schedule_window_icon(self, delay_ms=100)
        except Exception:  # noqa: BLE001
            pass
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

    def _on_hybrid_planner_toggle(self) -> None:
        """Persist Settings → Hybrid Broker (Cloud Planner); refresh DAG label."""
        try:
            from dana.settings import set_hybrid_planner_enabled

            enabled = bool(self._hybrid_planner_var.get())
            set_hybrid_planner_enabled(enabled)
            log(
                "UI",
                f"hybrid_planner_enabled={'True' if enabled else 'False'} (saved)",
            )
        except Exception as exc:  # noqa: BLE001
            log("UI", f"WARNING: could not save hybrid_planner_enabled ({exc})")
        self._refresh_planner_mode_label(warn_missing_key=True)

    def _refresh_planner_mode_label(self, *, warn_missing_key: bool = False) -> None:
        """Update DAG drawer 'Planner Mode: […]' from settings + API key presence."""
        try:
            from dana.graph.cloud_planner import (
                planner_mode_label,
                publish_planner_mode,
            )

            mode = publish_planner_mode(warn_missing_key=warn_missing_key)
            mode = mode or planner_mode_label()
        except Exception:  # noqa: BLE001
            mode = "LOCAL"
        lbl = getattr(self, "_dag_planner_mode_lbl", None)
        if lbl is None:
            return
        try:
            lbl.configure(text=f"Planner Mode: [{mode}]")
        except Exception:  # noqa: BLE001
            pass
        view = getattr(self, "dag_monitor_view", None)
        if view is not None and hasattr(view, "set_planner_mode"):
            try:
                view.set_planner_mode(mode)
            except Exception:  # noqa: BLE001
                pass

    def kill_dana_processes(self) -> dict[str, Any]:
        """Stage 8.9.2 — launch ``stop_dana.vbs`` / ``stop_dana.bat`` (non-blocking).

        Prefers the VBS silent runner (no console flash). Uses a detached
        ``subprocess.Popen`` so teardown can finish after this GUI process dies.
        """
        try:
            from dana.paths import PROJECT_ROOT

            root = Path(PROJECT_ROOT)
        except Exception:  # noqa: BLE001
            root = Path(__file__).resolve().parents[1]
        launchers = root / "scripts" / "launchers"
        vbs = launchers / "stop_dana.vbs"
        bat = launchers / "stop_dana.bat"
        if not vbs.is_file() and not bat.is_file():
            # Fallback: thin root wrappers / legacy layout / unit-test tmp roots.
            vbs = root / "stop_dana.vbs"
            bat = root / "stop_dana.bat"
        runner = vbs if vbs.is_file() else bat
        if not runner.is_file():
            msg = (
                f"stop_dana.vbs / stop_dana.bat not found under "
                f"{launchers} or {root}"
            )
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
            log("UI", f"STOP DANA — launched {runner.name} pid={proc.pid}")
            return {"ok": True, "pid": int(proc.pid), "path": str(runner)}
        except FileNotFoundError as exc:
            msg = f"Failed to launch {runner.name}: {exc}"
            log("UI", f"WARNING: {msg}")
            return {"ok": False, "error": "FileNotFoundError", "message": msg}
        except OSError as exc:
            msg = f"Failed to launch {runner.name}: {exc}"
            log("UI", f"WARNING: {msg}")
            return {"ok": False, "error": type(exc).__name__, "message": msg}

    def _halt_engine_full(self) -> None:
        """In-process full halt: terminate workers, cancel audio/LLM, STOPPED pill.

        Kill-switch ``stop_dana.*`` launch remains separate (see
        ``_on_stop_dana_clicked``); this path must run even if the batch is
        missing so the Local Engine goes inactive immediately.
        """
        # Drop dictation latch if hot.
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
        self._engine_stopped = True
        set_engine_engaged(False)
        # Termination latch — workers / conversation loop exit.
        try:
            stop_event.set()
        except Exception:  # noqa: BLE001
            pass
        # Cancel VAD / mic latch + pending TTS.
        try:
            is_recording.clear()
        except Exception:  # noqa: BLE001
            pass
        try:
            vad_capture_active.clear()
        except Exception:  # noqa: BLE001
            pass
        try:
            reset_tts_audio_state("stop_dana", ui_state="idle")
        except Exception:  # noqa: BLE001
            pass
        # Abort pending actuators / Ghost Typist / in-flight LLM actions.
        try:
            from dana.middleware.kill_switch import trigger_halt

            trigger_halt(reason="stop_dana")
        except Exception:  # noqa: BLE001
            pass
        # Detach daemon sidecar reconnect (do not hot_restart — hard stop).
        try:
            client = getattr(self, "_daemon_client", None)
            if client is not None:
                try:
                    client.stop_auto_reconnect()
                except Exception:  # noqa: BLE001
                    pass
                try:
                    client._drop_socket()
                except Exception:  # noqa: BLE001
                    pass
                self._daemon_client = None
        except Exception:  # noqa: BLE001
            pass
        self._set_behavior_controls_locked(False)
        self._refresh_engine_ui()
        try:
            from dana.ui.status_bus import emit_state_change

            emit_state_change("idle", message="● STOPPED | Local Engine")
        except Exception:  # noqa: BLE001
            pass
        log("UI", "STOP DANA — engine halted (STOPPED)")
        try:
            self.log_transcript(
                "Dana",
                "Engine STOPPED — Local Engine halted.",
                agent_id="broker",
            )
        except Exception:  # noqa: BLE001
            pass
        try:
            if self.assistive_orb is not None:
                self.assistive_orb.refresh_controls()
        except Exception:  # noqa: BLE001
            pass

    def _on_stop_dana_clicked(self) -> None:
        """Halt engine in-process, show TERMINATING…, then fire kill switch."""
        btn = getattr(self, "stop_dana_btn", None)
        try:
            self._halt_engine_full()
        except Exception as exc:  # noqa: BLE001
            log("UI", f"WARNING: in-process engine halt failed ({exc})")
        try:
            if btn is not None:
                btn.configure(
                    text="TERMINATING...",
                    state="disabled",
                    fg_color=_UI_ROSE_HOVER,
                )
        except Exception:  # noqa: BLE001
            pass
        try:
            self.update_idletasks()
        except Exception:  # noqa: BLE001
            pass

        def _launch() -> None:
            result = self.kill_dana_processes()
            if not result.get("ok"):
                try:
                    if btn is not None:
                        btn.configure(
                            text="STOP DANA",
                            state="normal",
                            fg_color=_UI_ROSE,
                        )
                except Exception:  # noqa: BLE001
                    pass
                try:
                    log(
                        "UI",
                        f"STOP DANA aborted: {result.get('message') or 'unknown'}",
                    )
                except Exception:  # noqa: BLE001
                    pass

        try:
            # Brief paint of TERMINATING… before the batch tears us down.
            self.after(80, _launch)
        except Exception:  # noqa: BLE001
            _launch()

    def _start_assistive_orb(self) -> None:
        """Stage 8.7 — spawn frameless topmost orb on the Tk main thread.

        DISABLED: experimental orb window is not launched at engine start.
        """
        # Experimental AssistiveTouch orb — leave commented until re-enabled.
        return
        # if self.assistive_orb is not None:
        #     return
        # try:
        #     from dana.ui.assistive_orb import AssistiveTouchOrb
        #
        #     def _active_agent() -> str:
        #         try:
        #             from dana.audio.multi_voice_tts import get_active_tts_agent
        #
        #             return str(get_active_tts_agent() or "broker")
        #         except Exception:  # noqa: BLE001
        #             return "broker"
        #
        #     self.assistive_orb = AssistiveTouchOrb(
        #         self,
        #         on_toggle_dictation=self._toggle_dictation_mode,
        #         on_open_dashboard=self.show_window,
        #         on_approve_ticket=self._orb_approve_ticket,
        #         on_deny_ticket=self._orb_deny_ticket,
        #         dictation_getter=lambda: bool(self._dictation_active),
        #         mode_getter=lambda: str(self._header_mode or "chat"),
        #         accent_getter=lambda: self._mode_accent(),
        #         agent_getter=_active_agent,
        #     )
        # except Exception as exc:  # noqa: BLE001
        #     log("UI", f"WARNING: AssistiveTouch orb failed to start ({exc})")
        #     self.assistive_orb = None

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
        description="Offline Dana voice agent (YOLO eyes + Ollama brain + Whisper + OpenWakeWord).",
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
        help="Delete dana_memory.enc and exit so the next run creates a fresh vault.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Headless mode: skip CustomTkinter Live Trace / tray UI.",
    )
    return parser.parse_args()


def agent_loop(args: Optional[argparse.Namespace] = None) -> int:
    if args is None:
        args = parse_args()
    local_files_only = not args.download

    log("Main", "=== CAMGRASPER Dana voice agent ===")
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

    # Headless CLI has no Dashboard ENGAGE button — arm the engine so
    # ``.trigger_ask`` injects are not left stranded in soft STANDBY.
    if getattr(args, "no_gui", False):
        os.environ.setdefault("DANA_NO_GUI", "1")
        os.environ.setdefault("DANA_HEADLESS", "1")
        set_engine_engaged(True)
        log("Main", "Headless mode: engine auto-ENGAGED for trigger injects")
        try:
            from dana.graph.meta_broker_process import start_headless_telemetry_drainer

            start_headless_telemetry_drainer()
            log("Main", "Headless Meta-Broker telemetry drainer started")
        except Exception as exc:  # noqa: BLE001
            log("Main", f"WARNING: headless telemetry drainer unavailable ({exc})")

    # Unlock encrypted long-term memory before loading models / audio threads.
    try:
        unlock_dana_memory()
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
    # Autonomous audio — always System Default regardless of settings.json.
    mic_id, speaker_id = None, None
    log(
        "Audio",
        f"Audio pipeline: mic={mic_id} speaker={speaker_id} rate={mic_rate}",
    )
    state.AUDIO_INPUT_DEVICE = None
    state.AUDIO_OUTPUT_DEVICE = None
    state.AUDIO_INPUT_RATE = mic_rate

    # Acoustic-aware: keep OS default when live; bind physical fallback when quiet.
    state.AUDIO_INPUT_DEVICE, state.AUDIO_INPUT_RATE = ensure_live_mic(
        None,
        state.AUDIO_INPUT_RATE,
        allow_fallback=True,
    )
    if not _validate_mic_id(state.AUDIO_INPUT_DEVICE):
        log("Main", "Aborting: configured microphone is not usable.")
        stop_event.set()
        return 2
    if not _validate_speaker_id(state.AUDIO_OUTPUT_DEVICE):
        log("Main", "Aborting: configured speaker is not usable.")
        stop_event.set()
        return 2

    # Adaptive noise floor before wake-word arming (3s ambient baseline).
    calibrate_noise_floor(duration_sec=3.0)

    device = select_device()
    dtype = select_dtype(device)

    # Single PortAudio InputStream producer before wake/VAD consumers start.
    ensure_mic_ingest_thread()
    if not mic_ingest_ready.wait(timeout=8.0):
        log(
            "Main",
            "WARNING: MicIngest not ready after 8s — wake/VAD will wait on the queue",
        )

    global _dual_wake_router, _dual_wake_poller
    _dual_wake_router = None
    _dual_wake_poller = None

    try:
        router = AudioRouter(
            sample_rate=16000,
            chunk_size=1280,
            sample_width=2,
            wakeword_model_path=str(resolve_wakeword_onnx()),
        )
        poller = WakePoller(
            router=router,
            whisper_model=router.whisper_model,
            standard_model=router.standard_model,
            callback=_trigger_dual_wake_event,
            threshold=0.5,
        )
        router.start()
        poller.start()
        _dual_wake_router = router
        _dual_wake_poller = poller
        log("WakeWord", "Dual-threshold wake polling listener started")
    except Exception as exc:
        log("WakeWord", f"WARNING: dual-threshold wake listener unavailable ({exc})")

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
    ]
    # Centralized TTSManager owns the Piper consumer thread (speech_queue).
    try:
        _tts_mgr_thread = _tts_manager.start(worker=tts_worker)
        if _tts_mgr_thread is not None:
            threads.append(_tts_mgr_thread)
            log("Main", f"Started thread: {_tts_mgr_thread.name}")
        else:
            fallback = threading.Thread(
                target=tts_worker, name="TTSWorker", daemon=True
            )
            threads.append(fallback)
    except Exception as _tts_start_exc:  # noqa: BLE001
        log("Main", f"WARNING: TTSManager start failed ({_tts_start_exc})")
        threads.append(
            threading.Thread(target=tts_worker, name="TTSWorker", daemon=True)
        )
    for t in threads:
        if t.is_alive():
            continue
        t.start()
        log("Main", f"Started thread: {t.name}")

    # Stage 7.2 — hardware panic button (F12 / DANA_KILL_HOTKEY).
    try:
        from dana.middleware.kill_switch import start_kill_switch_listener

        if start_kill_switch_listener():
            log("Main", "Kill switch listener armed (default hotkey F12)")
        else:
            log("Main", "Kill switch listener not started")
    except Exception as exc:  # noqa: BLE001
        log("Main", f"WARNING: kill switch listener failed: {exc}")

    # Phase 1/2 — adaptive compute governor (idle → research via input.txt queue).
    try:
        from dana.middleware.idle_monitor import start_idle_monitor

        if start_idle_monitor():
            log("Main", "IdleMonitor started (USER_ACTIVE/USER_AWAY compute governor)")
        else:
            log("Main", "IdleMonitor not started")
    except Exception as exc:  # noqa: BLE001
        log("Main", f"WARNING: IdleMonitor failed: {exc}")

    # Persist conversational mode so system jobs cannot steal it later.
    try:
        set_dana_mode(get_dana_mode(), as_voice=True)
        log("Main", f"Voice session mode seeded: {get_dana_mode()}")
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
        "Dana is ready. Say 'Dana' to wake. | Tray Quit / Ctrl+C=quit",
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
            load_dana_settings,
        )

        load_dana_settings(force_reload=True)
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
        from dana.workspace import ensure_dana_workspace

        ensure_dana_workspace(migrate=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[Workspace] WARNING: ensure_dana_workspace failed: {exc}")

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
        from dana.paths import DANA_WORKSPACE

        log("Main", f"DANA_WORKSPACE={DANA_WORKSPACE}")
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
        reset_dana_vault()
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
        os.environ.setdefault("DANA_NO_GUI", "1")
        os.environ.setdefault("DANA_HEADLESS", "1")
        try:
            from dana.graph.meta_broker_process import start_headless_telemetry_drainer

            start_headless_telemetry_drainer()
        except Exception:  # noqa: BLE001
            pass
        log("Main", "Headless mode (--no-gui): Live Trace UI disabled.")
        return agent_loop(args)

    # Stage 8.9.7 — GUI paints in STANDBY first; heavy agent loop deferred.
    gui = DanaGUI()
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
            mode=get_dana_mode(),
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

    # Tray owns its own daemon thread — never block the CTk mainloop.
    threading.Thread(
        target=run_system_tray,
        name="SystemTray",
        args=(gui,),
        daemon=True,
    ).start()

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
        try:
            unregister_transcript_listener(gui._on_transcript_event)
        except Exception:  # noqa: BLE001
            pass
        try:
            unregister_vault_prompt_listener(gui._on_vault_unlock_request)
        except Exception:  # noqa: BLE001
            pass
        try:
            unregister_spec_approval_listener(gui._on_spec_approval_requested)
        except Exception:  # noqa: BLE001
            pass
        try:
            unregister_dictation_sessions_listener(gui._on_dictation_sessions_changed)
        except Exception:  # noqa: BLE001
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
