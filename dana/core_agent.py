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


# _gui_instance / _tray_icon / _agent_loop_thread, the Live Trace telemetry
# queue, and the trace-status glyph table now live in dana.core.shared_state
# (Phase 5 decomposition) -- shared with dana.ui.app_gui's DanaGUI/TraceCell
# (consumer side) and dana.ui.tray_icon (tray side). The first three are
# reassigned values, so callers below go through state.get_gui_instance() /
# state.set_gui_instance() etc. instead of a bare name.
from dana.core.shared_state import _TRACE_STATUS_ICONS, gui_telemetry_queue


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

# TraceCell / DanaGUI (dana.ui.app_gui) and the tray icon (dana.ui.tray_icon)
# moved out in the Phase 5 core_agent.py decomposition -- re-exported below
# so every existing call site / test import keeps working unchanged.
from dana.ui.app_gui import DanaGUI, TraceCell, _UI_ACCENT, _UI_CANVAS
from dana.ui.tray_icon import (
    create_tray_image,
    request_dana_quit,
    run_system_tray,
    update_tray_icon_for_state,
    _device_menu_label,
    _parse_device_menu_label,
)

# ---------------------------------------------------------------------------
# Main - agent loop (background) + GUI main thread
# ---------------------------------------------------------------------------

# CLI parsing, process lifecycle, daemon-thread orchestration, vault
# unlock/reset, and execute_lockdown_shutdown() moved to
# dana.core.app_runtime (Phase 6 of the core_agent.py decomposition),
# re-exported below so every existing call site / test import keeps
# working unchanged.
from dana.core.app_runtime import (
    email_recovery_key,
    enforce_singleton,
    execute_lockdown_shutdown,
    main,
    agent_loop,
    parse_args,
    populate_vault_hot_cache,
    reset_dana_vault,
    unlock_dana_memory,
    _install_signal_handlers,
    _shutdown_agent_threads,
)

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


