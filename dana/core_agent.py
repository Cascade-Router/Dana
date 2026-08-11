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
    ensure_project_root_on_syspath,
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

# _gui_instance / _tray_icon / _agent_loop_thread, the Live Trace telemetry
# queue, and the trace-status glyph table now live in dana.core.shared_state
# (Phase 5 decomposition) -- shared with dana.ui.app_gui's DanaGUI/TraceCell
# (consumer side) and dana.ui.tray_icon (tray side). The first three are
# reassigned values, so callers below go through state.get_gui_instance() /
# state.set_gui_instance() etc. instead of a bare name.
from dana.core.shared_state import _TRACE_STATUS_ICONS, gui_telemetry_queue


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
# ---------------------------------------------------------------------------
# Secure encrypted memory vault (AES-256 via Fernet + PBKDF2)
# ---------------------------------------------------------------------------
# SecureMemory lives in dana.secure_memory.py (shared with vault daemon).

# Agent-loop / conversational FSM bucket (Phase 4 of the core_agent.py
# decomposition) now lives in dana.core.agent_loop, imported back here in
# full so every existing bare-name call site below keeps working unchanged.
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

# Vision (Thread 1), wake-word (Thread 2), and text-injection (Thread 4)
# buckets moved to dana.vision.tracker_worker, dana.audio.wakeword_worker,
# and dana.ingestion.text_injection in Phase 7 of the core_agent.py
# decomposition; dana.core.agent_loop imports them directly now (see its
# module docstring), so nothing needs re-exporting from here.

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


