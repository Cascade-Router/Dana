"""Agent-loop / conversational FSM bucket, extracted verbatim from ``dana.core_agent``.

Phase 4 of the core_agent.py decomposition (see the approved plan). Holds the
tool-dispatch/ReAct core: ``build_dana_system_prompt``, ``execute_tool_call``
(~45 nested tool-dispatch closures, moved as-is -- internals are not
refactored in this pass), ``tool_router``, ``ask_ollama_messages`` +
``_ask_ollama_messages_unlocked``, ``commit_agentic_turn``, and
``conversation_worker`` (the wake -> listen -> transcribe -> respond FSM),
plus small mid-task-drop helpers used only by ``conversation_worker``.

Reassigned shared_state names touched by this bucket (``active_vision_tool``,
``latest_frame``, ``latest_dets``, ``dana_profile``, ``VAULT_HOT_CACHE``,
``vault_client``, ``_shared_wakeword_model``, ``_active_mid_task_prompt``) are
also written/read by functions that stay in ``core_agent.py`` (``tracker_worker``,
``wakeword_worker``, ``unlock_dana_memory``, ``execute_lockdown_shutdown``,
``populate_vault_hot_cache``) -- every site on both sides goes through
``dana.core.shared_state`` (``state.X``), never a bare name, so neither module
sees a stale cross-module snapshot.

Functions still owned by ``core_agent.py`` (GUI, tray, tracker, wake-word,
CLI/process-lifecycle -- later phases) are reached via a lazy bridge import,
matching the sanctioned temporary-bridge pattern already used elsewhere in
this decomposition (e.g. ``_nt_hide_console_if_mp_child``).
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any, Optional, Union

import numpy as np
import requests
from spatial_context import SPATIAL_AGGREGATOR

import ingest  # noqa: E402 — scripts/ingest.py -> task_queue.json converter
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
from dana.audio.mic_input import flush_audio_buffer_queue, flush_input_buffer, record_utterance
from dana.audio.noise_floor import get_dynamic_speech_floor
from dana.audio.stt import (
    correct_known_stt_names,
    ensure_whisper_bundle,
    is_silent_non_speech_transcript,
    is_whisper_hallucination,
    is_whisper_rate_hallucination,
    start_whisper_background_load,
    transcribe_audio,
)
from dana.audio.tts_manager import enqueue_speech_impl as enqueue_speech
from dana.audio.tts_worker import maybe_play_boot_ready_audio, wait_for_speech_idle
from dana.core import shared_state as state
from dana.core.constants import (
    FOLLOWUP_FLUSH_SEC,
    FOLLOWUP_VAD_MAX_SECONDS,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT_SEC,
    OLLAMA_URL,
    POST_ACK_FLUSH_SEC,
    POST_ACK_IGNORE_ONSET_MS,
    POST_ACK_VAD_GRACE_SEC,
    SAMPLE_RATE,
    TTS_IDLE_WAIT_TIMEOUT,
    VAD_MAX_SECONDS,
    YOLO_CONF,
    YOLO_WEIGHTS,
)
from dana.core.shared_state import (  # noqa: F401
    _active_mid_task_lock,
    _tool_working_ack_sent,
    active_vision_lock,
    camera_tool,
    conversation_history,
    conversation_history_lock,
    emit_live_transcript,
    engine_engaged,
    HISTORY_MAX_MESSAGES,
    is_recording,
    latest_dets_lock,
    latest_frame_lock,
    mic_ingest_ready,
    notify_dictation_sessions_changed,
    notify_spec_approval_requested,
    ollama_ready,
    piper_voices_ready,
    quiet_mic_mode,
    screen_tool,
    set_subtitle,
    set_ui_state,
    spatial_memory,
    spatial_memory_lock,
    stop_event,
    tts_busy,
    vad_abort_event,
    vad_capture_active,
    wake_mic_released,
    wakeword_armed,
    WHISPER_AMBIENT_SILENT,
    WHISPER_HALLUCINATIONS,
    whisper_bundle_lock,
    whisper_ready,
)
from dana.logging import log, log_conversation, log_debug
from dana.prompts.spatial_synthesis import build_agent_system_prompt, spatial_focus_hint
from dana.sanitize import sanitize_tool_trace
from dana.tools import ToolCall, ToolValidationError, get_broker
from dana.vision_tools import ScreenAgent, VideoAgent

def build_dana_system_prompt(
    yolo_labels: list[str],
    profile: Optional[dict[str, Any]] = None,
    user_text: str = "",
) -> str:
    """System prompt: SpatialIR synthesis guide + ReAct protocol + language lock."""
    # Not yet migrated (GUI/tray/vision buckets, later phases) -- sanctioned
    # temporary bridge import (same pattern as _nt_hide_console_if_mp_child).
    from dana.core_agent import format_class_list

    if profile is None:
        profile = state.dana_profile
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
        vault_hot_cache=state.VAULT_HOT_CACHE or None,
    )
    # Inject distilled lessons_learned when the intent matches a prior failure domain.
    try:
        broker = get_broker()
        if state.vault_client is not None and state.vault_client.session_token:

            def _lessons_provider():
                from dana.reflector import load_lessons

                return load_lessons(state.vault_client)

            broker.set_lessons_provider(_lessons_provider)
        if user_text:
            prompt = broker.augment_system_prompt(prompt, user_text)
    except Exception:  # noqa: BLE001
        pass
    return prompt


def execute_tool_call(tc: ToolCall) -> str:
    """Dispatch a validated ToolCall IR; returns an Observation string for ReAct."""
    # Not yet migrated (GUI/tray/vision buckets, later phases) -- sanctioned
    # temporary bridge import (same pattern as _nt_hide_console_if_mp_child).
    from dana.core_agent import flush_conversation_memory

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
        source = str(call.arguments.get("source") or "")
        target: Optional[Union[ScreenAgent, VideoAgent]] = None
        if source == "camera":
            target = camera_tool
        elif source == "screen":
            target = screen_tool
        else:
            return "ERROR: source must be screen or camera"

        with active_vision_lock:
            current = state.active_vision_tool
            if current is target:
                return f"OK: vision already on {source}"
            if current is camera_tool and target is screen_tool:
                camera_tool.release()
            state.active_vision_tool = target

        with spatial_memory_lock:
            spatial_memory.clear()
        with latest_dets_lock:
            state.latest_dets.clear()
        with latest_frame_lock:
            state.latest_frame = None

        SPATIAL_AGGREGATOR.set_vision_source(source)
        log("Router", f"Vision tool -> {source} via IR {call.tool_id}")
        return f"OK: switched vision to {source}"

    def _handle_analyze_visual(call: ToolCall) -> str:
        """Screen → mss+pytesseract OCR; webcam → JIT YOLO object detection."""
        source = str(call.arguments.get("source") or "screen").strip().lower()
        if source not in {"screen", "webcam", "camera", "video"}:
            with active_vision_lock:
                source = (
                    "webcam" if state.active_vision_tool is camera_tool else "screen"
                )
        # Schema enum is screen|webcam; vision_tools also accepts camera.
        if source == "camera":
            source = "webcam"
        if source in {"webcam", "video"}:
            from dana.vision_tools import analyze_visual_context

            return analyze_visual_context(source=source)
        from dana.tools.vision import analyze_visual_context

        return analyze_visual_context()

    def _handle_ocr_with_region(call: ToolCall) -> str:
        """Florence-2 OCR grounding → text + ROI overlay highlight."""
        from dana.tools.visual_tools import ocr_with_region

        query = str(call.arguments.get("query") or "").strip()
        return ocr_with_region(query=query)

    def _handle_describe_spatial(call: ToolCall) -> str:
        # Prefer live JIT YOLO payload; keep SpatialIR as secondary context.
        from dana.vision_tools import analyze_visual_context

        with active_vision_lock:
            source = "webcam" if state.active_vision_tool is camera_tool else "screen"
        payload = analyze_visual_context(source=source)
        focus = str(call.arguments.get("focus") or "all")
        block = SPATIAL_AGGREGATOR.synthesize_prompt_block()
        hint = spatial_focus_hint(focus)
        return f"{payload} | SpatialIR={block} | {hint}"

    def _handle_read_vault(call: ToolCall) -> str:
        key = str(call.arguments.get("key") or "").strip()
        if not key:
            return "Error: Memory key not found in vault."
        # Pronoun / garbage keys must never crash the loop.
        if key.lower() in {"it", "this", "that", "them", "those", "these", "something"}:
            return "Error: Memory key not found in vault."
        try:
            if not state.vault_client.session_token:
                return "ERROR: vault session unavailable"
            value = state.vault_client.read_memory(key)
            state.dana_profile = dict(state.vault_client.profile)
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
        key = str(call.arguments.get("key") or "").strip()
        value = call.arguments.get("value")
        if not key:
            return "ERROR: missing key"
        if value is None:
            return "ERROR: missing value"
        try:
            if not state.vault_client.session_token:
                return "ERROR: vault session unavailable"
            state.vault_client.write_memory(key, value)
            state.dana_profile = dict(state.vault_client.profile)
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
            report = state.vault_client.last_consolidation
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
            if result.get("halted"):
                return f"HALTED: inject_keystrokes — {result.get('error')}"
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
        from dana.tools.os_tools import (
            cascade_git_tool_args,
            is_cascade_git_query,
        )
        from dana.tools.system_repl import shell_execute

        command = call.arguments.get("command")
        cwd = call.arguments.get("cwd")
        cwd_s = str(cwd).strip() if cwd is not None else ""
        raw = str(call.raw_text or "")
        if is_cascade_git_query(raw):
            forced = cascade_git_tool_args(raw)
            command = forced["command"]
            cwd_s = forced["cwd"]
        if command is None or not str(command).strip():
            return "ERROR: missing command"
        return shell_execute(str(command), cwd=cwd_s or None)

    def _handle_execute_powershell(call: ToolCall) -> str:
        from dana.tools.os_tools import (
            cascade_git_tool_args,
            is_cascade_git_query,
        )
        from dana.tools.powershell import execute_powershell

        command = call.arguments.get("command")
        cwd = call.arguments.get("cwd")
        cwd_s = str(cwd).strip() if cwd is not None else ""
        raw = str(call.raw_text or "")
        # Named-repo git asks: always retarget cwd + git log (ignore bad LLM args).
        if is_cascade_git_query(raw):
            forced = cascade_git_tool_args(raw)
            command = forced["command"]
            cwd_s = forced["cwd"]
        if command is None or not str(command).strip():
            return "ERROR: missing command"
        return execute_powershell(str(command), cwd=cwd_s or None)

    def _handle_write_to_file(call: ToolCall) -> str:
        from dana.tools.actuators import write_to_file

        filepath = call.arguments.get("filepath")
        if filepath is None or not str(filepath).strip():
            return "ERROR: missing filepath"
        content = call.arguments.get("content")
        content_s = "" if content is None else str(content)
        return write_to_file(str(filepath), content_s)

    def _handle_execute_command(call: ToolCall) -> str:
        from dana.tools.actuators import execute_command

        command = call.arguments.get("command")
        if command is None or not str(command).strip():
            return "ERROR: missing command"
        timeout_raw = call.arguments.get("timeout", 15)
        try:
            timeout_sec = int(timeout_raw) if timeout_raw is not None else 15
        except (TypeError, ValueError):
            timeout_sec = 15
        return execute_command(str(command), timeout=timeout_sec)

    def _handle_execute_python_script(call: ToolCall) -> str:
        from dana.tools.actuators import execute_python_script

        script_path = call.arguments.get("script_path")
        if script_path is None or not str(script_path).strip():
            return "ERROR: missing script_path"
        timeout_raw = call.arguments.get("timeout", 300)
        try:
            timeout_sec = int(timeout_raw) if timeout_raw is not None else 300
        except (TypeError, ValueError):
            timeout_sec = 300
        background = call.arguments.get("background", False)
        bg = bool(background) and str(background).strip().lower() not in {
            "0",
            "false",
            "no",
            "",
        }
        return execute_python_script(
            str(script_path),
            args=call.arguments.get("args"),
            timeout=timeout_sec,
            background=bg,
        )

    def _handle_get_sandbox_job_status(call: ToolCall) -> str:
        from dana.tools.actuators import get_sandbox_job_status

        job_id = call.arguments.get("job_id")
        key = None if job_id is None or not str(job_id).strip() else str(job_id).strip()
        return get_sandbox_job_status(key)

    def _handle_fetch_webpage(call: ToolCall) -> str:
        from dana.tools.browser import fetch_webpage

        url = call.arguments.get("url")
        if url is None or not str(url).strip():
            return "ERROR: missing url"
        selector = call.arguments.get("selector")
        limit = call.arguments.get("limit")
        extract_hn = call.arguments.get("extract_hn_titles", False)
        extract_hn_titles = bool(extract_hn) and str(extract_hn).strip().lower() not in (
            "0",
            "false",
            "no",
            "",
        )
        kwargs: dict[str, object] = {}
        if selector is not None and str(selector).strip():
            kwargs["selector"] = str(selector).strip()
        if limit is not None:
            kwargs["limit"] = limit
        if extract_hn_titles:
            kwargs["extract_hn_titles"] = True
        return fetch_webpage(str(url), **kwargs)  # type: ignore[arg-type]

    def _handle_file_editor(call: ToolCall) -> str:
        from dana.tools.os_tools import append_dependency_digest
        from dana.tools.system_repl import file_editor

        action = str(call.arguments.get("action") or "").strip()
        filepath = call.arguments.get("filepath")
        if filepath is None or not str(filepath).strip():
            return "ERROR: missing filepath"
        content = call.arguments.get("content")
        content_s = None if content is None else str(content)
        obs = file_editor(action, str(filepath), content_s)
        if str(action).lower() == "read":
            return append_dependency_digest(str(filepath), obs)
        return obs

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

    def _handle_ingest_local_directory(call: ToolCall) -> str:
        from dana.memory.vault import ingest_local_directory

        path = call.arguments.get("path")
        if path is None or not str(path).strip():
            raw = str(call.raw_text or "")
            m = re.search(
                r"([A-Za-z]:\\[^\s\"']+|/[^\s\"']+|\.{0,2}/[^\s\"']+)",
                raw,
            )
            path = m.group(1) if m else ""
        if not str(path or "").strip():
            return "ERROR: missing path for ingest_local_directory"
        return ingest_local_directory(str(path).strip())

    def _handle_search_vault(call: ToolCall) -> str:
        from dana.memory.vault import search_vault

        query = call.arguments.get("query")
        if query is None or not str(query).strip():
            query = str(call.raw_text or "").strip()
        if not str(query or "").strip():
            return "ERROR: missing query for search_vault"
        n_raw = call.arguments.get("n_results", 5)
        try:
            n_results = int(n_raw) if n_raw is not None else 5
        except (TypeError, ValueError):
            n_results = 5
        return search_vault(str(query).strip(), n_results=n_results)

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
        from dana.tools.os_tools import (
            append_dependency_digest,
            is_watchdog_graph_query,
            watchdog_graph_filepath,
        )

        filepath = call.arguments.get("filepath")
        if filepath is None or not str(filepath).strip():
            filepath = call.arguments.get("path")
        raw = str(call.raw_text or "")
        # Watchdog graph asks: force the exact module path (ignore bad LLM args).
        if is_watchdog_graph_query(raw):
            filepath = watchdog_graph_filepath()
        if filepath is None or not str(filepath).strip():
            return "ERROR: missing filepath"
        path_s = str(filepath).strip()
        result = read_local_file(path_s)
        if str(result).upper().startswith("ERROR"):
            return str(result)
        obs = f"OK: read_local_file path={path_s!r}\n{result}"
        return append_dependency_digest(path_s, obs)

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

    def _handle_meta_broker(call: ToolCall) -> str:
        """Closed-loop Meta-Broker DAG — bypasses Tool Forge / generic ReAct."""
        import logging
        import traceback as _tb

        from dana.graph.artifact_manifest import META_BROKER_STDLIB_RULE
        from dana.graph.meta_broker_process import run_meta_broker_isolated
        from dana.graph.monitor_bus import (
            get_monitor_bus,
            publish_graph_error,
            write_broker_crash_dump,
        )
        from dana.tools.broker import extract_meta_broker_prompt

        prompt = str(
            call.arguments.get("prompt")
            or call.arguments.get("goal")
            or call.arguments.get("query")
            or call.raw_text
            or ""
        ).strip()
        prompt = extract_meta_broker_prompt(prompt)
        if not prompt:
            return "ERROR: meta_broker missing prompt"
        raw_intent = prompt
        approved = bool(call.arguments.get("approved"))
        # Spec Compiler — plain English → strict /broker, or REJECT without spawn.
        try:
            from dana.graph.nodes.spec_compiler import (
                PENDING_USER_APPROVAL,
                build_spec_approval_payload,
                compile_user_spec,
                hitl_spec_approval_enabled,
                is_broker_ready_spec,
                is_reject_spec,
            )

            if not is_broker_ready_spec(prompt):
                compiled = compile_user_spec(prompt)
                if is_reject_spec(compiled):
                    log("MetaBroker", compiled)
                    return compiled
                prompt = compiled
                log(
                    "MetaBroker",
                    f"spec_compiler produced /broker chars={len(prompt)}",
                )
            # Human-in-the-loop: pause for Approve & Run unless already approved.
            if hitl_spec_approval_enabled() and not approved:
                payload = build_spec_approval_payload(
                    compiled_spec=prompt,
                    raw_intent=raw_intent,
                )
                log(
                    "MetaBroker",
                    f"{PENDING_USER_APPROVAL}: awaiting UI Approve & Run "
                    f"chars={len(prompt)}",
                )
                try:
                    notify_spec_approval_requested(payload)
                except Exception:  # noqa: BLE001
                    pass
                return (
                    f"{PENDING_USER_APPROVAL}: Spec compiled — "
                    "click Approve & Run on the Approval Card to dispatch.\n"
                    f"{prompt[:300]}"
                )
        except Exception as exc:  # noqa: BLE001
            log("MetaBroker", f"spec_compiler skipped ({exc})")
        # Prompt builder: stamp stdlib-only epic codegen constraint on the macro.
        if META_BROKER_STDLIB_RULE not in prompt:
            prompt = f"{META_BROKER_STDLIB_RULE}\n\n{prompt}"
        try:
            bus = get_monitor_bus(create=True)
            if bus is not None:
                bus.publish("status", status="meta_broker_running")
                bus.publish("tool", message=f"meta_broker start: {prompt[:160]}")
        except Exception:  # noqa: BLE001
            pass
        log(
            "MetaBroker",
            f"dispatch run_meta_broker (isolated process) chars={len(prompt)}",
        )

        def _on_broker_event(event: dict) -> None:
            if str(event.get("type") or "") != "telemetry":
                return
            msg = str(event.get("message") or "")
            phase = str(event.get("phase") or "")
            status = str(event.get("status") or "")
            epic_title = str(event.get("epic_title") or "")
            terminal = bool(event.get("terminal"))
            try:
                from dana.graph.task_tracker import emit_meta_broker_telemetry

                emit_meta_broker_telemetry(
                    task_id="meta_broker",
                    prompt=prompt,
                    phase=phase,
                    status=status,
                    message=msg,
                    epic_title=epic_title,
                    terminal=terminal,
                )
            except Exception:  # noqa: BLE001
                pass
            try:
                from dana.ui.status_bus import emit_state_change

                if terminal:
                    emit_state_change(
                        "idle",
                        tool="meta_broker",
                        message=msg[:120] or "Meta-Broker idle",
                    )
                else:
                    emit_state_change(
                        "processing",
                        tool="meta_broker",
                        message=msg[:120] or "Meta-Broker running",
                    )
            except Exception:  # noqa: BLE001
                pass
            try:
                from dana.audio.tts_manager import get_tts_manager

                mgr = get_tts_manager()
                low = msg.lower()
                if "dispatch epic" in low or low.startswith("starting epic"):
                    # "Starting Epic 1: Title" / legacy "Dispatch epic 1: …"
                    spoken = msg
                    if "dispatch epic" in low:
                        spoken = msg.replace("Dispatch epic", "Starting Epic", 1)
                        spoken = spoken.replace("dispatch epic", "Starting Epic", 1)
                    mgr.notify(spoken.split(":")[0].strip() + ".")
                elif terminal and status.lower() in {
                    "completed",
                    "done",
                    "ok",
                    "success",
                }:
                    mgr.notify("Task complete.")
                elif "ram" in low:
                    mgr.notify("Warning: RAM limit exceeded.")
                elif terminal and status.lower() in {"failed", "error"}:
                    mgr.notify("Task failed.")
            except Exception:  # noqa: BLE001
                pass
            try:
                bus2 = get_monitor_bus(create=False)
                if bus2 is not None:
                    bus2.publish(
                        "tool",
                        message=msg[:200],
                    )
            except Exception:  # noqa: BLE001
                pass

        try:
            final = run_meta_broker_isolated(
                prompt,
                on_event=_on_broker_event,
                timeout_s=float(
                    os.environ.get("DANA_META_BROKER_TIMEOUT_S") or "300"
                ),
            )
        except Exception as exc:  # noqa: BLE001
            tb_text = "".join(_tb.format_exception(type(exc), exc, exc.__traceback__))
            logging.getLogger("dana.meta_broker").exception(
                "run_meta_broker crashed: %s", exc
            )
            dump = write_broker_crash_dump(
                exc,
                context=f"_handle_meta_broker prompt={prompt[:200]!r}",
                traceback_text=tb_text,
            )
            try:
                publish_graph_error(
                    f"run_meta_broker raised {type(exc).__name__}: {exc}",
                    exc=exc,
                    node="meta_broker_dispatch",
                    dump=False,
                    prompt_preview=prompt[:200],
                )
            except Exception:  # noqa: BLE001
                pass
            err = (
                f"ERROR: meta_broker failed: {type(exc).__name__}: {exc}\n"
                f"crash_dump={dump}"
            )
            log("MetaBroker", err)
            return err
        status = str(final.get("status") or "")
        epics = list(final.get("epics") or [])
        response = str(final.get("final_response") or "").strip()
        epic_log = list(final.get("epic_log") or [])
        err_field = str(final.get("error") or "").strip()
        if status == "failed" or err_field:
            try:
                publish_graph_error(
                    err_field or response or "meta_broker finished with status=failed",
                    node="meta_broker_result",
                    dump=True,
                    status=status,
                )
            except Exception:  # noqa: BLE001
                pass
        summary = (
            f"OK: meta_broker status={status} epics={len(epics)}\n"
            f"{response or '(no final_response)'}\n"
            f"epic_log: {'; '.join(str(x) for x in epic_log[-8:])}"
        )
        if status == "failed":
            summary = (
                f"ERROR: meta_broker status=failed epics={len(epics)}\n"
                f"{err_field or response or '(no detail)'}\n"
                f"epic_log: {'; '.join(str(x) for x in epic_log[-8:])}"
            )
        try:
            bus = get_monitor_bus(create=False)
            if bus is not None:
                bus.publish(
                    "dag",
                    status=status,
                    tasks=[
                        {
                            "task_id": e.get("epic_id"),
                            "action": e.get("title") or e.get("goal"),
                            "status": e.get("status"),
                        }
                        for e in epics
                        if isinstance(e, dict)
                    ],
                )
                bus.publish("status", status=status or "completed")
                bus.publish("done", status=status or "completed")
                bus.publish("tool", message=summary[:400])
        except Exception:  # noqa: BLE001
            pass
        return summary

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

    def _handle_list_activity_for_day(call: ToolCall) -> str:
        from dana.tools.activity_index import handle_list_activity_for_day

        date_str = call.arguments.get("date_str")
        if date_str is None or not str(date_str).strip():
            # Infer from raw utterance when the LLM omits the arg.
            raw = str(call.raw_text or "")
            date_str = "yesterday"
            low = raw.lower()
            if "this morning" in low or "today" in low:
                date_str = "today"
            elif "last night" in low:
                date_str = "last night"
            elif "previous session" in low:
                date_str = "previous session"
        return handle_list_activity_for_day(str(date_str))

    def _handle_get_system_telemetry(call: ToolCall) -> str:
        from dana.tools.system_tools import handle_get_system_telemetry

        return handle_get_system_telemetry()

    def _handle_parse_idle_log_duration(call: ToolCall) -> str:
        from dana.tools.system_tools import handle_parse_idle_log_duration

        log_path = call.arguments.get("log_path")
        path_arg = str(log_path).strip() if log_path is not None else ""
        return handle_parse_idle_log_duration(path_arg or None)

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
            vault_client=state.vault_client if state.vault_client.session_token else None,
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
        "execute_powershell": _handle_execute_powershell,
        "write_to_file": _handle_write_to_file,
        "execute_command": _handle_execute_command,
        "execute_python_script": _handle_execute_python_script,
        "get_sandbox_job_status": _handle_get_sandbox_job_status,
        "fetch_webpage": _handle_fetch_webpage,
        "file_editor": _handle_file_editor,
        "python_repl": _handle_python_repl,
        "flush_memory": _handle_flush_memory,
        "ingest_local_directory": _handle_ingest_local_directory,
        "search_vault": _handle_search_vault,
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
        "meta_broker": _handle_meta_broker,
        "read_system_architecture": _handle_read_architecture,
        "list_activity_for_day": _handle_list_activity_for_day,
        "get_system_telemetry": _handle_get_system_telemetry,
        "parse_idle_log_duration": _handle_parse_idle_log_duration,
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
        raw_obs = str(broker.dispatch(tc, handlers))
    except ToolValidationError as exc:
        raw_obs = f"ERROR: dispatch failed ({exc})"
    # Scratchpad: compress before ReAct messages / blackboard / telemetry see it.
    try:
        from dana.middleware.scratchpad import compress_tool_output

        return compress_tool_output(raw_obs)
    except Exception:  # noqa: BLE001
        return raw_obs


def tool_router(whisper_text: str) -> tuple[str, Optional[ToolCall]]:
    """Fast-path bilingual IR router for immediate side effects (vision switch).

    Returns ``(possibly_corrected_text, deferred_tool_or_None)``.
    Bound deferred tools are forced into the agentic loop; unbound tools
    (e.g. describe_spatial_scene) only inject a hard visual-context constraint.

    Soft alias hits below ``HIGH_CONFIDENCE_TOOL_THRESHOLD`` are ignored unless
    ``requires_tool_graph`` already demands ReAct — keeps System-1 strict.
    """
    # Not yet migrated (GUI/tray/vision buckets, later phases) -- sanctioned
    # temporary bridge import (same pattern as _nt_hide_console_if_mp_child).
    from dana.core_agent import speak_tool_working_ack

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

    try:
        from dana.tools.broker import HIGH_CONFIDENCE_TOOL_THRESHOLD

        conf = float(getattr(call, "confidence", 0.0) or 0.0)
        if conf < float(HIGH_CONFIDENCE_TOOL_THRESHOLD) and not requires_tool_graph(
            whisper_text or ""
        ):
            log(
                "Router",
                f"Soft tool alias ignored for System-1 "
                f"({call.tool_id} conf={conf:.2f} < {HIGH_CONFIDENCE_TOOL_THRESHOLD})",
            )
            return whisper_text, None
    except Exception:  # noqa: BLE001
        pass

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


def _register_chat_soft_drop(
    *,
    prompt: str,
    reason: str,
    answer: str = "",
) -> str:
    """RECEIVED → IN_PROGRESS → DROPPED for lightweight-chat soft fails.

    Writes ``logs/dropped_tasks.log`` only (no production ledger spam).
    """
    import uuid

    from dana.graph.task_tracker import TaskStatus, get_shared_task_tracker

    tid = f"chat_{uuid.uuid4().hex[:12]}"
    tracker = get_shared_task_tracker()
    tracker.start_task(tid, prompt or "")
    tracker.update_status(tid, TaskStatus.IN_PROGRESS)
    tracker.log_dropped_task(
        tid,
        reason,
        last_state_buffer={
            "user_text": prompt or "",
            "final_raw": answer or "",
            "mode": "chat",
        },
        draft_ledger=False,
    )
    return tid


def _mark_mid_task_prompt(prompt: str | None) -> None:
    with _active_mid_task_lock:
        state._active_mid_task_prompt = (prompt or "").strip() or None


def _clear_mid_task_prompt() -> None:
    with _active_mid_task_lock:
        state._active_mid_task_prompt = None


def _drop_mid_task_on_vad_timeout(*, stop_reason: str, rms_raw: float) -> None:
    """If a chat mid-task was open when VAD emptied, record DROPPED."""
    with _active_mid_task_lock:
        prompt = state._active_mid_task_prompt
        state._active_mid_task_prompt = None
    if not prompt:
        return
    try:
        _register_chat_soft_drop(
            prompt=prompt,
            reason=(
                f"vad_{stop_reason}_mid_task rms_raw={float(rms_raw):.5f}"
            ),
            answer="",
        )
        log(
            "Conversation",
            f"Task tracker DROPPED mid-task on VAD {stop_reason} "
            f"(rms_raw={float(rms_raw):.5f})",
        )
    except Exception as exc:  # noqa: BLE001
        log("Conversation", f"WARNING: mid-task drop log failed ({exc})")


def ask_ollama_messages(
    messages: list[dict[str, str]],
    model: str = OLLAMA_MODEL,
    *,
    num_predict: Optional[int] = None,
    response_format: Any = None,
    temperature: Optional[float] = None,
) -> str:
    """Isolated Ollama chat call (no conversation_history mutation) for ReAct steps.

    Streams the response so LLM TTFT can be recorded to ``dana_performance.log``.

    ``response_format`` maps to Ollama ``format`` (``\"json\"`` or a JSON Schema
    dict) for structured small-model tool / supervisor outputs.
    """
    from dana.agentic import OLLAMA_UNREACHABLE_SPEECH
    from dana.system_health import llm_lock

    # Serialize generations — concurrent Ollama calls double VRAM and crash GPUs.
    with llm_lock:
        return _ask_ollama_messages_unlocked(
            messages,
            model,
            num_predict=num_predict,
            response_format=response_format,
            temperature=temperature,
            unreachable_speech=OLLAMA_UNREACHABLE_SPEECH,
        )


def _ask_ollama_messages_unlocked(
    messages: list[dict[str, str]],
    model: str,
    *,
    num_predict: Optional[int],
    response_format: Any,
    temperature: Optional[float],
    unreachable_speech: str,
) -> str:
    # Thermal guardrail: unload when USER_AWAY; keep warm 5m while USER_ACTIVE.
    # Never pass -1 / "-1" — Ollama returns HTTP 400 on those values.
    try:
        from dana.middleware.idle_monitor import ollama_keep_alive

        _keep_alive = ollama_keep_alive()
    except Exception:  # noqa: BLE001
        _keep_alive = 0
    from dana.llm_client import merge_ollama_options

    options: dict[str, Any] = {
        "num_ctx": 8192,
        "num_predict": 1024 if num_predict is None else int(num_predict),
    }
    if temperature is not None:
        options["temperature"] = float(temperature)

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "keep_alive": _keep_alive,
        # Hard caps for 8GB VRAM: shorter KV cache + bounded generation.
        # When DANA_SPECULATIVE_DECODING=1, merge_ollama_options injects
        # draft_num_predict (draft model pairing is Modelfile DRAFT / -md).
        "options": merge_ollama_options(options),
    }
    if response_format is not None:
        # Ollama structured outputs — schema dict or "json".
        payload["format"] = response_format
    t0 = time.perf_counter()
    ttft_logged = False
    parts: list[str] = []
    try:
        with requests.post(
            OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT_SEC, stream=True
        ) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.strip() if isinstance(raw_line, str) else raw_line.decode(
                    "utf-8", errors="replace"
                ).strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                message = data.get("message") or {}
                piece = str(message.get("content") or "")
                if piece:
                    if not ttft_logged:
                        ttft_logged = True
                        try:
                            from dana.perf import log_perf

                            log_perf(
                                "llm_ttft",
                                (time.perf_counter() - t0) * 1000.0,
                                model=model,
                            )
                        except Exception:  # noqa: BLE001
                            pass
                    parts.append(piece)
                if data.get("done"):
                    break
    except (requests.exceptions.ConnectionError, ConnectionError) as exc:
        raise ConnectionError(unreachable_speech) from exc
    except requests.exceptions.Timeout as exc:
        raise TimeoutError(
            f"{unreachable_speech} (timed out after {OLLAMA_TIMEOUT_SEC:.0f}s)"
        ) from exc
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(f"Ollama HTTP error: {exc}") from exc

    content = "".join(parts).strip()
    if not content:
        raise RuntimeError("Ollama returned empty content")
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


# ---------------------------------------------------------------------------
# Thread 4 - Conversational cascade (wake -> turns -> follow-up loop)
# ---------------------------------------------------------------------------

def conversation_worker(
    local_files_only: bool,
    device,
    dtype,
) -> None:
    # Not yet migrated (GUI/tray/vision/CLI buckets, later phases) --
    # sanctioned temporary bridge import (same pattern used in
    # dana.audio.mic_input/tts_worker for _nt_hide_console_if_mp_child).
    from dana.core_agent import (
        _nt_hide_console_if_mp_child,
        clear_context_spoken_reply,
        compile_and_append_voice_prompt,
        emit_trace,
        execute_lockdown_shutdown,
        flush_conversation_memory,
        format_class_list,
        format_vision_context_for_llm,
        get_spatial_memory_labels,
        is_clear_context_command,
        is_engine_engaged,
        is_lockdown_command,
        is_standby_command,
        is_time_command,
        parse_yolo_results,
        pop_injected_question_ex,
        remember_spatial_labels,
        speak_tool_working_ack,
        wall_clock_spoken_reply,
        yolo_device_arg,
    )

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
            if state._shared_wakeword_model is not None:
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
        f"Dana is ready (brain={OLLAMA_MODEL}). Say 'Dana' to wake — then ask "
        f"follow-ups without the wake word (~{FOLLOWUP_VAD_MAX_SECONDS:.0f}s silence -> Standing by).",
    )

    def _warmup_llm() -> None:
        """Background 1-token ping so qwen2.5-coder:7b weights land in VRAM before first turn.

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
                emit_live_transcript("Dana", message)
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
            log_conversation("Dana", ack)
            emit_live_transcript("User (Whisper)", whisper_text or "")
            emit_live_transcript("Dana", ack)
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
                        mode=get_dana_mode(),
                    )
                except Exception:  # noqa: BLE001
                    pass
                try:
                    notify_dictation_sessions_changed()
                except Exception:  # noqa: BLE001
                    pass
                log_conversation("Dana", ack)
                emit_live_transcript("Dana", ack)
                enqueue_speech(ack, interruptible=False)
                wait_for_speech_idle(timeout=8.0)
                set_subtitle("")
                return True
        except Exception as exc:  # noqa: BLE001
            log("Dictation", f"WARNING: dictation handler failed ({exc})")

        # Mode switch fast-path — no LLM, no YOLO, no tools.
        switched = parse_mode_switch(whisper_text or "")
        if switched is not None:
            active = set_dana_mode(switched)
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
            log_conversation("Dana", ack)
            emit_live_transcript("User (Whisper)", whisper_text or "")
            emit_live_transcript("Dana", ack)
            enqueue_speech(ack, interruptible=False)
            wait_for_speech_idle(timeout=8.0)
            set_subtitle("")
            return True

        # Meta-Broker keyword override — bypass LLM classifier / architect_new_tool.
        try:
            from dana.tools.broker import (
                extract_meta_broker_prompt,
                is_meta_broker_intent,
            )

            if is_meta_broker_intent(whisper_text or ""):
                mb_prompt = extract_meta_broker_prompt(whisper_text or "")
                log(
                    "Conversation",
                    "Meta-Broker override: bypassing intent classifier → meta_broker",
                )
                try:
                    from dana.telemetry import log_router

                    log_router(
                        "meta_broker override",
                        current_agent="MetaBroker",
                        active_intent="meta_broker",
                        payload={"chars": len(mb_prompt)},
                    )
                except Exception:  # noqa: BLE001
                    pass
                # Persist once to the conversation log. Do NOT emit_live_transcript
                # for the user line here — silent text inject / Whisper paths already
                # echoed the utterance into the Dashboard chat (double-print bug).
                log_conversation("User", whisper_text or "")
                obs = execute_tool_call(
                    ToolCall(
                        tool_id="meta_broker",
                        arguments={"prompt": mb_prompt},
                        raw_text=whisper_text or "",
                        confidence=0.99,
                    )
                )
                spoken = obs
                # HITL gate — Approval Card owns the next step; don't TTS the macro.
                if str(spoken or "").startswith("PENDING_USER_APPROVAL"):
                    log_conversation("Dana", "Awaiting spec approval.")
                    emit_live_transcript(
                        "Dana",
                        "Spec compiled — review the Approval Card, then Approve & Run.",
                    )
                    set_subtitle("")
                    return True
                if spoken.startswith("OK:"):
                    # Prefer the broker's final_response line for TTS.
                    lines = [
                        ln.strip()
                        for ln in spoken.splitlines()
                        if ln.strip() and not ln.startswith("epic_log:")
                    ]
                    spoken = lines[1] if len(lines) > 1 else lines[0]
                if len(spoken) > 420:
                    spoken = spoken[:417] + "..."
                log_conversation("Dana", spoken)
                emit_live_transcript("Dana", spoken)
                enqueue_speech(spoken, interruptible=False)
                wait_for_speech_idle(timeout=30.0)
                set_subtitle("")
                return True
        except Exception as exc:  # noqa: BLE001
            log("Conversation", f"WARNING: meta_broker override failed ({exc})")

        # Proportional compute: System-1 lightweight unless a tool is required.
        # Isolated jail tasks (BACKGROUND / queue) always take the ReAct path.
        use_chat = not isolated
        tool_force = requires_tool_graph(whisper_text or "")
        high_conf_tool = False
        peek_call = None
        try:
            from dana.tools.broker import HIGH_CONFIDENCE_TOOL_THRESHOLD

            peek_call = get_broker().parse_utterance(whisper_text or "")
            if peek_call is not None and float(
                getattr(peek_call, "confidence", 0.0) or 0.0
            ) >= float(HIGH_CONFIDENCE_TOOL_THRESHOLD):
                high_conf_tool = True
        except Exception:  # noqa: BLE001
            peek_call = None
        if use_chat and (tool_force or high_conf_tool):
            use_chat = False
            log(
                "Conversation",
                "Tool-graph escalation: high-confidence tool / system intent → "
                "ReAct/MoA (lightweight chat bypassed)",
            )
        elif use_chat and get_dana_mode() not in ("chat",):
            # Developer/vision/research without an explicit tool stay on System-1.
            log(
                "Conversation",
                f"Proportional System-1: mode={get_dana_mode()} with no "
                "high-confidence tool — lightweight chat",
            )
        routed_tool = None
        if not use_chat:
            whisper_text, routed_tool = tool_router(whisper_text)
            if routed_tool is None and high_conf_tool and peek_call is not None:
                routed_tool = peek_call
        route_tag = (
            "chat"
            if use_chat
            else ("tool" if (tool_force or high_conf_tool) else "developer")
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
        try:
            from dana.tools.broker import should_blindfold_vision

            _vision_blindfold = should_blindfold_vision(
                user_text=whisper_text or "",
                forced_tool_id=(
                    routed_tool.tool_id if routed_tool is not None else None
                ),
            )
        except Exception:  # noqa: BLE001
            _vision_blindfold = False
        if _vision_blindfold:
            log(
                "Conversation",
                "Vision blindfold: ignoring analyze_visual_context / screen "
                "context for research swarm or USER_AWAY",
            )
        if use_chat and not _vision_blindfold:
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
        elif use_chat and _vision_blindfold:
            vision_log = ""
        elif _vision_blindfold:
            vision_log = ""
        else:
            # Prefer a fresh frame from the active tool (important after a switch).
            with active_vision_lock:
                tool = state.active_vision_tool
            frame = None
            try:
                frame = tool.get_frame()
            except Exception as exc:  # noqa: BLE001
                log("Conversation", f"WARNING: active tool get_frame failed ({exc})")
            if frame is not None:
                with latest_frame_lock:
                    state.latest_frame = frame
            else:
                with latest_frame_lock:
                    frame = None if state.latest_frame is None else state.latest_frame.copy()
            if frame is None and not tool_force:
                log("Conversation", "No vision frame available; skipping turn.")
                return False

            if frame is not None:
                with latest_dets_lock:
                    dets = list(state.latest_dets)
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
                            state.latest_dets[:] = dets
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
            system_prompt = build_dana_system_prompt(
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
                log_conversation("Dana", answer)
                emit_live_transcript("Dana", answer)
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
                try:
                    from dana.cascade_router import lightweight_model_name

                    _lw_model = lightweight_model_name()
                except Exception:  # noqa: BLE001
                    _lw_model = OLLAMA_MODEL
                result = run_lightweight_chat(
                    user_text=whisper_text,
                    system_prompt=system_prompt,
                    model=_lw_model,
                    ask_fn=ask_ollama_messages,
                    visual_context=vision_log or None,
                    use_chat_memory=True,
                )
                answer = result.final_text
                # Completion gate: filler / verbal escalate → ReAct once (no END).
                chat_escalated_once = False
                try:
                    from dana.graph.completion_gate import (
                        is_filler_response,
                        should_reject_chat_final,
                    )

                    if should_reject_chat_final(answer, whisper_text):
                        reason = (
                            "chat_filler_end"
                            if is_filler_response(answer)
                            else "chat_verbal_tool_graph_escalate"
                        )
                        try:
                            _register_chat_soft_drop(
                                prompt=whisper_text or "",
                                reason=reason,
                                answer=answer or "",
                            )
                        except Exception as drop_exc:  # noqa: BLE001
                            log(
                                "Conversation",
                                f"WARNING: chat soft-drop log failed ({drop_exc})",
                            )
                        _mark_mid_task_prompt(whisper_text)
                        chat_escalated_once = True
                        use_chat = False
                        log(
                            "Conversation",
                            "Chat completion gate: "
                            f"{reason} → escalate once to ReAct/MoA",
                        )
                        try:
                            from dana.telemetry import log_router

                            log_router(
                                "tool-graph escalation",
                                current_agent="ReAct_Agent",
                                active_intent="chat_completion_gate",
                                payload={"reason": reason, "mode": "chat"},
                            )
                        except Exception:  # noqa: BLE001
                            pass
                        if not routed_tool:
                            whisper_text, routed_tool = tool_router(whisper_text)
                        result = run_react_loop(
                            user_text=whisper_text,
                            system_prompt=system_prompt,
                            execute_fn=execute_tool_call,
                            max_iters=REACT_MAX_ITERS,
                            vault_client=(
                                state.vault_client if state.vault_client.session_token else None
                            ),
                            reflect_fn=ask_ollama_messages,
                            prior_messages=prior_turns,
                            on_tool_start=speak_tool_working_ack,
                            visual_context=vision_log or None,
                            model=OLLAMA_MODEL,
                            forced_tool=routed_tool,
                            tts_callback=enqueue_speech,
                        )
                        answer = result.final_text
                        _clear_mid_task_prompt()
                except Exception as gate_exc:  # noqa: BLE001
                    log(
                        "Conversation",
                        f"WARNING: chat completion gate failed ({gate_exc})",
                    )
                try:
                    from dana.schema import TraceEvent
                    from dana.ui.trace_bus import TraceEventBus

                    TraceEventBus.instance().emit(
                        TraceEvent(
                            event_type="node_exit",
                            node="synthesis",
                            message=(
                                "Chat escalated to ReAct"
                                if chat_escalated_once
                                else "Chat complete"
                            ),
                            mode="chat" if not chat_escalated_once else "developer",
                            payload=(answer or "")[:800],
                            latency_ms=(time.perf_counter() - _chat_trace_t0)
                            * 1000.0,
                            state_keys=("final_raw", "messages"),
                        )
                    )
                except Exception:  # noqa: BLE001
                    pass
                if chat_escalated_once:
                    log(
                        "Conversation",
                        "Chat→tool-graph escalate complete "
                        f"(chat_memory={chat_memory_size()})",
                    )
                else:
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
                    vault_client=state.vault_client if state.vault_client.session_token else None,
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
                # Compact INFO tool ids; full sanitized observations only under DANA_DEBUG.
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
        log_conversation("Dana", answer or "", extra=f"{latency_ms:.0f} ms")
        emit_live_transcript("Dana", answer)
        SPATIAL_AGGREGATOR.update_transcript(assistant=answer)

        # Prefer live astream TTS; skip duplicate final enqueue when already spoken.
        if not getattr(result, "tts_streamed", False):
            enqueue_speech(answer if answer else "I'm not sure.")
        elif not (answer or "").strip():
            enqueue_speech("I'm not sure.")
        wait_for_speech_idle(timeout=30.0)
        time.sleep(0.15)
        set_subtitle("")

        # Phase 5 — compress completed idle research into dense vault summaries.
        try:
            if isolated and str(whisper_text or "").lstrip().startswith(
                "[BACKGROUND TASK]"
            ):
                from dana.middleware.idle_monitor import compress_idle_research_output

                blob = f"{whisper_text}\n\n{answer or ''}".strip()
                outcome = compress_idle_research_output(
                    blob, topic=str(whisper_text)[:200]
                )
                log_debug("IdleCompress", f"background task compress: {outcome}")
        except Exception as exc:  # noqa: BLE001
            log_debug("IdleCompress", f"WARNING: compress skipped ({exc})")
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
                # System/queue lane — never emit Conversation "User said:" (that
                # listener is reserved for live mic/text and caused recursive routing).
                log("TaskQueue", f'Dispatching isolated ReAct: "{preview}"')
                try:
                    set_subtitle(f'Background: "{preview}"')
                except Exception:  # noqa: BLE001
                    pass
                if is_standby_command(command):
                    emit_live_transcript("Dana", "Standing by.")
                    enqueue_speech("Standing by.", interruptible=False)
                    wait_for_speech_idle(timeout=8.0)
                    return
                if is_clear_context_command(command):
                    flush_conversation_memory(reason="task_queue")
                    reply = clear_context_spoken_reply(command)
                    emit_live_transcript("Dana", reply)
                    enqueue_speech(reply)
                    wait_for_speech_idle(timeout=8.0)
                    return
                if is_lockdown_command(command):
                    execute_lockdown_shutdown()
                    return
                if is_time_command(command):
                    reply = wall_clock_spoken_reply()
                    emit_live_transcript("Dana", reply)
                    enqueue_speech(reply)
                    wait_for_speech_idle(timeout=8.0)
                    return
                ok = run_brain_turn(command, time.perf_counter(), isolated=True)
                if not ok:
                    raise RuntimeError("isolated ReAct turn failed")

            # System job lane: escalate for ReAct jail without mutating voice mode.
            from dana.agentic import restore_voice_mode

            prior = get_dana_mode()
            escalated = False
            if prior == "chat":
                set_dana_mode("developer", as_voice=False)
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
                    log_conversation("Dana", reply)
                    emit_live_transcript("Dana", reply)
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
                    log_conversation("Dana", reply)
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

                    # Text/chat override: skip Whisper / 10s timeout; process inject/queue.
                    if stop_reason == "text_override":
                        log(
                            "Conversation",
                            "Follow-up VAD aborted for text — processing chat override",
                        )
                        set_ui_state("idle")
                        continue

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
                        _drop_mid_task_on_vad_timeout(
                            stop_reason=stop_reason,
                            rms_raw=rms_raw,
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

                    # Text/chat override: skip Whisper / 10s timeout; process inject/queue.
                    if stop_reason == "text_override":
                        log(
                            "Conversation",
                            "Wake VAD aborted for text — processing chat override",
                        )
                        set_ui_state("idle")
                        continue

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
                        try:
                            from dana.ui.status_bus import emit_state_change

                            emit_state_change(
                                "idle",
                                tool="vad_timeout",
                                message="No speech detected — disarmed",
                            )
                        except Exception:  # noqa: BLE001
                            pass
                        _drop_mid_task_on_vad_timeout(
                            stop_reason=stop_reason,
                            rms_raw=rms_raw,
                        )
                        end_session_to_idle()
                        break
                    if (not speech_started) or rms_raw < get_dynamic_speech_floor():
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
                # Low-RMS captures (often with max_timeout) must not contaminate
                # conversation history — Whisper hallucinates on hush/noise.
                speech_floor = get_dynamic_speech_floor()
                if rms_raw < speech_floor:
                    log(
                        "Conversation",
                        f"Discarding STT commit (reason={stop_reason}, "
                        f"rms_raw={rms_raw:.5f}) — below speech floor "
                        f"{speech_floor:.5f}"
                        f"{' / max_timeout' if stop_reason == 'max_timeout' else ''}"
                        "; not writing conversation history",
                    )
                    whisper_text = ""
                    set_subtitle("")
                    set_ui_state("listening")
                    follow_up = True
                    continue
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
                    log_conversation("Dana", reply)
                    emit_live_transcript("Dana", reply)
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
                    log_conversation("Dana", reply)
                    end_session_to_idle(reply)
                    break

                # Chat mode: never feed the task-queue / ReAct jail — lightweight chat only.
                if get_dana_mode() == "chat":
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
                log_conversation("Dana", reply)
                emit_live_transcript("Dana", reply)
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
                log_conversation("Dana", reply)
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
