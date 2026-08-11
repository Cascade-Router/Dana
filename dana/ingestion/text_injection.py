"""Text-injection state + the ``.trigger_ask`` / ``input.txt`` ingest thread.

Extracted verbatim from ``dana.core_agent`` (Phase 7 of the core_agent.py
decomposition; see docs/architecture/phase7_core_agent_decomposition.md).
``input_txt_ingest_worker`` is one of the four daemon threads
``dana.core.app_runtime.agent_loop()`` spawns; the rest of this module is
the injected-question state it and the Dashboard silent-chat bar
(``dana.ui.app_gui.DanaGUI.submit_text_command``) share, letting typed or
file-dropped text skip the mic/Whisper path entirely.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import ingest  # scripts/ingest.py -> task_queue.json converter

from dana.agentic import get_dana_mode, set_dana_mode
from dana.core.shared_state import (
    TRIGGER_FILE,
    get_ui_state,
    injected_question,
    injected_question_lock,
    is_recording,
    stop_event,
    vad_abort_event,
    vad_capture_active,
    _injected_already_logged,
    _injected_source,
)
from dana.logging import log
from dana.paths import TEXT_INJECTION_PATH

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
