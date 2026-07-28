"""Build LangChain tools from ``tools.json``, dispatching through Donna's ToolCall IR."""

from __future__ import annotations

import itertools
import os
import re
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Optional

from langchain_core.tools import StructuredTool, tool
from pydantic import Field, create_model

from dana.paths import PROJECT_ROOT, EXECUTION_JAIL_LIBRARY_DIR
from dana.tools.schema import ToolCall, ToolSpec, load_tool_registry

_WATCHDOG_TOOL_DESCRIPTION = (
    "USE THIS TOOL IMMEDIATELY if the user asks you to 'run a background task', "
    "'write a script', 'monitor the system', 'run a watchdog', 'activate the Titan "
    "initiative', or 'start the Titan Protocol'. This tool dispatches a background "
    "LangGraph swarm to write and execute Python code safely. Do not attempt to "
    "write the Python code yourself in the chat—you MUST call this tool to handle it. "
    "NEVER confuse Titan Protocol with a .json file — it is a spoken codename, not "
    "read_local_file."
)
_KILL_WATCHDOG_DESCRIPTION = (
    "Stop a background Watchdog by ID when the user asks to cancel, "
    "stop monitoring, or kill a watchdog."
)
_SAVE_SCRIPT_DESCRIPTION = (
    "Save a useful Watchdog or helper Python script into the sandboxed "
    "library at CAMGRASPER/execution_jail/library/ (never outside that folder)."
)

_REPO_ROOT = PROJECT_ROOT
_SANDBOX_LIBRARY = EXECUTION_JAIL_LIBRARY_DIR.resolve()
_SAFE_SCRIPT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,80}$")

# Global registry of live Watchdog jobs (thread + cancel handles + task text).
active_watchdogs: dict[str, dict[str, Any]] = {}
_watchdog_lock = threading.Lock()
_watchdog_id_seq = itertools.count(1)


def _new_watchdog_id() -> str:
    return str(next(_watchdog_id_seq))


def list_active_watchdog_summaries() -> list[tuple[str, str]]:
    """Return ``(task_id, task_description)`` for every live Watchdog."""
    with _watchdog_lock:
        rows = [
            (tid, str(entry.get("task") or "").strip() or "(no description)")
            for tid, entry in active_watchdogs.items()
        ]
    return rows


def format_active_watchdogs_xml() -> str:
    """Compact ``<active_watchdogs>`` block for recency-bias injection."""
    rows = list_active_watchdog_summaries()
    if not rows:
        return ""
    lines = [f"{tid}: {desc}" for tid, desc in rows]
    return "<active_watchdogs>\n" + "\n".join(lines) + "\n</active_watchdogs>"


def _register_watchdog_process(
    task_id: str, process: Any | None
) -> None:
    with _watchdog_lock:
        entry = active_watchdogs.get(task_id)
        if entry is not None:
            entry["process"] = process


def _emit_tts(
    tts_callback: Callable[[str], None] | None,
    text: str,
) -> None:
    """Speak via injected callback only (no core_agent import)."""
    if tts_callback is None:
        return
    phrase = (text or "").strip()
    if not phrase:
        return
    try:
        tts_callback(phrase)
    except Exception:  # noqa: BLE001
        pass


def _watchdog_worker(
    task_id: str,
    task: str,
    tts_callback: Callable[[str], None] | None = None,
) -> None:
    """Background: compile + invoke the Donna↔Titan Watchdog StateGraph."""
    with _watchdog_lock:
        entry = active_watchdogs.get(task_id)
        stop = entry.get("stop") if entry else None
    if stop is None:
        stop = threading.Event()

    def _on_process(proc: Any | None) -> None:
        _register_watchdog_process(task_id, proc)

    try:
        if stop.is_set():
            return
        from dana.swarm.watchdog_graph import build_watchdog_graph

        app = build_watchdog_graph(
            stop_event=stop,
            on_process=_on_process,
        )
        result = app.invoke(
            {
                "task": task,
                "code": "",
                "feedback": "",
                "lint_errors": "",
                "status": "pending",
                "revisions": 0,
                "history": [],
            },
        )
        if stop.is_set():
            return
        status = str((result or {}).get("status") or "")
        feedback = str((result or {}).get("feedback") or "")
        # Alerts during monitoring already TTS via repl_executor. Speak failures here.
        status_l = status.lower()
        if (
            status_l == "error"
            or status_l.startswith("error")
            or status.upper().startswith("REJECTED")
        ):
            try:
                from dana.logging import log

                tip = feedback.strip()[:160] or status
                log("Watchdog", f"Watchdog ended with failure: {tip}")
                # Avoid double-speaking the terminal_failure phrase.
                if "had to abort" not in tip.lower():
                    _emit_tts(
                        tts_callback,
                        f"Watchdog could not stay active: {tip}",
                    )
            except Exception:
                pass
    except Exception as exc:  # noqa: BLE001
        if not stop.is_set():
            try:
                from dana.logging import log_exception

                log_exception("Watchdog", "Watchdog deployment failed", exc=exc)
            except Exception:
                pass
            _emit_tts(tts_callback, f"Watchdog deployment failed: {exc}")
    finally:
        with _watchdog_lock:
            active_watchdogs.pop(task_id, None)


def dispatch_watchdog_impl(
    task: str,
    *,
    tts_callback: Callable[[str], None] | None = None,
    vault_client: Any | None = None,
) -> str:
    """Fire-and-forget Watchdog graph on a daemon thread (ReAct-safe)."""
    _ = vault_client  # reserved for callers that inject VaultClient via DI
    q = (task or "").strip()
    if not q:
        return "ERROR: missing task"

    # Pre-flight: sandbox must be writable before we spawn the coder/REPL thread.
    try:
        from dana.swarm.watchdog_graph import preflight_watchdog_write

        preflight_watchdog_write(require_code=False)
    except RuntimeError as exc:
        try:
            from dana.logging import log_exception

            log_exception(
                "Watchdog",
                "dispatch_watchdog preflight failed — refusing deploy",
                exc=exc,
            )
        except Exception:
            pass
        return f"ERROR: Watchdog preflight failed: {exc}"

    task_id = _new_watchdog_id()
    stop = threading.Event()
    thread = threading.Thread(
        target=_watchdog_worker,
        args=(task_id, q, tts_callback),
        name=f"DonnaWatchdog-{task_id}",
        daemon=True,
    )
    with _watchdog_lock:
        active_watchdogs[task_id] = {
            "thread": thread,
            "task": q,
            "stop": stop,
            "process": None,
        }
    thread.start()
    return f"OK: Watchdog deployed with ID: {task_id}"


def kill_watchdog_impl(task_id: str) -> str:
    """Signal stop + terminate the Watchdog subprocess/thread if present."""
    tid = str(task_id or "").strip()
    if not tid:
        return "ERROR: missing task_id"

    with _watchdog_lock:
        entry = active_watchdogs.get(tid)
        if entry is None:
            return f"ERROR: no active Watchdog with ID: {tid}"
        stop = entry.get("stop")
        proc = entry.get("process")
        thread = entry.get("thread")

    if stop is not None:
        stop.set()

    if proc is not None:
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=1.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    # Give the worker a moment to exit and self-remove from the registry.
    if isinstance(thread, threading.Thread) and thread.is_alive():
        thread.join(timeout=2.0)

    with _watchdog_lock:
        still = active_watchdogs.pop(tid, None)
        if still is not None and isinstance(still.get("thread"), threading.Thread):
            # Thread still hung (e.g. blocked in LLM); mark cancelled + drop visibility.
            pass

    return f"OK: Watchdog {tid} stopped."


_JASON_SUPERVISOR_TOOL_DESCRIPTION = (
    "Jason/Titan conversational supervisor. Call when the user addresses Jason/Titan "
    "to read/evaluate a prior Donna file (notes.txt) and have Donna write a Python "
    "script (e.g. scaling_metrics.py). Runs a linear Jason→Donna LangGraph handoff."
)


def dispatch_jason_supervisor_impl(query: str) -> str:
    """Synchronous Jason→Donna supervisor graph (ReAct-safe; no background thread)."""
    from dana.swarm.jason_supervisor_graph import dispatch_jason_supervisor_impl as _run

    return _run(query)


@tool(
    "dispatch_jason_supervisor",
    description=_JASON_SUPERVISOR_TOOL_DESCRIPTION,
)
def dispatch_jason_supervisor(query: str) -> str:
    """Jason reads/evaluates a file then hands off to Donna to write a Python script."""
    return dispatch_jason_supervisor_impl(query)


@tool(
    "dispatch_watchdog",
    description=_WATCHDOG_TOOL_DESCRIPTION,
)
def dispatch_watchdog(task: str) -> str:
    """USE THIS TOOL IMMEDIATELY if the user asks you to 'run a background task', 'write a script', 'monitor the system', 'run a watchdog', 'activate the Titan initiative', or 'start the Titan Protocol'. This tool dispatches a background LangGraph swarm to write and execute Python code safely. Do not attempt to write the Python code yourself in the chat—you MUST call this tool to handle it. NEVER confuse Titan Protocol with a .json file."""
    return dispatch_watchdog_impl(task)


@tool(
    "kill_watchdog",
    description=_KILL_WATCHDOG_DESCRIPTION,
)
def kill_watchdog(task_id: str) -> str:
    """Stop a running background Watchdog by its task ID."""
    return kill_watchdog_impl(task_id)


def save_script_to_library_impl(script_name: str, code: str) -> str:
    """Write ``code`` under ``CAMGRASPER/execution_jail/library/`` only (path-jail)."""
    name = (script_name or "").strip().replace("\\", "/").split("/")[-1]
    if name.lower().endswith(".py"):
        name = name[:-3]
    if not _SAFE_SCRIPT_NAME_RE.match(name):
        return (
            "ERROR: invalid script_name — use a plain basename "
            "(letters, digits, _ or -)"
        )
    body = (code or "").strip()
    if not body:
        return "ERROR: missing code"

    try:
        from donna_jason_loop.jason_critic import static_code_safety_reject

        blocked = static_code_safety_reject(body)
        if blocked:
            return f"ERROR: refused unsafe code: {blocked}"
    except Exception:
        pass

    from dana.paths import DONNA_WORKSPACE

    _SANDBOX_LIBRARY.mkdir(parents=True, exist_ok=True)
    target = (_SANDBOX_LIBRARY / f"{name}.py").resolve()
    try:
        target.relative_to(_SANDBOX_LIBRARY)
    except ValueError:
        return "ERROR: path escapes CAMGRASPER/execution_jail/library"
    if not str(target).startswith(str(_SANDBOX_LIBRARY)):
        return "ERROR: path escapes CAMGRASPER/execution_jail/library"

    target.write_text(body + ("\n" if not body.endswith("\n") else ""), encoding="utf-8")
    try:
        rel = os.path.relpath(str(target), str(DONNA_WORKSPACE)).replace("\\", "/")
    except ValueError:
        rel = str(target)
    return f"OK: saved script to CAMGRASPER/{rel}"


@tool(
    "save_script_to_library",
    description=_SAVE_SCRIPT_DESCRIPTION,
)
def save_script_to_library(script_name: str, code: str) -> str:
    """Save useful code strictly to CAMGRASPER/execution_jail/library/."""
    return save_script_to_library_impl(script_name, code)


def _pydantic_args_model(spec: ToolSpec):
    """Dynamic args schema so bind_tools exposes enums / required fields."""
    fields: dict[str, Any] = {}
    for param in spec.parameters:
        desc = param.description_en or param.name
        if param.enum:
            # Literal["a", "b"] — unpack concrete enum strings (3.11+).
            lit_vals = tuple(str(x) for x in param.enum)
            ann: Any = Literal[*lit_vals]  # type: ignore[misc, valid-type]
        else:
            ann = str
        if param.required:
            fields[param.name] = (ann, Field(description=desc))
        else:
            fields[param.name] = (
                Optional[ann],
                Field(default=None, description=desc),
            )
    if not fields:
        # No-arg tools still need a model for schema consistency.
        return create_model(f"{spec.id}Args")  # type: ignore[call-arg]
    return create_model(f"{spec.id}Args", **fields)  # type: ignore[call-arg]


def _make_structured_tool(
    spec: ToolSpec,
    execute_fn: Callable[[ToolCall], str],
) -> StructuredTool:
    tool_id = spec.id
    description = (spec.description_en or tool_id).strip()
    args_model = _pydantic_args_model(spec)

    def _run(**kwargs: Any) -> str:
        cleaned = {k: v for k, v in kwargs.items() if v is not None}
        call = ToolCall(
            tool_id=tool_id,
            arguments=cleaned,
            source_lang="en",
            raw_text=f"langchain:{tool_id}",
            confidence=1.0,
        )
        try:
            return str(execute_fn(call))
        except Exception as exc:  # noqa: BLE001
            return f"ERROR: tool {tool_id} failed: {exc}"

    _run.__name__ = tool_id
    _run.__doc__ = description

    return StructuredTool.from_function(
        func=_run,
        name=tool_id,
        description=description,
        args_schema=args_model,
    )


_NATIVE_TOOL_IDS = frozenset(
    {"dispatch_watchdog", "kill_watchdog", "save_script_to_library"}
)
# Removed from LLM bind_tools per architectural lockdown (vision/clipboard air-gap).
_UNBOUND_TOOL_IDS = frozenset({"describe_spatial_scene", "read_clipboard_context"})


def _make_dispatch_watchdog_tool(
    tts_callback: Callable[[str], None] | None,
    vault_client: Any | None,
) -> StructuredTool:
    """LangChain tool closed over injected TTS / vault (no core_agent import)."""

    def _run(task: str) -> str:
        return dispatch_watchdog_impl(
            task,
            tts_callback=tts_callback,
            vault_client=vault_client,
        )

    _run.__name__ = "dispatch_watchdog"
    _run.__doc__ = _WATCHDOG_TOOL_DESCRIPTION
    return StructuredTool.from_function(
        func=_run,
        name="dispatch_watchdog",
        description=_WATCHDOG_TOOL_DESCRIPTION,
    )


def build_langchain_tools(
    execute_fn: Callable[[ToolCall], str],
    *,
    registry: dict[str, ToolSpec] | None = None,
    tool_ids: set[str] | frozenset[str] | None = None,
    include_natives: bool = True,
    tts_callback: Callable[[str], None] | None = None,
    vault_client: Any | None = None,
) -> list[Any]:
    """Convert ``tools.json`` entries + native tools (e.g. Watchdog) for bind_tools.

    When ``tool_ids`` is set, only those registry tools are bound (Semantic RAG
    top-K injection). Native Watchdog helpers are still appended unless
    ``include_natives`` is False.
    """
    reg = registry if registry is not None else load_tool_registry()
    tools: list[Any] = []
    for spec in reg.values():
        if spec.id in _NATIVE_TOOL_IDS or spec.id in _UNBOUND_TOOL_IDS:
            continue
        if tool_ids is not None and spec.id not in tool_ids:
            continue
        tools.append(_make_structured_tool(spec, execute_fn))
    if include_natives:
        if tool_ids is None or "dispatch_watchdog" in tool_ids:
            tools.append(
                _make_dispatch_watchdog_tool(tts_callback, vault_client)
            )
        if tool_ids is None or "kill_watchdog" in tool_ids:
            tools.append(kill_watchdog)
        if tool_ids is None or "save_script_to_library" in tool_ids:
            tools.append(save_script_to_library)
    return tools
