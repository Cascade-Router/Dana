"""Background Process Management — Dana's "os_tools" capability domain
(see dana.core.react_dispatch's _OS_TOOLS_TOOL_IDS/is_mutating_tool).

Complements dana.plugins.os.process_manager's ``execute_terminal_command``
(synchronous, ~15s-timeout, made for tests/linters/build steps) with the
OTHER half of "run a project command": a genuinely long-running,
NON-BLOCKING service (a dev server, a file watcher, ...) that
``execute_terminal_command`` would simply time out — and fail the whole
ReAct turn — trying to wait on, orphaning the half-started server and
permanently locking whatever port it bound.

``start_background_service`` spawns the command, redirects its stdout/
stderr to a log file under ``AGENT_WORKSPACE_DIR/data/logs/{alias}.log``,
and returns IMMEDIATELY — the log is readable any time afterward via the
agent's ordinary ``read_file`` tool, no separate log-streaming machinery
needed. ``stop_background_service`` kills the ENTIRE process TREE it
started (not just its own immediate PID). ``list_background_services``
reports which aliases are still actually running.

Every spawned process is placed in its own process group — ``os.setsid``
on POSIX, ``CREATE_NEW_PROCESS_GROUP`` on Windows — SPECIFICALLY so
``stop_background_service`` can kill the whole tree later. A dev-server
command (``npm run dev``, ``python -m uvicorn``, ...) run via
``shell=True`` is very often a thin wrapper process around the actual
long-running server; killing only the wrapper's own PID would leave the
real server (and its bound port) running forever — exactly the "orphaned
grandchild, permanently locked port" failure mode this module exists to
prevent.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
from pathlib import Path
from typing import Any

from dana.plugins.os.file_system import PathEscapeError, resolve_sandboxed_path

# A plain module global, not a function-default value — tests read/mutate
# this directly (see tests/plugins/os/test_background_services.py) to
# assert on what's currently tracked, and to force-clean any process a test
# itself spawned. Maps a caller-chosen ``alias`` to the live ``Popen``
# handle start_background_service created for it.
_ACTIVE_PROCESSES: dict[str, subprocess.Popen] = {}

# aliases are used both as a dict key AND as a log FILENAME component
# (see _log_path below) — restricted to a safe, unambiguous charset so
# neither of those uses needs its own separate validation, and so a
# malicious/malformed alias can never be read as a path segment (no "/" or
# "\\" is ever permitted, so there is no traversal to even attempt).
_ALIAS_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")

# Wall-clock ceiling for stop_background_service's own kill+reap sequence —
# this is "how long do we wait for the OS to confirm the tree is actually
# gone," not a timeout on the SERVICE's own runtime (which has none by
# design; that's the whole point of a background service).
_STOP_TIMEOUT_S = 10.0


def _log_path(alias: str) -> Path:
    """Sandbox-relative log path for ``alias``, resolved through the SAME
    ``resolve_sandboxed_path`` every other os_tools operation uses —
    ``allowed_mounts`` is deliberately never threaded in here: a service's
    log always lives under the agent's OWN sandbox, never inside a mounted
    external directory, regardless of where the service's ``working_dir``
    itself points.
    """
    return resolve_sandboxed_path(f"data/logs/{alias}.log")


def start_background_service(
    command: str,
    alias: str,
    working_dir: str = ".",
    allowed_mounts: list[str] | None = None,
) -> dict[str, Any]:
    """Spawns ``command`` as a non-blocking background process — returns
    as soon as the process exists, never waits for it to finish (it isn't
    expected to). ``working_dir`` is validated exactly like
    ``execute_terminal_command``'s own cwd, via ``resolve_sandboxed_path``.

    MUTATING — gated by ``dana.core.react_dispatch.is_mutating_tool``, so the
    ReAct loop suspends for explicit user approval (reviewing the exact
    command string, same as ``execute_terminal_command``) before this ever
    actually runs.
    """
    cmd = (command or "").strip()
    if not cmd:
        return {"ok": False, "error": "command must not be empty"}

    alias = (alias or "").strip()
    if not alias:
        return {"ok": False, "error": "alias must not be empty"}
    if not _ALIAS_PATTERN.fullmatch(alias):
        return {"ok": False, "error": f"alias must contain only letters, digits, '_', '-', or '.': {alias!r}"}

    existing = _ACTIVE_PROCESSES.get(alias)
    if existing is not None and existing.poll() is None:
        return {"ok": False, "error": f"a service aliased {alias!r} is already running — stop it first"}

    try:
        cwd = resolve_sandboxed_path(working_dir, allowed_mounts)
    except PathEscapeError as exc:
        return {"ok": False, "error": str(exc)}
    if not cwd.exists():
        return {"ok": False, "error": f"working_dir does not exist: {working_dir!r}"}
    if not cwd.is_dir():
        return {"ok": False, "error": f"working_dir is not a directory: {working_dir!r}"}

    log_path = _log_path(alias)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return {"ok": False, "error": f"could not create log directory: {exc}"}

    # Its own process GROUP, not just its own process — see this module's
    # docstring for why: without this, stop_background_service's tree-kill
    # below has no group to target, and killing only this Popen's own PID
    # would leave a dev-server grandchild (spawned by a shell wrapper)
    # running forever.
    popen_kwargs: dict[str, Any] = {}
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["preexec_fn"] = os.setsid  # noqa: PLW1509 — single-threaded parent, no fork-safety hazard here

    try:
        with open(log_path, "wb") as log_handle:
            process = subprocess.Popen(  # noqa: S602 — shell=True is intentional, same as execute_terminal_command; HITL-gated
                cmd,
                cwd=str(cwd),
                stdout=log_handle,
                stderr=log_handle,
                shell=True,
                **popen_kwargs,
            )
    except OSError as exc:
        return {"ok": False, "error": f"could not start service: {exc}"}

    _ACTIVE_PROCESSES[alias] = process
    return {
        "ok": True,
        "alias": alias,
        "pid": process.pid,
        "log_path": f"data/logs/{alias}.log",
    }


def stop_background_service(alias: str) -> dict[str, Any]:
    """Kills the ENTIRE process tree started by ``start_background_service``
    under ``alias`` — not just the immediate child, which for a dev-server
    command is very often a thin wrapper around the actual long-running
    server. Uses the OS's process-GROUP kill primitive for exactly that
    reason: ``taskkill /T /F`` (Windows) or ``os.killpg`` (POSIX) reach
    every descendant the group's own launch (``start_background_service``'s
    ``CREATE_NEW_PROCESS_GROUP``/``os.setsid``) put there.

    ``taskkill /T /F`` is used rather than sending ``CTRL_BREAK_EVENT`` on
    Windows: a Ctrl+Break is a signal the target process tree can ignore
    or fail to handle (many console apps started detached from a real
    console never see it at all), whereas ``taskkill /T /F`` unconditionally
    terminates the whole tree — the guarantee this tool exists to provide.

    MUTATING — gated by ``dana.core.react_dispatch.is_mutating_tool``, same
    bar as starting one: an agent silently killing a service the user may
    still be relying on is exactly as consequential as starting one.
    """
    alias = (alias or "").strip()
    if not alias:
        return {"ok": False, "error": "alias must not be empty"}

    process = _ACTIVE_PROCESSES.get(alias)
    if process is None:
        return {"ok": False, "error": f"no active service aliased {alias!r}"}

    if process.poll() is not None:
        # Already exited on its own (crashed, or ran to completion) —
        # clean up bookkeeping and report it, not an error.
        _ACTIVE_PROCESSES.pop(alias, None)
        return {"ok": True, "alias": alias, "already_stopped": True}

    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(process.pid)],
                capture_output=True,
                text=True,
                timeout=_STOP_TIMEOUT_S,
            )
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        process.wait(timeout=_STOP_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": f"could not stop service {alias!r}: {exc}"}
    finally:
        _ACTIVE_PROCESSES.pop(alias, None)

    return {"ok": True, "alias": alias, "already_stopped": False}


def list_background_services() -> dict[str, Any]:
    """Every alias ever started this process's lifetime (server restart
    clears it — this is in-memory bookkeeping, not persisted), with its
    CURRENT liveness re-checked via ``Popen.poll()`` right now rather than
    trusting whatever was true the moment it was started. Read-only: never
    mutates ``_ACTIVE_PROCESSES`` (an already-exited entry is only ever
    cleaned up by ``stop_background_service``, or the next
    ``start_background_service`` reusing the same alias).
    """
    services = [
        {"alias": alias, "pid": process.pid, "running": process.poll() is None}
        for alias, process in _ACTIVE_PROCESSES.items()
    ]
    return {"ok": True, "services": services}


__all__ = ("start_background_service", "stop_background_service", "list_background_services")
