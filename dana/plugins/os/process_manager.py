"""Time-boxed, sandboxed process execution — Dana's "os_tools" capability
domain (see dana.core.react_dispatch's _OS_TOOLS_TOOL_IDS/is_mutating_tool).
Two tools live here: ``run_python_script`` (a fixed, non-shell argv list —
see its own docstring) and ``execute_terminal_command`` (Generalized
Terminal Execution — genuinely arbitrary shell commands, see ITS docstring
for why that's a materially different, higher-risk contract).

``run_python_script`` runs a ``.py`` file that must already exist inside
the sandbox (AGENT_WORKSPACE_DIR) — validated via the SAME
``resolve_sandboxed_path`` that file_system.py's list_directory/read_file/
write_file already use, not a reimplementation — as a subprocess of the
CURRENT interpreter (``sys.executable``), never a shell and never an
arbitrary string command. A strict timeout prevents an LLM-generated
infinite loop from hanging the ReAct loop forever; a non-zero exit code or
a timeout both come back as ``{"ok": False, "error": ...}`` carrying
stdout/stderr, so the model can read the traceback and self-correct rather
than the call ever raising an uncaught exception.

Naming note: dana/tools/tools.json already has a DIFFERENT, live
"execute_python_script" tool (dana/tools/actuators.py, jailed to
EXECUTION_JAIL_DIR, with its own timeout/background-job contract, used by
dana.core.agent_loop's separate broker). This module's tool is named
run_python_script instead, to avoid silently overwriting that shared
schema entry — see dana/plugins/web/research.py's docstring for the same
naming-collision reasoning applied to search_web/read_webpage.

Scope note: this validates and gates WHICH script file can be launched
(inside the sandbox, only ``.py``, only after explicit HITL approval —
this tool declares no "read_only": true, so is_mutating_tool's fail-closed
schema check gates it) and bounds HOW LONG it can run. It does not additionally
confine what the running script itself can access once started (no
container/seccomp/AppContainer-style runtime jail) — that's a materially
larger problem than "validate the launch, time-box it, gate it on human
approval," which is what was asked for here.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any

from dana.plugins.os.file_system import PathEscapeError, resolve_sandboxed_path

_DEFAULT_TIMEOUT_S = 10.0
_TERMINAL_DEFAULT_TIMEOUT_S = 15.0
_MAX_OUTPUT_CHARS = 8_000


def _truncate(text: str) -> str:
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text
    return text[:_MAX_OUTPUT_CHARS] + "\n…[truncated]"


def run_python_script(
    script_path: str,
    args: list[str] | None = None,
    *,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Executes ``script_path`` (must resolve to an existing ``.py`` file
    inside the sandbox) via ``[sys.executable, script_path, *args]``.

    ``timeout_s`` is a keyword-only override for tests — the LLM-facing
    tool schema (tools.json) and dana.core.react_dispatch's adapter never
    expose it, so the ~10s ceiling is not something a model call can raise.
    """
    try:
        target = resolve_sandboxed_path(script_path)
    except PathEscapeError as exc:
        return {"ok": False, "error": str(exc)}

    if target.suffix.lower() != ".py":
        return {"ok": False, "error": f"only .py files may be executed, got: {script_path!r}"}
    if not target.exists():
        return {"ok": False, "error": f"script does not exist: {script_path!r}"}
    if not target.is_file():
        return {"ok": False, "error": f"path is not a file: {script_path!r}"}
    if not sys.executable:
        return {"ok": False, "error": "no Python interpreter available (sys.executable is empty)"}

    safe_args = [str(a) for a in args] if isinstance(args, list) else []
    command = [sys.executable, str(target), *safe_args]
    cwd = resolve_sandboxed_path(".")

    try:
        completed = subprocess.run(  # noqa: S603 — argv list, shell=False, fixed interpreter + sandboxed path
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _truncate(exc.stdout) if isinstance(exc.stdout, str) else ""
        stderr = _truncate(exc.stderr) if isinstance(exc.stderr, str) else ""
        return {
            "ok": False,
            "error": f"script timed out after {timeout_s}s",
            "stdout": stdout,
            "stderr": stderr,
            "timed_out": True,
        }
    except OSError as exc:
        return {"ok": False, "error": f"could not execute script: {exc}"}

    stdout = _truncate(completed.stdout or "")
    stderr = _truncate(completed.stderr or "")
    if completed.returncode != 0:
        return {
            "ok": False,
            "error": stderr.strip() or f"script exited with code {completed.returncode}",
            "stdout": stdout,
            "stderr": stderr,
            "returncode": completed.returncode,
        }
    return {"ok": True, "stdout": stdout, "stderr": stderr, "returncode": 0}


def execute_terminal_command(
    command: str,
    working_dir: str = ".",
    allowed_mounts: list[str] | None = None,
    *,
    timeout_s: float = _TERMINAL_DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Generalized Terminal Execution: runs an arbitrary shell ``command``
    (a project's own tests/linters/build steps — ``npm test``,
    ``cargo check``, ``git diff``, ...) with its cwd confined to the
    sandbox or an explicitly-mounted directory, via the SAME
    ``resolve_sandboxed_path`` every other os_tools operation uses.

    Materially different (and higher-risk) than ``run_python_script``
    above: that tool builds a fixed, non-shell argv list around a single
    validated ``.py`` path, so it can never be anything other than "run
    THIS Python file." There is no equivalent way to generalize to "run
    whatever command a mounted project's own tooling actually uses on the
    command line" without accepting a raw string interpreted by a real
    shell (``shell=True``) — pipes, shell builtins, and multi-word
    commands (``npm test``, ``cargo check``) all require it. That is
    exactly why this tool declares no "read_only": true in tools.json, so
    ``dana.core.react_dispatch.is_mutating_tool``'s fail-closed schema
    check gates it: the sandboxed cwd and the timeout below bound the
    blast radius, but neither is the actual safety boundary — the HITL
    approval click showing the user the EXACT command string before it
    ever runs is. This tool must never be reachable with no human
    reviewing the string first.

    ``working_dir`` defaults to the sandbox root ('.'); pass an absolute
    path under a registered mount to run a command from inside that
    mounted directory instead. ``timeout_s`` is a keyword-only override
    for tests — the LLM-facing tool schema (tools.json) and
    dana.core.react_dispatch's adapter never expose it, so the ~15s
    ceiling is not something a model call can raise.
    """
    try:
        cwd = resolve_sandboxed_path(working_dir, allowed_mounts)
    except PathEscapeError as exc:
        return {"ok": False, "error": str(exc)}
    if not cwd.exists():
        return {"ok": False, "error": f"working_dir does not exist: {working_dir!r}"}
    if not cwd.is_dir():
        return {"ok": False, "error": f"working_dir is not a directory: {working_dir!r}"}

    cmd = (command or "").strip()
    if not cmd:
        return {"ok": False, "error": "command must not be empty"}

    try:
        completed = subprocess.run(  # noqa: S602 — shell=True IS the point (see docstring): HITL-gated, sandboxed cwd, time-boxed
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            shell=True,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _truncate(exc.stdout) if isinstance(exc.stdout, str) else ""
        stderr = _truncate(exc.stderr) if isinstance(exc.stderr, str) else ""
        return {
            "ok": False,
            "error": f"command timed out after {timeout_s}s",
            "stdout": stdout,
            "stderr": stderr,
            "timed_out": True,
        }
    except OSError as exc:
        return {"ok": False, "error": f"could not execute command: {exc}"}

    stdout = _truncate(completed.stdout or "")
    stderr = _truncate(completed.stderr or "")
    if completed.returncode != 0:
        return {
            "ok": False,
            "error": stderr.strip() or f"command exited with code {completed.returncode}",
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": completed.returncode,
        }
    return {"ok": True, "stdout": stdout, "stderr": stderr, "exit_code": 0}


__all__ = ("run_python_script", "execute_terminal_command")
