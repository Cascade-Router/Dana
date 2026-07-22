"""Hardened local shell / file / Python REPL actuators for the LangGraph agent.

Safeguards:
  - Hard 15s subprocess timeout (never hang the ReAct loop).
  - Aggressive 2000-char truncation on all returned text (context-window guard).
  - Workspace jail via ``Path.is_relative_to(PROJECT_ROOT)`` (no path traversal).
  - Agent Python never runs via ``exec()`` in-process — always ``python.exe`` subprocess.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from donna.paths import PROJECT_ROOT

TIMEOUT_SEC = 15
MAX_OUTPUT_CHARS = 2000
SANDBOX_SCRIPT_NAME = ".donna_sandbox.py"

_ROOT = Path(PROJECT_ROOT).resolve()
_SANDBOX_PATH = _ROOT / SANDBOX_SCRIPT_NAME


def _truncate_tail(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    raw = text if isinstance(text, str) else str(text or "")
    if len(raw) <= limit:
        return raw
    return f"...[truncated to last {limit} chars]\n{raw[-limit:]}"


def _truncate_file_body(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """Keep first+last 1000 chars when over limit (else full text ≤2000)."""
    raw = text if isinstance(text, str) else str(text or "")
    if len(raw) <= limit:
        return raw
    half = limit // 2
    return f"{raw[:half]}\n...[truncated]...\n{raw[-half:]}"


def _resolve_jailed(filepath: str) -> Path:
    """Resolve ``filepath`` under PROJECT_ROOT; raise ValueError on escape."""
    raw = Path(str(filepath or "").strip()).expanduser()
    if not str(filepath or "").strip():
        raise ValueError("empty filepath")
    if raw.is_absolute():
        candidate = raw.resolve()
    else:
        candidate = (_ROOT / raw).resolve()
    if not candidate.is_relative_to(_ROOT):
        raise ValueError(
            f"path traversal blocked: {filepath!r} is outside project root {_ROOT}"
        )
    return candidate


def _popen_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "cwd": str(_ROOT),
    }
    if os.name == "nt":
        kwargs["creationflags"] = int(
            getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        )
    return kwargs


def _kill_process_tree(pid: int) -> None:
    """Force-kill pid and children (Windows shell=True leaves orphaned grandchildren)."""
    if pid <= 0:
        return
    if os.name == "nt":
        try:
            subprocess.run(  # noqa: S603
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
                timeout=5,
                check=False,
            )
        except Exception:  # noqa: BLE001
            pass
        return
    try:
        import signal

        os.killpg(pid, signal.SIGKILL)
    except Exception:  # noqa: BLE001
        try:
            os.kill(pid, 9)
        except Exception:  # noqa: BLE001
            pass


def _run_shell(command: str) -> tuple[int, str, str] | str:
    """``subprocess.run``-equivalent with hard timeout + process-tree kill.

    Returns ``(returncode, stdout, stderr)`` or a timeout warning string.
    """
    popen_kwargs = _popen_kwargs()
    # New process group on POSIX so timeout can signal the whole tree.
    if os.name != "nt":
        popen_kwargs["start_new_session"] = True

    try:
        proc = subprocess.Popen(  # noqa: S602
            command,
            shell=True,
            **popen_kwargs,
        )
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: shell_execute failed: {exc}"

    try:
        stdout, stderr = proc.communicate(timeout=TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        _kill_process_tree(proc.pid)
        try:
            stdout, stderr = proc.communicate(timeout=3)
        except Exception:  # noqa: BLE001
            stdout, stderr = "", ""
        partial = ""
        if isinstance(stdout, str) and stdout.strip():
            partial = f"\npartial_stdout:\n{_truncate_tail(stdout)}"
        if isinstance(stderr, str) and stderr.strip():
            partial += f"\npartial_stderr:\n{_truncate_tail(stderr)}"
        return (
            f"WARNING: command timed out after {TIMEOUT_SEC} seconds and was killed."
            f"{partial}"
        )
    except Exception as exc:  # noqa: BLE001
        _kill_process_tree(proc.pid)
        return f"ERROR: shell_execute failed: {exc}"

    return int(proc.returncode or 0), stdout or "", stderr or ""


def shell_execute(command: str) -> str:
    """Run a shell command at project root with a hard 15s timeout."""
    cmd = (command or "").strip()
    if not cmd:
        return "ERROR: empty command"

    result = _run_shell(cmd)
    if isinstance(result, str):
        return result

    returncode, stdout_raw, stderr_raw = result
    stdout = _truncate_tail(stdout_raw.rstrip())
    stderr = _truncate_tail(stderr_raw.rstrip())
    parts = [
        f"exit_code={returncode}",
        f"stdout:\n{stdout or '(empty)'}",
    ]
    if stderr:
        parts.append(f"stderr:\n{stderr}")
    return "\n".join(parts)

def file_editor(action: str, filepath: str, content: str | None = None) -> str:
    """Read/write files strictly inside PROJECT_ROOT (blocks ../ and abs escapes)."""
    act = (action or "").strip().lower()
    if act not in {"read", "write"}:
        return "ERROR: action must be 'read' or 'write'"

    try:
        target = _resolve_jailed(filepath)
    except ValueError as exc:
        return f"ERROR: {exc}"

    if act == "read":
        if not target.is_file():
            return f"ERROR: file not found: {target.relative_to(_ROOT).as_posix()}"
        try:
            body = target.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            return f"ERROR: read failed: {exc}"
        return (
            f"OK: read {target.relative_to(_ROOT).as_posix()} "
            f"({len(body)} chars)\n{_truncate_file_body(body)}"
        )

    # write
    if content is None:
        return "ERROR: content is required for write"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(content), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: write failed: {exc}"
    return (
        f"OK: wrote {len(str(content))} chars to "
        f"{target.relative_to(_ROOT).as_posix()}"
    )


def python_repl(code: str) -> str:
    """Execute agent Python in a separate interpreter subprocess (never in-process exec)."""
    src = code if isinstance(code, str) else str(code or "")
    if not src.strip():
        return "ERROR: empty code"

    try:
        _SANDBOX_PATH.write_text(src, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: failed to write sandbox script: {exc}"

    popen_kwargs = _popen_kwargs()
    if os.name != "nt":
        popen_kwargs["start_new_session"] = True

    try:
        proc = subprocess.Popen(  # noqa: S603
            [sys.executable, str(_SANDBOX_PATH)],
            shell=False,
            **popen_kwargs,
        )
    except Exception as exc:  # noqa: BLE001
        try:
            if _SANDBOX_PATH.exists():
                _SANDBOX_PATH.unlink()
        except OSError:
            pass
        return f"ERROR: python_repl failed: {exc}"

    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_process_tree(proc.pid)
        try:
            stdout, stderr = proc.communicate(timeout=3)
        except Exception:  # noqa: BLE001
            stdout, stderr = "", ""
    except Exception as exc:  # noqa: BLE001
        _kill_process_tree(proc.pid)
        try:
            if _SANDBOX_PATH.exists():
                _SANDBOX_PATH.unlink()
        except OSError:
            pass
        return f"ERROR: python_repl failed: {exc}"
    finally:
        try:
            if _SANDBOX_PATH.exists():
                _SANDBOX_PATH.unlink()
        except OSError:
            pass

    if timed_out:
        partial = ""
        if isinstance(stdout, str) and stdout.strip():
            partial = f"\npartial_stdout:\n{_truncate_tail(stdout)}"
        if isinstance(stderr, str) and stderr.strip():
            partial += f"\npartial_stderr:\n{_truncate_tail(stderr)}"
        return (
            f"WARNING: python_repl timed out after {TIMEOUT_SEC} seconds and was killed."
            f"{partial}"
        )

    out = _truncate_tail((stdout or "").rstrip())
    err = _truncate_tail((stderr or "").rstrip())
    parts = [
        f"exit_code={int(proc.returncode or 0)}",
        f"stdout:\n{out or '(empty)'}",
    ]
    if err:
        parts.append(f"stderr:\n{err}")
    return "\n".join(parts)
