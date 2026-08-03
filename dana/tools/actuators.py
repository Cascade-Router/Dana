"""System actuators: write local files and run host shell/PowerShell commands."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from dana.tools.powershell import DANGEROUS_COMMANDS_RE, SECURITY_VIOLATION_MSG
from dana.vault_service import windows_no_window_creationflags

# CREATE_SUSPENDED — primary thread starts suspended until ResumeThread.
_CREATE_SUSPENDED = 0x00000004

DEFAULT_TIMEOUT_SEC = 15


def write_to_file(filepath: str, content: str) -> str:
    """Create parent dirs, write UTF-8 text, return absolute path + byte size."""
    try:
        from dana.ui.status_bus import emit_state_change

        emit_state_change("executing", tool="write_to_file")
    except Exception:  # noqa: BLE001
        pass

    raw = (filepath or "").strip()
    if not raw:
        return "ERROR: empty filepath"

    try:
        path = Path(raw).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = content if isinstance(content, str) else str(content or "")
        encoded = data.encode("utf-8")
        path.write_bytes(encoded)
        abs_path = str(path.resolve())
        return f"OK: wrote {abs_path} ({len(encoded)} bytes)"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: write_to_file failed: {exc}"


def execute_command(command: str, timeout: int = DEFAULT_TIMEOUT_SEC) -> str:
    """Run a host command; PowerShell on Windows with sandbox safety checks.

    Applies ``DANGEROUS_COMMANDS_RE`` before spawn. On Windows, uses
    ``CREATE_NO_WINDOW`` and, when Job APIs are available, ``WindowsJob``
    (CREATE_SUSPENDED → assign → resume) with a hard timeout.
    """
    try:
        from dana.ui.status_bus import emit_state_change

        emit_state_change("executing", tool="execute_command")
    except Exception:  # noqa: BLE001
        pass

    cmd = (command or "").strip()
    if not cmd:
        return "ERROR: empty command"

    if DANGEROUS_COMMANDS_RE.search(cmd):
        return SECURITY_VIOLATION_MSG

    try:
        timeout_sec = max(1, int(timeout))
    except (TypeError, ValueError):
        timeout_sec = DEFAULT_TIMEOUT_SEC

    try:
        if os.name == "nt":
            argv = [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                cmd,
            ]
            return _execute_windows(argv, timeout_sec)
        return _execute_run(["/bin/sh", "-c", cmd], timeout_sec, windows=False)
    except FileNotFoundError:
        return (
            "ERROR: execute_command failed: shell executable not found "
            "(host has no PowerShell/sh on PATH)."
        )
    except subprocess.TimeoutExpired:
        return f"ERROR: execute_command timed out after {timeout_sec}s"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: execute_command failed: {exc}"


def _format_observation(returncode: int, stdout: str, stderr: str) -> str:
    out = (stdout or "").rstrip()
    err = (stderr or "").rstrip()
    return (
        f"returncode={int(returncode)}\n"
        f"stdout:\n{out or '(empty)'}\n"
        f"stderr:\n{err or '(empty)'}"
    )


def _execute_windows(argv: list[str], timeout_sec: int) -> str:
    """Prefer WindowsJob sandbox; fall back to subprocess.run + CREATE_NO_WINDOW."""
    from dana.tools.win32_sandbox import (
        JOB_APIS_AVAILABLE,
        WindowsJob,
        resume_suspended_process,
    )

    if not JOB_APIS_AVAILABLE:
        return _execute_run(argv, timeout_sec, windows=True)

    creationflags = windows_no_window_creationflags(_CREATE_SUSPENDED)
    proc = subprocess.Popen(  # noqa: S603
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creationflags,
    )
    try:
        with WindowsJob() as job:
            if job.active:
                job.assign_pid(proc.pid)
            resume_suspended_process(proc.pid)
            stdout, stderr = proc.communicate(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass
        try:
            proc.communicate(timeout=5)
        except Exception:  # noqa: BLE001
            pass
        return f"ERROR: execute_command timed out after {timeout_sec}s"
    except Exception:
        try:
            resume_suspended_process(proc.pid)
        except Exception:  # noqa: BLE001
            pass
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass
        raise

    returncode = int(proc.returncode if proc.returncode is not None else 0)
    return _format_observation(returncode, stdout or "", stderr or "")


def _execute_run(argv: list[str], timeout_sec: int, *, windows: bool) -> str:
    """subprocess.run path (non-Windows or missing Job APIs)."""
    run_kwargs: dict[str, Any] = {
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": timeout_sec,
        "check": False,
    }
    if windows or os.name == "nt":
        run_kwargs["creationflags"] = windows_no_window_creationflags()

    completed = subprocess.run(argv, **run_kwargs)  # noqa: S603
    return _format_observation(
        int(completed.returncode),
        completed.stdout or "",
        completed.stderr or "",
    )
