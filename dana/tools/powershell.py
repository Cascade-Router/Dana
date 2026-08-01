"""PowerShell CLI actuator for the LangGraph ReAct agent (Windows OS context)."""

from __future__ import annotations

import os
import re
import subprocess
from typing import Any

# Import at module level so vault_service's subprocess.Popen patch runs once
# before tests (or callers) replace Popen — not mid-execution.
from dana.vault_service import windows_no_window_creationflags

# Destructive / network exfil patterns — matched before any subprocess spawn.
DANGEROUS_COMMANDS_RE = re.compile(
    r"(?ix)"
    r"(?:"
    r"\b(?:rm|del|erase|rmdir|rd)\b"
    r"|Remove-Item\b"
    r"|Stop-Process\b"
    r"|\bkill\b"
    r"|Restart-Computer\b"
    r"|Format-Volume\b"
    r"|Format-Disk\b"
    r"|Invoke-WebRequest\b"
    r"|\biwr\b"
    r"|Invoke-RestMethod\b"
    r"|\birm\b"
    r")"
)

SECURITY_VIOLATION_MSG = (
    "SECURITY_VIOLATION: Command blocked by Dānā sandbox policy."
)

# CREATE_SUSPENDED — primary thread starts suspended until ResumeThread.
_CREATE_SUSPENDED = 0x00000004


def execute_powershell(command: str) -> str:
    """Run a PowerShell command and return structured stdout/stderr/returncode.

    Windows OS context for the ReAct agent
    --------------------------------------
    This actuator targets the host PowerShell CLI (``powershell.exe`` on Windows).
    Prefer it when the user asks for PowerShell-native work: registry, WMI/CIM,
    Get-Process / Get-Service, environment variables, Object Pipeline filters,
    or any script that must use PowerShell syntax rather than ``cmd.exe``.

    Command formatting rules (agent self-correction)
    ------------------------------------------------
    - Pass **PowerShell script text only** in ``command`` — do **not** wrap it in
      ``powershell ... -Command`` again; the actuator already invokes
      ``powershell -NoProfile -NonInteractive -Command <command>``.
    - Prefer single-line or semicolon-joined statements for non-interactive runs
      (e.g. ``Get-Process | Select-Object -First 5 | ConvertTo-Json -Compress``).
    - Quote strings with single quotes inside PowerShell when possible so the
      outer JSON/tool-arg layer does not mangle escapes:
      ``Write-Output 'Actuator Online'``.
    - Avoid interactive prompts (``Read-Host``, ``pause``, ``Out-GridView``).
      Use ``-Force`` / ``-Confirm:$false`` only when the action is intentional
      and safe; never use this tool for destructive wipe commands.
    - On failure, read ``returncode``, ``stderr``, and ``stdout`` in the returned
      block, fix the script, and retry — do not invent success from an empty body.

    Security
    --------
    Commands matching ``DANGEROUS_COMMANDS_RE`` are rejected with
    ``SECURITY_VIOLATION`` and never executed. On Windows, allowed commands are
    started with ``CREATE_SUSPENDED``, assigned to a Job Object
    (``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``), then resumed.

    Returns
    -------
    A multi-line observation string::

        returncode=<int>
        stdout:
        <captured stdout or (empty)>
        stderr:
        <captured stderr or (empty)>
    """
    cmd = (command or "").strip()
    if not cmd:
        return "ERROR: empty command"

    if DANGEROUS_COMMANDS_RE.search(cmd):
        return SECURITY_VIOLATION_MSG

    argv = [
        "powershell",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        cmd,
    ]

    try:
        if os.name == "nt":
            return _execute_windows_sandboxed(argv)
        return _execute_plain(argv)
    except FileNotFoundError:
        return (
            "ERROR: execute_powershell failed: powershell executable not found "
            "(host has no PowerShell on PATH)."
        )
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: execute_powershell failed: {exc}"


def _format_observation(returncode: int, stdout: str, stderr: str) -> str:
    out = (stdout or "").rstrip()
    err = (stderr or "").rstrip()
    return (
        f"returncode={int(returncode)}\n"
        f"stdout:\n{out or '(empty)'}\n"
        f"stderr:\n{err or '(empty)'}"
    )


def _execute_windows_sandboxed(argv: list[str]) -> str:
    """Popen with CREATE_SUSPENDED + WindowsJob; plain Popen if job APIs missing."""
    from dana.tools.win32_sandbox import (
        JOB_APIS_AVAILABLE,
        WindowsJob,
        resume_suspended_process,
    )

    if not JOB_APIS_AVAILABLE:
        return _execute_plain(argv, windows=True)

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
            stdout, stderr = proc.communicate()
    except Exception:
        # Ensure a suspended process cannot be left hanging.
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


def _execute_plain(argv: list[str], *, windows: bool = False) -> str:
    """Execute without Job Object (non-Windows or missing job APIs)."""
    popen_kwargs: dict[str, Any] = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
    }
    if windows or os.name == "nt":
        popen_kwargs["creationflags"] = windows_no_window_creationflags()

    proc = subprocess.Popen(argv, **popen_kwargs)  # noqa: S603
    stdout, stderr = proc.communicate()
    returncode = int(proc.returncode if proc.returncode is not None else 0)
    return _format_observation(returncode, stdout or "", stderr or "")
