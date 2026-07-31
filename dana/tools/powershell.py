"""PowerShell CLI actuator for the LangGraph ReAct agent (Windows OS context)."""

from __future__ import annotations

import os
import subprocess
from typing import Any


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

    run_kwargs: dict[str, Any] = {
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
    }
    if os.name == "nt":
        from dana.vault_service import windows_no_window_creationflags

        run_kwargs["creationflags"] = windows_no_window_creationflags()

    try:
        completed = subprocess.run(  # noqa: S603
            [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                cmd,
            ],
            **run_kwargs,
        )
    except FileNotFoundError:
        return (
            "ERROR: execute_powershell failed: powershell executable not found "
            "(host has no PowerShell on PATH)."
        )
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: execute_powershell failed: {exc}"

    stdout = (completed.stdout or "").rstrip()
    stderr = (completed.stderr or "").rstrip()
    returncode = int(completed.returncode if completed.returncode is not None else 0)
    return (
        f"returncode={returncode}\n"
        f"stdout:\n{stdout or '(empty)'}\n"
        f"stderr:\n{stderr or '(empty)'}"
    )
