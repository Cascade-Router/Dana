"""Hermetic checks for the PowerShell CLI actuator."""

from __future__ import annotations

import os
import shutil
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from dana.tools.powershell import execute_powershell


def _powershell_available() -> bool:
    return shutil.which("powershell") is not None


@pytest.mark.skipif(
    os.name != "nt" or not _powershell_available(),
    reason="Real PowerShell actuator check requires Windows + powershell on PATH",
)
def test_execute_powershell_actuator_online_real() -> None:
    out = execute_powershell('Write-Output "Actuator Online"')
    assert "Actuator Online" in out
    assert "returncode=0" in out
    assert "stdout:" in out


def test_execute_powershell_hermetic_mock() -> None:
    """Offline CI path: mock subprocess so PowerShell need not be installed."""
    fake = MagicMock()
    fake.pid = 1234
    fake.returncode = 0
    fake.communicate.return_value = ("Actuator Online\n", "")

    # Keep hermetic: skip real Job Object / ResumeThread side effects on Windows.
    with (
        patch("dana.tools.powershell.subprocess.Popen", return_value=fake) as popen,
        patch("dana.tools.win32_sandbox.JOB_APIS_AVAILABLE", False),
    ):
        out = execute_powershell('Write-Output "Actuator Online"')

    assert "Actuator Online" in out
    assert "returncode=0" in out
    assert "stdout:" in out
    assert "stderr:\n(empty)" in out

    popen.assert_called_once()
    args, kwargs = popen.call_args
    assert args[0] == [
        "powershell",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        'Write-Output "Actuator Online"',
    ]
    assert kwargs.get("text") is True
    if os.name == "nt":
        assert int(kwargs.get("creationflags") or 0) & int(
            getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        )


def test_execute_powershell_empty_command() -> None:
    assert execute_powershell("").startswith("ERROR:")
    assert execute_powershell("   ").startswith("ERROR:")
