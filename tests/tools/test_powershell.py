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
    fake.returncode = 0
    fake.stdout = "Actuator Online\n"
    fake.stderr = ""

    with patch("dana.tools.powershell.subprocess.run", return_value=fake) as run:
        out = execute_powershell('Write-Output "Actuator Online"')

    assert "Actuator Online" in out
    assert "returncode=0" in out
    assert "stdout:" in out
    assert "stderr:\n(empty)" in out

    run.assert_called_once()
    args, kwargs = run.call_args
    assert args[0] == [
        "powershell",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        'Write-Output "Actuator Online"',
    ]
    assert kwargs.get("capture_output") is True
    assert kwargs.get("text") is True
    if os.name == "nt":
        assert int(kwargs.get("creationflags") or 0) & int(
            getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        )


def test_execute_powershell_empty_command() -> None:
    assert execute_powershell("").startswith("ERROR:")
    assert execute_powershell("   ").startswith("ERROR:")
