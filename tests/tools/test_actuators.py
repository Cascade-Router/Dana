"""Hermetic checks for write_to_file / execute_command system actuators."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dana.tools.actuators import execute_command, write_to_file
from dana.tools.powershell import SECURITY_VIOLATION_MSG


def _powershell_available() -> bool:
    return shutil.which("powershell") is not None


def test_write_to_file_creates_parents_and_reports_bytes(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "dir" / "hello.txt"
    content = "actuator write ok"
    with patch("dana.ui.status_bus.emit_state_change") as emit:
        out = write_to_file(str(target), content)

    assert target.is_file()
    assert target.read_text(encoding="utf-8") == content
    abs_path = str(target.resolve())
    assert abs_path in out
    assert f"{len(content.encode('utf-8'))} bytes" in out
    assert out.startswith("OK:")
    emit.assert_called()
    assert emit.call_args.kwargs.get("tool") == "write_to_file" or (
        len(emit.call_args.args) >= 1
    )


@pytest.mark.skipif(
    os.name != "nt" or not _powershell_available(),
    reason="Real execute_command check requires Windows + powershell on PATH",
)
def test_execute_command_write_output_real() -> None:
    with patch("dana.ui.status_bus.emit_state_change"):
        out = execute_command('Write-Output "Actuator Online"')
    assert "Actuator Online" in out
    assert "returncode=0" in out
    assert "stdout:" in out


def test_execute_command_hermetic_echo_mock() -> None:
    """Offline CI path: mock Popen/run so PowerShell need not be installed."""
    if os.name == "nt":
        fake = MagicMock()
        fake.pid = 4321
        fake.returncode = 0
        fake.communicate.return_value = ("Actuator Online\n", "")

        with (
            patch("dana.tools.actuators.subprocess.Popen", return_value=fake) as popen,
            patch("dana.tools.win32_sandbox.JOB_APIS_AVAILABLE", True),
            patch("dana.tools.win32_sandbox.WindowsJob") as job_cls,
            patch("dana.tools.win32_sandbox.resume_suspended_process"),
            patch("dana.ui.status_bus.emit_state_change") as emit,
        ):
            job = MagicMock()
            job.active = True
            job_cls.return_value.__enter__.return_value = job
            job_cls.return_value.__exit__.return_value = None
            out = execute_command('Write-Output "Actuator Online"')

        assert "Actuator Online" in out
        assert "returncode=0" in out
        assert "stdout:" in out
        assert "stderr:\n(empty)" in out
        emit.assert_called()
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
        assert int(kwargs.get("creationflags") or 0) & int(
            getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        )
        return

    completed = MagicMock()
    completed.returncode = 0
    completed.stdout = "Actuator Online\n"
    completed.stderr = ""
    with (
        patch("dana.tools.actuators.subprocess.run", return_value=completed) as run,
        patch("dana.ui.status_bus.emit_state_change") as emit,
    ):
        out = execute_command('echo Actuator Online')

    assert "Actuator Online" in out
    assert "returncode=0" in out
    emit.assert_called()
    run.assert_called_once()



def test_execute_command_blocks_dangerous() -> None:
    with patch("dana.ui.status_bus.emit_state_change"):
        out = execute_command("Remove-Item -Recurse C:\\temp")
    assert out == SECURITY_VIOLATION_MSG


def test_execute_command_empty() -> None:
    with patch("dana.ui.status_bus.emit_state_change"):
        assert execute_command("").startswith("ERROR:")
        assert execute_command("   ").startswith("ERROR:")


def test_write_to_file_empty_path() -> None:
    with patch("dana.ui.status_bus.emit_state_change"):
        assert write_to_file("", "x").startswith("ERROR:")
