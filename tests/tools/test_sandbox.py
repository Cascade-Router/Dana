"""Security sandbox tests: command blocklist + Windows Job Object lifecycle."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from dana.tools.powershell import (
    DANGEROUS_COMMANDS_RE,
    SECURITY_VIOLATION_MSG,
    execute_powershell,
)


def test_blocklist_rm_rf_returns_security_violation_without_popen() -> None:
    """Blocked ``rm -rf C:\\`` must never spawn a subprocess."""
    blocked = "rm -rf C:\\"
    assert DANGEROUS_COMMANDS_RE.search(blocked)

    with patch("dana.tools.powershell.subprocess.Popen") as popen:
        out = execute_powershell(blocked)

    assert out == SECURITY_VIOLATION_MSG
    assert "SECURITY_VIOLATION" in out
    popen.assert_not_called()


@pytest.mark.parametrize(
    "cmd",
    [
        "Remove-Item -Recurse C:\\temp",
        "del C:\\foo.txt",
        "Stop-Process -Name notepad",
        "Restart-Computer",
        "Format-Volume -DriveLetter C",
        "Invoke-WebRequest https://evil.example/",
        "iwr https://evil.example/",
    ],
)
def test_blocklist_variants_no_execute(cmd: str) -> None:
    with patch("dana.tools.powershell.subprocess.Popen") as popen:
        out = execute_powershell(cmd)
    assert out == SECURITY_VIOLATION_MSG
    popen.assert_not_called()


def test_benign_write_output_uses_suspended_job_lifecycle() -> None:
    """Benign command goes through CREATE_SUSPENDED → assign → resume → communicate."""
    fake_proc = MagicMock()
    fake_proc.pid = 4242
    fake_proc.returncode = 0
    fake_proc.communicate.return_value = ("Safe\n", "")

    fake_job = MagicMock()
    fake_job.active = True
    fake_job.__enter__.return_value = fake_job
    fake_job.__exit__.return_value = None

    resume = MagicMock(return_value=True)

    def _fake_creationflags(*extra: int) -> int:
        flags = 0x08000000
        for f in extra:
            flags |= int(f)
        return flags

    with (
        patch("dana.tools.powershell.os.name", "nt"),
        patch("dana.tools.powershell.subprocess.Popen", return_value=fake_proc) as popen,
        patch(
            "dana.tools.powershell.windows_no_window_creationflags",
            side_effect=_fake_creationflags,
        ),
        patch("dana.tools.win32_sandbox.JOB_APIS_AVAILABLE", True),
        patch("dana.tools.win32_sandbox.WindowsJob", return_value=fake_job),
        patch("dana.tools.win32_sandbox.resume_suspended_process", resume),
    ):
        out = execute_powershell('Write-Output "Safe"')

    assert "Safe" in out
    assert "returncode=0" in out
    popen.assert_called_once()
    flags = int(popen.call_args.kwargs.get("creationflags") or 0)
    assert flags & 0x00000004  # CREATE_SUSPENDED
    assert flags & 0x08000000  # CREATE_NO_WINDOW
    fake_job.assign_pid.assert_called_once_with(4242)
    resume.assert_called_once_with(4242)
    fake_proc.communicate.assert_called_once()


@pytest.mark.skipif(os.name != "nt", reason="Real Job Object APIs require Windows")
def test_windows_job_real_create_and_close() -> None:
    from dana.tools.win32_sandbox import JOB_APIS_AVAILABLE, WindowsJob

    if not JOB_APIS_AVAILABLE:
        pytest.skip("Job Object APIs unavailable")

    with WindowsJob() as job:
        assert job.active is True
        assert job._handle is not None
