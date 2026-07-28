"""Tests for Donna OS automation: run_terminal_command success / error / timeout."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from dana.os_automation import TERMINAL_COMMAND_TIMEOUT_SEC, run_terminal_command


def test_run_terminal_command_success_echo() -> None:
    """Success: echo returns stdout containing 'hello world'."""
    out = run_terminal_command("echo 'hello world'")
    # cmd.exe may include quotes in stdout; still must contain the message.
    assert "hello world" in out
    assert not out.upper().startswith("ERROR:")
    print(f"[PASS] Success: {out!r}")


def test_run_terminal_command_error_missing_directory() -> None:
    """Error catching: missing path must not crash; return stderr-backed ERROR string."""
    # Spec command. On Windows, cmd.exe has no `ls`, so use PowerShell's ls alias
    # so stderr reflects a missing directory rather than "command not recognized".
    command = "ls /directory_that_does_not_exist"
    if os.name == "nt":
        command = (
            'powershell -NoProfile -Command '
            '"ls /directory_that_does_not_exist"'
        )
    out = run_terminal_command(command)
    assert isinstance(out, str)
    assert out.upper().startswith("ERROR:")
    blob = out.lower()
    assert any(
        marker in blob
        for marker in (
            "no such file",
            "cannot find",
            "not found",
            "does not exist",
            "cannot find path",
            "cannot find the path",
            "cannot find the file",
            "directory_that_does_not_exist",
        )
    ), f"expected missing-directory stderr, got: {out!r}"
    print(f"[PASS] Error Catching: {out[:200]!r}")


def test_run_terminal_command_timeout_protection() -> None:
    """Timeout protection: TimeoutExpired → graceful string, no crash."""
    import subprocess

    with patch("dana.os_automation.subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired(
            cmd="sleep 999",
            timeout=TERMINAL_COMMAND_TIMEOUT_SEC,
            output=None,
            stderr=b"hung process",
        )
        out = run_terminal_command("sleep 999")

    assert isinstance(out, str)
    assert out.upper().startswith("ERROR:")
    assert "timed out" in out.lower() or "timeout" in out.lower()
    assert str(TERMINAL_COMMAND_TIMEOUT_SEC) in out
    print(f"[PASS] Timeout Protection: {out!r}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
