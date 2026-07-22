"""Unit checks for hardened system_repl actuators."""

from __future__ import annotations

import time

from donna.tools.system_repl import file_editor, python_repl, shell_execute


def test_file_editor_blocks_windows_hosts() -> None:
    out = file_editor("read", r"C:\Windows\System32\drivers\etc\hosts")
    assert out.startswith("ERROR:"), out
    assert "path traversal blocked" in out or "outside project root" in out


def test_file_editor_blocks_parent_traversal() -> None:
    out = file_editor("read", "../../Windows/System32/drivers/etc/hosts")
    assert out.startswith("ERROR:"), out


def test_shell_execute_timeout() -> None:
    # Windows hang command; 15s hard kill.
    t0 = time.monotonic()
    out = shell_execute("ping -t 8.8.8.8")
    elapsed = time.monotonic() - t0
    assert "timed out" in out.lower(), out
    assert 14.0 <= elapsed < 25.0, f"elapsed={elapsed}"


def test_python_repl_runs_subprocess() -> None:
    out = python_repl("print('donna-sandbox-ok')")
    assert "donna-sandbox-ok" in out
    assert "exit_code=0" in out


def test_output_truncation() -> None:
    out = python_repl("print('X' * 5000)")
    assert len(out) < 2500
    assert "truncated" in out.lower() or out.count("X") <= 2000
