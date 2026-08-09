"""Smoke checks for dana.logging (runtime vs conversation isolation)."""

from __future__ import annotations

import os
from pathlib import Path

from dana.logging import (
    CONVERSATION_LOG_PATH,
    RUNTIME_LOG_PATH,
    debug_logging_enabled,
    enable_runtime_file_logging,
    log,
    log_conversation,
    log_debug,
    log_exception,
    reset_conversation_log,
)
import dana.logging as dana_logging


def test_conversation_log_clears_on_reset() -> None:
    path = Path(reset_conversation_log())
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "Dana conversation session" in text
    assert "system noise excluded" in text.lower() or "Latest User" in text

    log_conversation("User", "Hello Dana")
    log_conversation("Dana", "Hi there.")
    body = path.read_text(encoding="utf-8")
    assert "User: Hello Dana" in body
    assert "Dana: Hi there." in body

    # New run clears previous turns.
    reset_conversation_log()
    cleared = path.read_text(encoding="utf-8")
    assert "Hello Dana" not in cleared
    assert "Dana conversation session" in cleared
    print("[PASS] conversation log clears on reset")


def test_log_debug_silent_by_default(monkeypatch) -> None:
    monkeypatch.delenv("DANA_DEBUG", raising=False)
    assert debug_logging_enabled() is False
    # Should not raise; silent when debug off.
    log_debug("Tracker", "Alive - should be invisible in normal mode")
    log("Main", "essential heartbeat")
    print("[PASS] log_debug silent without DANA_DEBUG")


def test_enable_runtime_resets_conversation() -> None:
    log_conversation("User", "stale turn before enable")
    enable_runtime_file_logging()
    body = Path(CONVERSATION_LOG_PATH).read_text(encoding="utf-8")
    assert "stale turn before enable" not in body
    assert os.path.isfile(RUNTIME_LOG_PATH)
    print("[PASS] enable_runtime_file_logging clears conversation log")


def test_runtime_log_keeps_last_100_lines(monkeypatch, tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    runtime_path = log_dir / "dana_runtime.log"
    monkeypatch.setattr(dana_logging, "RUNTIME_LOG_DIR", str(log_dir))
    monkeypatch.setattr(dana_logging, "RUNTIME_LOG_PATH", str(runtime_path))
    max_lines = dana_logging.RUNTIME_LOG_MAX_LINES

    for i in range(max_lines + 5):
        dana_logging.append_runtime_log(f"line {i}\n")

    lines = runtime_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == max_lines
    assert lines[0] == f"line {5}"
    assert lines[-1] == f"line {max_lines + 4}"

    dana_logging.append_runtime_log(f"line {max_lines + 5}\n")
    lines = runtime_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == max_lines
    assert lines[0] == "line 6"
    assert lines[-1] == f"line {max_lines + 5}"
    print(f"[PASS] runtime log keeps last {max_lines} lines")


def test_log_exception_writes_stack_trace(monkeypatch, tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    runtime_path = log_dir / "dana_runtime.log"
    monkeypatch.setattr(dana_logging, "RUNTIME_LOG_DIR", str(log_dir))
    monkeypatch.setattr(dana_logging, "RUNTIME_LOG_PATH", str(runtime_path))

    try:
        raise RuntimeError("synthetic TTS Engine Failure")
    except RuntimeError as exc:
        dana_logging.log_exception("Audio", "TTS Engine Failure", exc=exc)

    body = runtime_path.read_text(encoding="utf-8")
    assert "TTS Engine Failure" in body
    assert "RuntimeError: synthetic TTS Engine Failure" in body
    assert "Traceback" in body
    print("[PASS] log_exception writes stack trace")


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
