"""Tests for open_application + read_local_file OS tools."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from dana.os_automation import (
    MAX_LOCAL_FILE_CHARS,
    TRUNCATION_SUFFIX,
    open_application,
    read_local_file,
)
from dana.prompts.spatial_synthesis import REACT_PROTOCOL


def test_os_automation_rules_mention_new_tools() -> None:
    assert "## OS Automation Rules" in REACT_PROTOCOL
    assert "## Silent context" in REACT_PROTOCOL
    assert "<visual_context>" in REACT_PROTOCOL
    assert "<memory>" in REACT_PROTOCOL
    # Obsolete prompt-bleed / JSON Initiative rules removed for native tools.
    assert "## Action forcing" not in REACT_PROTOCOL
    assert "JSON Initiative" not in REACT_PROTOCOL
    assert "## Native tool calling" not in REACT_PROTOCOL
    assert "JSON tool calling" not in REACT_PROTOCOL
    assert "open_application" in REACT_PROTOCOL
    assert "Do NOT use `run_terminal_command` to open UI apps" in REACT_PROTOCOL
    assert "read_local_file" in REACT_PROTOCOL
    assert 'User: "Open Notepad"' in REACT_PROTOCOL
    notepad_idx = REACT_PROTOCOL.index('User: "Open Notepad"')
    next_bullet = REACT_PROTOCOL.find("\n- User:", notepad_idx + 1)
    notepad_block = REACT_PROTOCOL[notepad_idx:next_bullet if next_bullet > 0 else None]
    assert "open_application" in notepad_block
    assert "FINAL:" not in notepad_block
    print("[PASS] OS Automation Rules updated for open_application / read_local_file")


def test_open_application_chrome_mocked_popen() -> None:
    with patch("dana.os_automation.subprocess.Popen") as mock_popen:
        result = open_application("chrome")
        mock_popen.assert_called_once()
    assert result == "OK: Launched chrome."
    print(f"[PASS] open_application chrome: {result}")


def test_open_application_unknown() -> None:
    result = open_application("not_a_real_donna_app_xyz")
    assert result.startswith("ERROR:")
    print(f"[PASS] open_application unknown: {result}")


def test_read_local_file_truncates() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "big.txt"
        path.write_text("x" * (MAX_LOCAL_FILE_CHARS + 500), encoding="utf-8")
        out = read_local_file(str(path))
        assert TRUNCATION_SUFFIX in out
        assert len(out) >= MAX_LOCAL_FILE_CHARS
    print("[PASS] read_local_file truncation")


if __name__ == "__main__":
    test_os_automation_rules_mention_new_tools()
    test_open_application_chrome_mocked_popen()
    test_open_application_unknown()
    test_read_local_file_truncates()
    print("OK")
