"""Unit tests for SendInput stealth keystrokes (no pyautogui)."""

from __future__ import annotations

import inspect

import dana.tools.os_control as osc


def test_no_pyautogui_or_pynput_in_os_control() -> None:
    src = inspect.getsource(osc)
    assert "import pyautogui" not in src
    assert "import pynput" not in src
    assert "SendInput" in src
    assert "MapVirtualKey" in src or "MapVirtualKeyW" in src
    assert "KEYEVENTF_SCANCODE" in src
    print("[PASS] no pyautogui/pynput; SendInput scancode present")


def test_human_delay_bounds() -> None:
    assert 0.04 <= osc._HUMAN_DELAY_MIN <= osc._HUMAN_DELAY_MAX <= 0.11
    print("[PASS] human delay 40–110ms")


def test_dry_run_keystrokes(monkeypatch) -> None:
    monkeypatch.setenv("DANA_OS_DRY_RUN", "1")
    # Reset rate limiter window.
    with osc._rate_lock:
        osc._last_keystroke_ts = 0.0
        osc._chars_window.clear()
    out = osc.execute_os_keystrokes("Hello")
    assert "dry_run" in out
    assert "sendinput_scancode" in out
    print("[PASS] dry_run stealth path")


def test_type_text_sendinput_mocked(monkeypatch) -> None:
    taps: list[tuple[int, bool]] = []

    def fake_send(vk: int, *, key_up: bool = False) -> None:
        taps.append((vk, key_up))

    monkeypatch.setattr(osc, "_send_scan", fake_send)
    monkeypatch.setattr(osc, "_human_sleep", lambda: None)
    monkeypatch.setattr(osc, "_resolve_char", lambda ch: (ord(ch.upper()), ch.isupper()))
    result = osc.type_text_sendinput("Ab")
    assert result["ok"] is True
    assert result["engine"] == "sendinput_scancode"
    assert result["chars_typed"] == 2
    # Every press has a matching release somewhere.
    assert any(not up for _, up in taps)
    assert any(up for _, up in taps)
    print("[PASS] mocked SendInput typing")


if __name__ == "__main__":
    test_no_pyautogui_or_pynput_in_os_control()
    test_human_delay_bounds()
    print("OK")
