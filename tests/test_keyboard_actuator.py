"""Pytest coverage for the foundational keyboard actuator (dry-run, no real hardware input).

Clears ``DANA_OS_DRY_RUN`` from the ambient environment before each test:
at least one other test module in this suite (``tests/test_e2e_lifecycle.py``)
sets it at import time via a raw ``os.environ[...] = ...`` (not
``monkeypatch``), which leaks into every test that runs afterward in the
same pytest process. Without this, tests here that expect non-dry-run
behavior by default silently get dry-run instead when the full suite runs
in an order where that file is collected first.
"""
from __future__ import annotations

import dana.tools.keyboard_actuator as keyboard_actuator
from dana.tools import rate_limiter
from dana.tools.keyboard_actuator import KeyboardActuator, execute_shortcut, type_text

import pytest


@pytest.fixture(autouse=True)
def _reset_rate_limiter(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DANA_OS_DRY_RUN", raising=False)
    rate_limiter.reset()
    yield
    rate_limiter.reset()


def _stub_actuator():
    calls: list[str] = []

    def type_fn(text):
        calls.append(text)
        return {"ok": True, "chars_typed": len(text)}

    return KeyboardActuator(type_fn=type_fn), calls


def test_type_text_calls_backend_with_exact_text() -> None:
    actuator, calls = _stub_actuator()
    result = actuator.type_text("hello world")
    assert result == {"ok": True, "chars_typed": 11, "dry_run": False}
    assert calls == ["hello world"]


def test_type_text_rejects_empty_text_without_calling_backend() -> None:
    actuator, calls = _stub_actuator()
    result = actuator.type_text("   ")
    assert result == {"ok": False, "error": "empty text"}
    assert calls == []


def test_type_text_rejects_oversized_text_without_calling_backend() -> None:
    actuator, calls = _stub_actuator()
    result = actuator.type_text("x" * 2001)
    assert result["ok"] is False
    assert "too long" in result["error"]
    assert calls == []


def test_type_text_rate_limits_rapid_successive_calls() -> None:
    actuator, calls = _stub_actuator()
    first = actuator.type_text("hello")
    second = actuator.type_text("world")
    assert first["ok"] is True
    assert second["ok"] is False
    assert "rate_limited" in second["error"]
    assert calls == ["hello"]


def test_type_text_surfaces_backend_failure() -> None:
    def failing_type_fn(text):
        return {"ok": False, "error": "SendInput failed (sent=0)"}

    actuator = KeyboardActuator(type_fn=failing_type_fn)
    result = actuator.type_text("hello")
    assert result["ok"] is False
    assert result["error"] == "SendInput failed (sent=0)"


def test_type_text_dry_run_validates_but_never_calls_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DANA_OS_DRY_RUN", "1")
    actuator, calls = _stub_actuator()
    result = actuator.type_text("hello world")
    assert result == {"ok": True, "chars_typed": 11, "dry_run": True}
    assert calls == []


def test_type_text_dry_run_still_enforces_empty_text_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DANA_OS_DRY_RUN", "1")
    actuator, calls = _stub_actuator()
    result = actuator.type_text("")
    assert result["ok"] is False


def test_type_text_respects_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    import dana.middleware.kill_switch as kill_switch

    monkeypatch.setattr(kill_switch, "halt_if_requested", lambda: True)
    actuator, calls = _stub_actuator()
    result = actuator.type_text("hello world")
    assert result["ok"] is False
    assert result["halted"] is True
    assert calls == []


def test_module_level_type_text_uses_real_os_control_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Dry-run end-to-end: no dependency injection, no real hardware call.
    monkeypatch.setenv("DANA_OS_DRY_RUN", "1")
    result = type_text("hello world")
    assert result == {"ok": True, "chars_typed": 11, "dry_run": True}


# --------------------------------------------------------------------------
# execute_shortcut
# --------------------------------------------------------------------------


def _stub_combo_actuator():
    calls: list[list[str]] = []

    def combo_fn(keys):
        calls.append(keys)

    return KeyboardActuator(combo_fn=combo_fn), calls


def test_execute_shortcut_sends_parsed_keys_to_backend() -> None:
    actuator, calls = _stub_combo_actuator()
    result = actuator.execute_shortcut("ctrl+c")
    assert result == {"ok": True, "keys": ["ctrl", "c"], "dry_run": False}
    assert calls == [["ctrl", "c"]]


def test_execute_shortcut_is_case_and_whitespace_insensitive() -> None:
    actuator, calls = _stub_combo_actuator()
    result = actuator.execute_shortcut("  CTRL + SHIFT + N  ")
    assert result["ok"] is True
    assert result["keys"] == ["ctrl", "shift", "n"]
    assert calls == [["ctrl", "shift", "n"]]


def test_execute_shortcut_rejects_empty_combo_without_calling_backend() -> None:
    actuator, calls = _stub_combo_actuator()
    result = actuator.execute_shortcut("   ")
    assert result["ok"] is False
    assert "non-empty" in result["error"]
    assert calls == []


def test_execute_shortcut_rejects_unresolvable_key_without_calling_backend() -> None:
    actuator, calls = _stub_combo_actuator()
    result = actuator.execute_shortcut("ctrl+not_a_real_key")
    assert result["ok"] is False
    assert "unsupported key" in result["error"]
    assert calls == []


def test_execute_shortcut_rejects_too_many_keys_without_calling_backend() -> None:
    actuator, calls = _stub_combo_actuator()
    result = actuator.execute_shortcut("ctrl+alt+shift+win+a")
    assert result["ok"] is False
    assert "too many keys" in result["error"]
    assert calls == []


def test_execute_shortcut_rate_limits_rapid_successive_calls() -> None:
    actuator, calls = _stub_combo_actuator()
    first = actuator.execute_shortcut("ctrl+c")
    second = actuator.execute_shortcut("ctrl+v")
    assert first["ok"] is True
    assert second["ok"] is False
    assert "rate_limited" in second["error"]
    assert calls == [["ctrl", "c"]]


def test_execute_shortcut_and_type_text_share_the_module_wide_rate_limiter() -> None:
    type_actuator, type_calls = _stub_actuator()
    combo_actuator, combo_calls = _stub_combo_actuator()
    first = type_actuator.type_text("hello")
    second = combo_actuator.execute_shortcut("ctrl+c")
    assert first["ok"] is True
    assert second["ok"] is False
    assert "rate_limited" in second["error"]
    assert combo_calls == []


def test_execute_shortcut_surfaces_backend_failure() -> None:
    def failing_combo_fn(keys):
        raise OSError("SendInput failed (sent=0)")

    actuator = KeyboardActuator(combo_fn=failing_combo_fn)
    result = actuator.execute_shortcut("ctrl+c")
    assert result["ok"] is False
    assert "SendInput failed" in result["error"]


def test_execute_shortcut_dry_run_validates_but_never_calls_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DANA_OS_DRY_RUN", "1")
    actuator, calls = _stub_combo_actuator()
    result = actuator.execute_shortcut("ctrl+c")
    assert result == {"ok": True, "keys": ["ctrl", "c"], "dry_run": True}
    assert calls == []


def test_execute_shortcut_dry_run_still_rejects_unresolvable_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DANA_OS_DRY_RUN", "1")
    actuator, calls = _stub_combo_actuator()
    result = actuator.execute_shortcut("ctrl+not_a_real_key")
    assert result["ok"] is False
    assert calls == []


def test_execute_shortcut_respects_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    import dana.middleware.kill_switch as kill_switch

    monkeypatch.setattr(kill_switch, "halt_if_requested", lambda: True)
    actuator, calls = _stub_combo_actuator()
    result = actuator.execute_shortcut("ctrl+c")
    assert result["ok"] is False
    assert result["halted"] is True
    assert calls == []


def test_module_level_execute_shortcut_uses_real_os_control_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Dry-run end-to-end: no dependency injection, no real hardware call.
    monkeypatch.setenv("DANA_OS_DRY_RUN", "1")
    result = execute_shortcut("ctrl+c")
    assert result == {"ok": True, "keys": ["ctrl", "c"], "dry_run": True}


# --------------------------------------------------------------------------
# os_control.resolve_key_name / press_key_combo (pure logic, no hardware)
# --------------------------------------------------------------------------


def test_resolve_key_name_recognizes_modifiers_and_named_keys() -> None:
    from dana.tools.os_control import (
        VK_CONTROL,
        VK_LWIN,
        VK_MENU,
        resolve_key_name,
    )

    assert resolve_key_name("ctrl") == VK_CONTROL
    assert resolve_key_name("  Alt ") == VK_MENU
    assert resolve_key_name("WIN") == VK_LWIN


def test_resolve_key_name_recognizes_letters_and_digits() -> None:
    from dana.tools.os_control import resolve_key_name

    assert resolve_key_name("c") == ord("C")
    assert resolve_key_name("N") == ord("N")
    assert resolve_key_name("9") == ord("9")


def test_resolve_key_name_recognizes_function_keys() -> None:
    from dana.tools.os_control import VK_F1, resolve_key_name

    assert resolve_key_name("f1") == VK_F1
    assert resolve_key_name("F24") == VK_F1 + 23
    assert resolve_key_name("f25") is None


def test_resolve_key_name_rejects_unknown_names() -> None:
    from dana.tools.os_control import resolve_key_name

    assert resolve_key_name("not_a_real_key") is None
    assert resolve_key_name("") is None
    assert resolve_key_name("  ") is None


def test_press_key_combo_rejects_empty_list_without_pressing_anything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dana.tools.os_control as os_control

    sent: list[tuple[int, bool]] = []
    monkeypatch.setattr(
        os_control, "_send_scan", lambda vk, key_up=False: sent.append((vk, key_up))
    )

    with pytest.raises(ValueError, match="at least one key"):
        os_control.press_key_combo([])
    assert sent == []


def test_press_key_combo_validates_all_keys_before_pressing_any(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dana.tools.os_control as os_control

    sent: list[tuple[int, bool]] = []
    monkeypatch.setattr(
        os_control, "_send_scan", lambda vk, key_up=False: sent.append((vk, key_up))
    )
    monkeypatch.setattr(os_control, "_human_sleep", lambda: None)

    with pytest.raises(ValueError, match="unsupported key"):
        os_control.press_key_combo(["ctrl", "not_a_real_key"])
    assert sent == []


def test_press_key_combo_presses_down_in_order_and_releases_in_reverse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dana.tools.os_control as os_control

    sent: list[tuple[int, bool]] = []
    monkeypatch.setattr(
        os_control, "_send_scan", lambda vk, key_up=False: sent.append((vk, key_up))
    )
    monkeypatch.setattr(os_control, "_human_sleep", lambda: None)

    os_control.press_key_combo(["ctrl", "shift", "c"])

    vk_ctrl = os_control.resolve_key_name("ctrl")
    vk_shift = os_control.resolve_key_name("shift")
    vk_c = os_control.resolve_key_name("c")
    assert sent == [
        (vk_ctrl, False),
        (vk_shift, False),
        (vk_c, False),
        (vk_c, True),
        (vk_shift, True),
        (vk_ctrl, True),
    ]
