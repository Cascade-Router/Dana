"""Pytest coverage for the foundational mouse actuator (dry-run, no real hardware input).

Every test injects ``move_fn``/``click_fn``/``screen_size_fn`` stubs so the
geometry + safety pipeline runs with no real SendInput calls, and resets the
module-wide rate limiter between tests so cases don't interfere with each other.
"""
from __future__ import annotations

import dana.tools.mouse_actuator as mouse_actuator
from dana.tools.mouse_actuator import MouseActuator, click_target_bbox

import pytest


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    mouse_actuator._last_actuation_ts = 0.0
    yield
    mouse_actuator._last_actuation_ts = 0.0


def _stub_actuator(screen_size=(1920, 1080)):
    calls: dict[str, list] = {"move": [], "click": []}

    def move_fn(x, y):
        calls["move"].append((x, y))

    def click_fn():
        calls["click"].append(True)

    actuator = MouseActuator(
        move_fn=move_fn,
        click_fn=click_fn,
        screen_size_fn=lambda: screen_size,
    )
    return actuator, calls


def test_click_bbox_moves_to_inset_centroid_and_clicks() -> None:
    actuator, calls = _stub_actuator()
    result = actuator.click_bbox([0, 0, 100, 100], padding_percent=0.0)
    assert result["ok"] is True
    assert result["point"] == (50, 50)
    assert calls["move"] == [(50, 50)]
    assert calls["click"] == [True]


def test_click_bbox_applies_padding_before_centroid() -> None:
    # Padding only shrinks the box; a symmetric box's centroid is unchanged.
    actuator, calls = _stub_actuator()
    result = actuator.click_bbox([0, 0, 100, 100], padding_percent=20.0)
    assert result["ok"] is True
    assert result["point"] == (50, 50)


def test_click_bbox_off_center_box() -> None:
    actuator, calls = _stub_actuator()
    result = actuator.click_bbox([10, 20, 30, 60], padding_percent=0.0)
    assert result["ok"] is True
    assert result["point"] == (20, 40)
    assert calls["move"] == [(20, 40)]


def test_click_bbox_normalizes_resolution_before_clicking() -> None:
    actuator, calls = _stub_actuator(screen_size=(4000, 4000))
    # Centroid in a 1000x1000 model space is (50, 50); scaled to a 2000x2000
    # target that should land at (100, 100).
    result = actuator.click_bbox(
        [0, 0, 100, 100],
        padding_percent=0.0,
        source_resolution=(1000, 1000),
        target_resolution=(2000, 2000),
    )
    assert result["ok"] is True
    assert result["point"] == (100, 100)
    assert calls["move"] == [(100, 100)]


def test_click_bbox_failsafe_aborts_when_target_off_screen() -> None:
    actuator, calls = _stub_actuator(screen_size=(100, 100))
    result = actuator.click_bbox([500, 500, 600, 600], padding_percent=0.0)
    assert result["ok"] is False
    assert "failsafe" in result["error"]
    assert calls["move"] == []
    assert calls["click"] == []


def test_click_bbox_failsafe_boundary_is_exclusive() -> None:
    # A centroid exactly on the screen edge is out-of-bounds (half-open range).
    actuator, calls = _stub_actuator(screen_size=(100, 100))
    result = actuator.click_bbox([100, 0, 100, 0], padding_percent=0.0)
    assert result["ok"] is False
    assert calls["move"] == []


def test_click_bbox_rejects_malformed_bbox_without_crashing() -> None:
    actuator, calls = _stub_actuator()
    result = actuator.click_bbox([0, 0, 10], padding_percent=0.0)
    assert result["ok"] is False
    assert "invalid bbox" in result["error"]
    assert calls["move"] == []


def test_click_bbox_rate_limits_rapid_successive_calls() -> None:
    actuator, calls = _stub_actuator()
    first = actuator.click_bbox([0, 0, 100, 100])
    second = actuator.click_bbox([0, 0, 100, 100])
    assert first["ok"] is True
    assert second["ok"] is False
    assert "rate_limited" in second["error"]
    # Only the first call actually moved/clicked.
    assert len(calls["move"]) == 1
    assert len(calls["click"]) == 1


def test_click_bbox_dry_run_validates_but_never_moves(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DONNA_OS_DRY_RUN", "1")
    actuator, calls = _stub_actuator()
    result = actuator.click_bbox([0, 0, 100, 100])
    assert result["ok"] is True
    assert result["dry_run"] is True
    assert result["point"] == (50, 50)
    assert calls["move"] == []
    assert calls["click"] == []


def test_click_bbox_dry_run_still_enforces_failsafe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DONNA_OS_DRY_RUN", "1")
    actuator, calls = _stub_actuator(screen_size=(100, 100))
    result = actuator.click_bbox([500, 500, 600, 600])
    assert result["ok"] is False
    assert "failsafe" in result["error"]


def test_click_bbox_respects_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    import dana.middleware.kill_switch as kill_switch

    monkeypatch.setattr(kill_switch, "halt_if_requested", lambda: True)
    actuator, calls = _stub_actuator()
    result = actuator.click_bbox([0, 0, 100, 100])
    assert result["ok"] is False
    assert result["halted"] is True
    assert calls["move"] == []
    assert calls["click"] == []


def test_click_target_bbox_tool_wrapper_returns_ok_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Dry-run end-to-end: no dependency injection, real (read-only) screen size.
    monkeypatch.setenv("DONNA_OS_DRY_RUN", "1")
    out = click_target_bbox([0, 0, 100, 100])
    assert out.startswith("OK: click_target_bbox")
    assert "dry_run=True" in out


def test_click_target_bbox_tool_wrapper_returns_error_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DONNA_OS_DRY_RUN", "1")
    out = click_target_bbox([0, 0, 10])  # malformed bbox
    assert out.startswith("ERROR: click_target_bbox failed")
