"""Stage 7.4 — Yield to Human soft-interrupt."""

from __future__ import annotations

import time

from dana.middleware.human_yield import (
    note_physical_input,
    reset_physical_input_clock,
    yield_check,
)
from dana.operators.ghost_typist import GhostTypistOperator


def setup_function() -> None:  # noqa: D103
    reset_physical_input_clock()


def teardown_function() -> None:  # noqa: D103
    reset_physical_input_clock()


def test_yield_check_waits_quiet_window(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("DONNA_DISABLE_TOAST", "1")
    reset_physical_input_clock()
    note_physical_input(source="test")
    events: list[str] = []

    t0 = time.perf_counter()
    did = yield_check(
        operator="test_op",
        quiet_s=0.35,
        sleep_s=0.05,
        on_pause=lambda: events.append("pause"),
        on_resume=lambda: events.append("resume"),
    )
    elapsed = time.perf_counter() - t0
    assert did is True
    assert events == ["pause", "resume"]
    assert elapsed >= 0.30


def test_yield_check_noop_without_physical_input() -> None:
    reset_physical_input_clock()
    events: list[str] = []
    did = yield_check(
        operator="test_op",
        quiet_s=3.0,
        on_pause=lambda: events.append("pause"),
        on_resume=lambda: events.append("resume"),
    )
    assert did is False
    assert events == []


def test_ghost_typist_yields_mid_type(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("DONNA_OS_DRY_RUN", "1")
    monkeypatch.setenv("DONNA_GHOST_SKIP_HOTKEY", "1")
    monkeypatch.setenv("DONNA_DISABLE_TOAST", "1")
    reset_physical_input_clock()

    typed: list[str] = []
    pauses: list[str] = []

    def _type(ch: str) -> bool:
        typed.append(ch)
        if len(typed) == 2:
            note_physical_input(source="simulated_mouse")
        return True

    import dana.middleware.human_yield as hy

    _orig = hy.yield_check

    def _yc(*, operator: str = "operator", quiet_s: float = 0.25, **kwargs):  # noqa: ANN003
        return _orig(
            operator=operator,
            quiet_s=0.25,
            sleep_s=0.05,
            on_pause=lambda: pauses.append("pause"),
            on_resume=lambda: pauses.append("resume"),
        )

    monkeypatch.setattr(hy, "yield_check", _yc)

    op = GhostTypistOperator(
        type_char=_type,
        wait_hotkey_fn=lambda *_a, **_k: True,
        read_visual=lambda: "stable",
        chunk_size=40,
    )
    result = op.run("ABCDEFGH", wait_hotkey=False)
    assert result.get("ok") is True
    assert len(typed) == len("ABCDEFGH")
    assert "pause" in pauses and "resume" in pauses
    # At least one yield after the simulated mouse move.
    assert pauses.count("pause") >= 1
