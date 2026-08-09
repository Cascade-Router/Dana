"""Stage 8.7 — AssistiveTouch floating orb (frameless / drag / hover expand)."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _dry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DANA_OS_DRY_RUN", "1")


def test_assistive_orb_disabled_by_design() -> None:
    """AssistiveTouch orb is an experimental feature, deliberately disabled.

    ``DanaGUI._start_assistive_orb`` (core_agent.py) has its whole body
    commented out behind ``return`` with an explicit
    "DISABLED: ... leave commented until re-enabled" docstring/comment.
    This is a regression guard on that decision, not a functional test of
    the orb (drag/hover/dictation-toggle) — write those against
    ``dana.ui.assistive_orb.AssistiveTouchOrb`` directly once the feature
    is re-enabled here.
    """
    from dana.memory.blackboard import set_dictation_mode

    set_dictation_mode(False)

    try:
        from dana.core_agent import DanaGUI
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"DanaGUI unavailable: {exc}")

    try:
        app = DanaGUI()
        app.update_idletasks()
        app.engage_engine()
        app.update_idletasks()
        app._start_assistive_orb()
        app.update_idletasks()

        assert app.assistive_orb is None

        set_dictation_mode(False)
        app.destroy()
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).lower()
        if "tk" in msg or "tcl" in msg or "pyimage" in msg:
            pytest.skip(f"Tk isolation: {exc}")
        raise
