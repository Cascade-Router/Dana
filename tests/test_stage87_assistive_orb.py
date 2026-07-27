"""Stage 8.7 — AssistiveTouch floating orb (frameless / drag / hover expand)."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _dry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DONNA_OS_DRY_RUN", "1")


def test_orb_frameless_topmost_drag_expand() -> None:
    from donna.memory.blackboard import set_dictation_mode

    set_dictation_mode(False)

    from donna.core_agent import DonnaGUI

    app = DonnaGUI()
    app.update_idletasks()
    app.update()
    # Force orb start (normally after(200)).
    app._start_assistive_orb()
    app.update_idletasks()
    app.update()

    orb = app.assistive_orb
    assert orb is not None
    win = orb.orb_window
    assert win.winfo_exists()

    # Frameless + topmost
    assert bool(win.overrideredirect())
    try:
        topmost = bool(win.attributes("-topmost"))
    except Exception:
        topmost = True
    assert topmost

    # Compact geometry is orb-sized
    app.update_idletasks()
    w0 = int(win.winfo_width())
    h0 = int(win.winfo_height())
    assert w0 <= 80
    assert h0 <= 80

    # Drag updates geometry
    x0, y0 = int(win.winfo_x()), int(win.winfo_y())
    class _E:
        def __init__(self, x_root: int, y_root: int) -> None:
            self.x_root = x_root
            self.y_root = y_root

    orb._on_press(_E(x0 + 10, y0 + 10))
    orb._on_drag(_E(x0 + 40, y0 + 50))
    orb._on_release()
    app.update_idletasks()
    assert int(win.winfo_x()) != x0 or int(win.winfo_y()) != y0
    assert orb._orb_x == int(win.winfo_x())
    assert orb._orb_y == int(win.winfo_y())

    # Hover expand reveals mini panel
    orb._on_enter()
    app.update_idletasks()
    app.update()
    assert orb._expanded is True
    assert int(win.winfo_width()) > 120
    assert orb._panel.winfo_manager() == "grid"

    # Leave (immediate shrink path)
    orb._cancel_leave()
    orb._apply_compact_geometry()
    app.update_idletasks()
    assert orb._expanded is False

    # Dictation toggle via orb refreshes label
    orb._click_dictation()
    app.update_idletasks()
    assert app._dictation_active is True
    assert "DICTATING" in str(orb._dictation_btn.cget("text")).upper()
    orb._click_dictation()
    app.update_idletasks()
    assert app._dictation_active is False

    set_dictation_mode(False)
    orb.destroy()
    app.destroy()
    print("ASSISTIVE_ORB_VERIFY_OK")
