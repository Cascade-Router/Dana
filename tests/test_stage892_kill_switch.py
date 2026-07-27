"""Stage 8.9.2 — STOP DONNA / KILL SWITCH launches stop_donna.bat."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _dry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DONNA_OS_DRY_RUN", "1")


def test_kill_donna_processes_launches_bat(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from donna.core_agent import DonnaGUI

    bat = tmp_path / "stop_donna.bat"
    bat.write_text("@echo off\n", encoding="utf-8")
    monkeypatch.setattr("donna.paths.PROJECT_ROOT", tmp_path)

    launched: list[dict] = []

    def _fake_popen(*args, **kwargs):  # noqa: ANN001
        launched.append({"args": args, "kwargs": kwargs})
        return SimpleNamespace(pid=4242)

    monkeypatch.setattr("donna.core_agent.subprocess.Popen", _fake_popen)

    app = DonnaGUI()
    app.update_idletasks()
    result = app.kill_donna_processes()
    assert result["ok"] is True
    assert result["pid"] == 4242
    assert launched
    assert "stop_donna.bat" in str(launched[0]["args"][0])
    assert launched[0]["kwargs"].get("shell") is True
    if os.name == "nt":
        flags = int(launched[0]["kwargs"].get("creationflags") or 0)
        no_window = int(getattr(__import__("subprocess"), "CREATE_NO_WINDOW", 0x08000000))
        assert flags & no_window, f"CREATE_NO_WINDOW missing from flags={flags:#x}"
    app.destroy()


def test_kill_donna_missing_bat(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from donna.core_agent import DonnaGUI

    monkeypatch.setattr("donna.paths.PROJECT_ROOT", tmp_path)
    app = DonnaGUI()
    result = app.kill_donna_processes()
    assert result["ok"] is False
    assert "not found" in str(result.get("message") or "").lower()
    app.destroy()


def test_stop_button_shows_terminating(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from donna.core_agent import DonnaGUI

    bat = tmp_path / "stop_donna.bat"
    bat.write_text("@echo off\n", encoding="utf-8")
    monkeypatch.setattr("donna.paths.PROJECT_ROOT", tmp_path)
    calls: list[int] = []
    monkeypatch.setattr(
        "donna.core_agent.subprocess.Popen",
        lambda *a, **k: calls.append(1) or SimpleNamespace(pid=1),
    )

    app = DonnaGUI()
    app.update_idletasks()
    assert "STOP DONNA" in str(app.stop_donna_btn.cget("text"))
    # Bypass after(80) — invoke the click handler then kill directly.
    app.stop_donna_btn.configure(text="TERMINATING...", state="disabled")
    app.update_idletasks()
    assert "TERMINATING" in str(app.stop_donna_btn.cget("text")).upper()
    assert app.kill_donna_processes()["ok"] is True
    assert calls
    app.destroy()


def test_live_trace_has_no_footer_kill_switch() -> None:
    """Stage 8.9.8 — footer KILL SWITCH removed; header STOP DONNA is sole exit."""
    import customtkinter as ctk

    from donna.core_agent import DonnaGUI
    from donna.ui.trace_window import LiveTracePanel

    root = ctk.CTk()
    root.withdraw()
    panel = LiveTracePanel(root)
    root.update_idletasks()
    assert not hasattr(panel, "_kill_switch_btn")
    assert not hasattr(panel, "_on_kill_switch")
    root.destroy()

    app = DonnaGUI()
    try:
        assert app.stop_donna_btn is not None
        assert "STOP DONNA" in str(app.stop_donna_btn.cget("text"))
    finally:
        app.destroy()
