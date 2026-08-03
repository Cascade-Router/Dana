"""open_window_on_startup setting load/save + kill-switch NO_WINDOW flag."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_open_window_on_startup_default_true(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings_file = tmp_path / "settings.json"
    monkeypatch.setattr("dana.settings.SETTINGS_PATH", str(settings_file))
    monkeypatch.setattr("dana.settings._CACHE", None)

    from dana import settings as ds

    assert ds.is_open_window_on_startup() is True


def test_open_window_on_startup_persist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings_file = tmp_path / "settings.json"
    settings_file.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr("dana.settings.SETTINGS_PATH", str(settings_file))
    monkeypatch.setattr("dana.settings._CACHE", None)

    from dana import settings as ds

    ds.set_open_window_on_startup(False)
    monkeypatch.setattr("dana.settings._CACHE", None)
    assert ds.is_open_window_on_startup() is False
    raw = json.loads(settings_file.read_text(encoding="utf-8"))
    assert raw.get("open_window_on_startup") is False

    ds.set_open_window_on_startup(True)
    monkeypatch.setattr("dana.settings._CACHE", None)
    assert ds.is_open_window_on_startup() is True
    raw = json.loads(settings_file.read_text(encoding="utf-8"))
    assert raw.get("open_window_on_startup") is True


def test_stop_dana_bat_hides_powershell() -> None:
    root = Path(__file__).resolve().parents[1]
    launchers = root / "scripts" / "launchers"
    bat = launchers / "stop_dana.bat"
    vbs = launchers / "stop_dana.vbs"
    text = bat.read_text(encoding="utf-8", errors="replace")
    assert bat.is_file()
    assert vbs.is_file()
    assert "-WindowStyle Hidden" in text
    assert ">nul" in text.lower() or "2>&1" in text
    assert "stop_dana.bat" in vbs.read_text(encoding="utf-8", errors="replace")
    # Root wrappers forward to scripts/launchers.
    assert "scripts\\launchers\\stop_dana.bat" in (
        root / "stop_dana.bat"
    ).read_text(encoding="utf-8", errors="replace")
