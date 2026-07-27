"""open_window_on_startup setting load/save + kill-switch NO_WINDOW flag."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_open_window_on_startup_default_true(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings_file = tmp_path / "settings.json"
    monkeypatch.setattr("donna.settings.SETTINGS_PATH", str(settings_file))
    monkeypatch.setattr("donna.settings._CACHE", None)

    from donna import settings as ds

    assert ds.is_open_window_on_startup() is True


def test_open_window_on_startup_persist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings_file = tmp_path / "settings.json"
    settings_file.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr("donna.settings.SETTINGS_PATH", str(settings_file))
    monkeypatch.setattr("donna.settings._CACHE", None)

    from donna import settings as ds

    ds.set_open_window_on_startup(False)
    monkeypatch.setattr("donna.settings._CACHE", None)
    assert ds.is_open_window_on_startup() is False
    raw = json.loads(settings_file.read_text(encoding="utf-8"))
    assert raw.get("open_window_on_startup") is False

    ds.set_open_window_on_startup(True)
    monkeypatch.setattr("donna.settings._CACHE", None)
    assert ds.is_open_window_on_startup() is True
    raw = json.loads(settings_file.read_text(encoding="utf-8"))
    assert raw.get("open_window_on_startup") is True


def test_stop_donna_bat_hides_powershell() -> None:
    bat = Path(__file__).resolve().parents[1] / "stop_donna.bat"
    text = bat.read_text(encoding="utf-8", errors="replace")
    assert "-WindowStyle Hidden" in text
    assert text.count("-WindowStyle Hidden") >= 2
