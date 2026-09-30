"""dana.tools.setup_startup must register the tracked launcher, never rewrite it."""

from __future__ import annotations

from pathlib import Path

import pytest

import dana.tools.setup_startup as startup

_LAUNCHER = "@echo off\r\nREM tracked launcher -- must survive registration untouched\r\n"


@pytest.fixture()
def fake_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(startup, "project_root", lambda: tmp_path)
    return tmp_path


def test_ensure_start_bat_leaves_tracked_launcher_unchanged(fake_root: Path) -> None:
    bat = fake_root / "scripts" / "launchers" / "start_dana.bat"
    bat.parent.mkdir(parents=True)
    bat.write_bytes(_LAUNCHER.encode("utf-8"))

    assert startup.ensure_start_bat() == bat
    assert bat.read_bytes() == _LAUNCHER.encode("utf-8")
    # The gitignored root wrapper the Tauri close handler looks for.
    assert "scripts\\launchers\\start_dana.bat" in (fake_root / "start_dana.bat").read_text(encoding="utf-8")


def test_ensure_start_bat_fails_loudly_when_launcher_is_missing(fake_root: Path) -> None:
    with pytest.raises(FileNotFoundError, match="launcher missing"):
        startup.ensure_start_bat()
    assert not (fake_root / "scripts").exists()


def test_entry_script_is_the_real_backend_launcher() -> None:
    entry = startup.entry_script()
    assert entry.name == "launch_api_server.py"
    assert entry.is_file()


def test_unix_autostart_entries_do_not_pass_legacy_flags(fake_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(startup, "macos_plist_path", lambda: fake_root / "dana.plist")
    monkeypatch.setattr(startup, "linux_desktop_path", lambda: fake_root / "dana.desktop")
    for path in (startup._write_macos_plist(), startup._write_linux_desktop()):
        body = path.read_text(encoding="utf-8")
        assert "launch_api_server.py" in body
        assert "run.py" not in body
        assert "--no-gui" not in body
