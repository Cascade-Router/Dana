"""Tests for cross-platform startup registration + tray listening cue."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dana.tools import setup_startup


def test_write_start_bat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(setup_startup, "project_root", lambda: tmp_path)
    monkeypatch.setattr(setup_startup, "_system", lambda: "Windows")
    (tmp_path / "run.py").write_text("# stub entry\n", encoding="utf-8")
    venv_scripts = tmp_path / ".venv" / "Scripts"
    venv_scripts.mkdir(parents=True)
    (venv_scripts / "python.exe").write_bytes(b"")

    # GUI launcher prefers pythonw when present.
    (venv_scripts / "pythonw.exe").write_bytes(b"")
    bat = setup_startup.write_start_bat()
    assert bat == tmp_path / "scripts" / "launchers" / "start_dana.bat"
    text = bat.read_text(encoding="utf-8")
    assert "run.py" in text
    assert "core_agent.py" not in text
    assert "pythonw.exe" in text
    assert "--no-gui" not in text
    assert "cd /d" in text
    assert "start \"Dana\"" in text or 'start "Dana"' in text
    wrap = tmp_path / "start_dana.bat"
    assert wrap.is_file()
    assert "scripts\\launchers\\start_dana.bat" in wrap.read_text(encoding="utf-8")
    print(f"[PASS] start_dana.bat written: {bat}")


def test_enable_disable_windows_mocked() -> None:
    fake_bat = Path("C:/fake/start_dana.bat")
    fake_winreg = MagicMock()
    fake_winreg.HKEY_CURRENT_USER = object()
    fake_winreg.KEY_SET_VALUE = 2
    fake_winreg.KEY_READ = 1
    fake_winreg.REG_SZ = 1
    ctx = MagicMock()
    fake_winreg.OpenKey.return_value.__enter__.return_value = ctx
    fake_winreg.QueryValueEx.return_value = (f'"{fake_bat}"', fake_winreg.REG_SZ)

    with (
        patch.object(setup_startup, "_system", return_value="Windows"),
        patch.object(setup_startup, "write_start_bat", return_value=fake_bat),
        patch.object(setup_startup, "_migrate_legacy_windows_artifacts", lambda: None),
        patch.dict("sys.modules", {"winreg": fake_winreg}),
    ):
        assert setup_startup.enable_startup() == 0
        fake_winreg.SetValueEx.assert_called_once()
        args = fake_winreg.SetValueEx.call_args[0]
        assert args[1] == setup_startup.VALUE_NAME
        assert args[1] == "DanaAssistant"
        assert "start_dana.bat" in str(args[4])

        assert setup_startup.startup_status() == 0
        assert setup_startup.disable_startup() == 0
        fake_winreg.DeleteValue.assert_called_once()
    print("[PASS] enable/status/disable (mocked winreg)")


def test_enable_disable_macos(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(setup_startup, "_system", lambda: "Darwin")
    monkeypatch.setattr(setup_startup, "project_root", lambda: tmp_path)
    plist = tmp_path / "Library" / "LaunchAgents" / setup_startup.MACOS_PLIST_NAME
    monkeypatch.setattr(setup_startup, "macos_plist_path", lambda: plist)
    (tmp_path / "run.py").write_text("# stub\n", encoding="utf-8")
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / ".venv" / "bin" / "python3").write_text("", encoding="utf-8")

    assert setup_startup.enable_startup() == 0
    assert plist.is_file()
    text = plist.read_text(encoding="utf-8")
    assert setup_startup.MACOS_LABEL in text
    assert "run.py" in text
    assert "--no-gui" in text
    assert "WorkingDirectory" in text
    assert "StandardOutPath" in text
    assert "StandardErrorPath" in text
    assert "/tmp/dana_startup.log" in text
    assert setup_startup.startup_status() == 0
    assert setup_startup.disable_startup() == 0
    assert not plist.exists()
    print("[PASS] macOS LaunchAgent enable/disable")


def test_enable_disable_linux(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(setup_startup, "_system", lambda: "Linux")
    monkeypatch.setattr(setup_startup, "project_root", lambda: tmp_path)
    desktop = tmp_path / ".config" / "autostart" / setup_startup.LINUX_DESKTOP_NAME
    monkeypatch.setattr(setup_startup, "linux_desktop_path", lambda: desktop)
    (tmp_path / "run.py").write_text("# stub\n", encoding="utf-8")
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / ".venv" / "bin" / "python3").write_text("", encoding="utf-8")

    assert setup_startup.enable_startup() == 0
    assert desktop.is_file()
    text = desktop.read_text(encoding="utf-8")
    assert "[Desktop Entry]" in text
    assert "Name=Dana" in text
    assert "run.py" in text
    assert "--no-gui" in text
    assert f"Path={tmp_path.resolve()}" in text or "Path=" in text
    assert "/tmp/dana_startup.log" in text
    assert "2>&1" in text
    assert "/bin/bash -c" in text
    assert setup_startup.startup_status() == 0
    assert setup_startup.disable_startup() == 0
    assert not desktop.exists()
    print("[PASS] Linux autostart enable/disable")


def test_audio_switcher_imports_cleanly() -> None:
    """Module must import on every OS without raising at import time."""
    import dana.tools.audio_switcher as audio_switcher

    assert hasattr(audio_switcher, "toggle_audio_endpoint")
    if audio_switcher._WINDOWS is False:
        msg = audio_switcher.toggle_audio_endpoint("wired")
        assert "Windows-only" in msg
    print("[PASS] audio_switcher import is cross-platform safe")


def test_tray_icon_listening_vs_idle() -> None:
    pytest.importorskip("PIL")
    try:
        from dana.core_agent import create_tray_image
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"core_agent unavailable in this environment: {exc}")

    idle = create_tray_image("idle")
    listening = create_tray_image("listening")
    assert idle.size == listening.size == (64, 64)
    assert idle.getpixel((32, 10)) != listening.getpixel((32, 10)) or idle.tobytes() != listening.tobytes()
    print("[PASS] tray listening icon differs from idle")


def test_app_ico_multi_resolution_exists() -> None:
    """assets/dana_logo.ico must exist with standard Windows sizes."""
    import struct

    from dana.paths import PROJECT_ROOT
    from dana.ui.logo import app_icon_path, load_app_icon_pil, resolve_app_icon_path

    ico = resolve_app_icon_path()
    assert ico is not None
    assert ico == app_icon_path()
    assert ico == Path(PROJECT_ROOT) / "assets" / "dana_logo.ico"
    raw = ico.read_bytes()
    count = struct.unpack_from("<H", raw, 4)[0]
    assert count >= 4
    img = load_app_icon_pil((32, 32))
    assert img is not None
    assert img.size == (32, 32)
    print(f"[PASS] dana_logo.ico entries={count} bytes={len(raw)}")


def test_appusermodelid_helper_present_in_entrypoints() -> None:
    """Entry points must set explicit AppUserModelID before GUI boot."""
    root = Path(__file__).resolve().parents[1]
    run_txt = (root / "run.py").read_text(encoding="utf-8")
    core_txt = (root / "dana" / "core_agent.py").read_text(encoding="utf-8", errors="replace")
    needle = "SetCurrentProcessExplicitAppUserModelID"
    assert needle in run_txt
    assert needle in core_txt
    assert "dana.assistant.desktop.v1" in run_txt
    assert "dana.assistant.desktop.v1" in core_txt
    from dana.ui import logo as logo_mod

    assert hasattr(logo_mod, "apply_window_icon")
    assert hasattr(logo_mod, "force_apply_window_icon")
    assert hasattr(logo_mod, "schedule_window_icon")
    print("[PASS] AppUserModelID wired in run.py + core_agent")


def test_write_desktop_shortcut_sets_icon(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(setup_startup, "project_root", lambda: tmp_path)
    monkeypatch.setattr(setup_startup, "_system", lambda: "Windows")
    monkeypatch.setattr(setup_startup, "desktop_shortcut_path", lambda: tmp_path / "Dana.lnk")
    (tmp_path / "run.py").write_text("# stub\n", encoding="utf-8")
    venv = tmp_path / ".venv" / "Scripts"
    venv.mkdir(parents=True)
    (venv / "pythonw.exe").write_bytes(b"")
    ico = tmp_path / "assets" / "dana_logo.ico"
    ico.parent.mkdir(parents=True)
    ico.write_bytes(b"\x00\x00\x01\x00")

    calls: list[list[str]] = []

    def _fake_run(cmd, **kwargs):  # noqa: ANN001
        calls.append(list(cmd))
        setup_startup.desktop_shortcut_path().write_text("lnk", encoding="utf-8")
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr("subprocess.run", _fake_run)
    lnk = setup_startup.write_desktop_shortcut()
    assert lnk is not None
    assert calls
    joined = " ".join(calls[0])
    assert "IconLocation" in joined
    # PowerShell must receive an absolute IconLocation (...\dana_logo.ico,0).
    assert "dana_logo.ico,0" in joined.replace("''", "'")
    assert "assets" in joined.lower()
    print("[PASS] desktop shortcut PowerShell includes absolute IconLocation")


def test_app_icon_path_is_absolute() -> None:
    path = setup_startup.app_icon_path()
    assert path.is_absolute()
    assert path.as_posix().endswith("assets/dana_logo.ico")
    assert path.is_file()
    print(f"[PASS] app_icon_path absolute: {path}")
