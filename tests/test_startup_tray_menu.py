"""Tests for tray Run-on-Startup checkmark binding."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from donna.tools import setup_startup
from donna.ui import startup_tray


def test_is_startup_enabled_windows_on(monkeypatch) -> None:
    monkeypatch.setattr(setup_startup, "_system", lambda: "Windows")
    fake_winreg = MagicMock()
    fake_winreg.HKEY_CURRENT_USER = object()
    fake_winreg.KEY_READ = 1
    fake_winreg.QueryValueEx.return_value = ("C:/start.bat", 1)
    fake_winreg.OpenKey.return_value.__enter__.return_value = MagicMock()
    with patch.dict("sys.modules", {"winreg": fake_winreg}):
        assert setup_startup.is_startup_enabled() is True
        assert setup_startup.check_startup_registry_status() is True


def test_is_startup_enabled_windows_off(monkeypatch) -> None:
    monkeypatch.setattr(setup_startup, "_system", lambda: "Windows")
    fake_winreg = MagicMock()
    fake_winreg.HKEY_CURRENT_USER = object()
    fake_winreg.KEY_READ = 1
    fake_winreg.OpenKey.return_value.__enter__.side_effect = FileNotFoundError()
    with patch.dict("sys.modules", {"winreg": fake_winreg}):
        assert setup_startup.is_startup_enabled() is False


def test_toggle_run_on_startup_flips_and_updates_menu(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        startup_tray,
        "check_startup_registry_status",
        lambda _item=None: False,
    )
    monkeypatch.setattr(
        "donna.tools.setup_startup.is_startup_enabled",
        lambda: False,
    )
    monkeypatch.setattr(
        "donna.tools.setup_startup.enable_startup",
        lambda: calls.append("enable") or 0,
    )
    monkeypatch.setattr(
        "donna.tools.setup_startup.disable_startup",
        lambda: calls.append("disable") or 0,
    )
    icon = MagicMock()
    startup_tray.toggle_run_on_startup(icon, None)
    assert calls == ["enable"]
    icon.update_menu.assert_called_once()


def test_tray_checked_binding_is_callable() -> None:
    assert callable(startup_tray.check_startup_registry_status)
    assert isinstance(startup_tray.check_startup_registry_status(None), bool)
