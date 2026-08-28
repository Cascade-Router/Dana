"""Regression coverage for Win32ControlPlane.resync_workspace's window
filter — it must target FreeCAD/AutoCAD by their actual owning process, not
by a substring match against the window's title text, which false-positives
on any unrelated window (e.g. a code editor with this repo open) whose title
happens to mention "freecad"/"autocad"/"acad".
"""

from __future__ import annotations

from unittest.mock import patch

from dana.platform.win32 import Win32ControlPlane, _process_exe_name


def _fake_process(name: str):
    class _Proc:
        def name(self_inner) -> str:
            return name

    return _Proc()


def test_resync_workspace_moves_the_real_freecad_process() -> None:
    windows = [
        {"hwnd": 111, "title": "* box - FreeCAD 1.1.3", "pid": 5936},
    ]
    with (
        patch("dana.tools.os_control.get_active_windows", return_value=windows),
        patch("dana.tools.os_control.get_secondary_monitor", return_value={"left": 1920, "top": 0, "width": 1920, "height": 1080}),
        patch("dana.tools.os_control.move_window_no_activate", return_value=True) as mock_move,
        patch("psutil.Process", return_value=_fake_process("freecad.exe")),
    ):
        result = Win32ControlPlane().resync_workspace()

    assert result["ok"] is True
    assert [m["hwnd"] for m in result["moved"]] == [111]
    mock_move.assert_called_once()


def test_resync_workspace_never_moves_an_unrelated_window_that_merely_mentions_freecad_in_its_title() -> None:
    """The exact live bug this fix addresses: a code editor's title can
    contain "freecad" (e.g. editing dana/plugins/freecad/engine.py) without
    the editor itself being FreeCAD — it must never be relocated."""
    windows = [
        {"hwnd": 222, "title": "FreeCAD plugin boolean cut fix - DANA - Visual Studio Code", "pid": 26164},
    ]
    with (
        patch("dana.tools.os_control.get_active_windows", return_value=windows),
        patch("dana.tools.os_control.get_secondary_monitor", return_value={"left": 1920, "top": 0, "width": 1920, "height": 1080}),
        patch("dana.tools.os_control.move_window_no_activate", return_value=True) as mock_move,
        patch("psutil.Process", return_value=_fake_process("Code.exe")),
    ):
        result = Win32ControlPlane().resync_workspace()

    assert result["ok"] is True
    assert result["moved"] == []
    mock_move.assert_not_called()


def test_resync_workspace_reports_single_monitor_and_touches_nothing() -> None:
    with patch("dana.tools.os_control.get_secondary_monitor", return_value=None):
        result = Win32ControlPlane().resync_workspace()
    assert result == {"ok": True, "moved": [], "note": "single monitor — nothing to resync"}


def test_process_exe_name_is_lowercased() -> None:
    with patch("psutil.Process", return_value=_fake_process("FreeCAD.EXE")):
        assert _process_exe_name(5936) == "freecad.exe"


def test_process_exe_name_returns_empty_string_when_process_is_gone() -> None:
    with patch("psutil.Process", side_effect=Exception("no such process")):
        assert _process_exe_name(999999) == ""
