"""Regression coverage for dana.tools.cad_vision's window-targeting fix —
_find_cad_window must match FreeCAD/AutoCAD by their actual owning process,
not by a substring match against the window's title text, which
false-positives on any unrelated window (e.g. a code editor with this repo
open) whose title happens to mention "freecad"/"autocad"/"acad". Same bug
class/fix as dana.platform.win32.resync_workspace (see tests/platform/test_win32.py).
"""

from __future__ import annotations

from unittest.mock import patch

from dana.tools.cad_vision import _find_cad_window, _process_exe_name


def _fake_process(name: str):
    class _Proc:
        def name(self_inner) -> str:
            return name

    return _Proc()


def test_find_cad_window_matches_the_real_freecad_process() -> None:
    windows = [{"hwnd": 111, "title": "* box - FreeCAD 1.1.3", "pid": 5936}]
    with (
        patch("dana.tools.cad_vision.get_active_windows", return_value=windows),
        patch("psutil.Process", return_value=_fake_process("freecad.exe")),
    ):
        assert _find_cad_window() == windows[0]


def test_find_cad_window_ignores_an_unrelated_window_that_merely_mentions_freecad_in_its_title() -> None:
    """The exact live bug this fix addresses: an editor's title can contain
    "freecad" (e.g. editing dana/plugins/freecad/engine.py) without the
    editor itself being FreeCAD — it must never be selected."""
    windows = [{"hwnd": 222, "title": "FreeCAD plugin fix - DANA - Visual Studio Code", "pid": 26164}]
    with (
        patch("dana.tools.cad_vision.get_active_windows", return_value=windows),
        patch("psutil.Process", return_value=_fake_process("Code.exe")),
    ):
        assert _find_cad_window() is None


def test_process_exe_name_is_lowercased() -> None:
    with patch("psutil.Process", return_value=_fake_process("FreeCAD.EXE")):
        assert _process_exe_name(5936) == "freecad.exe"


def test_process_exe_name_returns_empty_string_when_process_is_gone() -> None:
    with patch("psutil.Process", side_effect=Exception("no such process")):
        assert _process_exe_name(999999) == ""
