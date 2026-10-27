"""show_in_freecad_gui must only ever manage the FreeCAD GUI process Dana
itself launched: never terminate or move a FreeCAD window the user opened,
never match an unrelated window by title, and never write to the user's own
FreeCAD preferences."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from dana.plugins.freecad import engine


@pytest.fixture(autouse=True)
def _no_tracked_gui(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine, "_dana_gui_proc", None)


def _sleeper() -> subprocess.Popen[bytes]:
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])


def test_terminate_only_kills_the_process_dana_launched(monkeypatch: pytest.MonkeyPatch) -> None:
    ours = _sleeper()
    users_own = _sleeper()
    try:
        monkeypatch.setattr(engine, "_dana_gui_proc", ours)
        engine._terminate_freecad_gui(timeout=5.0)
        assert ours.poll() is not None
        assert users_own.poll() is None
        assert engine._dana_gui_proc is None
    finally:
        for proc in (ours, users_own):
            if proc.poll() is None:
                proc.kill()
                proc.wait()


def test_terminate_without_a_tracked_process_is_a_no_op() -> None:
    engine._terminate_freecad_gui()
    assert engine._is_freecad_gui_running() is False


def test_find_window_matches_dana_pid_not_title(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Proc:
        pid = 4242

    windows = [
        {"hwnd": 1, "pid": 100, "title": "engine.py - dana/plugins/freecad - Visual Studio Code"},
        {"hwnd": 2, "pid": 200, "title": "FreeCAD 1.1 - users_own_part.FCStd"},
        {"hwnd": 3, "pid": 4242, "title": "FreeCAD"},
    ]
    import dana.tools.os_control as os_control

    monkeypatch.setattr(os_control, "get_active_windows", lambda: windows)
    monkeypatch.setattr(engine, "_dana_gui_proc", _Proc())
    assert engine._find_freecad_window() == windows[2]

    monkeypatch.setattr(engine, "_dana_gui_proc", None)
    assert engine._find_freecad_window() is None


def test_gui_launch_uses_private_user_config_and_fixed_macro(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    doc = tmp_path / "Session_Active.FCStd"
    doc.write_bytes(b"fcstd")
    launched: list[list[str]] = []
    terminated: list[int] = []

    class _FakePopen:
        pid = 999

        def __init__(self, argv: list[str]) -> None:
            launched.append(argv)
            self.alive = True

        def poll(self) -> int | None:
            return None if self.alive else 0

        def terminate(self) -> None:
            terminated.append(self.pid)
            self.alive = False

        def wait(self, timeout: float | None = None) -> int:
            return 0

    monkeypatch.setenv("DANA_HEADLESS", "false")
    monkeypatch.setattr(engine, "_GUI_USER_CFG_PATH", tmp_path / "dana_gui_user.cfg")
    monkeypatch.setattr(engine, "_FIT_VIEW_MACRO_PATH", tmp_path / "dana_fit_view.FCMacro")
    monkeypatch.setattr(engine, "get_freecad_gui_path", lambda: "FreeCAD.exe")
    monkeypatch.setattr(engine.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(engine, "_WINDOW_POLL_TIMEOUT_S", 0.0)
    monkeypatch.setattr(engine, "_notify_cad_update_ready", lambda *_a, **_k: None)

    for _ in range(2):
        engine.show_in_freecad_gui(str(doc))

    assert launched[0] == [
        "FreeCAD.exe",
        "-u",
        str(tmp_path / "dana_gui_user.cfg"),
        str(doc),
        str(tmp_path / "dana_fit_view.FCMacro"),
    ]
    assert launched[0] == launched[1]
    assert terminated == [999]  # the relaunch closed Dana's previous window only
    assert (tmp_path / "dana_fit_view.FCMacro").read_text(encoding="utf-8") == engine._FIT_VIEW_MACRO
    assert list(tmp_path.glob("*.FCMacro")) == [tmp_path / "dana_fit_view.FCMacro"]


def test_install_globs_are_only_defined_per_platform() -> None:
    """A module-level re-assignment of _COMMON_INSTALL_GLOBS (outside the
    sys.platform branches) used to overwrite the Linux/macOS paths with the
    Windows ones."""
    tree = ast.parse(Path(engine.__file__).read_text(encoding="utf-8"))
    top_level: list[Any] = [
        node
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(t, ast.Name) and t.id == "_COMMON_INSTALL_GLOBS"
            for t in (node.targets if isinstance(node, ast.Assign) else [node.target])
        )
    ]
    assert top_level == []
