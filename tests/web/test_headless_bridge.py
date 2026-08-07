"""HF Space / headless bridge smoke tests (no Tkinter, no Ollama required)."""

from __future__ import annotations

import os
import sys


def test_headless_env_flags_and_no_tkinter() -> None:
    os.environ["DONNA_NO_GUI"] = "1"
    os.environ["DONNA_HEADLESS"] = "1"
    for name in (
        "customtkinter",
        "dana.ui.assistive_orb",
        "dana.ui.main",
        "dana.ui.trace_window",
        "dana.core_agent",
    ):
        sys.modules.pop(name, None)

    from dana.web.headless_bridge import (
        assert_no_tkinter_loaded,
        get_bridge,
        load_manifest_dict,
        status_label,
    )

    assert status_label("idle") == "● Idle"
    assert status_label("processing") == "● Processing"
    assert status_label("epic_executing") == "● Epic Executing"
    bridge = get_bridge()
    assert bridge.status() == "idle"
    assert not bridge.is_running
    assert isinstance(load_manifest_dict(), dict)
    assert_no_tkinter_loaded()
    ok, note = bridge.stop()
    assert ok is False
    assert "No Meta-Broker" in note


def test_app_module_imports_without_tkinter() -> None:
    for name in (
        "customtkinter",
        "dana.ui.assistive_orb",
        "dana.ui.main",
        "dana.ui.trace_window",
        "dana.core_agent",
    ):
        sys.modules.pop(name, None)
    os.environ["DONNA_NO_GUI"] = "1"
    os.environ["DONNA_HEADLESS"] = "1"
    import importlib

    import dana.web.headless_bridge as hb

    importlib.reload(hb)
    hb.assert_no_tkinter_loaded()
    ok, note = hb.get_bridge().submit("")
    assert ok is False
    assert "Empty" in note


def test_app_py_is_control_plane_not_legacy_florence() -> None:
    from pathlib import Path

    src = Path(__file__).resolve().parents[2] / "app.py"
    text = src.read_text(encoding="utf-8")
    low = text.lower()
    assert "annotatedimage" not in low
    assert "sample desktop" not in low
    assert "florence_vision" not in low
    assert "ticket_card" not in low
    assert "Dānā Control Plane" in text or "Dana Control Plane" in text
    assert "headless_bridge" in text
    assert "Submit Command" in text
    assert "Stop Execution" in text
    assert "Artifact Manifest" in text
    assert "Live Workspace Viewer" in text
    assert "Epic Task Tracker" in text
    assert "Spec Approval" in text
    assert "_chat_append_user" in text
    assert "DONNA_FORCE_LOCAL" in text
    assert "customtkinter" not in low
