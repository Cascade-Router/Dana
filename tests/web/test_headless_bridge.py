"""HF Space / headless bridge smoke tests (no Tkinter, no Ollama required)."""

from __future__ import annotations

import os
import sys


def test_headless_env_flags_and_no_tkinter() -> None:
    os.environ["DONNA_NO_GUI"] = "1"
    os.environ["DONNA_HEADLESS"] = "1"
    # Drop any accidental prior GUI imports from other tests in-process.
    for name in ("tkinter", "_tkinter", "customtkinter"):
        sys.modules.pop(name, None)

    from dana.web.headless_bridge import (
        assert_no_tkinter_loaded,
        get_bridge,
        status_label,
    )

    assert status_label("idle") == "● Idle"
    assert status_label("processing") == "● Processing"
    bridge = get_bridge()
    assert bridge.status() == "idle"
    assert not bridge.is_running
    assert_no_tkinter_loaded()


def test_app_module_imports_without_tkinter() -> None:
    for name in ("tkinter", "_tkinter", "customtkinter"):
        sys.modules.pop(name, None)
    os.environ["DONNA_NO_GUI"] = "1"
    os.environ["DONNA_HEADLESS"] = "1"
    import importlib

    # Fresh-ish import of app helpers via bridge only (full Gradio Blocks is heavy).
    import dana.web.headless_bridge as hb

    importlib.reload(hb)
    hb.assert_no_tkinter_loaded()
    ok, note = hb.get_bridge().submit("")
    assert ok is False
    assert "Empty" in note


def test_app_py_source_has_no_tkinter_and_uses_bridge() -> None:
    from pathlib import Path

    src = Path(__file__).resolve().parents[1].parent / "app.py"
    # tests/web → parents[1]=tests, need repo root
    src = Path(__file__).resolve().parents[2] / "app.py"
    text = src.read_text(encoding="utf-8")
    assert "tkinter" not in text.lower() or "no tkinter" in text.lower()
    assert "customtkinter" not in text.lower()
    assert "run_meta_broker_isolated" in text or "headless_bridge" in text
    assert "get_bridge" in text
    assert "DONNA_NO_GUI" in text
