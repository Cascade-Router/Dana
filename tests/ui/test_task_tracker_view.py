"""Headless-safe TaskTrackerView + Donna-string purge smoke tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from dana.graph.task_tracker import (
    TaskStatus,
    TaskTracker,
    get_shared_task_tracker,
    humanize_activity,
    set_shared_task_tracker,
)
from dana.logging import CONVERSATION_LOG_PATH, RUNTIME_LOG_PATH
from dana.paths import WAKEWORD_ONNX, resolve_wakeword_onnx


def test_humanize_activity_labels() -> None:
    assert "Python REPL" in humanize_activity(
        TaskStatus.TOOL_EXECUTING, {"tool": "python_repl"}
    )
    assert humanize_activity(TaskStatus.COMPLETED) == "Completed"
    assert "Received" in humanize_activity(TaskStatus.RECEIVED, prompt="hello world")


def test_shared_tracker_injectable(tmp_path: Path) -> None:
    custom = TaskTracker(
        dropped_log_path=tmp_path / "dropped.log",
        ledger_path=tmp_path / "ledger.md",
    )
    set_shared_task_tracker(custom)
    try:
        assert get_shared_task_tracker() is custom
        custom.start_task("ui-1", "click the search button")
        custom.update_status(
            "ui-1",
            TaskStatus.TOOL_EXECUTING,
            metadata={"tool": "nav_and_click"},
        )
        acts = custom.list_activities()
        assert acts
        assert any("Navigating" in a.message or "nav_and_click" in a.message for a in acts)
    finally:
        set_shared_task_tracker(None)


def test_task_tracker_view_refresh_headless(tmp_path: Path) -> None:
    ctk = pytest.importorskip("customtkinter")
    from dana.ui.task_tracker_view import TaskTrackerView

    tracker = TaskTracker(
        dropped_log_path=tmp_path / "dropped.log",
        ledger_path=tmp_path / "ledger.md",
    )
    tracker.start_task("t1", "Grounding search button")
    tracker.append_activity("t1", "Grounding search button")
    tracker.update_status("t1", TaskStatus.IN_PROGRESS)
    tracker.update_status(
        "t1", TaskStatus.TOOL_EXECUTING, metadata={"tool": "python_repl"}
    )

    try:
        root = ctk.CTk()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Tk unavailable: {exc}")
    try:
        view = TaskTrackerView(root, tracker=tracker, poll_ms=10_000)
        view.pack(fill="both", expand=True)
        view.refresh()
        assert view._rows, "expected at least one timeline row"
        assert view._last_sig
    finally:
        try:
            root.destroy()
        except Exception:  # noqa: BLE001
            pass


def test_gui_title_and_banner_are_dana() -> None:
    try:
        from dana.core_agent import DonnaGUI
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"DonnaGUI unavailable: {exc}")

    try:
        app = DonnaGUI()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Tk unavailable: {exc}")
    try:
        title = str(app.title())
        assert "Dana" in title
        assert "Donna" not in title
        raw = str(app.transcript_box.get("1.0", "end"))
        assert "[Dana]" in raw
        assert "[Donna]" not in raw
        assert "STOP DANA" in str(app.stop_donna_btn.cget("text"))
    finally:
        try:
            app.destroy()
        except Exception:  # noqa: BLE001
            pass


def test_runtime_log_paths_are_dana_branded() -> None:
    assert RUNTIME_LOG_PATH.replace("\\", "/").endswith("logs/dana_runtime.log")
    assert CONVERSATION_LOG_PATH.replace("\\", "/").endswith(
        "logs/dana_conversation.log"
    )
    assert "donna_runtime" not in RUNTIME_LOG_PATH
    assert "donna_conversation" not in CONVERSATION_LOG_PATH


def test_wakeword_onnx_prefers_dana_with_legacy_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import dana.paths as paths

    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(paths, "WAKEWORD_ONNX", tmp_path / "dana.onnx")
    monkeypatch.setattr(paths, "WAKEWORD_ONNX_ALT", tmp_path / "wake_word_model.onnx")
    monkeypatch.setattr(paths, "WAKEWORD_ONNX_LEGACY", tmp_path / "donna.onnx")

    assert paths.resolve_wakeword_onnx() == tmp_path / "dana.onnx"
    (tmp_path / "donna.onnx").write_bytes(b"legacy")
    assert paths.resolve_wakeword_onnx() == tmp_path / "donna.onnx"
    (tmp_path / "dana.onnx").write_bytes(b"modern")
    assert paths.resolve_wakeword_onnx() == tmp_path / "dana.onnx"


def test_key_modules_ui_strings_not_donna() -> None:
    root = Path(__file__).resolve().parents[2]
    checks = {
        root / "dana" / "logging.py": ("dana_runtime.log", "Dana conversation"),
        root / "dana" / "ui" / "__init__.py": ("Dana CustomTkinter",),
        root / "dana" / "vision" / "overlay.py": ('title("Dana ROI")',),
    }
    for path, needles in checks.items():
        text = path.read_text(encoding="utf-8")
        for needle in needles:
            assert needle in text, f"{path.name} missing {needle!r}"
        # Product banner / window title must not say Donna.
        assert 'title("Donna' not in text
        assert "Donna conversation session" not in text

    assert WAKEWORD_ONNX.name == "dana.onnx"
    assert resolve_wakeword_onnx().name in {
        "dana.onnx",
        "wake_word_model.onnx",
        "donna.onnx",
    }
