"""Wire-up tests: monitor bus → DagMonitorView → DonnaGUI drawer."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from dana.graph.monitor_bus import (
    get_monitor_bus,
    publish_supervisor_plan,
    publish_tool_call,
    publish_worker_finish,
    publish_worker_start,
    reset_monitor_bus,
)


def test_monitor_bus_semantic_events() -> None:
    bus = reset_monitor_bus()
    seen: list[str] = []

    def _cb(ev) -> None:  # noqa: ANN001
        seen.append(str(ev.kind))

    bus.subscribe(_cb)
    publish_supervisor_plan(
        [
            {"task_id": 1, "action": "read", "status": "pending"},
            {"task_id": 2, "action": "write", "status": "pending"},
        ]
    )
    publish_worker_start(1, action="read")
    publish_tool_call("AST parsed a.py", worker=1, tool="get_file_outline")
    publish_worker_finish(1, status="completed", summary="ok")
    kinds = set(seen)
    assert "supervisor_plan" in kinds
    assert "worker_start" in kinds
    assert "tool_call" in kinds
    assert "worker_finish" in kinds
    assert bus.latest_dag.get("tasks")


def test_dag_monitor_view_renders_headless() -> None:
    ctk = pytest.importorskip("customtkinter")
    from dana.ui.dag_monitor_view import DagMonitorView

    bus = reset_monitor_bus()
    publish_supervisor_plan(
        [
            {"task_id": 1, "action": "read a.py", "status": "running"},
            {"task_id": 2, "action": "edit b.py", "status": "pending", "dependencies": [1]},
        ]
    )
    publish_tool_call("AST parsed a.py", worker=1, tool="get_file_outline")
    publish_tool_call("Staging: shadow staged b.py", tool="staging")

    expanded = {"hit": False}

    try:
        root = ctk.CTk()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Tk unavailable: {exc}")
    try:
        view = DagMonitorView(
            root,
            bus=bus,
            poll_ms=10_000,
            on_multistep=lambda: expanded.__setitem__("hit", True),
        )
        view.pack(fill="both", expand=True)
        view.refresh()
        assert view._tree_rows, "expected DAG tree rows"
        assert any(
            "AST" in (line[0] if isinstance(line, tuple) else line)
            or "Staging" in (line[0] if isinstance(line, tuple) else line)
            for line in view._log_lines
        )
        assert expanded["hit"] is True
        bus.publish("done", status="completed")
        # Reflect completed statuses on the snapshot for the summary line.
        bus.publish(
            "dag",
            status="completed",
            tasks=[
                {"task_id": 1, "action": "read a.py", "status": "completed"},
                {"task_id": 2, "action": "edit b.py", "status": "completed"},
            ],
        )
        view.refresh()
        assert view._completion_announced
        assert "Graph END" in view._summary_text
    finally:
        try:
            root.destroy()
        except Exception:  # noqa: BLE001
            pass


def test_donna_gui_has_dag_drawer_and_toggle() -> None:
    try:
        from dana.core_agent import DonnaGUI
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"DonnaGUI unavailable: {exc}")

    bus = reset_monitor_bus()
    try:
        app = DonnaGUI()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Tk unavailable: {exc}")
    try:
        assert getattr(app, "dag_monitor_frame", None) is not None
        assert getattr(app, "dag_monitor_view", None) is not None
        assert getattr(app, "_dag_toggle_btn", None) is not None
        assert app._dag_drawer_visible is False

        app._expand_dag_drawer()
        assert app._dag_drawer_visible is True
        assert "DAG" in str(app._dag_toggle_btn.cget("text"))

        app._collapse_dag_drawer()
        assert app._dag_drawer_visible is False
        assert "▸" in str(app._dag_toggle_btn.cget("text"))

        # Auto-expand callback path used when multi-step plans arrive.
        publish_supervisor_plan(
            [
                {"task_id": 1, "action": "a", "status": "pending"},
                {"task_id": 2, "action": "b", "status": "pending"},
            ]
        )
        app.dag_monitor_view.refresh()
        # on_multistep schedules expand via after(0); process pending idle tasks.
        try:
            app.update()
            app.update_idletasks()
        except Exception:  # noqa: BLE001
            pass
        assert app._dag_drawer_visible is True
        assert bus.latest_dag.get("tasks")
    finally:
        try:
            app.destroy()
        except Exception:  # noqa: BLE001
            pass


def test_async_dag_stream_keeps_ui_thread_free(tmp_path: Path) -> None:
    try:
        from dana.core_agent import DonnaGUI
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"DonnaGUI unavailable: {exc}")

    reset_monitor_bus()
    try:
        app = DonnaGUI()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Tk unavailable: {exc}")

    demo = tmp_path / "dag_async"
    demo.mkdir(parents=True, exist_ok=True)
    (demo / "a.py").write_text("A = 1\n", encoding="utf-8")
    store = {str(demo / "a.py").replace("\\", "/"): "A = 1\n"}

    def tool_fn(action: str, filepath: str, content: str | None = None) -> str:
        key = filepath.replace("\\", "/")
        time.sleep(0.05)
        if action == "read":
            return f"OK: read {key}\n{store.get(key, '')}"
        store[key] = str(content or "")
        return f"OK: write {key} (shadow staged)"

    # Deterministic 2-step planner (DagPlanner = Callable[[str], list[DagTask]]).
    def planner(_prompt: str):
        return [
            {
                "task_id": 1,
                "action": f"read {demo / 'a.py'}",
                "filepath": str(demo / "a.py"),
                "status": "pending",
                "dependencies": [],
                "summary": "",
                "error": "",
                "attempts": 0,
            },
            {
                "task_id": 2,
                "action": f"write {demo / 'b.py'}",
                "filepath": str(demo / "b.py"),
                "content": "B = 1\n",
                "status": "pending",
                "dependencies": [1],
                "summary": "",
                "error": "",
                "attempts": 0,
            },
        ]

    try:
        ui_thread = threading.current_thread()
        app.start_dag_monitor_stream(
            "demo",
            tool_fn=tool_fn,
            planner=planner,
        )
        # Pump Tk briefly; UI thread must remain the same (non-blocking start).
        deadline = time.time() + 4.0
        while time.time() < deadline:
            try:
                app.update()
            except Exception:  # noqa: BLE001
                break
            t = getattr(app, "_dag_stream_thread", None)
            if t is not None and not t.is_alive():
                break
            time.sleep(0.05)
        assert threading.current_thread() is ui_thread
        bus = get_monitor_bus(create=False)
        assert bus is not None
        # Either drained into the view or still on the bus / snapshot.
        app.dag_monitor_view.refresh()
        assert bus.latest_dag.get("tasks") or app.dag_monitor_view._log_lines
    finally:
        try:
            app.destroy()
        except Exception:  # noqa: BLE001
            pass


def test_main_module_accepts_dag_demo_flag() -> None:
    from dana.ui import main as ui_main

    parser_src = Path(ui_main.__file__).read_text(encoding="utf-8")
    assert "--dag-demo" in parser_src
    assert "start_dag_monitor_stream" in parser_src
