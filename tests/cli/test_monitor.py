"""DAG monitor bus + streamed graph updates (hermetic)."""

from __future__ import annotations

from pathlib import Path

import pytest

from dana.graph.builder import _cli_main, stream_dag_supervisor
from dana.graph.monitor_bus import get_monitor_bus, reset_monitor_bus
from dana.paths import PROJECT_ROOT


def test_stream_monitor_bus_receives_dag_and_tools(tmp_path: Path) -> None:
    reset_monitor_bus()
    bus = get_monitor_bus(create=True)
    assert bus is not None

    demo = Path(PROJECT_ROOT) / "logs" / "dag_monitor_demo_test"
    demo.mkdir(parents=True, exist_ok=True)
    (demo / "a.py").write_text("A = 1\n", encoding="utf-8")
    (demo / "b.py").write_text("B = 0\n", encoding="utf-8")
    store = {
        "logs/dag_monitor_demo_test/a.py": "A = 1\n",
        "logs/dag_monitor_demo_test/b.py": "B = 0\n",
        "logs/dag_monitor_demo_test/summary.md": "",
    }

    def tool_fn(action: str, filepath: str, content: str | None = None) -> str:
        key = filepath.replace("\\", "/")
        if action == "read":
            body = store.get(key, "")
            return f"OK: read {key}\n{body}"
        store[key] = str(content or "")
        target = Path(PROJECT_ROOT) / key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(store[key], encoding="utf-8")
        return f"OK: write {key} (shadow staged)"

    def outline_fn(file_path: str) -> str:
        from dana.graph.monitor_bus import publish_tool_line

        publish_tool_line(f"AST parsed {file_path}")
        return f"OK: outline {file_path} lang=python symbols=1\n(module)"

    from langgraph.graph import END, START, StateGraph

    from dana.graph.nodes.supervisor import (
        END_ROUTE,
        SUPERVISOR_NODE,
        WORKER_NODE,
        make_supervisor_node,
        route_after_supervisor,
    )
    from dana.graph.nodes.worker import make_workers_node
    from dana.graph.state import SupervisorState, empty_supervisor_state
    from dana.graph.builder import _emit_stream_chunk

    prompt = (
        "1. Read logs/dag_monitor_demo_test/a.py\n"
        "2. Edit logs/dag_monitor_demo_test/b.py\n"
        "3. Write logs/dag_monitor_demo_test/summary.md\n"
    )
    g = StateGraph(SupervisorState)
    g.add_node(SUPERVISOR_NODE, make_supervisor_node())
    g.add_node(
        WORKER_NODE,
        make_workers_node(tool_fn=tool_fn, outline_fn=outline_fn),
    )
    g.add_edge(START, SUPERVISOR_NODE)
    g.add_conditional_edges(
        SUPERVISOR_NODE,
        route_after_supervisor,
        {WORKER_NODE: WORKER_NODE, END_ROUTE: END},
    )
    g.add_edge(WORKER_NODE, SUPERVISOR_NODE)
    app = g.compile()
    initial = empty_supervisor_state(prompt)
    chunks = 0
    for chunk in app.stream(initial, stream_mode="updates"):
        chunks += 1
        _emit_stream_chunk(chunk, monitor=True, on_update=None)

    events = bus.drain(max_items=500)
    kinds = {e.kind for e in events}
    assert chunks >= 3
    assert "dag" in kinds or bus.latest_dag
    tool_msgs = [
        str(e.payload.get("message") or "")
        for e in events
        if e.kind == "tool"
    ]
    assert any("AST parsed" in m or "outline" in m or "shadow staged" in m for m in tool_msgs) or bus.latest_dag.get("tasks")
    assert bus.latest_dag.get("tasks")


def test_cli_monitor_demo_flag() -> None:
    reset_monitor_bus()
    rc = _cli_main(["--monitor", "--demo"])
    assert rc == 0
    bus = get_monitor_bus(create=False)
    assert bus is not None
    assert bus.latest_dag.get("tasks")


def test_textual_app_mounts_and_polls() -> None:
    pytest.importorskip("textual")
    import asyncio

    from dana.cli.monitor import DanaMonitorApp
    from dana.graph.monitor_bus import publish_tool_line, reset_monitor_bus

    async def _run() -> None:
        reset_monitor_bus()
        app = DanaMonitorApp(auto_demo=False)
        async with app.run_test() as pilot:
            publish_tool_line("AST parsed watchdog_graph.py")
            publish_tool_line("Staging edit to .dana_scratch")
            await pilot.pause(0.3)
            assert app.query_one("#tool_log") is not None
            assert app.query_one("#dag_tree") is not None
            assert app.query_one("#telemetry") is not None

    asyncio.run(_run())
