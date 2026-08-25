"""Non-blocking smoke: Textual pilot + background demo DAG."""

from __future__ import annotations

import asyncio

from dana.cli.monitor import DanaMonitorApp
from dana.graph.monitor_bus import get_monitor_bus, reset_monitor_bus


async def main() -> None:
    reset_monitor_bus()
    app = DanaMonitorApp(auto_demo=True)
    async with app.run_test(size=(120, 40)) as pilot:
        for _ in range(40):
            await pilot.pause(0.2)
            bus = get_monitor_bus(create=False)
            if (
                bus
                and bus.latest_dag.get("tasks")
                and str(bus.status) in {"completed", "failed"}
            ):
                break
        bus = get_monitor_bus(create=False)
        assert bus is not None
        tasks = bus.latest_dag.get("tasks") or []
        print("tasks", len(tasks), "status", bus.status)
        tel = app.query_one("#telemetry")
        print("telemetry_widget", type(tel).__name__)
        assert len(tasks) >= 3
        assert app.query_one("#dag_tree") is not None
        assert app.query_one("#tool_log") is not None
        print("MONITOR_UI_OK")


if __name__ == "__main__":
    asyncio.run(main())
