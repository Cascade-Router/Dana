"""Textual TUI — live LangGraph DAG / worker / staging monitor.

Usage:
  python -m dana.cli.monitor
  python -m dana.cli.monitor --demo
  # In another terminal (or via --demo which spawns a background DAG):
  python -m dana.graph.builder --monitor --demo
"""

from __future__ import annotations

import argparse
import threading
import time
from typing import Any

try:
    import psutil
except Exception:  # noqa: BLE001
    psutil = None  # type: ignore[assignment]

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, RichLog, Static, Tree


_STATUS_GLYPH = {
    "pending": "○",
    "ready": "○",
    "running": "●",
    "completed": "✔",
    "failed": "✖",
    "blocked": "■",
}


def _vector_sync_status() -> str:
    try:
        from dana.memory.vector_sync import get_vector_sync

        sync = get_vector_sync()
        if sync is None:
            return "vector_sync=off"
        st = sync.stats
        return (
            f"vector_sync={'on' if getattr(sync, '_started', False) else 'off'} "
            f"events={st.get('events', 0)} reembed={st.get('reembedded', 0)} "
            f"purged={st.get('purged', 0)}"
        )
    except Exception:  # noqa: BLE001
        return "vector_sync=unknown"


def _memory_status() -> str:
    if psutil is None:
        return "mem=n/a"
    try:
        proc = psutil.Process()
        rss_mb = proc.memory_info().rss / (1024 * 1024)
        sys_pct = psutil.virtual_memory().percent
        return f"rss={rss_mb:.0f}MB sys={sys_pct:.0f}%"
    except Exception:  # noqa: BLE001
        return "mem=n/a"


class DanaMonitorApp(App[None]):
    """Three-pane live view of supervisor DAG execution."""

    TITLE = "Dānā DAG Monitor"
    CSS = """
    Screen {
        layout: vertical;
    }
    #body {
        height: 1fr;
    }
    #dag_panel {
        width: 42%;
        border: solid $accent;
        padding: 0 1;
    }
    #stream_panel {
        width: 58%;
        border: solid $primary;
        padding: 0 1;
    }
    #telemetry {
        height: 3;
        dock: bottom;
        background: $surface;
        color: $text;
        padding: 0 1;
        content-align: left middle;
    }
    Tree {
        height: 1fr;
    }
    RichLog {
        height: 1fr;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("d", "run_demo", "Demo DAG"),
        ("c", "clear_log", "Clear log"),
    ]

    def __init__(self, *, auto_demo: bool = False) -> None:
        super().__init__()
        self._auto_demo = bool(auto_demo)
        self._demo_thread: threading.Thread | None = None
        self._tree_labels: dict[str, Any] = {}

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            with Vertical(id="dag_panel"):
                yield Static("DAG Tree", classes="title")
                yield Tree("Supervisor plan", id="dag_tree")
            with Vertical(id="stream_panel"):
                yield Static("Live Tool Stream", classes="title")
                yield RichLog(id="tool_log", highlight=True, markup=True)
        yield Static("telemetry…", id="telemetry")
        yield Footer()

    def on_mount(self) -> None:
        from dana.graph.monitor_bus import get_monitor_bus, reset_monitor_bus

        reset_monitor_bus()
        get_monitor_bus(create=True)
        log = self.query_one("#tool_log", RichLog)
        log.write("[bold cyan]Dānā monitor online[/]. Press [bold]d[/] for demo DAG, [bold]q[/] to quit.")
        self.set_interval(0.15, self._poll_bus)
        self.set_interval(1.0, self._refresh_telemetry)
        self._refresh_telemetry()
        if self._auto_demo:
            self.action_run_demo()

    def action_clear_log(self) -> None:
        self.query_one("#tool_log", RichLog).clear()

    def action_run_demo(self) -> None:
        if self._demo_thread and self._demo_thread.is_alive():
            self.query_one("#tool_log", RichLog).write(
                "[yellow]Demo already running…[/]"
            )
            return
        self.query_one("#tool_log", RichLog).write(
            "[bold green]Starting background demo DAG (non-blocking)…[/]"
        )
        self._demo_thread = threading.Thread(
            target=self._run_demo_dag,
            name="dana-monitor-demo",
            daemon=True,
        )
        self._demo_thread.start()

    def _run_demo_dag(self) -> None:
        try:
            from dana.graph.builder import _cli_main

            # Re-use hermetic demo path with monitor bus enabled.
            rc = _cli_main(["--monitor", "--demo"])
            from dana.graph.monitor_bus import get_monitor_bus

            bus = get_monitor_bus(create=False)
            if bus is not None:
                bus.publish(
                    "tool",
                    message=f"Demo DAG finished rc={rc}",
                )
                bus.publish("status", status="completed" if rc == 0 else "failed")
        except Exception as exc:  # noqa: BLE001
            from dana.graph.monitor_bus import get_monitor_bus

            bus = get_monitor_bus(create=False)
            if bus is not None:
                bus.publish("tool", message=f"Demo error: {exc}")
                bus.publish("status", status="failed")

    def _refresh_telemetry(self) -> None:
        from dana.graph.monitor_bus import get_monitor_bus

        bus = get_monitor_bus(create=False)
        status = bus.status if bus else "idle"
        tel = bus.latest_telemetry if bus else {}
        vs = tel.get("vector_sync") or _vector_sync_status()
        mem = _memory_status()
        line = f" status={status} │ {vs} │ {mem} │ q quit · d demo "
        self.query_one("#telemetry", Static).update(line)
        if bus is not None:
            from dana.graph.monitor_bus import publish_telemetry

            publish_telemetry(
                vector_sync=_vector_sync_status(),
                memory=_memory_status(),
                ts=time.time(),
            )

    def _poll_bus(self) -> None:
        from dana.graph.monitor_bus import get_monitor_bus

        bus = get_monitor_bus(create=False)
        if bus is None:
            return
        log = self.query_one("#tool_log", RichLog)
        for ev in bus.drain(max_items=100):
            if ev.kind == "tool":
                msg = str(ev.payload.get("message") or "")
                if msg:
                    log.write(msg)
            elif ev.kind == "dag":
                self._render_dag(ev.payload)
                node = ev.payload.get("node") or ""
                st = ev.payload.get("status") or ""
                if node:
                    log.write(f"[dim]graph[/] node={node} status={st}")
            elif ev.kind == "status":
                self._refresh_telemetry()
            elif ev.kind == "done":
                log.write(
                    f"[bold]DONE[/] status={ev.payload.get('status')}"
                )
                self._refresh_telemetry()
            elif ev.kind == "telemetry":
                pass
        # Also refresh tree from latest snapshot if events were coalesced away.
        latest = bus.latest_dag
        if latest:
            self._render_dag(latest)

    def _render_dag(self, payload: dict[str, Any]) -> None:
        tree = self.query_one("#dag_tree", Tree)
        tree.clear()
        root = tree.root
        root.expand()
        status = str(payload.get("status") or "planning")
        cycles = payload.get("supervisor_cycles")
        root.set_label(f"Supervisor [{status}] cycles={cycles}")
        active = set(int(x) for x in (payload.get("active_task_ids") or []) if str(x).isdigit() or isinstance(x, int))
        tasks = list(payload.get("tasks") or [])
        if not tasks:
            root.add_leaf("(no tasks yet)")
            return
        for t in tasks:
            tid = t.get("task_id")
            st = str(t.get("status") or "pending")
            glyph = _STATUS_GLYPH.get(st, "?")
            action = str(t.get("action") or "")[:48]
            mark = " «active»" if tid in active else ""
            label = f"{glyph} #{tid} [{st}]{mark} {action}"
            leaf = root.add(label, expand=False)
            deps = t.get("dependencies") or []
            if deps:
                leaf.add_leaf(f"deps → {deps}")
            summary = str(t.get("summary") or "").strip()
            if summary:
                leaf.add_leaf(f"∑ {summary[:80]}")
            err = str(t.get("error") or "").strip()
            if err:
                leaf.add_leaf(f"! {err[:80]}")
        for line in payload.get("checkpoint_log") or []:
            root.add_leaf(f"ckpt: {str(line)[:70]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dānā LangGraph DAG TUI monitor")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Auto-start a hermetic background DAG on launch",
    )
    args = parser.parse_args(argv)
    app = DanaMonitorApp(auto_demo=bool(args.demo))
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
