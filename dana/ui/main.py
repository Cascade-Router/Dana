"""Smoke entrypoint for Dana Control Dashboard UI.

Usage::

    python -m dana.ui.main
    python -m dana.ui.main --stay --dag-demo

Headless / CI: constructs ``DanaGUI`` and exits after a short idle so the
process does not hang forever. Pass ``--stay`` to keep the window open.
"""

from __future__ import annotations

import argparse
import sys
import tkinter as tk

from dana.core.telemetry import AsyncRingBuffer, NeuralStreamEmitter
from dana.utils.adaptive_poller import AdaptivePoller


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dana Control Dashboard smoke UI")
    parser.add_argument(
        "--stay",
        action="store_true",
        help="Keep the window open (interactive). Default: destroy after paint.",
    )
    parser.add_argument(
        "--ms",
        type=int,
        default=400,
        help="Milliseconds to idle before auto-close when not --stay (default 400).",
    )
    parser.add_argument(
        "--dag-demo",
        action="store_true",
        help="After paint, run a background multi-step DAG into the live monitor bus.",
    )
    args = parser.parse_args(argv)

    # Ensure GEMINI_API_KEY / etc. from .env are visible when Hybrid Broker is on.
    try:
        from dana.graph.cloud_planner import ensure_dotenv_loaded

        ensure_dotenv_loaded()
    except Exception:  # noqa: BLE001
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except Exception:  # noqa: BLE001
            pass

    try:
        from dana.ui.theme import apply_dana_ctk_theme

        apply_dana_ctk_theme()
    except Exception:  # noqa: BLE001
        pass

    try:
        from dana.core_agent import DanaGUI
    except Exception as exc:  # noqa: BLE001
        print(f"DanaGUI unavailable: {exc}", file=sys.stderr)
        return 1

    try:
        app = DanaGUI()
        try:
            app.withdraw()
        except Exception:
            pass

        buffer = AsyncRingBuffer(capacity=500)
        emitter = NeuralStreamEmitter(buffer)
        telemetry_text = tk.Text(app, height=12, width=80)
        telemetry_text.pack(fill=tk.BOTH, expand=True)

        def _refresh_canvas() -> bool:
            events = buffer.snapshot()
            telemetry_text.delete("1.0", tk.END)
            if events:
                telemetry_text.insert(tk.END, "\n".join(repr(item) for item in events[-12:]))
            else:
                telemetry_text.insert(tk.END, "(no telemetry yet)")
            return bool(events)

        # AdaptivePoller.start() runs callbacks on a background thread, which
        # is unsafe here since the callback touches Tk widgets — drive it
        # from the widget's own self.after() chain instead (same pattern as
        # DanaGUI._master_telemetry_tick).
        poller = AdaptivePoller(_refresh_canvas)

        def _telemetry_tick() -> None:
            if not app.winfo_exists():
                return
            try:
                had_activity = _refresh_canvas()
            except Exception:
                had_activity = None
            delay_s = poller.note_activity(had_activity)
            try:
                app.after(max(1, int(delay_s * 1000)), _telemetry_tick)
            except Exception:
                pass

        app.after(50, _telemetry_tick)
        emitter.emit("ui_ready", {"mode": "Unified Canvas"})
    except Exception as exc:  # noqa: BLE001
        print(f"Tk unavailable: {exc}", file=sys.stderr)
        return 1

    try:
        # Show briefly for smoke; constructor withdraws by default.
        try:
            app.deiconify()
            app.lift()
        except Exception:  # noqa: BLE001
            pass

        if args.dag_demo:
            try:
                from dana.graph.monitor_bus import reset_monitor_bus

                reset_monitor_bus()
            except Exception:  # noqa: BLE001
                pass
            prompt = (
                "1. Read logs/dag_ui_demo/a.py\n"
                "2. Edit logs/dag_ui_demo/b.py\n"
                "3. Write logs/dag_ui_demo/summary.md\n"
            )

            def _kick() -> None:
                try:
                    from pathlib import Path

                    from dana.paths import PROJECT_ROOT

                    demo = Path(PROJECT_ROOT) / "logs" / "dag_ui_demo"
                    demo.mkdir(parents=True, exist_ok=True)
                    (demo / "a.py").write_text("A = 1\n", encoding="utf-8")
                    (demo / "b.py").write_text("B = 0\n", encoding="utf-8")
                    store = {
                        "logs/dag_ui_demo/a.py": "A = 1\n",
                        "logs/dag_ui_demo/b.py": "B = 0\n",
                        "logs/dag_ui_demo/summary.md": "",
                    }

                    def tool_fn(
                        action: str, filepath: str, content: str | None = None
                    ) -> str:
                        key = filepath.replace("\\", "/")
                        if action == "read":
                            return f"OK: read {key}\n{store.get(key, '')}"
                        store[key] = str(content or "")
                        target = Path(PROJECT_ROOT) / key
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_text(store[key], encoding="utf-8")
                        return f"OK: write {key} (shadow staged)"

                    app.start_dag_monitor_stream(prompt, tool_fn=tool_fn)
                except Exception as exc:  # noqa: BLE001
                    print(f"DAG demo failed to start: {exc}", file=sys.stderr)

            try:
                app.after(120, _kick)
            except Exception:  # noqa: BLE001
                _kick()

        if args.stay:
            app.mainloop()
            return 0

        idle_ms = max(50, int(args.ms))
        if args.dag_demo:
            idle_ms = max(idle_ms, 2500)

        def _quit() -> None:
            try:
                app.destroy()
            except Exception:  # noqa: BLE001
                pass

        app.after(idle_ms, _quit)
        app.mainloop()
        print("ui smoke ok")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"UI smoke failed: {exc}", file=sys.stderr)
        try:
            app.destroy()
        except Exception:  # noqa: BLE001
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
