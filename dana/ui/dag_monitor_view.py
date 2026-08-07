"""Live DAG execution panel for the Dana Control Dashboard."""

from __future__ import annotations

import json
import os
from typing import Any, Callable

import customtkinter as ctk

from dana.ui import theme as T

_STATUS_COLORS: dict[str, str] = {
    "pending": T.MUTED,
    "ready": T.MUTED,
    "dispatched": T.AMBER,
    "running": T.AMBER,
    "in_progress": T.AMBER,
    "active": T.AMBER,
    "completed": T.EMERALD,
    "failed": T.ROSE,
    "error": T.ROSE,
}


def _pick_workspace_file() -> tuple[str, str]:
    """Prefer newest scratch artifact, then ``.dana_scratch/manifest.json``."""
    candidates: list[str] = []
    try:
        from dana.web.headless_bridge import load_manifest_dict

        man = load_manifest_dict()
        for art in man.get("artifacts") or []:
            if isinstance(art, dict) and art.get("file_path"):
                candidates.append(str(art["file_path"]))
    except Exception:  # noqa: BLE001
        man = None

    seen: set[str] = set()
    ordered: list[str] = []
    for rel in candidates:
        key = rel.replace("\\", "/").lstrip("./")
        if not key or key in seen:
            continue
        if key.startswith(("dana/", "donna_security/", "website/")):
            continue
        seen.add(key)
        ordered.append(key)

    for rel in reversed(ordered):
        path = rel
        if not os.path.isfile(path):
            alt = os.path.join(".dana_scratch", rel)
            if os.path.isfile(alt):
                path = alt
            else:
                continue
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        if len(text) > 12000:
            text = text[:12000] + "\n...[truncated]...\n"
        return rel, text

    if man is not None:
        try:
            body = json.dumps(man, indent=2, ensure_ascii=False)
            return ".dana_scratch/manifest.json", body
        except Exception:  # noqa: BLE001
            pass
    return (
        "(waiting for artifacts…)",
        "# Live workspace will appear here as files are generated.\n",
    )


class DagMonitorView(ctk.CTkFrame):
    """DAG step telemetry + broker log (left) and Live Workspace Viewer (right).

    Polls ``dana.graph.monitor_bus`` on the Tk main thread so graph workers
    can publish from background threads without blocking the UI.
    """

    def __init__(
        self,
        master: Any,
        *,
        bus: Any | None = None,
        bus_factory: Callable[[], Any] | None = None,
        poll_ms: int = 250,
        max_log_lines: int = 80,
        on_multistep: Callable[[], None] | None = None,
        on_complete: Callable[[str], None] | None = None,
        show_header: bool = True,
        status_label: Any | None = None,
        external_tick: bool = False,
    ) -> None:
        super().__init__(master, fg_color=T.BG, corner_radius=12)
        self._bus = bus
        self._bus_factory = bus_factory
        self._poll_ms = max(80, int(poll_ms))
        self._external_tick = bool(external_tick)
        self._max_log_lines = max(20, int(max_log_lines))
        self._on_multistep = on_multistep
        self._on_complete = on_complete
        self._show_header = bool(show_header)
        self._status_label = status_label
        self._tree_rows: list[ctk.CTkFrame] = []
        # (text, optional text_color) — errors render in rose/red.
        self._log_lines: list[tuple[str, str | None]] = []
        self._last_tree_sig = ""
        self._last_log_sig = ""
        self._last_workspace_sig = ""
        self._completion_announced = False
        self._multistep_announced = False
        self._summary_text = ""
        self._planner_mode = "LOCAL"
        self._build()
        if not self._external_tick:
            self.after(self._poll_ms, self._poll)

    def _resolve_bus(self) -> Any | None:
        if self._bus is not None:
            return self._bus
        if self._bus_factory is not None:
            try:
                return self._bus_factory()
            except Exception:  # noqa: BLE001
                return None
        try:
            from dana.graph.monitor_bus import get_monitor_bus

            return get_monitor_bus(create=True)
        except Exception:  # noqa: BLE001
            return None

    def set_bus(self, bus: Any | None) -> None:
        """Tests / DI: swap the backing bus and force a redraw."""
        self._bus = bus
        self._last_tree_sig = ""
        self._last_log_sig = ""
        self._completion_announced = False
        self._multistep_announced = False
        self.refresh()

    def set_planner_mode(self, mode: str) -> None:
        """Show Planner Mode: [LOCAL] / [HYBRID CLOUD] in the summary strip."""
        text = str(mode or "LOCAL").strip() or "LOCAL"
        self._planner_mode = text
        line = f"Planner Mode: [{text}]"
        self._append_log(line)
        try:
            self._summary_lbl.configure(text=line, text_color=T.MUTED)
        except Exception:  # noqa: BLE001
            pass
        try:
            self._render_log()
        except Exception:  # noqa: BLE001
            pass

    def _build(self) -> None:
        if self._show_header:
            header = ctk.CTkFrame(self, fg_color=T.CARD, corner_radius=10)
            header.pack(fill="x", padx=8, pady=(8, 4))
            ctk.CTkLabel(
                header,
                text="DAG Execution",
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color=T.TEXT,
                anchor="w",
            ).pack(side="left", padx=12, pady=8)
            if self._status_label is None:
                self._status_label = ctk.CTkLabel(
                    header,
                    text="Idle",
                    font=ctk.CTkFont(size=11),
                    text_color=T.MUTED,
                    anchor="e",
                )
                self._status_label.pack(side="right", padx=12, pady=8)

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=6, pady=(0, 8))
        try:
            body.grid_columnconfigure(0, weight=1)
            body.grid_columnconfigure(1, weight=1)
            body.grid_rowconfigure(0, weight=1)
        except Exception:  # noqa: BLE001
            pass

        # Left: DAG step telemetry + broker / tool micro-log
        left = ctk.CTkFrame(body, fg_color=T.CARD, corner_radius=8)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 3))
        try:
            left.grid_rowconfigure(1, weight=1)
            left.grid_rowconfigure(3, weight=2)
            left.grid_columnconfigure(0, weight=1)
        except Exception:  # noqa: BLE001
            pass
        ctk.CTkLabel(
            left,
            text="DAG steps",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=T.MUTED,
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 2))
        self._tree_scroll = ctk.CTkScrollableFrame(
            left, fg_color="transparent", corner_radius=0, height=120
        )
        self._tree_scroll.grid(row=1, column=0, sticky="nsew", padx=4, pady=(0, 4))
        ctk.CTkLabel(
            left,
            text="Broker / tool stream",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=T.MUTED,
            anchor="w",
        ).grid(row=2, column=0, sticky="ew", padx=8, pady=(4, 2))
        self._log_box = ctk.CTkTextbox(
            left,
            fg_color=T.BG,
            text_color=T.TEXT,
            font=ctk.CTkFont(family="Consolas", size=11),
            wrap="word",
            activate_scrollbars=True,
        )
        self._log_box.grid(row=3, column=0, sticky="nsew", padx=6, pady=(0, 6))
        try:
            self._log_box.configure(state="disabled")
        except Exception:  # noqa: BLE001
            pass

        # Right: Live Workspace Viewer (fills previously empty half)
        right = ctk.CTkFrame(body, fg_color=T.CARD, corner_radius=8)
        right.grid(row=0, column=1, sticky="nsew", padx=(3, 0))
        ctk.CTkLabel(
            right,
            text="Live Workspace Viewer",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=T.MUTED,
            anchor="w",
        ).pack(fill="x", padx=8, pady=(6, 2))
        self._workspace_label = ctk.CTkLabel(
            right,
            text="(waiting for artifacts…)",
            font=ctk.CTkFont(size=10),
            text_color=T.ACCENT,
            anchor="w",
        )
        self._workspace_label.pack(fill="x", padx=10, pady=(0, 2))
        self._workspace_box = ctk.CTkTextbox(
            right,
            fg_color=T.BG,
            text_color=T.TEXT,
            font=ctk.CTkFont(family="Consolas", size=11),
            wrap="none",
            activate_scrollbars=True,
        )
        self._workspace_box.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        try:
            self._workspace_box.configure(state="disabled")
        except Exception:  # noqa: BLE001
            pass
        self._set_workspace(
            "(waiting for artifacts…)",
            "# Live workspace will appear here as files are generated.\n",
        )

        self._summary_lbl = ctk.CTkLabel(
            self,
            text="",
            font=ctk.CTkFont(size=11),
            text_color=T.EMERALD,
            anchor="w",
        )
        self._summary_lbl.pack(fill="x", padx=12, pady=(0, 8))

    def _set_status(self, text: str) -> None:
        lbl = self._status_label
        if lbl is None:
            return
        try:
            lbl.configure(text=text)
        except Exception:  # noqa: BLE001
            pass

    def _set_workspace(self, label: str, body: str) -> None:
        sig = f"{label}|{len(body)}|{hash(body)}"
        if sig == self._last_workspace_sig:
            return
        self._last_workspace_sig = sig
        try:
            self._workspace_label.configure(text=label)
        except Exception:  # noqa: BLE001
            pass
        try:
            self._workspace_box.configure(state="normal")
            self._workspace_box.delete("1.0", "end")
            self._workspace_box.insert("1.0", body)
            self._workspace_box.configure(state="disabled")
        except Exception:  # noqa: BLE001
            pass

    def _refresh_workspace(self) -> None:
        try:
            label, body = _pick_workspace_file()
            self._set_workspace(label, body)
        except Exception:  # noqa: BLE001
            pass

    def _append_log(self, line: str, *, color: str | None = None) -> None:
        text = str(line).strip()
        if not text:
            return
        if self._log_lines and self._log_lines[-1][0] == text:
            return
        self._log_lines.append((text, color))
        if len(self._log_lines) > self._max_log_lines:
            self._log_lines = self._log_lines[-self._max_log_lines :]

    def _render_log(self) -> None:
        sig = "\n".join(f"{c or ''}|{t}" for t, c in self._log_lines)
        if sig == self._last_log_sig:
            return
        self._last_log_sig = sig
        try:
            self._log_box.configure(state="normal")
            self._log_box.delete("1.0", "end")
            try:
                self._log_box.tag_config("err", foreground=T.ROSE)
            except Exception:  # noqa: BLE001
                pass
            for text, color in self._log_lines:
                if color:
                    self._log_box.insert("end", text + "\n", ("err",))
                else:
                    self._log_box.insert("end", text + "\n")
            self._log_box.see("end")
            self._log_box.configure(state="disabled")
        except Exception:  # noqa: BLE001
            pass

    def _render_tree(self, tasks: list[dict[str, Any]]) -> None:
        sig = "|".join(
            f"{t.get('task_id')}:{t.get('status')}:{t.get('action')}" for t in tasks
        )
        if sig == self._last_tree_sig:
            return
        self._last_tree_sig = sig
        for row in self._tree_rows:
            try:
                row.destroy()
            except Exception:  # noqa: BLE001
                pass
        self._tree_rows.clear()
        if not tasks:
            empty = ctk.CTkLabel(
                self._tree_scroll,
                text="No active DAG",
                text_color=T.MUTED,
                font=ctk.CTkFont(size=11),
                anchor="w",
            )
            empty.pack(fill="x", padx=6, pady=4)
            self._tree_rows.append(empty)  # type: ignore[arg-type]
            return
        for task in tasks:
            tid = task.get("task_id")
            action = str(task.get("action") or "task")
            status = str(task.get("status") or "pending").lower()
            deps = task.get("dependencies") or []
            indent = "  " * min(3, len(list(deps)))
            color = _STATUS_COLORS.get(status, T.TEXT)
            row = ctk.CTkFrame(self._tree_scroll, fg_color="transparent")
            row.pack(fill="x", padx=2, pady=1)
            ctk.CTkLabel(
                row,
                text=f"{indent}#{tid} [{status}] {action}",
                text_color=color,
                font=ctk.CTkFont(size=11),
                anchor="w",
            ).pack(fill="x", padx=4)
            summary = str(task.get("summary") or task.get("error") or "").strip()
            if summary:
                ctk.CTkLabel(
                    row,
                    text=f"{indent}  → {summary[:100]}",
                    text_color=T.MUTED,
                    font=ctk.CTkFont(size=10),
                    anchor="w",
                ).pack(fill="x", padx=4)
            self._tree_rows.append(row)

    def refresh(self) -> None:
        """Drain the bus and refresh tree + micro-log."""
        bus = self._resolve_bus()
        if bus is None:
            return
        events = []
        try:
            events = bus.drain(max_items=300)
        except Exception:  # noqa: BLE001
            events = []

        for ev in events:
            kind = getattr(ev, "kind", "")
            payload = getattr(ev, "payload", {}) or {}
            if kind in {"tool", "tool_call"}:
                msg = str(payload.get("message") or "")
                tool = str(payload.get("tool") or "")
                worker = payload.get("worker")
                if kind == "tool_call" and tool and not msg.startswith(tool):
                    prefix = f"[worker {worker}] " if worker is not None else ""
                    self._append_log(f"{prefix}{tool}: {msg}")
                elif msg:
                    self._append_log(msg)
            elif kind == "worker_start":
                tid = payload.get("task_id")
                action = str(payload.get("action") or "")
                self._append_log(f"[worker {tid}] start {action}".strip())
            elif kind == "worker_finish":
                tid = payload.get("task_id")
                status = str(payload.get("status") or "completed")
                summary = str(payload.get("summary") or "")[:120]
                self._append_log(f"[worker {tid}] {status}: {summary}".strip())
            elif kind == "supervisor_plan":
                tasks = list(payload.get("tasks") or [])
                if len(tasks) >= 2 and not self._multistep_announced:
                    self._multistep_announced = True
                    self._append_log(f"Supervisor plan: {len(tasks)} tasks")
                    if self._on_multistep is not None:
                        try:
                            self._on_multistep()
                        except Exception:  # noqa: BLE001
                            pass
            elif kind in {"graph_error", "graph_failed"}:
                node = str(payload.get("node") or "")
                err_type = str(payload.get("error_type") or "Error")
                message = str(payload.get("message") or "graph failure")
                dump = str(payload.get("dump_path") or "")
                line = f"[GRAPH ERROR] {node}: {err_type}: {message}"
                if dump:
                    line = f"{line} (dump={dump})"
                self._append_log(line, color=T.ROSE)
                self._set_status(f"failed · {err_type}")
                try:
                    self._summary_lbl.configure(
                        text=line[:220],
                        text_color=T.ROSE,
                    )
                except Exception:  # noqa: BLE001
                    pass
            elif kind == "done":
                status = str(payload.get("status") or "completed")
                self._announce_complete(status, bus)

        try:
            snap = dict(bus.latest_dag or {})
        except Exception:  # noqa: BLE001
            snap = {}
        tasks = list(snap.get("tasks") or [])
        self._render_tree(tasks)
        self._render_log()
        self._refresh_workspace()

        if len(tasks) >= 2 and not self._multistep_announced:
            self._multistep_announced = True
            if self._on_multistep is not None:
                try:
                    self._on_multistep()
                except Exception:  # noqa: BLE001
                    pass

        try:
            latest_err = dict(getattr(bus, "latest_error", None) or {})
        except Exception:  # noqa: BLE001
            latest_err = {}
        if latest_err and not tasks:
            err_msg = str(latest_err.get("message") or "graph failure")[:120]
            self._set_status(f"failed · {err_msg}")
        else:
            status = str(snap.get("status") or getattr(bus, "status", "") or "idle")
            if tasks:
                done = sum(
                    1
                    for t in tasks
                    if str(t.get("status") or "").lower()
                    in {"completed", "failed", "error"}
                )
                self._set_status(f"{status} · {done}/{len(tasks)}")
            else:
                self._set_status(status or "Idle")

        if status.lower() in {"completed", "failed", "done", "end"} and tasks:
            self._announce_complete(status, bus)

    def _announce_complete(self, status: str, bus: Any) -> None:
        if self._completion_announced:
            return
        self._completion_announced = True
        try:
            tasks = list((bus.latest_dag or {}).get("tasks") or [])
        except Exception:  # noqa: BLE001
            tasks = []
        completed = sum(
            1 for t in tasks if str(t.get("status") or "").lower() == "completed"
        )
        failed = sum(
            1
            for t in tasks
            if str(t.get("status") or "").lower() in {"failed", "error"}
        )
        summary = (
            f"Graph END · status={status} · "
            f"{completed} completed, {failed} failed, {len(tasks)} total"
        )
        self._summary_text = summary
        try:
            self._summary_lbl.configure(text=summary)
        except Exception:  # noqa: BLE001
            pass
        self._append_log(summary)
        if self._on_complete is not None:
            try:
                self._on_complete(summary)
            except Exception:  # noqa: BLE001
                pass

    def _poll(self) -> None:
        try:
            if not self.winfo_exists():
                return
        except Exception:  # noqa: BLE001
            return
        try:
            self.refresh()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.after(self._poll_ms, self._poll)
        except Exception:  # noqa: BLE001
            pass


__all__ = ("DagMonitorView",)
