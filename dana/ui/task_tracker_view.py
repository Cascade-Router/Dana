"""Human-readable Task Tracker timeline for the Dana control dashboard."""

from __future__ import annotations

from typing import Any, Callable

import customtkinter as ctk

from dana.graph.task_tracker import (
    TaskStatus,
    TaskTracker,
    get_shared_task_tracker,
)
from dana.ui import theme as T

_STATUS_PILLS: dict[str, tuple[str, str]] = {
    TaskStatus.RECEIVED.value: ("RECEIVED", T.BORDER),
    TaskStatus.IN_PROGRESS.value: ("IN PROGRESS", T.AMBER),
    TaskStatus.TOOL_EXECUTING.value: ("IN PROGRESS", T.AMBER),
    TaskStatus.COMPLETED.value: ("COMPLETED", T.EMERALD),
    TaskStatus.FAILED.value: ("FAILED", T.ROSE),
    TaskStatus.DROPPED.value: ("FAILED", T.ROSE),
    "ABORTED": ("ABORTED", T.ROSE),
}


def _detail_mono_font() -> ctk.CTkFont:
    """Prefer JetBrains Mono / Fira Code; fall back to Consolas."""
    for family in ("JetBrains Mono", "Fira Code", "Cascadia Mono", "Consolas"):
        try:
            return ctk.CTkFont(family=family, size=11)
        except Exception:  # noqa: BLE001
            continue
    return ctk.CTkFont(size=11)


def _pill_width(label: str) -> int:
    """Enough horizontal room so status text never clips inside the badge."""
    return max(96, len(label) * 8 + 24)


class TaskTrackerView(ctk.CTkFrame):
    """Scrollable activity timeline with a full-width Task Detail panel below."""

    def __init__(
        self,
        master: Any,
        *,
        tracker: TaskTracker | None = None,
        tracker_factory: Callable[[], TaskTracker] | None = None,
        poll_ms: int = 200,
        max_rows: int = 48,
        show_header: bool = True,
        status_label: Any | None = None,
        external_tick: bool = False,
    ) -> None:
        super().__init__(master, fg_color=T.BG, corner_radius=12)
        self._tracker = tracker
        self._tracker_factory = tracker_factory or get_shared_task_tracker
        self._poll_ms = max(100, int(poll_ms))
        self._external_tick = bool(external_tick)
        self._max_rows = max(8, int(max_rows))
        self._show_header = bool(show_header)
        self._rows: list[ctk.CTkFrame] = []
        self._last_sig = ""
        self._empty_lbl = status_label
        self._selected_id: str | None = None
        self._build()
        if not self._external_tick:
            self.after(self._poll_ms, self._poll)

    def _resolve_tracker(self) -> TaskTracker:
        if self._tracker is not None:
            return self._tracker
        return self._tracker_factory()

    def set_tracker(self, tracker: TaskTracker | None) -> None:
        """Tests / DI: swap the backing tracker and force a redraw."""
        self._tracker = tracker
        self._last_sig = ""
        self.refresh()

    def _build(self) -> None:
        if self._show_header:
            header = ctk.CTkFrame(self, fg_color=T.CARD, corner_radius=10)
            header.pack(fill="x", padx=8, pady=(8, 4))
            ctk.CTkLabel(
                header,
                text="Task Tracker",
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color=T.TEXT,
                anchor="w",
            ).pack(side="left", padx=12, pady=8)
            if self._empty_lbl is None:
                self._empty_lbl = ctk.CTkLabel(
                    header,
                    text="No active tasks",
                    font=ctk.CTkFont(size=11),
                    text_color=T.MUTED,
                    anchor="e",
                )
                self._empty_lbl.pack(side="right", padx=12, pady=8)

        # Full-width stack: task cards on top, Task Detail below (uses sidebar width).
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=6, pady=(0, 8))
        try:
            body.grid_columnconfigure(0, weight=1)
            body.grid_rowconfigure(0, weight=1)
            body.grid_rowconfigure(1, weight=1)
        except Exception:  # noqa: BLE001
            pass

        left = ctk.CTkFrame(body, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", pady=(0, 4))
        self._scroll = ctk.CTkScrollableFrame(
            left,
            fg_color="transparent",
            corner_radius=0,
        )
        self._scroll.pack(fill="both", expand=True)

        right = ctk.CTkFrame(body, fg_color=T.CARD, corner_radius=8)
        right.grid(row=1, column=0, sticky="nsew", pady=(4, 0))
        ctk.CTkLabel(
            right,
            text="Task detail",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=T.MUTED,
            anchor="w",
        ).pack(fill="x", padx=12, pady=(8, 4))
        self._detail = ctk.CTkTextbox(
            right,
            fg_color=T.BG,
            text_color=T.TEXT,
            font=_detail_mono_font(),
            wrap="word",
            activate_scrollbars=True,
        )
        self._detail.pack(fill="both", expand=True, padx=12, pady=(0, 10))
        try:
            self._detail.configure(state="disabled")
        except Exception:  # noqa: BLE001
            pass
        self._set_detail(
            "Click a task card to inspect prompt, status, and activity payload."
        )

    def _set_detail(self, text: str) -> None:
        try:
            self._detail.configure(state="normal")
            self._detail.delete("1.0", "end")
            self._detail.insert("1.0", text)
            self._detail.configure(state="disabled")
        except Exception:  # noqa: BLE001
            pass

    def _select_task(self, rec: Any) -> None:
        tid = str(getattr(rec, "task_id", "") or "")
        self._selected_id = tid
        status_val = getattr(rec.status, "value", str(rec.status))
        prompt = str(getattr(rec, "prompt", "") or "").strip()
        # Avoid dumping enormous /broker macros twice in the UI — truncate.
        if len(prompt) > 600:
            prompt = prompt[:600] + "\n…[truncated]"
        activities = list(getattr(rec, "activities", None) or [])
        lines = [
            f"task_id: {tid}",
            f"status:  {status_val}",
            f"updated: {getattr(rec, 'updated_at', '')}",
            "",
            "── prompt ──",
            prompt or "(none)",
            "",
            "── activity ──",
        ]
        if not activities:
            lines.append("(no activity yet)")
        else:
            for act in activities[-12:]:
                msg = str(act.get("message") or "").strip()
                ts = str(act.get("timestamp") or "").strip()
                lines.append(f"[{ts}] {msg}" if ts else msg)
        meta = getattr(rec, "metadata", None) or {}
        if meta:
            lines.extend(["", "── metadata ──", str(meta)])
        self._set_detail("\n".join(lines))

    def refresh(self) -> None:
        """Rebuild timeline rows from the current tracker snapshot."""
        try:
            tracker = self._resolve_tracker()
        except Exception:  # noqa: BLE001
            return
        try:
            activities = tracker.list_activities(limit=self._max_rows)
            tasks = tracker.list_tasks(limit=12)
        except Exception:  # noqa: BLE001
            return

        sig = f"{tracker.revision()}|" + "|".join(
            f"{a.timestamp}:{a.task_id}:{a.status}:{a.message}" for a in activities
        )
        if sig == self._last_sig:
            return
        self._last_sig = sig

        for row in self._rows:
            try:
                row.destroy()
            except Exception:  # noqa: BLE001
                pass
        self._rows.clear()

        try:
            lbl = self._empty_lbl
            if lbl is not None:
                if activities:
                    lbl.configure(text=f"{len(tasks)} task(s)")
                else:
                    lbl.configure(text="No active tasks")
        except Exception:  # noqa: BLE001
            pass

        if tasks:
            selected = None
            for rec in tasks:
                self._rows.append(self._make_task_row(rec))
                if self._selected_id and str(rec.task_id) == self._selected_id:
                    selected = rec
            if selected is not None:
                self._select_task(selected)
            elif tasks and self._selected_id is None:
                self._select_task(tasks[0])
        else:
            for event in activities:
                self._rows.append(
                    self._make_activity_row(
                        message=event.message,
                        status=event.status,
                        timestamp=event.timestamp,
                        task_id=event.task_id,
                    )
                )

    def _pill_style(self, status: str) -> tuple[str, str]:
        key = str(status or "").strip().upper()
        return _STATUS_PILLS.get(key, (key or "UNKNOWN", T.MUTED))

    def _make_status_pill(self, parent: Any, label: str, color: str) -> ctk.CTkLabel:
        pill = ctk.CTkLabel(
            parent,
            text=label,
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=T.TEXT_ON_ACCENT,
            fg_color=color,
            corner_radius=8,
            width=_pill_width(label),
            height=22,
            anchor="center",
        )
        return pill

    def _make_task_row(self, rec: Any) -> ctk.CTkFrame:
        status_val = getattr(rec.status, "value", str(rec.status))
        label, color = self._pill_style(status_val)
        selected = self._selected_id == str(getattr(rec, "task_id", ""))
        card = ctk.CTkFrame(
            self._scroll,
            fg_color=T.CARD,
            corner_radius=10,
            border_width=2 if selected else 1,
            border_color=T.ACCENT if selected else T.BORDER,
        )
        card.pack(fill="x", padx=4, pady=4)

        def _on_click(_event: Any = None, record: Any = rec) -> None:
            self._select_task(record)
            self._last_sig = ""
            self.refresh()

        # Header row: status badge | title | timestamp (grid so badge never squeezes).
        top = ctk.CTkFrame(card, fg_color="transparent")
        top.pack(fill="x", padx=10, pady=(8, 2))
        try:
            top.grid_columnconfigure(0, weight=0)
            top.grid_columnconfigure(1, weight=1)
            top.grid_columnconfigure(2, weight=0)
        except Exception:  # noqa: BLE001
            pass

        pill = self._make_status_pill(top, label, color)
        pill.grid(row=0, column=0, sticky="w", padx=(0, 8), pady=2)

        tid = str(getattr(rec, "task_id", "") or "task")
        title_lbl = ctk.CTkLabel(
            top,
            text=tid,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=T.TEXT,
            anchor="w",
        )
        title_lbl.grid(row=0, column=1, sticky="ew", padx=(0, 8), pady=2)

        ts_lbl = ctk.CTkLabel(
            top,
            text=str(getattr(rec, "updated_at", "") or ""),
            font=ctk.CTkFont(size=10),
            text_color=T.MUTED,
            anchor="e",
        )
        ts_lbl.grid(row=0, column=2, sticky="e", pady=2)

        activities = list(getattr(rec, "activities", None) or [])
        preview = ""
        if activities:
            preview = str(activities[-1].get("message") or "")[:120]
        elif status_val == TaskStatus.COMPLETED.value:
            preview = "Completed"
        elif status_val == TaskStatus.FAILED.value:
            preview = "Failed"
        else:
            preview = "Waiting… — click for details"
        preview_lbl = ctk.CTkLabel(
            card,
            text=f"• {preview}",
            font=ctk.CTkFont(size=11),
            text_color=T.MUTED,
            anchor="w",
            wraplength=360,
            justify="left",
        )
        preview_lbl.pack(fill="x", padx=14, pady=(0, 8))

        for widget in (card, top, pill, title_lbl, ts_lbl, preview_lbl):
            try:
                widget.bind("<Button-1>", _on_click)
            except Exception:  # noqa: BLE001
                pass
        try:
            ctk.CTkButton(
                card,
                text="Inspect",
                width=72,
                height=24,
                fg_color="transparent",
                border_width=1,
                border_color=T.BORDER,
                text_color=T.ACCENT,
                command=_on_click,
            ).pack(anchor="e", padx=10, pady=(0, 8))
        except Exception:  # noqa: BLE001
            pass
        return card

    def _make_activity_row(
        self,
        *,
        message: str,
        status: str,
        timestamp: str,
        task_id: str,
    ) -> ctk.CTkFrame:
        label, color = self._pill_style(status)
        row = ctk.CTkFrame(
            self._scroll,
            fg_color=T.CARD,
            corner_radius=8,
            border_width=1,
            border_color=T.BORDER,
        )
        row.pack(fill="x", padx=4, pady=3)
        try:
            row.grid_columnconfigure(0, weight=0)
            row.grid_columnconfigure(1, weight=1)
        except Exception:  # noqa: BLE001
            pass
        pill = self._make_status_pill(row, label, color)
        pill.grid(row=0, column=0, sticky="nw", padx=(8, 8), pady=8)
        body = ctk.CTkFrame(row, fg_color="transparent")
        body.grid(row=0, column=1, sticky="ew", padx=(0, 8), pady=4)
        ctk.CTkLabel(
            body,
            text=message,
            font=ctk.CTkFont(size=12),
            text_color=T.TEXT,
            anchor="w",
            wraplength=320,
            justify="left",
        ).pack(fill="x")
        ctk.CTkLabel(
            body,
            text=f"{task_id} · {timestamp}",
            font=ctk.CTkFont(size=10),
            text_color=T.MUTED,
            anchor="w",
        ).pack(fill="x")
        return row

    def tick(self) -> None:
        """One polling step — safe to call from an external scheduler.

        Drains tracker notifications then refreshes rows. Does not
        reschedule itself; that's the caller's job.
        """
        try:
            if not self.winfo_exists():
                return
        except Exception:  # noqa: BLE001
            return
        try:
            tracker = self._resolve_tracker()
            if hasattr(tracker, "drain_notifications"):
                notes: list = []
                for _ in range(4):
                    batch = tracker.drain_notifications(max_items=64)
                    if not batch:
                        break
                    notes.extend(batch)
                if notes:
                    self._last_sig = ""
        except Exception:  # noqa: BLE001
            pass
        try:
            self.refresh()
        except Exception:  # noqa: BLE001
            pass

    def _poll(self) -> None:
        """Self-scheduling wrapper around ``tick()`` (internal-poll mode only)."""
        if not self.winfo_exists():
            return
        self.tick()
        try:
            self.after(self._poll_ms, self._poll)
        except Exception:  # noqa: BLE001
            pass


__all__ = ("TaskTrackerView",)
