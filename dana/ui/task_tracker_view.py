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
}


class TaskTrackerView(ctk.CTkFrame):
    """Scrollable activity timeline backed by an injectable ``TaskTracker``."""

    def __init__(
        self,
        master: Any,
        *,
        tracker: TaskTracker | None = None,
        tracker_factory: Callable[[], TaskTracker] | None = None,
        poll_ms: int = 400,
        max_rows: int = 48,
    ) -> None:
        super().__init__(master, fg_color=T.BG, corner_radius=12)
        self._tracker = tracker
        self._tracker_factory = tracker_factory or get_shared_task_tracker
        self._poll_ms = max(100, int(poll_ms))
        self._max_rows = max(8, int(max_rows))
        self._rows: list[ctk.CTkFrame] = []
        self._last_sig = ""
        self._build()
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
        header = ctk.CTkFrame(self, fg_color=T.CARD, corner_radius=10)
        header.pack(fill="x", padx=8, pady=(8, 4))
        ctk.CTkLabel(
            header,
            text="Task Tracker",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=T.TEXT,
            anchor="w",
        ).pack(side="left", padx=12, pady=8)
        self._empty_lbl = ctk.CTkLabel(
            header,
            text="No active tasks",
            font=ctk.CTkFont(size=11),
            text_color=T.MUTED,
            anchor="e",
        )
        self._empty_lbl.pack(side="right", padx=12, pady=8)

        self._scroll = ctk.CTkScrollableFrame(
            self,
            fg_color="transparent",
            corner_radius=0,
        )
        self._scroll.pack(fill="both", expand=True, padx=6, pady=(0, 8))

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

        sig = "|".join(
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
            if activities:
                self._empty_lbl.configure(text=f"{len(tasks)} task(s)")
            else:
                self._empty_lbl.configure(text="No active tasks")
        except Exception:  # noqa: BLE001
            pass

        # Prefer task cards when present; fall back to flat activity log.
        if tasks:
            for rec in tasks:
                self._rows.append(self._make_task_row(rec))
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

    def _make_task_row(self, rec: Any) -> ctk.CTkFrame:
        status_val = getattr(rec.status, "value", str(rec.status))
        label, color = self._pill_style(status_val)
        card = ctk.CTkFrame(
            self._scroll,
            fg_color=T.CARD,
            corner_radius=10,
            border_width=1,
            border_color=T.BORDER,
        )
        card.pack(fill="x", padx=4, pady=4)

        top = ctk.CTkFrame(card, fg_color="transparent")
        top.pack(fill="x", padx=10, pady=(8, 2))
        ctk.CTkLabel(
            top,
            text=f"  {label}  ",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=T.TEXT_ON_ACCENT,
            fg_color=color,
            corner_radius=999,
        ).pack(side="left")
        ctk.CTkLabel(
            top,
            text=str(getattr(rec, "updated_at", "") or ""),
            font=ctk.CTkFont(size=10),
            text_color=T.MUTED,
            anchor="e",
        ).pack(side="right")

        prompt = str(getattr(rec, "prompt", "") or "").strip()
        if prompt:
            short = prompt if len(prompt) <= 96 else prompt[:93] + "..."
            ctk.CTkLabel(
                card,
                text=short,
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=T.TEXT,
                anchor="w",
                wraplength=520,
                justify="left",
            ).pack(fill="x", padx=12, pady=(2, 2))

        activities = list(getattr(rec, "activities", None) or [])
        if not activities:
            msg = "Waiting…"
            if status_val == TaskStatus.COMPLETED.value:
                msg = "Completed"
            elif status_val == TaskStatus.FAILED.value:
                msg = "Failed"
            ctk.CTkLabel(
                card,
                text=f"• {msg}",
                font=ctk.CTkFont(size=11),
                text_color=T.MUTED,
                anchor="w",
            ).pack(fill="x", padx=14, pady=(0, 8))
        else:
            for act in activities[-4:]:
                line = str(act.get("message") or "").strip() or "Activity"
                ts = str(act.get("timestamp") or "").strip()
                step = f"• {line}" + (f"  ·  {ts}" if ts else "")
                ctk.CTkLabel(
                    card,
                    text=step,
                    font=ctk.CTkFont(size=11),
                    text_color=T.MUTED,
                    anchor="w",
                    wraplength=500,
                    justify="left",
                ).pack(fill="x", padx=14, pady=0)
            ctk.CTkFrame(card, fg_color="transparent", height=6).pack()
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
        ctk.CTkLabel(
            row,
            text=f"  {label}  ",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=T.TEXT_ON_ACCENT,
            fg_color=color,
            corner_radius=999,
        ).pack(side="left", padx=(8, 6), pady=8)
        body = ctk.CTkFrame(row, fg_color="transparent")
        body.pack(side="left", fill="x", expand=True, padx=(0, 8), pady=4)
        ctk.CTkLabel(
            body,
            text=message,
            font=ctk.CTkFont(size=12),
            text_color=T.TEXT,
            anchor="w",
        ).pack(fill="x")
        ctk.CTkLabel(
            body,
            text=f"{task_id} · {timestamp}",
            font=ctk.CTkFont(size=10),
            text_color=T.MUTED,
            anchor="w",
        ).pack(fill="x")
        return row

    def _poll(self) -> None:
        if not self.winfo_exists():
            return
        try:
            self.refresh()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.after(self._poll_ms, self._poll)
        except Exception:  # noqa: BLE001
            pass


__all__ = ("TaskTrackerView",)
