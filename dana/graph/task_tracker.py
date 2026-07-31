"""Task state tracker — surfaces dropped / ghosted turns instead of silent END."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Lifecycle states for a Dana ReAct turn."""

    RECEIVED = "RECEIVED"
    IN_PROGRESS = "IN_PROGRESS"
    TOOL_EXECUTING = "TOOL_EXECUTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    DROPPED = "DROPPED"


_TERMINAL = frozenset(
    {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.DROPPED}
)

# Human-readable tool activity lines for the GUI Task Tracker timeline.
_TOOL_ACTIVITY_LABELS: dict[str, str] = {
    "python_repl": "Running Python REPL",
    "execute_powershell": "Running PowerShell",
    "web_search": "Grounding web search",
    "analyze_visual_context": "Analyzing visual context",
    "ocr_with_region": "Running OCR on region",
    "file_editor": "Editing file",
    "nav_and_click": "Navigating UI element",
    "draft_cursor_prompt": "Drafting Cursor handoff",
    "architect_new_tool": "Architecting new tool",
    "read_vault_memory": "Reading vault memory",
    "read_local_file": "Reading local file",
    "spreadsheet_query": "Querying spreadsheet",
}


@dataclass
class TaskRecord:
    task_id: str
    prompt: str
    status: TaskStatus = TaskStatus.RECEIVED
    metadata: dict[str, Any] = field(default_factory=dict)
    updated_at: str = ""
    activities: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.updated_at:
            self.updated_at = _utc_now()


@dataclass(frozen=True)
class ActivityEvent:
    """One human-readable timeline row for the GUI."""

    task_id: str
    message: str
    status: str
    timestamp: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _default_dropped_log_path() -> Path:
    try:
        from dana.paths import LOGS_DIR

        return Path(LOGS_DIR) / "dropped_tasks.log"
    except Exception:  # noqa: BLE001
        return Path("logs") / "dropped_tasks.log"


def humanize_activity(
    status: TaskStatus | str,
    metadata: dict[str, Any] | None = None,
    *,
    prompt: str = "",
) -> str:
    """Map a lifecycle status (+ optional tool metadata) to a short UI line."""
    if isinstance(status, TaskStatus):
        st = status
    else:
        try:
            st = TaskStatus(str(status).strip().upper())
        except ValueError:
            return str(status or "Activity")
    meta = dict(metadata or {})
    tool = str(meta.get("tool") or meta.get("tool_id") or "").strip()
    if st == TaskStatus.RECEIVED:
        preview = (prompt or "").strip()
        if preview:
            short = preview if len(preview) <= 64 else preview[:61] + "..."
            return f"Received: {short}"
        return "Task received"
    if st == TaskStatus.IN_PROGRESS:
        return "Working on request"
    if st == TaskStatus.TOOL_EXECUTING:
        if tool:
            return _TOOL_ACTIVITY_LABELS.get(tool, f"Running {tool}")
        return "Executing tool"
    if st == TaskStatus.COMPLETED:
        return "Completed"
    if st == TaskStatus.FAILED:
        err = str(meta.get("error") or meta.get("detail") or "").strip()
        return f"Failed: {err}" if err else "Failed"
    if st == TaskStatus.DROPPED:
        why = str(meta.get("drop_reason") or "").strip()
        return f"Dropped: {why}" if why else "Dropped"
    return st.value


_shared_tracker: TaskTracker | None = None
_shared_lock = threading.Lock()


def get_shared_task_tracker() -> TaskTracker:
    """Process-wide tracker shared by the ReAct graph and GUI (DI-friendly)."""
    global _shared_tracker
    with _shared_lock:
        if _shared_tracker is None:
            _shared_tracker = TaskTracker()
        return _shared_tracker


def set_shared_task_tracker(tracker: TaskTracker | None) -> None:
    """Tests / DI: replace or clear the shared tracker."""
    global _shared_tracker
    with _shared_lock:
        _shared_tracker = tracker


class TaskTracker:
    """Thread-safe in-memory task lifecycle tracker with dropped-task logging.

    ``dropped_log_path`` / ``ledger_path`` are injectable so unit tests never
    write into the production ``logs/`` or ``dana_security/patch_ledger.md``.
    """

    def __init__(
        self,
        *,
        dropped_log_path: Path | str | None = None,
        ledger_path: Path | str | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._tasks: dict[str, TaskRecord] = {}
        self._activity_log: list[ActivityEvent] = []
        self.dropped_log_path = (
            Path(dropped_log_path)
            if dropped_log_path is not None
            else _default_dropped_log_path()
        )
        self.ledger_path = Path(ledger_path) if ledger_path is not None else None

    def start_task(self, task_id: str, prompt: str) -> TaskRecord:
        """Register a new task in ``RECEIVED`` (idempotent overwrite)."""
        tid = str(task_id or "").strip() or "unknown"
        with self._lock:
            rec = TaskRecord(
                task_id=tid,
                prompt=str(prompt or ""),
                status=TaskStatus.RECEIVED,
            )
            self._tasks[tid] = rec
            self._append_activity_locked(
                tid,
                humanize_activity(TaskStatus.RECEIVED, prompt=rec.prompt),
                TaskStatus.RECEIVED,
            )
            return rec

    def update_status(
        self,
        task_id: str,
        status: TaskStatus | str,
        metadata: dict[str, Any] | None = None,
    ) -> TaskRecord:
        """Advance lifecycle status; merges optional metadata."""
        tid = str(task_id or "").strip() or "unknown"
        if isinstance(status, TaskStatus):
            new_status = status
        else:
            new_status = TaskStatus(str(status).strip().upper())
        with self._lock:
            rec = self._tasks.get(tid)
            if rec is None:
                rec = TaskRecord(task_id=tid, prompt="", status=TaskStatus.RECEIVED)
                self._tasks[tid] = rec
            rec.status = new_status
            rec.updated_at = _utc_now()
            if metadata:
                rec.metadata.update(dict(metadata))
            self._append_activity_locked(
                tid,
                humanize_activity(new_status, rec.metadata, prompt=rec.prompt),
                new_status,
                dict(rec.metadata),
            )
            return rec

    def append_activity(
        self,
        task_id: str,
        message: str,
        *,
        status: TaskStatus | str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ActivityEvent:
        """Push a custom human-readable activity line (GUI / tests)."""
        tid = str(task_id or "").strip() or "unknown"
        st = status
        if st is None:
            with self._lock:
                rec = self._tasks.get(tid)
                st = rec.status if rec is not None else TaskStatus.IN_PROGRESS
        if isinstance(st, TaskStatus):
            st_enum = st
        else:
            st_enum = TaskStatus(str(st).strip().upper())
        with self._lock:
            return self._append_activity_locked(
                tid, str(message or "").strip() or "Activity", st_enum, metadata
            )

    def _append_activity_locked(
        self,
        task_id: str,
        message: str,
        status: TaskStatus,
        metadata: dict[str, Any] | None = None,
    ) -> ActivityEvent:
        event = ActivityEvent(
            task_id=task_id,
            message=message,
            status=status.value,
            timestamp=_utc_now(),
            metadata=dict(metadata or {}),
        )
        self._activity_log.append(event)
        if len(self._activity_log) > 200:
            self._activity_log = self._activity_log[-200:]
        rec = self._tasks.get(task_id)
        if rec is not None:
            rec.activities.append(
                {
                    "message": event.message,
                    "status": event.status,
                    "timestamp": event.timestamp,
                }
            )
            if len(rec.activities) > 40:
                rec.activities = rec.activities[-40:]
        return event

    def get_task(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            return self._tasks.get(str(task_id or "").strip())

    def list_tasks(self, *, limit: int = 40) -> list[TaskRecord]:
        """Newest-first snapshot of tracked tasks (shallow copies)."""
        with self._lock:
            items = list(self._tasks.values())
        items.sort(key=lambda r: r.updated_at, reverse=True)
        return items[: max(1, int(limit))]

    def list_activities(self, *, limit: int = 80) -> list[ActivityEvent]:
        """Newest-first activity timeline for the GUI."""
        with self._lock:
            tail = list(self._activity_log[-max(1, int(limit)) :])
        tail.reverse()
        return tail

    def log_dropped_task(
        self,
        task_id: str,
        reason: str,
        last_state_buffer: dict[str, Any] | None = None,
        *,
        draft_ledger: bool = True,
    ) -> dict[str, Any]:
        """Append a dropped-task line and optionally draft a ``[PENDING]`` ticket.

        Log format (one line): ``Timestamp | Task ID | Prompt | Reason``

        Set ``draft_ledger=False`` for chat soft-drops that should only hit
        ``dropped_tasks.log`` (avoid spamming the production patch ledger).
        """
        tid = str(task_id or "").strip() or "unknown"
        why = str(reason or "").strip() or "incomplete trajectory"
        with self._lock:
            rec = self._tasks.get(tid)
            prompt = rec.prompt if rec is not None else ""
            if rec is not None:
                rec.status = TaskStatus.DROPPED
                rec.updated_at = _utc_now()
                rec.metadata["drop_reason"] = why
                if last_state_buffer is not None:
                    rec.metadata["last_state_buffer"] = last_state_buffer
                prompt = rec.prompt
            else:
                prompt = str(
                    (last_state_buffer or {}).get("prompt")
                    or (last_state_buffer or {}).get("user_text")
                    or ""
                )
                self._tasks[tid] = TaskRecord(
                    task_id=tid,
                    prompt=prompt,
                    status=TaskStatus.DROPPED,
                    metadata={
                        "drop_reason": why,
                        "last_state_buffer": last_state_buffer or {},
                    },
                )
            self._append_activity_locked(
                tid,
                humanize_activity(
                    TaskStatus.DROPPED,
                    {"drop_reason": why},
                    prompt=prompt,
                ),
                TaskStatus.DROPPED,
                {"drop_reason": why},
            )

        stamp = _utc_now()
        line = f"{stamp} | {tid} | {prompt} | {why}\n"
        log_path = self.dropped_log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()

        ticket_meta: dict[str, Any] = {"ok": False, "skipped": True}
        if draft_ledger:
            ticket_meta = self._draft_dropped_ticket(
                task_id=tid,
                prompt=prompt,
                reason=why,
                last_state_buffer=last_state_buffer,
                stamp=stamp,
            )
        return {
            "ok": True,
            "task_id": tid,
            "log_path": str(log_path),
            "ledger": ticket_meta,
        }

    def _draft_dropped_ticket(
        self,
        *,
        task_id: str,
        prompt: str,
        reason: str,
        last_state_buffer: dict[str, Any] | None,
        stamp: str,
    ) -> dict[str, Any]:
        """Draft a ``[PENDING]`` ticket via ``append_pending_ticket`` (injectable path)."""
        try:
            from dana_security.ledger_writer import (
                append_pending_ticket,
                format_escalation_ticket,
            )

            buf = last_state_buffer or {}
            trace_bits = [
                f"prompt={prompt!r}",
                f"reason={reason}",
            ]
            if buf:
                trace_bits.append(f"last_state_buffer={buf!r}")
            ticket = format_escalation_ticket(
                task_id=f"drop_{task_id}",
                error_trace="\n".join(trace_bits),
                recommended_fix=(
                    "Investigate incomplete ReAct trajectory; ensure completion "
                    "gate blocks END while pending_synthesis / unresolved tools."
                ),
                objective=f"DROPPED task: {reason}",
                timestamp=stamp,
            )
            dest = append_pending_ticket(ticket, ledger_path=self.ledger_path)
            return {"ok": True, "ledger_path": str(dest)}
        except Exception as exc:  # noqa: BLE001 — never crash the graph
            logger.warning(
                "task_tracker: failed to draft dropped ticket id=%s (%s: %s)",
                task_id,
                type(exc).__name__,
                exc,
            )
            return {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }


__all__ = (
    "ActivityEvent",
    "TaskRecord",
    "TaskStatus",
    "TaskTracker",
    "get_shared_task_tracker",
    "humanize_activity",
    "set_shared_task_tracker",
)
