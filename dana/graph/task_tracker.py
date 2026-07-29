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


@dataclass
class TaskRecord:
    task_id: str
    prompt: str
    status: TaskStatus = TaskStatus.RECEIVED
    metadata: dict[str, Any] = field(default_factory=dict)
    updated_at: str = ""

    def __post_init__(self) -> None:
        if not self.updated_at:
            self.updated_at = _utc_now()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _default_dropped_log_path() -> Path:
    try:
        from dana.paths import LOGS_DIR

        return Path(LOGS_DIR) / "dropped_tasks.log"
    except Exception:  # noqa: BLE001
        return Path("logs") / "dropped_tasks.log"


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
            return rec

    def get_task(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            return self._tasks.get(str(task_id or "").strip())

    def log_dropped_task(
        self,
        task_id: str,
        reason: str,
        last_state_buffer: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append a dropped-task line and draft a ``[PENDING]`` ledger ticket.

        Log format (one line): ``Timestamp | Task ID | Prompt | Reason``
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

        stamp = _utc_now()
        line = f"{stamp} | {tid} | {prompt} | {why}\n"
        log_path = self.dropped_log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()

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
    "TaskRecord",
    "TaskStatus",
    "TaskTracker",
)
