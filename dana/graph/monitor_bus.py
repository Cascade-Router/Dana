"""Thread-safe event bus bridging LangGraph streams → Textual / Control Dashboard."""

from __future__ import annotations

import logging
import queue
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

# Low-level kinds (TUI) + semantic aliases requested by the Control Dashboard.
EventKind = Literal[
    "dag",
    "tool",
    "telemetry",
    "status",
    "done",
    "supervisor_plan",
    "worker_start",
    "tool_call",
    "worker_finish",
    "graph_error",
    "graph_failed",
]

_logger = logging.getLogger("dana.graph.monitor_bus")


@dataclass
class MonitorEvent:
    kind: EventKind
    payload: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


class MonitorBus:
    """Fan-out queue for DAG / tool / telemetry updates."""

    def __init__(self, *, maxsize: int = 2000) -> None:
        self._q: queue.Queue[MonitorEvent] = queue.Queue(maxsize=maxsize)
        self._lock = threading.Lock()
        self._latest_dag: dict[str, Any] = {}
        self._latest_telemetry: dict[str, Any] = {}
        self._latest_error: dict[str, Any] = {}
        self._status: str = "idle"
        self._subscribers: list[Any] = []

    def subscribe(self, callback: Any) -> None:
        """Register ``callback(event: MonitorEvent)`` (best-effort, non-blocking)."""
        with self._lock:
            if callback not in self._subscribers:
                self._subscribers.append(callback)

    def unsubscribe(self, callback: Any) -> None:
        with self._lock:
            self._subscribers = [c for c in self._subscribers if c is not callback]

    def publish(self, kind: EventKind, **payload: Any) -> None:
        ev = MonitorEvent(kind=kind, payload=dict(payload))
        with self._lock:
            if kind in {"dag", "supervisor_plan"}:
                # Keep a unified snapshot for tree widgets.
                merged = dict(self._latest_dag)
                merged.update(payload)
                if kind == "supervisor_plan" and "tasks" in payload:
                    merged["tasks"] = list(payload.get("tasks") or [])
                self._latest_dag = merged
            elif kind == "telemetry":
                self._latest_telemetry = dict(payload)
            elif kind in {"status", "done"}:
                self._status = str(payload.get("status") or self._status)
            elif kind in {"graph_error", "graph_failed"}:
                self._latest_error = dict(payload)
                self._status = "failed"
            subs = list(self._subscribers)
        try:
            self._q.put_nowait(ev)
        except queue.Full:
            try:
                _ = self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(ev)
            except queue.Full:
                pass
        for cb in subs:
            try:
                cb(ev)
            except Exception:  # noqa: BLE001
                pass

    def drain(self, *, max_items: int = 200) -> list[MonitorEvent]:
        out: list[MonitorEvent] = []
        for _ in range(max(1, int(max_items))):
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                break
        return out

    @property
    def latest_dag(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._latest_dag)

    @property
    def latest_telemetry(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._latest_telemetry)

    @property
    def latest_error(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._latest_error)

    @property
    def status(self) -> str:
        with self._lock:
            return self._status


_BUS: MonitorBus | None = None
_BUS_LOCK = threading.Lock()


def get_monitor_bus(*, create: bool = True) -> MonitorBus | None:
    global _BUS
    with _BUS_LOCK:
        if _BUS is None and create:
            _BUS = MonitorBus()
        return _BUS


def reset_monitor_bus() -> MonitorBus:
    global _BUS
    with _BUS_LOCK:
        _BUS = MonitorBus()
        return _BUS


def broker_crash_dump_path() -> Path:
    try:
        from dana.paths import LOGS_DIR

        return Path(LOGS_DIR) / "broker_crash_dump.txt"
    except Exception:  # noqa: BLE001
        return Path("logs") / "broker_crash_dump.txt"


def write_broker_crash_dump(
    exc: BaseException | None = None,
    *,
    context: str = "",
    traceback_text: str | None = None,
    tasks_json: Any = None,
    raw_llm_output: str | None = None,
) -> Path:
    """Persist a Meta-Broker crash dump under ``logs/broker_crash_dump.txt``."""
    import json as _json

    path = broker_crash_dump_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tb = traceback_text
    if tb is None and exc is not None:
        tb = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
    if tb is None:
        tb = traceback.format_exc()
    stamp = time.strftime("%Y-%m-%d %H:%M:%S %z")
    if not (tb or "").strip() or (tb or "").strip() == "NoneType: None":
        tb = "(no exception traceback — soft failure / status=failed)\n"
    tasks_block = ""
    if tasks_json is not None:
        try:
            pretty = _json.dumps(tasks_json, indent=2, ensure_ascii=False, default=str)
        except Exception:  # noqa: BLE001
            pretty = repr(tasks_json)
        tasks_block = f"\n--- generated tasks JSON ---\n{pretty[:12000]}\n"
    raw_block = ""
    if raw_llm_output:
        raw_block = (
            f"\n--- raw LLM output ---\n{str(raw_llm_output)[:12000]}\n"
        )
    body = (
        f"=== Meta-Broker crash dump @ {stamp} ===\n"
        f"context: {context or '(none)'}\n"
        f"exception: {type(exc).__name__ if exc else 'n/a'}: {exc}\n\n"
        f"{tb}\n"
        f"{tasks_block}"
        f"{raw_block}"
    )
    try:
        path.write_text(body, encoding="utf-8")
    except Exception as write_exc:  # noqa: BLE001
        _logger.error("failed to write broker crash dump: %s", write_exc)
    return path


def publish_graph_error(
    message: str,
    *,
    exc: BaseException | None = None,
    node: str = "",
    dump: bool = True,
    terminal: bool = True,
    **extra: Any,
) -> None:
    """Publish ``graph_error`` (+ ``graph_failed``) and optionally write a dump.

    When ``terminal`` is false (soft mid-graph failure), skip the ``done`` event
    so the DAG monitor does not treat the run as finished while repair continues.
    """
    tb = ""
    if exc is not None:
        tb = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        _logger.exception("graph_error node=%s: %s", node, message)
    else:
        _logger.error("graph_error node=%s: %s", node, message)
    dump_path = ""
    tasks_json = extra.pop("tasks_json", None)
    raw_llm_output = extra.pop("raw_llm_output", None)
    if dump:
        try:
            dump_path = str(
                write_broker_crash_dump(
                    exc,
                    context=f"node={node}; {message}",
                    traceback_text=tb or None,
                    tasks_json=tasks_json,
                    raw_llm_output=(
                        str(raw_llm_output) if raw_llm_output is not None else None
                    ),
                )
            )
        except Exception:  # noqa: BLE001
            dump_path = ""
    bus = get_monitor_bus(create=True)
    assert bus is not None
    payload = {
        "message": str(message)[:2000],
        "node": node,
        "error_type": type(exc).__name__ if exc else "Error",
        "traceback": (tb or "")[:8000],
        "dump_path": dump_path,
        **extra,
    }
    if tasks_json is not None:
        try:
            payload["tasks_json"] = tasks_json
        except Exception:  # noqa: BLE001
            pass
    bus.publish("graph_error", **payload)
    bus.publish("graph_failed", **payload)
    bus.publish("status", status="failed")
    bus.publish(
        "tool",
        message=f"[GRAPH ERROR] {node}: {str(message)[:300]}",
    )
    if terminal:
        bus.publish("done", status="failed")


def publish_dag_snapshot(state: dict[str, Any] | None, *, node: str = "") -> None:
    bus = get_monitor_bus(create=False)
    if bus is None or not state:
        return
    tasks = []
    for t in state.get("dag") or []:
        if not isinstance(t, dict):
            continue
        tasks.append(
            {
                "task_id": t.get("task_id"),
                "action": t.get("action"),
                "status": t.get("status"),
                "dependencies": list(t.get("dependencies") or []),
                "summary": (t.get("summary") or "")[:160],
                "error": (t.get("error") or "")[:120],
            }
        )
    payload = {
        "node": node,
        "status": state.get("status"),
        "supervisor_cycles": state.get("supervisor_cycles"),
        "active_task_ids": list(state.get("active_task_ids") or []),
        "pending_tasks": list(state.get("pending_tasks") or []),
        "checkpoint_log": list(state.get("checkpoint_log") or [])[-5:],
        "tasks": tasks,
    }
    bus.publish("dag", **payload)
    # Semantic alias for Control Dashboard subscribers.
    if tasks and str(state.get("status") or "") in {
        "planning",
        "dispatching",
        "awaiting_workers",
        "evaluating",
    }:
        bus.publish("supervisor_plan", **payload)


def publish_supervisor_plan(tasks: list[dict[str, Any]], **extra: Any) -> None:
    bus = get_monitor_bus(create=True)
    assert bus is not None
    bus.publish("supervisor_plan", tasks=list(tasks), **extra)
    bus.publish("dag", tasks=list(tasks), **extra)


def publish_worker_start(task_id: int | str, *, action: str = "") -> None:
    bus = get_monitor_bus(create=False)
    if bus is None:
        return
    bus.publish("worker_start", task_id=task_id, action=action)
    bus.publish("tool", message=f"[worker {task_id}] start {action}".strip())


def publish_tool_call(
    message: str,
    *,
    worker: str | int | None = None,
    tool: str = "",
) -> None:
    bus = get_monitor_bus(create=False)
    if bus is None:
        return
    bus.publish(
        "tool_call",
        message=message,
        worker=worker,
        tool=tool,
    )
    prefix = f"[worker {worker}] " if worker is not None else ""
    tool_bit = f"{tool}: " if tool else ""
    bus.publish("tool", message=f"{prefix}{tool_bit}{message}")


def publish_worker_finish(
    task_id: int | str,
    *,
    status: str = "completed",
    summary: str = "",
) -> None:
    bus = get_monitor_bus(create=False)
    if bus is None:
        return
    bus.publish(
        "worker_finish",
        task_id=task_id,
        status=status,
        summary=summary,
    )
    bus.publish(
        "tool",
        message=f"[worker {task_id}] {status}: {(summary or '')[:120]}",
    )


def publish_tool_line(message: str, *, worker: str | int | None = None) -> None:
    bus = get_monitor_bus(create=False)
    if bus is None:
        return
    prefix = f"[worker {worker}] " if worker is not None else ""
    bus.publish("tool", message=f"{prefix}{message}")


def publish_telemetry(**fields: Any) -> None:
    bus = get_monitor_bus(create=False)
    if bus is None:
        return
    bus.publish("telemetry", **fields)


__all__ = (
    "MonitorBus",
    "MonitorEvent",
    "broker_crash_dump_path",
    "get_monitor_bus",
    "publish_dag_snapshot",
    "publish_graph_error",
    "publish_supervisor_plan",
    "publish_telemetry",
    "publish_tool_call",
    "publish_tool_line",
    "publish_worker_finish",
    "publish_worker_start",
    "reset_monitor_bus",
    "write_broker_crash_dump",
)
