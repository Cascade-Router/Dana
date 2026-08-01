"""Thread-safe STATE_CHANGE bus for VAD / supervisor status indicators.

Workers call ``emit_state_change`` (non-blocking). The CustomTkinter dashboard
drains via ``drain_state_changes`` on the Tk main thread. Headless-safe: no Tk
imports; queue drops are silent.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any, Literal

StateStatus = Literal["idle", "listening", "routing", "executing"]

_VALID = frozenset({"idle", "listening", "routing", "executing"})

_TOOL_LABELS: dict[str, str] = {
    "execute_powershell": "PowerShell",
    "shell_execute": "PowerShell",
    "run_terminal_command": "PowerShell",
    "analyze_visual_context": "Vision",
    "ocr_with_region": "OCR",
    "draft_cursor_prompt": "Ticket Draft",
    "web_search": "Web Search",
    "dispatch_research_swarm": "Research Swarm",
    "dispatch_jason_supervisor": "Jason",
    "dispatch_watchdog": "Watchdog",
}


class StatusEventBus:
    """Process-wide singleton queue for STATE_CHANGE events."""

    _instance: StatusEventBus | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=128)
        self._snap_lock = threading.Lock()
        self._snapshot: dict[str, Any] = {
            "event": "STATE_CHANGE",
            "status": "idle",
            "tool": "",
            "message": "",
            "ts": 0.0,
        }

    @classmethod
    def instance(cls) -> StatusEventBus:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def emit(
        self,
        status: str,
        *,
        tool: str = "",
        message: str = "",
    ) -> dict[str, Any]:
        st = str(status or "idle").strip().lower()
        if st not in _VALID:
            st = "idle"
        payload = {
            "event": "STATE_CHANGE",
            "status": st,
            "tool": str(tool or "").strip(),
            "message": str(message or "").strip(),
            "ts": time.time(),
        }
        with self._snap_lock:
            prev = (
                self._snapshot.get("status"),
                self._snapshot.get("tool"),
            )
            cur = (payload["status"], payload["tool"])
            if prev == cur:
                return dict(self._snapshot)
            self._snapshot = dict(payload)
        try:
            self._q.put_nowait(payload)
        except queue.Full:
            try:
                _ = self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(payload)
            except queue.Full:
                pass
        except Exception:  # noqa: BLE001
            pass
        return payload

    def drain(self, *, max_items: int = 32) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for _ in range(max(1, int(max_items))):
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                break
        return out

    def snapshot(self) -> dict[str, Any]:
        with self._snap_lock:
            return dict(self._snapshot)


def get_status_bus() -> StatusEventBus:
    return StatusEventBus.instance()


def emit_state_change(
    status: str,
    *,
    tool: str = "",
    message: str = "",
) -> dict[str, Any]:
    """Emit a STATE_CHANGE event (always thread-safe / headless-safe)."""
    return get_status_bus().emit(status, tool=tool, message=message)


def drain_state_changes(*, max_items: int = 32) -> list[dict[str, Any]]:
    return get_status_bus().drain(max_items=max_items)


def status_snapshot() -> dict[str, Any]:
    return get_status_bus().snapshot()


def friendly_tool_label(tool: str) -> str:
    key = str(tool or "").strip()
    if not key:
        return "Tool"
    if key in _TOOL_LABELS:
        return _TOOL_LABELS[key]
    # execute_foo_bar → Foo Bar
    pretty = key.replace("_", " ").strip()
    if pretty.lower().startswith("execute "):
        pretty = pretty[8:].strip()
    return pretty.title() if pretty else "Tool"


def format_system_status_line(
    status: str,
    *,
    tool: str = "",
    message: str = "",
) -> str:
    """Human label for the System Status line above chat input."""
    st = str(status or "").strip().lower()
    if message:
        return str(message)
    if st == "routing":
        return "Supervisor Routing..."
    if st == "executing":
        return f"Executing {friendly_tool_label(tool)}..."
    if st == "listening":
        return "Listening..."
    return ""


__all__ = (
    "StatusEventBus",
    "drain_state_changes",
    "emit_state_change",
    "format_system_status_line",
    "friendly_tool_label",
    "get_status_bus",
    "status_snapshot",
)
