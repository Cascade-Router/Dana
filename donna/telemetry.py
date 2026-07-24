"""Donna telemetry: live dashboard + tagged JSONL structured logger.

Dashboard (``CAMGRASPER/dashboard.md``):
  A daemon thread refreshes the markdown table every ~45 seconds.

Tagged telemetry (``CAMGRASPER/logs/donna_telemetry.jsonl``):
  Queryable JSON lines with bureaucratic tags:
  ``[VOICE_ASR]``, ``[ROUTER]``, ``[REASONING_TRACE]``, ``[TOOL_EXECUTION]``, ``[HANDOFF]``.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from donna.paths import DASHBOARD_PATH, DONNA_WORKSPACE, LOGS_DIR

_LOCK = threading.Lock()
_STATUS = "Healthy"
_PID = os.getpid()
_RECENT_TOOLS: deque[str] = deque(maxlen=3)
_CASCADE_LAT_MS: float | None = None
_CASCADE_MODEL: str = ""
_CASCADE_OVER_THRESHOLD: bool = False
_DASHBOARD_THREAD: threading.Thread | None = None
_DASHBOARD_STOP = threading.Event()
DASHBOARD_INTERVAL_SEC = 45.0

# --- Tagged JSONL telemetry (Module 1) ---
TELEMETRY_JSONL_PATH: Path = LOGS_DIR / "donna_telemetry.jsonl"
TelemetryTag = Literal[
    "VOICE_ASR",
    "ROUTER",
    "REASONING_TRACE",
    "TOOL_EXECUTION",
    "HANDOFF",
    "SENSOR_VISION",
    "ACTUATOR_START",
    "ACTUATOR_DONE",
    "NOTIFICATION_TOAST",
    "NOTIFICATION_PIGGYBACK",
    "OPERATOR_NAV_CLICK_COMPLETE",
]
_VALID_TAGS = frozenset(
    {
        "VOICE_ASR",
        "ROUTER",
        "REASONING_TRACE",
        "TOOL_EXECUTION",
        "HANDOFF",
        "SENSOR_VISION",
        "ACTUATOR_START",
        "ACTUATOR_DONE",
        "NOTIFICATION_TOAST",
        "NOTIFICATION_PIGGYBACK",
        "OPERATOR_NAV_CLICK_COMPLETE",
    }
)
_JSONL_LOCK = threading.Lock()


def cascade_latency_threshold_ms() -> float:
    """Warn when a high-complexity DeepSeek call exceeds this many ms."""
    try:
        return max(
            1000.0,
            float(os.environ.get("DONNA_CASCADE_LATENCY_THRESHOLD_MS", "120000") or "120000"),
        )
    except ValueError:
        return 120000.0


def note_cascade_latency(latency_ms: float, *, model: str = "") -> None:
    """Record last high-complexity DeepSeek latency for ``dashboard.md``."""
    global _CASCADE_LAT_MS, _CASCADE_MODEL, _CASCADE_OVER_THRESHOLD
    try:
        ms = float(latency_ms)
    except (TypeError, ValueError):
        return
    thr = cascade_latency_threshold_ms()
    with _LOCK:
        _CASCADE_LAT_MS = ms
        _CASCADE_MODEL = (model or "").strip()[:80]
        _CASCADE_OVER_THRESHOLD = ms >= thr


def set_system_status(status: str) -> None:
    """Healthy | Intercepting | Restarting"""
    global _STATUS
    with _LOCK:
        _STATUS = (status or "Healthy").strip() or "Healthy"


def note_tool_event(label: str) -> None:
    text = (label or "").strip()
    if not text:
        return
    with _LOCK:
        _RECENT_TOOLS.appendleft(text[:120])


def _bug_counts() -> tuple[int, int]:
    try:
        from donna.bug_tracker import PENDING_STATUS, load_bug_tracker

        bugs = load_bug_tracker()
        pending = 0
        patched = 0
        for entry in bugs:
            st = str(entry.get("status") or PENDING_STATUS).upper()
            if st in ("PENDING", "OPEN"):
                pending += 1
            elif st == "PATCHED":
                patched += 1
        return pending, patched
    except Exception:  # noqa: BLE001
        return 0, 0


def _resolve_donna_pid(explicit: int | None = None) -> int:
    if explicit:
        return int(explicit)
    # Prefer the live singleton listener on :47474 over this process (monitor scripts
    # may call write_dashboard from a short-lived Python and must not overwrite PID).
    try:
        import socket

        # Windows: query via PowerShell-less netstat parse is heavy; use psutil if present.
        try:
            import psutil  # type: ignore

            for conn in psutil.net_connections(kind="inet"):
                if (
                    conn.laddr
                    and getattr(conn.laddr, "port", None) == 47474
                    and conn.status == "LISTEN"
                    and conn.pid
                ):
                    return int(conn.pid)
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        pass
    with _LOCK:
        return int(_PID or os.getpid())


def write_dashboard(
    *,
    status: str | None = None,
    pid: int | None = None,
) -> str:
    """Overwrite ``CAMGRASPER/dashboard.md`` with a clean status table."""
    DONNA_WORKSPACE.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        cur_status = status or _STATUS
        recent = list(_RECENT_TOOLS)
        cascade_ms = _CASCADE_LAT_MS
        cascade_model = _CASCADE_MODEL
        cascade_over = _CASCADE_OVER_THRESHOLD
    cur_pid = _resolve_donna_pid(pid)
    pending, patched = _bug_counts()
    tools_cell = ", ".join(recent) if recent else "—"
    thr = cascade_latency_threshold_ms()
    if cascade_ms is None:
        cascade_cell = "—"
    else:
        flag = " OVER THRESHOLD" if cascade_over else ""
        model_bit = f" `{cascade_model}`" if cascade_model else ""
        cascade_cell = f"{cascade_ms:.0f} ms{model_bit} (threshold {thr:.0f} ms){flag}"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    body = (
        "# Donna Live Telemetry\n\n"
        f"_Updated: {stamp}_\n\n"
        "| Field | Value |\n"
        "| --- | --- |\n"
        f"| System Status | {cur_status} |\n"
        f"| Active PID | `{cur_pid}` |\n"
        f"| Last 3 tools (executed / forged) | {tools_cell} |\n"
        f"| Last high-complexity DeepSeek latency | {cascade_cell} |\n"
        f"| Bugs PENDING | {pending} |\n"
        f"| Bugs PATCHED | {patched} |\n"
    )
    tmp = DASHBOARD_PATH.with_suffix(".tmp")
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(DASHBOARD_PATH)
    return str(DASHBOARD_PATH)


def _dashboard_loop() -> None:
    while not _DASHBOARD_STOP.wait(DASHBOARD_INTERVAL_SEC):
        try:
            write_dashboard()
        except Exception:  # noqa: BLE001
            pass


def start_dashboard_thread() -> None:
    """Start the 45s dashboard writer (idempotent)."""
    global _DASHBOARD_THREAD, _PID
    _PID = os.getpid()
    set_system_status("Healthy")
    try:
        write_dashboard()
    except Exception:  # noqa: BLE001
        pass
    if _DASHBOARD_THREAD is not None and _DASHBOARD_THREAD.is_alive():
        return
    _DASHBOARD_STOP.clear()
    _DASHBOARD_THREAD = threading.Thread(
        target=_dashboard_loop,
        name="DonnaDashboard",
        daemon=True,
    )
    _DASHBOARD_THREAD.start()


def stop_dashboard_thread() -> None:
    _DASHBOARD_STOP.set()


def snapshot() -> dict[str, Any]:
    with _LOCK:
        return {
            "status": _STATUS,
            "pid": _PID,
            "recent_tools": list(_RECENT_TOOLS),
            "cascade_latency_ms": _CASCADE_LAT_MS,
            "cascade_model": _CASCADE_MODEL,
            "cascade_over_threshold": _CASCADE_OVER_THRESHOLD,
        }


def _normalize_tag(tag: str) -> str:
    raw = (tag or "").strip().upper()
    raw = raw.strip("[]")
    if raw not in _VALID_TAGS:
        raise ValueError(
            f"Unknown telemetry tag {tag!r}; expected one of {sorted(_VALID_TAGS)}"
        )
    return raw


def emit_tagged(
    tag: str,
    message: str = "",
    *,
    session_id: str = "",
    current_agent: str = "",
    active_intent: str = "",
    payload: dict[str, Any] | None = None,
    latency_ms: float | None = None,
) -> dict[str, Any]:
    """Append one structured JSON line to ``logs/donna_telemetry.jsonl``.

    Tags are written as ``[VOICE_ASR]``-style strings for greppable diagnostics.
    """
    tag_norm = _normalize_tag(tag)
    record: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "tag": f"[{tag_norm}]",
        "message": (message or "").strip(),
        "session_id": (session_id or "").strip(),
        "current_agent": (current_agent or "").strip(),
        "active_intent": (active_intent or "").strip(),
        "pid": os.getpid(),
        "workspace": str(DONNA_WORKSPACE),
    }
    if latency_ms is not None:
        try:
            record["latency_ms"] = float(latency_ms)
        except (TypeError, ValueError):
            pass
    if payload:
        record["payload"] = payload

    path = TELEMETRY_JSONL_PATH
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with _JSONL_LOCK:
            with path.open("a", encoding="utf-8", errors="replace") as fh:
                fh.write(line + "\n")
    except Exception:  # noqa: BLE001 — never break the agent loop
        pass
    return record


def log_voice_asr(
    transcript: str,
    *,
    session_id: str = "",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return emit_tagged(
        "VOICE_ASR",
        transcript,
        session_id=session_id,
        payload=payload,
    )


def log_router(
    message: str,
    *,
    session_id: str = "",
    current_agent: str = "",
    active_intent: str = "",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return emit_tagged(
        "ROUTER",
        message,
        session_id=session_id,
        current_agent=current_agent,
        active_intent=active_intent,
        payload=payload,
    )


def log_reasoning_trace(
    think_text: str,
    *,
    session_id: str = "",
    clean_text: str = "",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body = (think_text or "")[:4000]
    extra = dict(payload or {})
    if clean_text:
        extra.setdefault("clean_preview", (clean_text or "")[:500])
    return emit_tagged(
        "REASONING_TRACE",
        body,
        session_id=session_id,
        payload=extra or None,
    )


def log_tool_execution(
    tool_id: str,
    *,
    session_id: str = "",
    current_agent: str = "",
    active_intent: str = "",
    ok: bool = True,
    latency_ms: float | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return emit_tagged(
        "TOOL_EXECUTION",
        f"{tool_id} {'ok' if ok else 'error'}",
        session_id=session_id,
        current_agent=current_agent,
        active_intent=active_intent,
        latency_ms=latency_ms,
        payload=payload,
    )


def log_handoff(
    target_agent: str,
    *,
    session_id: str = "",
    current_agent: str = "",
    active_intent: str = "",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return emit_tagged(
        "HANDOFF",
        f"-> {target_agent}",
        session_id=session_id,
        current_agent=current_agent,
        active_intent=active_intent,
        payload=payload,
    )


def log_sensor_vision(
    message: str = "",
    *,
    latency_ms: float | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Stage 4.1 — vision poller published ``latest_visual_context``."""
    return emit_tagged(
        "SENSOR_VISION",
        message or "latest_visual_context updated",
        current_agent="Vision_Sensor",
        active_intent="sensor_publish",
        latency_ms=latency_ms,
        payload=payload,
    )


def log_actuator_start(
    tool_name: str,
    *,
    action_id: int | None = None,
    session_id: str = "",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Stage 4.2 — actuator claimed a pending action_queue row."""
    extra = dict(payload or {})
    if action_id is not None:
        extra.setdefault("action_id", int(action_id))
    return emit_tagged(
        "ACTUATOR_START",
        f"{tool_name} action_id={action_id}",
        session_id=session_id,
        current_agent="Actuator",
        active_intent=tool_name,
        payload=extra or None,
    )


def log_actuator_done(
    tool_name: str,
    *,
    action_id: int | None = None,
    session_id: str = "",
    ok: bool = True,
    latency_ms: float | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Stage 4.2 — actuator wrote completed/failed result to the Blackboard."""
    extra = dict(payload or {})
    if action_id is not None:
        extra.setdefault("action_id", int(action_id))
    extra.setdefault("ok", bool(ok))
    return emit_tagged(
        "ACTUATOR_DONE",
        f"{tool_name} {'ok' if ok else 'error'} action_id={action_id}",
        session_id=session_id,
        current_agent="Actuator",
        active_intent=tool_name,
        latency_ms=latency_ms,
        payload=extra or None,
    )


def log_notification_toast(
    message: str = "",
    *,
    action_id: int | None = None,
    session_id: str = "",
    tool_name: str = "",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Stage 4.3 — silent Windows toast fired after actuator resolve."""
    extra = dict(payload or {})
    if action_id is not None:
        extra.setdefault("action_id", int(action_id))
    if tool_name:
        extra.setdefault("tool_name", tool_name)
    return emit_tagged(
        "NOTIFICATION_TOAST",
        message or f"toast {tool_name}",
        session_id=session_id,
        current_agent="Actuator",
        active_intent=tool_name or "toast",
        payload=extra or None,
    )


def log_notification_piggyback(
    message: str = "",
    *,
    session_id: str = "",
    count: int = 0,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Stage 4.3 — unread action results spliced into the Chat Node prompt."""
    extra = dict(payload or {})
    extra.setdefault("count", int(count))
    return emit_tagged(
        "NOTIFICATION_PIGGYBACK",
        message or f"piggyback count={count}",
        session_id=session_id,
        current_agent="Chat_Agent",
        active_intent="notification_piggyback",
        payload=extra or None,
    )


def log_operator_nav_click_complete(
    *,
    query: str = "",
    session_id: str = "",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Stage 6.3 — NavigationOperator completed a closed-loop click."""
    extra = dict(payload or {})
    if query:
        extra.setdefault("query", query)
    return emit_tagged(
        "OPERATOR_NAV_CLICK_COMPLETE",
        f"navigate_and_click complete query={query!r}",
        session_id=session_id,
        current_agent="Navigation_Operator",
        active_intent="navigate_and_click",
        payload=extra or None,
    )

