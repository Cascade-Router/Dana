"""Three-tier orchestration telemetry (INFO / DEBUG / TRACE).

The ReAct orchestration layer (``dana.core.react_dispatch``,
``dana.api.server``) used to log through bare ``print(..., file=sys.stderr)``
calls with no severity distinction at all: an internal diagnostic printed at
the exact same volume as an actual state transition. A per-turn "[Turn
Context] Available tools for LLM: [... every candidate tool_id ...]" line, a
raw generated Python/FreeCAD script, or a verbatim tool-result payload
(base64 image data, a compiled skill's full source) drowned out the handful
of lines that actually matter: a tool call dispatching, a plan advancing, an
error. This module makes that distinction explicit instead of leaving it to
each call site's own judgment.

Three tiers, in order of what a normal run should show:

  INFO  (default) -- ONLY the six deterministic state events below. Nothing
          else may log at INFO from the orchestration layer:
            REQUEST, PLAN_CREATED, TOOL_CALL, TOOL_RESULT,
            TASK_STATE_CHANGE, ERROR
          ``TOOL_RESULT`` never carries a raw base64/source-code dump (see
          ``_strip_raw_dumps``) -- only a lean, LLM-result-shaped summary.
  DEBUG -- internal diagnostics a developer actively investigating a turn
          wants to see: candidate tool-id lists, "continuing loop" iteration
          breadcrumbs, HITL suspend/resume chatter, retries.
  TRACE -- everything else: full topology-graph dumps, raw generated
          script/source text, byte-for-byte payload echoes.

Level is read once from ``DANA_LOG_LEVEL`` (default ``INFO``) via
``logging`` itself, so a disabled tier's call sites pay only the cost of one
disabled-level check -- no call site needs to guard itself.
"""

from __future__ import annotations

import logging
import os
import threading
from collections import deque
from typing import Any


class AsyncRingBuffer:
    """Thread-safe fixed-size ring buffer for low-latency telemetry."""

    def __init__(self, *, capacity: int = 500) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._lock = threading.Lock()
        self._events: deque[dict[str, Any]] = deque(maxlen=capacity)

    def append(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._events.append(event)

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()


class NeuralStreamEmitter:
    """O(1), non-blocking structured event emitter for the UI/telemetry stream."""

    def __init__(self, buffer: AsyncRingBuffer) -> None:
        self._buffer = buffer

    def emit(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        event = {"type": event_type, "payload": dict(payload or {})}
        self._buffer.append(event)


TRACE = 5
logging.addLevelName(TRACE, "TRACE")

_LOGGER_NAME = "dana.orchestration"
_logger = logging.getLogger(_LOGGER_NAME)

# Fields whose value is a known raw dump (compiled-skill source, a base64
# data URI, an embedded FreeCAD/Python script) -- always replaced with a
# short placeholder in a TOOL_RESULT event, regardless of length.
_RAW_PAYLOAD_KEYS = frozenset(
    {"python_code", "source", "script", "image_b64", "mesh_b64", "data_uri", "audio_b64"}
)
# Any OTHER string field longer than this is truncated too -- catches a raw
# dump nobody explicitly flagged (e.g. a new tool's own large text field)
# without needing this table to enumerate every possible key up front.
_MAX_INLINE_STR = 300

# Fix #3 -- Expose Geometry Post-Conditions: these keys carry the
# Deterministic Post-Conditions data (dana.plugins.freecad.engine's
# `_execute_ir_tool`-derived {length, width, height, volume}, or a step's
# own post_condition check) an LLM's spatial claim ("a horizontal cylinder")
# actually gets verified against. They are NEVER subject to the generic
# string-truncation/raw-dump stripping above (a plain dict, so today's
# _strip_raw_dumps already passes them through unchanged -- this table
# makes that guarantee EXPLICIT and future-proof against a broader filter
# later swallowing them by accident) and are additionally promoted to their
# own TOP-LEVEL field on the TOOL_RESULT event itself (see log_tool_result),
# not left buried inside the nested "result" dict where a human scanning
# the line for what geometry the LLM actually received could miss them.
_GEOMETRY_KEYS = ("geometry", "post_condition")


def _resolve_level() -> int:
    raw = (os.environ.get("DANA_LOG_LEVEL") or "INFO").strip().upper()
    if raw == "TRACE":
        return TRACE
    return logging.getLevelName(raw) if isinstance(logging.getLevelName(raw), int) else logging.INFO


def _configure() -> logging.Logger:
    if not _logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
        _logger.addHandler(handler)
        _logger.propagate = False
    _logger.setLevel(_resolve_level())
    return _logger


_configure()


def set_level(level: str) -> None:
    """Explicit override (tests, or a caller that wants to raise verbosity
    mid-process) -- otherwise ``DANA_LOG_LEVEL`` at import time wins."""
    os.environ["DANA_LOG_LEVEL"] = level
    _configure()


def trace(msg: str, *args: Any) -> None:
    _logger.log(TRACE, msg, *args)


def debug(msg: str, *args: Any) -> None:
    _logger.debug(msg, *args)


def _render(event: str, fields: dict[str, Any]) -> str:
    rendered = " ".join(f"{k}={v!r}" for k, v in fields.items())
    return f"{event} {rendered}".rstrip()


def log_request(**fields: Any) -> None:
    _logger.info(_render("REQUEST", fields))


def log_plan_created(**fields: Any) -> None:
    _logger.info(_render("PLAN_CREATED", fields))


def log_tool_call(tool_id: str, *, arguments: Any = None, **fields: Any) -> None:
    # save_new_skill/compile_plan_as_skill's own arguments can themselves
    # carry a full python_code source string -- the same raw-dump risk
    # log_tool_result guards against, so TOOL_CALL reuses the same filter
    # rather than only cleaning the RESULT half of a call.
    _logger.info(_render("TOOL_CALL", {"tool_id": tool_id, "arguments": _strip_raw_dumps(arguments), **fields}))


def _strip_raw_dumps(payload: Any) -> Any:
    """Fix #1 -- TOOL_RESULT must never carry a raw base64/source-code dump
    into the log. Only the TOP-LEVEL dict is inspected (matching the shallow
    result shape every tool in this codebase returns): a known raw-dump key
    (``_RAW_PAYLOAD_KEYS``) or any string over ``_MAX_INLINE_STR`` chars is
    replaced with a short placeholder noting the omitted size -- everything
    else (numbers, names, bounding boxes) passes through unchanged, since
    those are exactly the deterministic fields worth seeing at INFO.

    ``_GEOMETRY_KEYS`` (Fix #3) are explicitly exempted FIRST, before any
    other rule gets a chance to apply -- they are always a plain dict, never
    a string, so today's other rules would never actually touch them, but
    this makes "these are never stripped" an explicit invariant rather than
    an accident of the current field shape.
    """
    if not isinstance(payload, dict):
        return payload
    cleaned: dict[str, Any] = {}
    for key, value in payload.items():
        if key in _GEOMETRY_KEYS:
            cleaned[key] = value
        elif key in _RAW_PAYLOAD_KEYS and isinstance(value, str):
            cleaned[key] = f"<omitted {len(value)} chars>"
        elif isinstance(value, str) and len(value) > _MAX_INLINE_STR:
            cleaned[key] = f"{value[:120]}...<{len(value)} chars omitted>"
        else:
            cleaned[key] = value
    return cleaned


def log_tool_result(
    tool_id: str,
    ok: bool,
    *,
    payload: Any = None,
    message: str = "",
    duration_ms: int | None = None,
) -> None:
    fields: dict[str, Any] = {"tool_id": tool_id, "ok": ok}
    if duration_ms is not None:
        fields["duration_ms"] = duration_ms
    if ok:
        fields["result"] = _strip_raw_dumps(payload)
        # Fix #3 -- Expose Geometry Post-Conditions: promoted to their OWN
        # top-level field too, not left only inside the nested "result"
        # dict above -- this is the actual BoundBox-derived length/width/
        # height and Shape.Volume data an LLM's spatial claim ("a
        # horizontal cylinder") gets checked against, and it must be
        # impossible to scan past in the INFO line the way an ordinary
        # nested result field can be.
        if isinstance(payload, dict):
            for key in _GEOMETRY_KEYS:
                if key in payload:
                    fields[key] = payload[key]
    else:
        fields["error"] = message
    _logger.info(_render("TOOL_RESULT", fields))


def log_task_state_change(**fields: Any) -> None:
    _logger.info(_render("TASK_STATE_CHANGE", fields))


def log_error(**fields: Any) -> None:
    _logger.error(_render("ERROR", fields))


__all__ = (
    "AsyncRingBuffer",
    "NeuralStreamEmitter",
    "TRACE",
    "debug",
    "log_error",
    "log_plan_created",
    "log_request",
    "log_task_state_change",
    "log_tool_call",
    "log_tool_result",
    "set_level",
    "trace",
)
