"""JSON-lines IPC protocol for the Agent Engine sidecar.

Framing: one UTF-8 JSON object per line (``\\n`` terminated) over loopback TCP.

Request::

    {"id": "<corr>", "method": "stream_chat"|"system_status"|"hot_restart", "params": {}}

Response / event stream::

    {"id": "<corr>", "ok": true, "type": "event"|"result"|"error",
     "event": "<optional>", "data": {}, "error": "<optional>"}
"""

from __future__ import annotations

from typing import Any, Literal

MethodName = Literal["stream_chat", "system_status", "hot_restart", "ping"]
FrameType = Literal["event", "result", "error"]

METHODS: frozenset[str] = frozenset(
    {"stream_chat", "system_status", "hot_restart", "ping"}
)


def make_request(
    method: str,
    *,
    req_id: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": str(req_id),
        "method": str(method),
        "params": dict(params or {}),
    }


def make_event(
    req_id: str,
    event: str,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": str(req_id),
        "ok": True,
        "type": "event",
        "event": str(event),
        "data": dict(data or {}),
    }


def make_result(
    req_id: str,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": str(req_id),
        "ok": True,
        "type": "result",
        "data": dict(data or {}),
    }


def make_error(req_id: str, message: str) -> dict[str, Any]:
    return {
        "id": str(req_id),
        "ok": False,
        "type": "error",
        "error": str(message),
        "data": {},
    }
