"""REST API for the frontend's Services Manager plugin — direct user
visibility into (and control over) the background processes the agent
started via dana.plugins.os.background_services's ``start_background_service``
ReAct tool, without needing to ask the LLM to check on or kill one.

This router adds NO independent process-tracking or kill logic of its
own: ``GET /api/services`` and ``DELETE /api/services/{alias}`` both
delegate straight to ``background_services.list_background_services``/
``stop_background_service`` — the EXACT SAME functions the agent's own
ReAct tool calls dispatch through (``dana.core.react_dispatch``) — so a
service killed from this UI and one killed by the agent itself can never
behave differently, and the complex cross-platform process-TREE kill
logic (see ``stop_background_service``'s own docstring) stays unified in
exactly one place, never duplicated here.

``GET /api/services/{alias}/logs`` is the one endpoint with logic of its
own: tailing the log file at the SAME sandbox-relative
``data/logs/{alias}.log`` path ``start_background_service`` itself writes
to (via ``dana.plugins.os.file_system.resolve_sandboxed_path`` — no
separate path-validation logic), bounded to a max trailing-byte window
before ever being read into memory, so this can't be used to pull an
arbitrarily large log file into a single response. A missing log file
(alias never started, or hasn't produced output yet) is NOT an error:
it comes back as ``{"exists": False, "lines": []}``, so the frontend's log
viewer can render a clean empty state instead of special-casing an error
response on every poll.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from dana.plugins.os.background_services import list_background_services, stop_background_service
from dana.plugins.os.file_system import PathEscapeError, resolve_sandboxed_path

router = APIRouter(prefix="/api/services", tags=["services"])

# Default/hard-cap on how many trailing LINES a single /logs call returns —
# same "don't blow up the response for no benefit" reasoning as
# dana.plugins.os.file_system.search_files's own _MAX_SEARCH_MATCHES.
_DEFAULT_LOG_LINES = 200
_MAX_LOG_LINES = 1000

# Bounds how many trailing BYTES of a (possibly huge, long-running-service)
# log file are ever read into memory before line-splitting — independent
# of how many lines the caller actually asked for, so a multi-gigabyte log
# still only ever costs a bounded read.
_MAX_LOG_TAIL_BYTES = 200_000


@router.get("")
def get_services() -> dict[str, Any]:
    return list_background_services()


@router.get("/{alias}/logs")
def get_service_logs(alias: str, lines: int = _DEFAULT_LOG_LINES) -> dict[str, Any]:
    line_count = max(1, min(lines, _MAX_LOG_LINES))
    log_path_str = f"data/logs/{alias}.log"
    try:
        target = resolve_sandboxed_path(log_path_str)
    except PathEscapeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not target.is_file():
        return {"ok": True, "alias": alias, "log_path": log_path_str, "exists": False, "lines": []}

    try:
        size = target.stat().st_size
        with open(target, "rb") as f:
            if size > _MAX_LOG_TAIL_BYTES:
                f.seek(size - _MAX_LOG_TAIL_BYTES)
            raw = f.read()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"could not read log file: {exc}") from exc

    text = raw.decode("utf-8", errors="replace")
    tail = text.splitlines()[-line_count:]
    return {"ok": True, "alias": alias, "log_path": log_path_str, "exists": True, "lines": tail}


@router.delete("/{alias}")
def delete_service(alias: str) -> dict[str, Any]:
    result = stop_background_service(alias)
    if not result.get("ok"):
        raise HTTPException(status_code=404, detail=result.get("error") or f"no active service aliased {alias!r}")
    return result


__all__ = ("router",)
