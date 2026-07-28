"""Zero-copy raw state buffer — full diagnostics without LLM truncation."""

from __future__ import annotations

import traceback
from typing import Any


def store_raw_trace(
    state: dict[str, Any] | None,
    exception: BaseException | str,
    context_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write full raw traceback + error metadata into ``raw_state_buffer``.

    Does **not** truncate or summarize for an LLM. Returns a state patch dict
    merging ``last_error`` into any existing ``raw_state_buffer``.
    """
    prev = dict((state or {}).get("raw_state_buffer") or {})
    meta = dict(context_metadata or {})

    if isinstance(exception, BaseException):
        tb_text = "".join(
            traceback.format_exception(
                type(exception),
                exception,
                exception.__traceback__,
            )
        )
        exc_type = type(exception).__name__
        exc_msg = str(exception)
    else:
        tb_text = str(exception)
        # If called inside an ``except`` block, append live ``format_exc``.
        live = traceback.format_exc()
        if live and live.strip() and live.strip() != "NoneType: None":
            if live not in tb_text:
                tb_text = f"{tb_text}\n{live}" if tb_text else live
        exc_type = ""
        exc_msg = str(exception)

    last_error: dict[str, Any] = {
        "traceback": tb_text,
        "exception_type": exc_type,
        "exception_message": exc_msg,
        "context": meta,
    }
    return {
        "raw_state_buffer": {
            **prev,
            "last_error": last_error,
        }
    }


def get_raw_trace(state: dict[str, Any] | None) -> dict[str, Any] | None:
    """Fetch the raw diagnostic blob for supervisor / critic consumers."""
    buf = (state or {}).get("raw_state_buffer")
    if not isinstance(buf, dict):
        return None
    last = buf.get("last_error")
    if last is None:
        return None
    if isinstance(last, dict):
        return last
    return {"traceback": str(last)}
