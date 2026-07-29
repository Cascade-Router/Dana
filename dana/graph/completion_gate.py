"""Completion gate — block END on filler / unresolved tools; tool timeout watchdog."""

from __future__ import annotations

import concurrent.futures
import re
from collections.abc import Callable
from typing import Any

# Spoken fallback when a tool hangs past the watchdog.
TOOL_TIMEOUT_MESSAGE = (
    "That took longer than expected and timed out..."
)
DEFAULT_TOOL_TIMEOUT_S = 30.0

# Soft acknowledgements that historically caused silent END ("ghosting").
_FILLER_PHRASES: tuple[str, ...] = (
    "let me check",
    "looking into that",
    "looking into it",
    "i'll look into",
    "ill look into",
    "let me look",
    "let me see",
    "one moment",
    "just a moment",
    "give me a second",
    "give me a moment",
    "hang on",
    "checking now",
    "i'll check",
    "ill check",
    "one sec",
)

_FILLER_RE = re.compile(
    r"|".join(re.escape(p) for p in _FILLER_PHRASES),
    re.IGNORECASE,
)


def is_filler_response(text: str | None) -> bool:
    """True when assistant text is a stall / filler acknowledgement."""
    raw = (text or "").strip()
    if not raw:
        return False
    # Short filler-only turns (ignore long substantive replies that mention the phrase).
    if len(raw) > 160:
        return False
    return bool(_FILLER_RE.search(raw))


def message_text(msg: Any) -> str:
    """Best-effort string content from a LangChain / dict message."""
    if msg is None:
        return ""
    content = getattr(msg, "content", None)
    if content is None and isinstance(msg, dict):
        content = msg.get("content")
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text") or block.get("content") or ""))
            else:
                parts.append(str(getattr(block, "text", "") or ""))
        return " ".join(p for p in parts if p).strip()
    return str(content or "").strip()


def latest_assistant_text(state: dict[str, Any] | None) -> str:
    """Scan messages backward for the latest AI / assistant text."""
    messages = list((state or {}).get("messages") or [])
    for msg in reversed(messages):
        role = getattr(msg, "type", None) or getattr(msg, "role", None)
        if role is None and isinstance(msg, dict):
            role = msg.get("type") or msg.get("role")
        name = type(msg).__name__ if msg is not None else ""
        is_ai = (
            str(role or "").lower() in {"ai", "assistant"}
            or name in {"AIMessage", "AIMessageChunk"}
        )
        if not is_ai:
            continue
        text = message_text(msg)
        if text:
            return text
    # Fallback: final_raw / last_obs spoken fields.
    st = state or {}
    return str(st.get("final_raw") or st.get("last_obs") or "").strip()


def has_unresolved_tool_calls(state: dict[str, Any] | None) -> bool:
    """True when the latest AIMessage still has tool_calls awaiting ToolMessages."""
    messages = list((state or {}).get("messages") or [])
    if not messages:
        return False
    # Find last AI message with tool_calls.
    last_ai_idx = -1
    tool_call_ids: set[str] = set()
    for i, msg in enumerate(messages):
        tcs = getattr(msg, "tool_calls", None)
        if tcs is None and isinstance(msg, dict):
            tcs = msg.get("tool_calls")
        if not tcs:
            continue
        last_ai_idx = i
        tool_call_ids = set()
        for tc in tcs:
            if isinstance(tc, dict):
                cid = str(tc.get("id") or "").strip()
            else:
                cid = str(getattr(tc, "id", "") or "").strip()
            if cid:
                tool_call_ids.add(cid)
    if last_ai_idx < 0:
        return False
    if not tool_call_ids:
        # Nameless tool_calls still count as unresolved until a later non-AI turn.
        return True
    answered: set[str] = set()
    for msg in messages[last_ai_idx + 1 :]:
        name = type(msg).__name__ if msg is not None else ""
        role = getattr(msg, "type", None) or getattr(msg, "role", None)
        if role is None and isinstance(msg, dict):
            role = msg.get("type") or msg.get("role")
        is_tool = (
            str(role or "").lower() == "tool"
            or name in {"ToolMessage"}
        )
        if not is_tool:
            continue
        cid = getattr(msg, "tool_call_id", None)
        if cid is None and isinstance(msg, dict):
            cid = msg.get("tool_call_id")
        if cid:
            answered.add(str(cid))
    return not tool_call_ids.issubset(answered)


def should_block_end(state: dict[str, Any] | None) -> bool:
    """Graph must not reach END while synthesis is pending or tools are open."""
    st = state or {}
    if bool(st.get("pending_synthesis")):
        return True
    if has_unresolved_tool_calls(st):
        return True
    return False


def flag_pending_synthesis_from_text(text: str | None) -> dict[str, Any]:
    """Return a state patch setting ``pending_synthesis`` from filler detection."""
    if is_filler_response(text):
        return {"pending_synthesis": True}
    return {"pending_synthesis": False}


def format_tool_failure_message(
    *,
    tool_id: str | None = None,
    detail: str | None = None,
    timed_out: bool = False,
) -> str:
    """Explicit user-facing failure (never silent terminate)."""
    if timed_out:
        return TOOL_TIMEOUT_MESSAGE
    tid = str(tool_id or "").strip() or "tool"
    extra = str(detail or "").strip()
    if extra:
        return f"Tool `{tid}` failed: {extra}"
    return f"Tool `{tid}` failed. I could not complete that step."


def tool_failure_state_patch(
    *,
    tool_id: str | None = None,
    detail: str | None = None,
    timed_out: bool = False,
    spoken: str | None = None,
) -> dict[str, Any]:
    """State patch for tool fail/timeout — routes via critic/supervisor corridor."""
    msg = spoken or format_tool_failure_message(
        tool_id=tool_id,
        detail=detail,
        timed_out=timed_out,
    )
    err = detail or msg
    return {
        "execution_error": str(err),
        "last_obs": msg,
        "final_raw": msg,
        "halt": False,
        "pending_synthesis": True,
        "fatal_block": False,
    }


def apply_timeout_failure(
    tracker: Any | None,
    task_id: str,
    *,
    tool_id: str | None = None,
) -> dict[str, Any]:
    """Mark task FAILED and return the timeout state patch (never silent drop)."""
    if tracker is not None:
        try:
            from dana.graph.task_tracker import TaskStatus

            tracker.update_status(
                str(task_id or "").strip() or "unknown",
                TaskStatus.FAILED,
                metadata={"timeout": True, "tool": tool_id},
            )
        except Exception:  # noqa: BLE001
            pass
    return tool_failure_state_patch(tool_id=tool_id, timed_out=True)


def run_with_tool_timeout(
    fn: Callable[[], Any],
    *,
    timeout_s: float = DEFAULT_TOOL_TIMEOUT_S,
) -> tuple[bool, Any, str | None]:
    """Run ``fn`` with a hard timeout (default 30s).

    Returns ``(ok, result, error_message)``. On timeout, ``error_message`` is
    :data:`TOOL_TIMEOUT_MESSAGE` and ``ok`` is False.
    """
    limit = max(0.05, float(timeout_s))
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(fn)
        try:
            return True, fut.result(timeout=limit), None
        except concurrent.futures.TimeoutError:
            return False, None, TOOL_TIMEOUT_MESSAGE
        except Exception as exc:  # noqa: BLE001
            return False, None, f"{type(exc).__name__}: {exc}"


async def run_async_with_tool_timeout(
    awaitable_factory: Callable[[], Any],
    *,
    timeout_s: float = DEFAULT_TOOL_TIMEOUT_S,
) -> tuple[bool, Any, str | None]:
    """Async counterpart using ``asyncio.wait_for`` (default 30s)."""
    import asyncio

    limit = max(0.05, float(timeout_s))
    try:
        result = await asyncio.wait_for(awaitable_factory(), timeout=limit)
        return True, result, None
    except asyncio.TimeoutError:
        return False, None, TOOL_TIMEOUT_MESSAGE
    except Exception as exc:  # noqa: BLE001
        return False, None, f"{type(exc).__name__}: {exc}"


def gate_route_after_agent(
    state: dict[str, Any] | None,
    *,
    proposed: str,
    end_sentinel: str = "__end__",
) -> str:
    """Rewrite a proposed post-agent route so filler / open tools never END."""
    if proposed != end_sentinel and str(proposed).upper() != "END":
        return proposed
    if should_block_end(state):
        return "agent"
    return proposed


def gate_route_after_execution(
    state: dict[str, Any] | None,
    *,
    proposed: str,
    end_sentinel: str = "__end__",
) -> str:
    """Rewrite post-tools route: block silent END; prefer agent/critic on failure."""
    st = state or {}
    err = st.get("execution_error")
    if err is not None and str(err).strip():
        # Preserve critic / fail_closed when already chosen.
        if proposed in {"critic", "fail_closed"}:
            return proposed
        # Non-silent: send to critic (supervisor self-heal corridor).
        return "critic"
    if proposed != end_sentinel and str(proposed).upper() != "END":
        if should_block_end(st) and proposed not in {"critic", "fail_closed", "agent"}:
            return "agent"
        return proposed
    if should_block_end(st):
        return "agent"
    return proposed


__all__ = (
    "DEFAULT_TOOL_TIMEOUT_S",
    "TOOL_TIMEOUT_MESSAGE",
    "apply_timeout_failure",
    "flag_pending_synthesis_from_text",
    "format_tool_failure_message",
    "gate_route_after_agent",
    "gate_route_after_execution",
    "has_unresolved_tool_calls",
    "is_filler_response",
    "latest_assistant_text",
    "message_text",
    "run_async_with_tool_timeout",
    "run_with_tool_timeout",
    "should_block_end",
    "tool_failure_state_patch",
)
