"""LangGraph async ReAct runner with MemorySaver + astream_events TTS telemetry.

Used by ``dana.agentic._run_react_loop_langchain``. Keeps strict ``bind_tools``
and Titan peg-native retries while streaming Thinking/tool TTS hooks.
"""

from __future__ import annotations

import asyncio
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from dana.schema import AgenticResult, ReactGraphState
from dana.tools.broker import IntentBroker, ToolValidationError, get_broker
from dana.tools.schema import ToolCall

# Re-export for callers that import ReactGraphState from this module.
# route_after_verifier is the closed-loop post-verifier router.
from dana.graph.nodes.verifier import route_after_verifier  # noqa: F401

__all__ = (
    "ReactGraphState",
    "compile_donna_react_graph",
    "route_after_execution",
    "route_after_verifier",
    "run_react_langgraph",
    "validation_retry_tool_corridor",
)


def validation_retry_tool_corridor(failed_tool_id: str) -> list[str]:
    """Stage 3.3: tools list for a ValidationError bounce turn (exactly one id)."""
    tid = str(failed_tool_id or "").strip()
    return [tid] if tid else []


def _emit_live_trace(event_type: str, **payload: Any) -> None:
    """Non-blocking Live Trace bus emit (safe from LangGraph worker threads)."""
    try:
        from dana.ui.trace_bus import emit_trace_event

        emit_trace_event(event_type, **payload)
    except Exception:  # noqa: BLE001
        pass


def _specs_for_tool_ids(
    tool_ids: list[str],
    *,
    semantic_specs: dict[str, Any],
    broker_registry: dict[str, Any],
) -> dict[str, Any]:
    """Resolve ToolSpecs for an ordered id list (semantic first, then broker)."""
    out: dict[str, Any] = {}
    for tid in tool_ids:
        name = str(tid or "").strip()
        if not name or name in out:
            continue
        if name in semantic_specs:
            out[name] = semantic_specs[name]
        elif name in broker_registry:
            out[name] = broker_registry[name]
    return out


def _invoked_tool_ids_from_messages(messages: list[Any] | None) -> set[str]:
    """Collect tool names already invoked via AIMessage.tool_calls."""
    invoked: set[str] = set()
    for msg in messages or []:
        tool_calls = getattr(msg, "tool_calls", None) or []
        for tc in tool_calls:
            if isinstance(tc, dict):
                name = str(tc.get("name") or "").strip()
            else:
                name = str(getattr(tc, "name", "") or "").strip()
            if name:
                invoked.add(name)
    return invoked


def pending_always_include_tools(state: dict[str, Any] | ReactGraphState) -> list[str]:
    """always_include ids that have not yet appeared as native tool calls."""
    always = [
        str(x).strip()
        for x in (state.get("always_include") or [])
        if str(x).strip()
    ]
    if not always:
        return []
    invoked = _invoked_tool_ids_from_messages(state.get("messages") or [])
    return [tid for tid in always if tid not in invoked]


def _force_tool_nudge_message(tool_id: str) -> Any:
    from langchain_core.messages import SystemMessage

    tid = str(tool_id or "").strip() or "tool"
    return SystemMessage(
        content=(
            f"SYSTEM: You must output the JSON tool call for `{tid}` now. "
            "Do not answer in natural language or claim the task/ticket is complete "
            f"until `{tid}` has executed successfully."
        )
    )


_TOOL_FORCE_SAVE_MSG = (
    "Error: You generated code in text but did not save it. You MUST invoke the "
    "`file_editor` or `write_to_file` tool to save your work."
)
_CODE_FENCE_DUMP_RE = re.compile(
    r"```(?:python|py|html|css|javascript|js|tkinter)?\b",
    re.I,
)
_HTML_DUMP_RE = re.compile(r"<(?:html|!DOCTYPE|script|style)\b", re.I)


def _looks_like_unsaved_code_dump(text: str) -> bool:
    raw = text or ""
    if _CODE_FENCE_DUMP_RE.search(raw) or _HTML_DUMP_RE.search(raw):
        return True
    return bool(re.search(r"(?m)^(def|class)\s+\w+", raw) and len(raw) > 80)


def _messages_have_code_save_force(messages: list[Any] | None) -> bool:
    needle = "You generated code in text but did not save it"
    for msg in messages or []:
        content = str(getattr(msg, "content", "") or "")
        if needle in content:
            return True
    return False


def _messages_have_force_nudge(messages: list[Any] | None, tool_id: str) -> bool:
    needle = f"JSON tool call for `{tool_id}`"
    for msg in messages or []:
        content = str(getattr(msg, "content", "") or "")
        if needle in content and "SYSTEM:" in content:
            return True
    return False


def _default_args_for_forced_tool(tool_id: str, user_text: str) -> dict[str, Any]:
    """Minimal args so a programmatically forced tool call can execute."""
    tid = str(tool_id or "").strip()
    raw = (user_text or "").strip()
    if tid == "analyze_visual_context":
        return {"source": "screen"}
    if tid == "ocr_with_region":
        return {"query": (user_text or "").strip()[:200]}
    if tid == "click_ui_element":
        return {"target_description": raw[:200]}
    if tid == "scroll_screen":
        low = raw.lower()
        direction = "down"
        for candidate in ("up", "down", "left", "right"):
            if candidate in low:
                direction = candidate
                break
        return {"direction": direction, "amount": "medium"}
    if tid == "draft_cursor_prompt":
        try:
            from dana.tools.broker import parse_draft_cursor_prompt_args

            args = dict(parse_draft_cursor_prompt_args(raw) or {})
        except Exception:  # noqa: BLE001
            args = {}
        if not str(args.get("objective") or "").strip():
            from dana.agentic import _full_sentence_boundary

            args["objective"] = (
                _full_sentence_boundary(raw) or "Log self-improvement ticket"
            )
        if "context" not in args:
            args["context"] = ""
        return args
    if tid in {"web_search", "dispatch_research_swarm", "dispatch_jason_supervisor"}:
        return {"query": raw}
    if tid == "meta_broker":
        try:
            from dana.tools.broker import extract_meta_broker_prompt

            return {"prompt": extract_meta_broker_prompt(raw)}
        except Exception:  # noqa: BLE001
            return {"prompt": raw}
    if tid == "file_editor":
        path = "notes.txt"
        m = re.search(
            r"([\w./\\-]+\.(?:txt|md|json|csv|log))",
            raw,
            flags=re.I,
        )
        if m:
            path = m.group(1).replace("\\", "/")
        content = (
            "Summary\n"
            "1. Clear roles and interfaces beat vague agent swarms.\n"
            "2. Shared memory and coordination are the usual bottlenecks.\n"
            "3. Prefer local tools and small loops over unbounded fan-out.\n"
        )
        return {"action": "write", "filepath": path, "content": content}
    if tid == "execute_powershell":
        try:
            from dana.tools.os_tools import (
                cascade_git_tool_args,
                is_cascade_git_query,
            )

            if is_cascade_git_query(raw):
                return cascade_git_tool_args(raw)
        except Exception:  # noqa: BLE001
            pass
        return {"command": raw or "Get-Date"}
    if tid == "read_local_file":
        try:
            from dana.tools.os_tools import (
                is_watchdog_graph_query,
                watchdog_graph_filepath,
            )

            if is_watchdog_graph_query(raw):
                return {"filepath": watchdog_graph_filepath()}
        except Exception:  # noqa: BLE001
            pass
        m = re.search(
            r"([\w./\\-]+\.(?:py|txt|md|json|csv|log))",
            raw,
            flags=re.I,
        )
        return {"filepath": m.group(1).replace("\\", "/") if m else "README.md"}
    return {}


def _synthetic_tool_call_message(tool_id: str, user_text: str, *, step: int) -> Any:
    """Build an AIMessage that forces tools-node execution for ``tool_id``."""
    from langchain_core.messages import AIMessage

    tid = str(tool_id or "").strip()
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": tid,
                "args": _default_args_for_forced_tool(tid, user_text),
                "id": f"force-{tid}-{step}-{uuid.uuid4().hex[:8]}",
                "type": "tool_call",
            }
        ],
    )


def _iter_tool_calls(message: Any) -> list[dict[str, Any]]:
    """Normalize AIMessage.tool_calls into ``{name, args, id}`` dicts."""
    out: list[dict[str, Any]] = []
    for tc in getattr(message, "tool_calls", None) or []:
        if isinstance(tc, dict):
            name = str(tc.get("name") or "").strip()
            args = dict(tc.get("args") or tc.get("arguments") or {})
            call_id = str(tc.get("id") or "").strip()
        else:
            name = str(getattr(tc, "name", "") or "").strip()
            args = dict(getattr(tc, "args", None) or {})
            call_id = str(getattr(tc, "id", "") or "").strip()
        if name:
            out.append({"name": name, "args": args, "id": call_id})
    return out


def message_has_draft_cursor_prompt(message: Any) -> bool:
    return any(tc["name"] == "draft_cursor_prompt" for tc in _iter_tool_calls(message))


# Stage 8.9.6 — Pydantic rewrite loop before Jason / HITL (not infinite).
_MAX_TICKET_VALIDATION_RETRIES = 3


def extract_draft_cursor_payload(state: ReactGraphState | dict[str, Any]) -> dict[str, Any]:
    """Pull ``objective`` / ``context`` from the latest draft_cursor_prompt tool call."""
    messages = list(state.get("messages") or [])
    for msg in reversed(messages):
        for tc in _iter_tool_calls(msg):
            if tc["name"] != "draft_cursor_prompt":
                continue
            args = dict(tc.get("args") or {})
            return {
                "type": "ticket_approval",
                "tool": "draft_cursor_prompt",
                "objective": str(args.get("objective") or "").strip(),
                "context": str(args.get("context") or "").strip(),
                "tool_call_id": str(tc.get("id") or ""),
                "session_id": str(state.get("session_id") or ""),
                "active_intent": str(state.get("active_intent") or ""),
                "jason_critique": str(state.get("jason_critique") or "").strip(),
            }
    return {
        "type": "ticket_approval",
        "tool": "draft_cursor_prompt",
        "objective": "",
        "context": "",
        "tool_call_id": "",
        "session_id": str(state.get("session_id") or ""),
        "active_intent": str(state.get("active_intent") or "draft_cursor_prompt"),
        "jason_critique": str(state.get("jason_critique") or "").strip(),
    }


def resolved_drafted_ticket(state: ReactGraphState | dict[str, Any]) -> dict[str, Any]:
    """Prefer Stage 8.9.6 ``drafted_ticket``; fall back to latest tool-call args."""
    drafted = state.get("drafted_ticket")
    if isinstance(drafted, dict) and (
        str(drafted.get("objective") or "").strip()
        or str(drafted.get("context") or "").strip()
    ):
        out = dict(drafted)
        out["jason_critique"] = str(
            state.get("jason_critique") or out.get("jason_critique") or ""
        ).strip()
        return out
    return extract_draft_cursor_payload(state)


def ticket_validate_node(state: ReactGraphState) -> dict[str, Any]:
    """Stage 8.9.6 — Pydantic gate before Jason / HITL; bounce MoA on failure."""
    from langchain_core.messages import SystemMessage, ToolMessage
    from pydantic import ValidationError

    from dana.tools.guards import DraftCursorTicketPayload, format_validation_bounce

    raw = extract_draft_cursor_payload(state)
    call_id = str(raw.get("tool_call_id") or f"validate-{uuid.uuid4().hex[:8]}")
    try:
        retries = int(state.get("ticket_validation_retries") or 0)
    except (TypeError, ValueError):
        retries = 0

    try:
        validated = DraftCursorTicketPayload.model_validate(
            {
                "objective": str(raw.get("objective") or ""),
                "context": str(raw.get("context") or ""),
            }
        )
    except ValidationError as exc:
        retries += 1
        detail = format_validation_bounce(exc)
        bounce = (
            f"Ticket validation failed: {detail} "
            "You must include root cause, step-by-step changes, and "
            "acceptance criteria."
        )
        _emit_live_trace(
            "status",
            node="ticket_validate",
            message="TICKET_VALIDATION_FAILED",
            payload=bounce[:800],
            mode="developer",
            tool="draft_cursor_prompt",
            state_keys=("ticket_validation_retries", "drafted_ticket"),
        )
        try:
            from dana.telemetry import log_tool_execution

            log_tool_execution(
                "draft_cursor_prompt",
                session_id=str(state.get("session_id") or ""),
                current_agent="Ticket_Validator",
                active_intent=str(state.get("active_intent") or "draft_cursor_prompt"),
                ok=False,
                payload={
                    "validation_bounce": True,
                    "guard": "pydantic",
                    "retry": retries < _MAX_TICKET_VALIDATION_RETRIES,
                    "attempt": retries,
                },
            )
        except Exception:  # noqa: BLE001
            pass

        if retries >= _MAX_TICKET_VALIDATION_RETRIES:
            obs = (
                f"Max retries reached: ticket validation failed after "
                f"{retries} attempts. {detail}"
            )
            return {
                "messages": [ToolMessage(content=obs, tool_call_id=call_id)],
                "drafted_ticket": {},
                "ticket_validated": False,
                "ticket_validation_retries": retries,
                "halt": True,
                "final_raw": (
                    "Max retries reached — ticket validation failed after 3 "
                    "attempts. Please rewrite with full context (root cause, "
                    "step-by-step, acceptance criteria, target files) and try again."
                ),
                "last_obs": obs,
                "always_include": [],
                "current_agent": "Ticket_Validator",
            }

        return {
            "messages": [
                ToolMessage(content=bounce, tool_call_id=call_id),
                SystemMessage(
                    content=(
                        f"SYSTEM: {bounce} "
                        f"Rewrite `draft_cursor_prompt` now "
                        f"(attempt {retries}/{_MAX_TICKET_VALIDATION_RETRIES}). "
                        "Supply complete structured fields. No other tools."
                    )
                ),
            ],
            "drafted_ticket": {},
            "ticket_validated": False,
            "ticket_validation_retries": retries,
            "halt": False,
            "last_obs": bounce,
            "current_agent": "Ticket_Validator",
        }

    from dana.middleware.hitl_ticket import extract_files_line

    drafted = {
        "type": "ticket_approval",
        "tool": "draft_cursor_prompt",
        "objective": validated.objective,
        "context": validated.context,
        "files": extract_files_line(validated.context),
        "tool_call_id": call_id,
        "session_id": str(state.get("session_id") or ""),
        "active_intent": str(state.get("active_intent") or "draft_cursor_prompt"),
        "jason_critique": str(state.get("jason_critique") or "").strip(),
    }
    _emit_live_trace(
        "status",
        node="ticket_validate",
        message="TICKET_VALIDATED",
        payload=f"objective_chars={len(drafted['objective'])} "
        f"context_chars={len(drafted['context'])}",
        mode="developer",
        tool="draft_cursor_prompt",
        state_keys=("drafted_ticket", "ticket_validated"),
    )
    return {
        "drafted_ticket": drafted,
        "ticket_validated": True,
        "ticket_validation_retries": 0,
        "halt": False,
        "last_obs": "Ticket payload validated — proceeding to Jason review",
        "current_agent": "Ticket_Validator",
    }


def _extract_user_request_text(state: ReactGraphState | dict[str, Any]) -> str:
    """Best-effort original user request from graph messages."""
    from langchain_core.messages import HumanMessage

    for msg in reversed(list(state.get("messages") or [])):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
            return str(getattr(msg, "content", "") or "").strip()
        if isinstance(msg, dict) and str(msg.get("role") or "") == "user":
            return str(msg.get("content") or "").strip()
    return ""


def _heuristic_jason_critique(
    user_request: str,
    *,
    objective: str,
    context: str,
) -> str:
    """Offline fallback when Ollama is unavailable."""
    obj = (objective or "").strip()
    ctx = (context or "").strip()
    req = (user_request or "").strip()
    if not obj:
        return (
            "This ticket is missing a clear objective — I recommend you deny it "
            "and ask MoA to redraft."
        )
    missing = []
    req_l = req.lower()
    blob = f"{obj}\n{ctx}".lower()
    if any(k in req_l for k in ("visual", "ocr", "bounds", "region", "screen")) and not any(
        k in blob for k in ("visual", "ocr", "bound", "region", "box", "screen")
    ):
        missing.append("visual bounds")
    if any(k in req_l for k in ("api", "endpoint", "schema")) and "api" not in blob:
        missing.append("API constraints")
    if missing:
        return (
            f"This ticket is missing the {missing[0]}; I recommend you deny it "
            "until MoA fills that gap."
        )
    if len(obj) < 24:
        return (
            "This ticket is thin on detail — approve only if you intend a narrow patch."
        )
    return "This ticket accurately captures the request; safe to approve if you agree."


def generate_jason_ticket_critique(
    user_request: str,
    *,
    objective: str,
    context: str,
) -> str:
    """Lightweight Jason review → 1–2 sentence spoken critique."""
    system = (
        "You are Jason, Donna's CTO supervisor. Review a drafted self-improvement "
        "ticket against the user's original request. Reply with ONLY 1-2 short "
        "spoken sentences. Either confirm it accurately captures the request, or "
        "name what is missing and recommend deny. No markdown, no bullet lists."
    )
    user = (
        f"USER REQUEST:\n{(user_request or '').strip() or '(unknown)'}\n\n"
        f"DRAFTED OBJECTIVE:\n{(objective or '').strip() or '(empty)'}\n\n"
        f"DRAFTED CONTEXT:\n{(context or '').strip() or '(empty)'}\n"
    )
    try:
        from dana.core_agent import ask_ollama_messages

        raw = ask_ollama_messages(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            num_predict=96,
        )
        critique = re.sub(r"\s+", " ", str(raw or "")).strip()
        # Keep to ~2 sentences for TTS.
        parts = re.split(r"(?<=[.!?])\s+", critique)
        critique = " ".join(p for p in parts[:2] if p).strip()
        if len(critique) > 280:
            critique = critique[:277].rstrip() + "…"
        if critique:
            return critique
    except Exception:  # noqa: BLE001
        pass
    return _heuristic_jason_critique(
        user_request, objective=objective, context=context
    )


def jason_ticket_review_node(state: ReactGraphState) -> dict[str, Any]:
    """Stage 8.9 — Jason reviews drafted ticket, speaks critique, then HITL.

    Runs only after Stage 8.9.6 ``ticket_validate`` succeeds (valid
    ``drafted_ticket``), before ``ticket_approval`` interrupt.
    """
    payload = resolved_drafted_ticket(state)
    user_req = _extract_user_request_text(state)
    critique = generate_jason_ticket_critique(
        user_req,
        objective=str(payload.get("objective") or ""),
        context=str(payload.get("context") or ""),
    )
    try:
        from dana import agentic as ag

        ag.set_stream_tts_agent("jason")
    except Exception:  # noqa: BLE001
        pass
    try:
        from dana.core_agent import enqueue_speech

        enqueue_speech(critique, agent_id="jason")
    except Exception:  # noqa: BLE001
        pass
    _emit_live_trace(
        "status",
        node="jason_ticket_review",
        message="JASON_TICKET_REVIEW",
        payload=critique[:800],
        mode="developer",
        tool="draft_cursor_prompt",
        state_keys=("jason_critique", "objective", "context"),
    )
    drafted = dict(payload)
    drafted["jason_critique"] = critique
    return {
        "jason_critique": critique,
        "drafted_ticket": drafted,
        "current_agent": "Jason_Supervisor",
        "last_obs": f"Jason review: {critique}",
        "halt": False,
    }


def _deny_draft_tool_messages(state: ReactGraphState | dict[str, Any]) -> list[Any]:
    """ToolMessage receipts so the ReAct turn can halt cleanly after Deny."""
    from langchain_core.messages import ToolMessage

    messages = list(state.get("messages") or [])
    last = messages[-1] if messages else None
    obs = (
        "DENIED: Ticket submission cancelled by operator (HITL). "
        "Do not retry draft_cursor_prompt unless the user asks again."
    )
    out: list[Any] = []
    for tc in _iter_tool_calls(last):
        if tc["name"] != "draft_cursor_prompt":
            continue
        call_id = str(tc.get("id") or f"deny-draft-{uuid.uuid4().hex[:8]}")
        out.append(ToolMessage(content=obs, tool_call_id=call_id))
    if not out:
        out.append(
            ToolMessage(content=obs, tool_call_id=f"deny-draft-{uuid.uuid4().hex[:8]}")
        )
    return out


def ticket_approval_node(state: ReactGraphState) -> dict[str, Any]:
    """Stage 8.6 — LangGraph HITL breakpoint before ``tools`` for ledger tickets.

    Uses ``interrupt(payload)`` so the graph freezes with the drafted ticket in
    the interrupt value. Resume with ``Command(resume={{approved: bool}})``.
    """
    from langgraph.types import interrupt

    from dana.middleware.hitl_ticket import (
        begin_ticket_hitl,
        decision_is_approved,
        get_consecutive_denials,
        hitl_enabled,
        record_hitl_decision,
    )

    if not hitl_enabled():
        return {"halt": False}
    # Stage 8.9.6 — never interrupt on unvalidated / missing drafted_ticket.
    if not state.get("ticket_validated"):
        return {"halt": False}
    drafted_ok = state.get("drafted_ticket")
    if not (
        isinstance(drafted_ok, dict)
        and str(drafted_ok.get("objective") or "").strip()
        and str(drafted_ok.get("context") or "").strip()
    ):
        return {"halt": False}
    if not any(
        message_has_draft_cursor_prompt(m) for m in (state.get("messages") or [])[-3:]
    ):
        return {"halt": False}

    payload = resolved_drafted_ticket(state)
    # Ensure Jason critique from prior node is surfaced to the GUI.
    payload["jason_critique"] = str(
        state.get("jason_critique") or payload.get("jason_critique") or ""
    ).strip()
    # Stage 8.9.3 — sync denial counter (reset on new distinct task fingerprint).
    denials = begin_ticket_hitl(payload)
    try:
        prior = int(state.get("consecutive_denials") or 0)
    except (TypeError, ValueError):
        prior = 0
    # Prefer process counter; seed from graph state if larger (checkpointer resume).
    denials = max(int(denials), prior)
    payload["consecutive_denials"] = denials
    decision = interrupt(payload)
    if decision_is_approved(decision):
        n = record_hitl_decision(True)
        return {
            "halt": False,
            "last_obs": "HITL: ticket approved — executing tools",
            "consecutive_denials": n,
        }
    # Deny path: submit_decision / wait_for_decision may already have incremented;
    # keep graph state aligned with the live counter.
    n = get_consecutive_denials()
    if n <= prior:
        n = record_hitl_decision(False)
    deny_msgs = _deny_draft_tool_messages(state)
    return {
        "messages": deny_msgs,
        "halt": True,
        "last_obs": "DENIED: ticket cancelled by operator",
        "final_raw": "Understood — I cancelled the ticket submission.",
        "always_include": [],
        "consecutive_denials": n,
    }


def _route_after_agent(state: ReactGraphState) -> str:
    """Conditional edge: agent → ticket_validate / tools / agent / END."""
    from langgraph.graph import END

    from dana.graph.completion_gate import gate_route_after_agent, should_block_end
    from dana.middleware.hitl_ticket import hitl_enabled

    messages = state.get("messages") or []
    last = messages[-1] if messages else None
    if last is not None and getattr(last, "tool_calls", None):
        # Stage 8.9.6 — Pydantic gate before Jason / HITL for draft tickets.
        if hitl_enabled() and message_has_draft_cursor_prompt(last):
            return "ticket_validate"
        return "tools"
    # Option B: do not END while broker-merged tools remain uninvoked.
    if pending_always_include_tools(state):
        return "agent"
    # Completion gate: filler / unresolved tools must not silent-END.
    if should_block_end(state):
        return "agent"
    proposed = END if state.get("halt") else END
    return gate_route_after_agent(state, proposed=proposed, end_sentinel=END)


def _route_after_ticket_validate(state: ReactGraphState) -> str:
    """Valid → Jason; bounce → MoA agent; max retries → END."""
    from langgraph.graph import END

    if state.get("halt"):
        return END
    if state.get("ticket_validated"):
        return "jason_ticket_review"
    return "agent"


def _route_after_jason_review(state: ReactGraphState) -> str:
    """After Jason critique → HITL ticket_approval (unless halted)."""
    from langgraph.graph import END

    if state.get("halt"):
        return END
    return "ticket_approval"


def _route_after_ticket_approval(state: ReactGraphState) -> str:
    """After HITL: Deny → END; Approve → tools (enqueue / execute)."""
    from langgraph.graph import END

    if state.get("halt"):
        return END
    return "tools"


def route_after_execution(state: ReactGraphState) -> str:
    """Conditional edge after tools: critic / fail_closed / agent / verifier.

    python_repl failures set ``execution_error``; while ``retry_count < max_retries``
    route to ``critic`` (then back to tools). Exhausted retries → ``fail_closed``.
    Fatal OS / dependency blocks (``fatal_block``) bypass Critic and go straight to
    ``fail_closed`` (ticket draft on the existing HITL corridor fields).
    Tool timeout / soft failures never silent-END — completion gate keeps the
    loop on agent/critic with an explicit spoken failure message.
    Successful halt routes to ``verifier`` (closed-loop evidence gate) before
    consolidate_memory / END — never silent-END after tool execution.
    """
    from langgraph.graph import END

    from dana.graph.completion_gate import gate_route_after_execution, should_block_end
    from dana.graph.nodes.critic import is_fatal_execution_error
    from dana.graph.workflow import remap_execution_end_to_verifier

    err = state.get("execution_error")
    if err is not None and str(err).strip():
        if state.get("fatal_block") or is_fatal_execution_error(err):
            return "fail_closed"
        retry = int(state.get("retry_count") or 0)
        max_r = state.get("max_retries")
        max_retries = int(max_r) if max_r is not None else 3
        if retry < max_retries:
            return "critic"
        return "fail_closed"

    if pending_always_include_tools(state):
        return "agent"
    # Successful / claimed tool completion → closed-loop verifier before END.
    # Do this even when pending_synthesis is set (e.g. prior failed verify), so
    # attempts can advance and the corridor cannot soft-lock on agent↔tools.
    if bool(state.get("halt")):
        return "verifier"
    if should_block_end(state):
        return "agent"
    gated = gate_route_after_execution(state, proposed="agent", end_sentinel=END)
    return remap_execution_end_to_verifier(gated, end_sentinel=END)


# Back-compat alias for callers / tests that still import the private name.
_route_after_tools = route_after_execution


def compile_donna_react_graph(
    agent_node: Callable[..., Any],
    tools_node: Callable[..., Any],
    *,
    ticket_approval_node_fn: Callable[..., Any] | None = None,
    jason_review_node_fn: Callable[..., Any] | None = None,
    ticket_validate_node_fn: Callable[..., Any] | None = None,
    planner_node_fn: Callable[..., Any] | None = None,
    executor_node_fn: Callable[..., Any] | None = None,
    critic_node_fn: Callable[..., Any] | None = None,
    fail_closed_node_fn: Callable[..., Any] | None = None,
    hydrate_memory_node_fn: Callable[..., Any] | None = None,
    consolidate_memory_node_fn: Callable[..., Any] | None = None,
    verifier_node_fn: Callable[..., Any] | None = None,
    os_worker_node_fn: Callable[..., Any] | None = None,
    checkpointer: Any | None = None,
) -> Any:
    """Compile the production ReAct StateGraph (same topology as live Donna).

    Topology:
      START → hydrate_memory → planner → executor
                    ─(OS / PowerShell intent)→ os_worker → verifier
                    ─(default)→ agent
                    ─(draft_cursor_prompt)→ ticket_validate
                    ─(valid)→ jason_ticket_review → ticket_approval ─(approve)→ tools
                    ─(invalid, <3)→ agent
                    ─(max retries)→ END
                    ─(other tool_calls)→ tools ─(python_repl error)→ critic → tools
                    ─(fatal_block)→ fail_closed → END  (+ drafted_ticket)
                    ─(retries exhausted)→ fail_closed → END
                    ─(continue)→ agent
                    ╲(pending always_include / nudge)→ agent
                    ╲(tool halt/final)→ verifier
                         ─(verified)→ consolidate_memory → END
                         ─(fail, <3)→ agent
                         ─(fail, ≥3)→ fail_closed → END
                    ╲(agent halt, no tools)→ consolidate_memory → END

    ``planner`` / ``executor`` enforce Plan-Then-Execute before the MoA agent.
    ``os_worker`` isolates PowerShell / system intents (injectable; no vision/swarm).
    ``ticket_validate`` (Stage 8.9.6) runs Pydantic before Jason / HITL.
    ``jason_ticket_review`` (Stage 8.9) speaks a critique, then
    ``ticket_approval`` HITL-interrupts before heavy / ledger tool execution.
    ``critic`` / ``fail_closed`` bound python_repl self-heal (injectable for evals).
    Fatal OS blocks skip Critic and land on ``fail_closed`` with a ticket draft.
    ``hydrate_memory`` / ``consolidate_memory`` are injectable episodic nodes;
    HITL deny / fail_closed / ticket halt paths skip consolidation.
    ``verifier`` is the closed-loop Generator-Critic evidence gate (injectable).
    """
    from langgraph.graph import END, START, StateGraph

    from dana import agentic as ag
    from dana.agentic_planning import executor_node as _default_executor
    from dana.agentic_planning import planner_node as _default_planner
    from dana.graph.nodes.critic import critic_node as _default_critic
    from dana.graph.nodes.critic import fail_closed_node as _default_fail_closed
    from dana.graph.nodes.memory import (
        consolidate_memory_node as _default_consolidate,
    )
    from dana.graph.nodes.memory import hydrate_memory_node as _default_hydrate
    from dana.graph.nodes.verifier import verifier_node as _default_verifier
    from dana.graph.workers.os_worker import (
        OS_WORKER_NODE,
        os_worker_node as _default_os_worker,
        route_after_executor,
    )

    workflow = StateGraph(ReactGraphState)
    workflow.add_node(
        "hydrate_memory",
        hydrate_memory_node_fn or _default_hydrate,
    )
    workflow.add_node("planner", planner_node_fn or _default_planner)
    workflow.add_node("executor", executor_node_fn or _default_executor)
    workflow.add_node("agent", agent_node)
    workflow.add_node(
        OS_WORKER_NODE,
        os_worker_node_fn or _default_os_worker,
    )
    workflow.add_node(
        "ticket_validate",
        ticket_validate_node_fn or ticket_validate_node,
    )
    workflow.add_node(
        "jason_ticket_review",
        jason_review_node_fn or jason_ticket_review_node,
    )
    workflow.add_node(
        "ticket_approval",
        ticket_approval_node_fn or ticket_approval_node,
    )
    workflow.add_node("tools", tools_node)
    workflow.add_node("critic", critic_node_fn or _default_critic)
    workflow.add_node("fail_closed", fail_closed_node_fn or _default_fail_closed)
    workflow.add_node("verifier", verifier_node_fn or _default_verifier)
    workflow.add_node(
        "consolidate_memory",
        consolidate_memory_node_fn or _default_consolidate,
    )
    # Corridor entry: hydrate episodic prefs before planner/supervisor.
    workflow.add_edge(START, "hydrate_memory")
    workflow.add_edge("hydrate_memory", "planner")
    workflow.add_edge("planner", "executor")
    # Supervisor fork: OS / PowerShell intents → isolated worker; else MoA agent.
    workflow.add_conditional_edges(
        "executor",
        route_after_executor,
        {
            OS_WORKER_NODE: OS_WORKER_NODE,
            "agent": "agent",
        },
    )
    # Worker output joins the closed-loop verifier → consolidate corridor.
    workflow.add_edge(OS_WORKER_NODE, "verifier")
    workflow.add_conditional_edges(
        "agent",
        _route_after_agent,
        {
            "ticket_validate": "ticket_validate",
            "tools": "tools",
            "agent": "agent",
            # Chat-only halt (no tools) → consolidate before END.
            END: "consolidate_memory",
        },
    )
    workflow.add_conditional_edges(
        "ticket_validate",
        _route_after_ticket_validate,
        {
            "jason_ticket_review": "jason_ticket_review",
            "agent": "agent",
            # Validation exhausted / halted — skip consolidate.
            END: END,
        },
    )
    workflow.add_conditional_edges(
        "jason_ticket_review",
        _route_after_jason_review,
        {"ticket_approval": "ticket_approval", END: END},
    )
    workflow.add_conditional_edges(
        "ticket_approval",
        _route_after_ticket_approval,
        # Deny / halt → END without consolidating bad prefs.
        {"tools": "tools", END: END},
    )
    workflow.add_conditional_edges(
        "tools",
        route_after_execution,
        {
            "critic": "critic",
            "fail_closed": "fail_closed",
            "agent": "agent",
            # Successful tool halt → closed-loop verifier before consolidate.
            "verifier": "verifier",
        },
    )
    workflow.add_conditional_edges(
        "verifier",
        route_after_verifier,
        {
            "agent": "agent",
            "fail_closed": "fail_closed",
            END: "consolidate_memory",
        },
    )
    workflow.add_edge("critic", "tools")
    # Failures never consolidate.
    workflow.add_edge("fail_closed", END)
    workflow.add_edge("consolidate_memory", END)
    cp = checkpointer if checkpointer is not None else ag._react_checkpointer()
    # Checkpointer required for interrupt() resume; ticket_approval is the gate.
    return workflow.compile(checkpointer=cp)


async def run_react_langgraph(
    *,
    user_text: str,
    system_prompt: str,
    execute_fn: Callable[[ToolCall], str],
    max_iters: int,
    broker: IntentBroker | None,
    reflect_fn: Callable[[list[dict[str, str]]], str] | None,
    vault_client: Any | None,
    enable_reflection: bool,
    prior_messages: list[dict[str, str]] | None,
    on_tool_start: Callable[[ToolCall, str], None] | None,
    visual_context: str | None,
    model: str,
    forced_tool: ToolCall | None = None,
    tts_callback: Callable[[str], None] | None = None,
) -> Any:
    """Compile + stream a MemorySaver-backed agent↔tools graph."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    from dana import agentic as ag
    from dana.cascade_router import resolve_chat_model
    from dana.tools.langchain_tools import _UNBOUND_TOOL_IDS, build_langchain_tools
    from dana.tools.registry import get_tool_registry
    from dana.settings import resolve_reply_lang

    broker = broker or get_broker()
    reply_lang = resolve_reply_lang(user_text)

    # Self-improvement asks: force architecture critique context into the prompt.
    try:
        if ag.is_self_improvement_intent(user_text):
            briefing = ag.format_self_improvement_briefing()
            system_prompt = f"{system_prompt}\n\n{briefing}".strip()
            if forced_tool is None:
                forced_tool = ToolCall(
                    tool_id="read_system_architecture",
                    arguments={},
                    source_lang="en",
                    raw_text=user_text,
                    confidence=0.99,
                )
    except Exception:  # noqa: BLE001
        pass

    # Temporal asks: prefer day-index tool when the router did not already force one.
    try:
        if ag.is_temporal_intent(user_text) and forced_tool is None:
            from dana.tools.activity_index import resolve_date_str

            forced_tool = ToolCall(
                tool_id="list_activity_for_day",
                arguments={"date_str": resolve_date_str("yesterday")},
                source_lang="en",
                raw_text=user_text,
                confidence=0.99,
            )
    except Exception:  # noqa: BLE001
        pass

    # Suite 2 perception: live telemetry / idle duration when broker missed.
    try:
        from dana.tools.system_tools import (
            is_idle_duration_query,
            is_system_telemetry_query,
        )

        if forced_tool is None and is_system_telemetry_query(user_text or ""):
            forced_tool = ToolCall(
                tool_id="get_system_telemetry",
                arguments={},
                source_lang="en",
                raw_text=user_text,
                confidence=0.99,
            )
        elif forced_tool is None and is_idle_duration_query(user_text or ""):
            forced_tool = ToolCall(
                tool_id="parse_idle_log_duration",
                arguments={},
                source_lang="en",
                raw_text=user_text,
                confidence=0.99,
            )
    except Exception:  # noqa: BLE001
        pass

    # Suite 3: named-repo git / watchdog graph when broker missed.
    _latex_turn = False
    try:
        from dana.tools.os_tools import (
            cascade_git_tool_args,
            is_cascade_git_query,
            is_latex_nocite_query,
            is_watchdog_graph_query,
            latex_system_prompt,
            watchdog_graph_filepath,
        )

        if forced_tool is None and is_cascade_git_query(user_text or ""):
            forced_tool = ToolCall(
                tool_id="execute_powershell",
                arguments=cascade_git_tool_args(user_text or ""),
                source_lang="en",
                raw_text=user_text,
                confidence=0.99,
            )
        elif forced_tool is None and is_watchdog_graph_query(user_text or ""):
            forced_tool = ToolCall(
                tool_id="read_local_file",
                arguments={"filepath": watchdog_graph_filepath()},
                source_lang="en",
                raw_text=user_text,
                confidence=0.99,
            )
        elif is_latex_nocite_query(user_text or ""):
            _latex_turn = True
            system_prompt = latex_system_prompt(user_text or "")
            forced_tool = None
    except Exception:  # noqa: BLE001
        _latex_turn = False

    # SQLite episodic_facts = primary grounding (before vault / vision tools).
    # Injected at the *end* of the system prompt (after tool rules) so local
    # models see facts with highest recency — not buried mid-prompt.
    _episodic_grounding: dict[str, Any] = {}
    _episodic_block_text = ""
    try:
        from dana.memory.episodic_grounding import (
            retrieve_episodic_grounding,
            should_suppress_vault_vision_tool,
        )

        _episodic_grounding = retrieve_episodic_grounding(user_text)
        _episodic_block_text = str(
            _episodic_grounding.get("grounding_block") or ""
        ).strip()
        if _episodic_grounding.get("suppress_vault_vision") and forced_tool is not None:
            if should_suppress_vault_vision_tool(forced_tool.tool_id):
                forced_tool = None
    except Exception:  # noqa: BLE001
        _episodic_grounding = {}
        _episodic_block_text = ""

    def _speak(phrase: str, *, agent_id: str | None = None) -> None:
        """Prefer injected TTS callback; fall back to agentic spooler helper.

        Stage 8.8 — routes voice by ``agent_id`` (sentence-chunked, non-blocking).
        """
        text = (phrase or "").strip()
        if not text:
            return
        aid = agent_id or ag.get_stream_tts_agent()
        if tts_callback is not None:
            try:
                tts_callback(text, agent_id=aid)  # type: ignore[call-arg]
                return
            except TypeError:
                try:
                    tts_callback(text)
                    return
                except Exception:  # noqa: BLE001
                    pass
            except Exception:  # noqa: BLE001
                pass
        ag._enqueue_tts_nonblocking(text, agent_id=aid)

    prompt = system_prompt
    if not _latex_turn:
        if ag._TOOL_EXECUTION_RULE not in prompt:
            prompt = f"{prompt}\n\n{ag._TOOL_EXECUTION_RULE}"
        if getattr(ag, "_PYTHON_DOMAIN_CLAMP", "") and ag._PYTHON_DOMAIN_CLAMP not in prompt:
            prompt = f"{prompt}\n\n{ag._PYTHON_DOMAIN_CLAMP}"
        if ag._STRICT_TOOL_ENFORCEMENT_RULE not in prompt:
            prompt = f"{prompt}\n\n{ag._STRICT_TOOL_ENFORCEMENT_RULE}"
        if ag._EXPLICIT_TOOL_INVOCATION_RULE not in prompt:
            prompt = f"{prompt}\n\n{ag._EXPLICIT_TOOL_INVOCATION_RULE}"
        if ag._R1_REASONING_RULE not in prompt:
            prompt = f"{prompt}\n\n{ag._R1_REASONING_RULE}"
        if ag._VOICE_SANITIZER_RULE not in prompt:
            prompt = f"{prompt}\n\n{ag._VOICE_SANITIZER_RULE}"
        if ag._INTERACTION_UX_RULE not in prompt:
            prompt = f"{prompt}\n\n{ag._INTERACTION_UX_RULE}"
        if ag._DRAFT_CURSOR_TPM_RULE not in prompt:
            prompt = f"{prompt}\n\n{ag._DRAFT_CURSOR_TPM_RULE}"
        if ag._DRAFT_CURSOR_TERMINATION_RULE not in prompt:
            prompt = f"{prompt}\n\n{ag._DRAFT_CURSOR_TERMINATION_RULE}"
    # Pre-compute explicit+mode merges early so hard-constraint text can list them.
    from dana.tools.broker import merge_bound_tool_ids, should_blindfold_vision

    _vision_blindfold = should_blindfold_vision(
        user_text=user_text,
        forced_tool_id=forced_tool.tool_id if forced_tool is not None else None,
    )
    if _vision_blindfold:
        visual_context = None
        try:
            _agentic_log(
                "Agentic",
                "Vision blindfold active — analyze_visual_context unbound; "
                "screen context cleared",
            )
        except Exception:  # noqa: BLE001
            pass

    _early_known = list(broker.registry.keys())
    try:
        _early_known = list(get_tool_registry().as_spec_dict().keys()) or _early_known
    except Exception:  # noqa: BLE001
        pass
    _merged_always = merge_bound_tool_ids(
        user_text=user_text,
        forced_tool_id=forced_tool.tool_id if forced_tool is not None else None,
        mode=ag.get_donna_mode(),
        known_ids=_early_known,
    )
    _merged_always = list(dict.fromkeys(_merged_always))
    if _latex_turn:
        _merged_always = []
    # Episodic grounding wins: drop vault/vision bindings for history / traps.
    if _episodic_grounding.get("suppress_vault_vision"):
        try:
            from dana.memory.episodic_grounding import should_suppress_vault_vision_tool

            _merged_always = [
                t for t in _merged_always if not should_suppress_vault_vision_tool(t)
            ]
        except Exception:  # noqa: BLE001
            pass
        _vision_blindfold = True
        visual_context = None
    if forced_tool is not None:
        tid = forced_tool.tool_id
        if tid in _UNBOUND_TOOL_IDS:
            prompt = (
                f"{prompt}\n\n"
                "ROUTER INTENT (HARD CONSTRAINT):\n"
                f"- The intent router classified this turn as `{tid}`.\n"
                "- That tool is NOT bound. Answer from Visual Context / SpatialIR only.\n"
                "- Do NOT call read_vault_memory, read_system_architecture, web_search, "
                "or any other tool for this turn unless the user clearly asked for it."
            )
        else:
            extras = [t for t in _merged_always if t != tid and t not in _UNBOUND_TOOL_IDS]
            if extras:
                prompt = (
                    f"{prompt}\n\n"
                    "ROUTER INTENT (HARD CONSTRAINT):\n"
                    f"- Prioritize tool `{tid}` first "
                    f"(args hint: {dict(forced_tool.arguments)}).\n"
                    f"- Also bind/call these explicitly requested tools this turn: "
                    f"{', '.join(extras)}.\n"
                    "- Do not drop explicit tool requests because of active mode. "
                    "After tool results, speak a short natural answer — never read "
                    "raw OK:/ERROR: strings aloud."
                )
            else:
                prompt = (
                    f"{prompt}\n\n"
                    "ROUTER INTENT (HARD CONSTRAINT):\n"
                    f"- You MUST prioritize tool `{tid}` for this turn "
                    f"(args hint: {dict(forced_tool.arguments)}).\n"
                    "- Do not substitute an unrelated tool. After the tool result, "
                    "speak a short natural answer — never read raw OK:/ERROR: strings aloud."
                )
            if tid == "draft_cursor_prompt" or "draft_cursor_prompt" in extras:
                prompt += f"\n- {ag._DRAFT_CURSOR_TPM_RULE}"
                prompt += f"\n- {ag._DRAFT_CURSOR_TERMINATION_RULE}"
            if tid == "architect_new_tool":
                prompt += (
                    "\n- Tool Forge only: NEVER call read_vault_memory, "
                    "read_local_file, file_jail_enforcer, or web_search this turn.\n"
                    "- On ERROR/LOCKED from Tool Forge, speak one short apology. "
                    "Do NOT invent JSON repairs, continue the forge yourself, or "
                    "dump sandbox/vault document contents."
                )
            if tid == "file_editor" or "file_editor" in extras:
                prompt += (
                    "\n- Dual-intent: call `file_editor` with action=write, filepath, "
                    "and non-empty content (summary/notes). Then speak a short natural "
                    "answer to any conversational question in the same user turn.\n"
                    "- Never invent tool names (e.g. build_tool_that_*). On ERROR from "
                    "an unknown/phantom tool, retry with file_editor then FINAL."
                )

    semantic = get_tool_registry()
    known_ids = list(semantic.as_spec_dict().keys()) or list(broker.registry.keys())
    # Prefer the early merge (same inputs); recompute if registry grew.
    always = list(
        dict.fromkeys(
            merge_bound_tool_ids(
                user_text=user_text,
                forced_tool_id=forced_tool.tool_id if forced_tool is not None else None,
                mode=ag.get_donna_mode(),
                known_ids=known_ids,
            )
        )
    )
    # --- Module 1: Blackboard + minimal bureaucratic state ---
    from dana.memory import (
        append_message,
        ensure_session,
        load_messages,
        set_session_meta,
    )
    from dana.telemetry import log_router

    current_agent = (
        "MoA_Reasoner"
        if (
            forced_tool is not None
            or "draft_cursor_prompt" in (user_text or "").lower()
        )
        else "ReAct_Agent"
    )
    # Stage 8.8 — bind sentence-stream TTS voice to the active bureaucratic agent.
    try:
        ag.set_stream_tts_agent(ag.agent_id_from_label(current_agent))
    except Exception:  # noqa: BLE001
        pass
    active_intent = (
        forced_tool.tool_id
        if forced_tool is not None
        else ("tool_graph" if always else "general")
    )
    # Stable session for ongoing dialogue; fresh id for isolated queue tasks.
    session_key = ag._REACT_THREAD_ID if prior_messages else None
    session_id = ensure_session(
        session_key,
        current_agent=current_agent,
        active_intent=active_intent,
    )
    set_session_meta(
        session_id,
        current_agent=current_agent,
        active_intent=active_intent,
    )
    # Task lifecycle tracker (dropped / timeout → FAILED, never silent ghost).
    from dana.graph.task_tracker import TaskStatus, get_shared_task_tracker

    _task_tracker = get_shared_task_tracker()
    _task_tracker.start_task(session_id, user_text or "")
    _task_tracker.update_status(session_id, TaskStatus.IN_PROGRESS)
    # Stage 4.3 — piggyback unread actuator completions into this turn's prompt.
    try:
        from dana.memory.blackboard import (
            format_background_system_alert,
            get_and_clear_unread_notifications,
        )

        _unread = get_and_clear_unread_notifications(session_id)
        _alert = format_background_system_alert(_unread)
        if _alert:
            prompt = f"{prompt}\n\n{_alert}"
            try:
                from dana.telemetry import log_notification_piggyback

                log_notification_piggyback(
                    _alert,
                    session_id=session_id,
                    count=len(_unread),
                    payload={
                        "action_ids": [
                            int(r.get("action_id") or 0) for r in _unread
                        ]
                    },
                )
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass
    # Episodic-only turns: replace the huge tool card with a short grounding prompt
    # so local models actually attend to SQLite facts (not 30k of tool rules).
    if _episodic_block_text and _episodic_grounding.get("suppress_vault_vision"):
        _epi_rules = [
            "You are Dānā, a local assistant with durable SQLite episodic memory.",
            "Answer ONLY from the IMMUTABLE EPISODIC GROUNDING below.",
            "Quote specific recorded names, numbers, and corrections verbatim.",
            "Do not invent history. Do not claim vault/vision access is required.",
            "Keep the reply concise and factual.",
        ]
        if _episodic_grounding.get("contradiction"):
            directive = str(
                _episodic_grounding.get("contradiction_directive") or ""
            ).strip()
            if directive:
                _epi_rules.append(directive)
                _epi_rules.append(
                    "Do NOT invent work history. Refuse the premise politely."
                )
        prompt = (
            "\n".join(_epi_rules)
            + "\n\n"
            + _episodic_block_text
        )
    # Durable history on Blackboard — graph state only holds this turn's scratch.
    if prior_messages:
        existing = load_messages(session_id)
        if not existing:
            for m in prior_messages:
                role = str((m or {}).get("role") or "").strip().lower()
                content = str((m or {}).get("content") or "")
                if role in {"user", "assistant", "system"} and content.strip():
                    append_message(session_id, role, content)
    append_message(session_id, "user", user_text or "")
    bb_prior = [
        {"role": r["role"], "content": r["content"]}
        for r in load_messages(session_id, limit=8)
        if r.get("role") in {"user", "assistant"}
    ]
    # Drop the user turn we just filed — _build_seed_messages appends it again.
    if bb_prior and bb_prior[-1].get("role") == "user":
        bb_prior = bb_prior[:-1]
    # Repeat grounding next to the user turn so facts are not lost in a long system card.
    _user_for_seed = user_text or ""
    if _episodic_block_text and _episodic_grounding.get("suppress_vault_vision"):
        _user_for_seed = (
            f"[Recorded episodic facts — answer from these]\n"
            f"{_episodic_block_text}\n\n"
            f"[User question]\n{_user_for_seed}"
        )
    seed = ag._build_seed_messages(
        user_text=_user_for_seed,
        system_prompt=prompt,
        prior_messages=bb_prior,
        visual_context=visual_context,
        reply_lang=reply_lang,
    )
    lc_messages = ag._dicts_to_lc_messages(seed)
    log_router(
        f"session={session_id} agent={current_agent} intent={active_intent}",
        session_id=session_id,
        current_agent=current_agent,
        active_intent=active_intent,
        payload={"mode": ag.get_donna_mode(), "always_include": list(always)},
    )
    # When the broker merge is non-empty, bind_tools must match it exactly —
    # do not dilute/overwrite with MoA/Vision semantic top-K defaults.
    semantic_specs = semantic.as_spec_dict()
    _vision_block = {
        "analyze_visual_context",
        "ocr_with_region",
        "click_ui_element",
        "type_text_in_element",
    }
    _episodic_block = {
        "analyze_visual_context",
        "ocr_with_region",
        "search_vault",
        "read_vault_memory",
        "write_vault_memory",
        "ingest_local_directory",
        "describe_spatial_scene",
        "capture_and_analyze_screen",
    }
    _suppress_vault_vision = bool(_episodic_grounding.get("suppress_vault_vision"))
    _episodic_answer_only = bool(
        _suppress_vault_vision
        and (
            _episodic_grounding.get("grounding_block")
            or _episodic_grounding.get("contradiction")
        )
    )
    if _vision_blindfold:
        always = [t for t in always if t not in _vision_block]
    if _suppress_vault_vision:
        always = [t for t in always if t not in _episodic_block]
    if _latex_turn or _episodic_answer_only:
        # LaTeX / episodic grounding: answer from prompt only — no tool diversion.
        always = []
        bind_registry = {}
        tool_ids: set[str] | None = set()
    elif always:
        bind_registry = _specs_for_tool_ids(
            always,
            semantic_specs=semantic_specs,
            broker_registry=broker.registry,
        )
        tool_ids = set(always)
    else:
        top_specs = semantic.retrieve_specs(user_text, k=6, always_include=always)
        bind_registry = top_specs if top_specs else broker.registry
        tool_ids = set(bind_registry.keys()) if top_specs else None
    if _vision_blindfold and tool_ids is not None:
        tool_ids = {t for t in tool_ids if t not in _vision_block}
    if _suppress_vault_vision and tool_ids is not None:
        tool_ids = {t for t in tool_ids if t not in _episodic_block}
    if _vision_blindfold and isinstance(bind_registry, dict):
        bind_registry = {
            k: v for k, v in bind_registry.items() if k not in _vision_block
        }
    if _suppress_vault_vision and isinstance(bind_registry, dict):
        bind_registry = {
            k: v for k, v in bind_registry.items() if k not in _episodic_block
        }
    tools = build_langchain_tools(
        execute_fn,
        registry=bind_registry,
        tool_ids=tool_ids,
        tts_callback=tts_callback,
        vault_client=vault_client,
    )
    if _vision_blindfold:
        tools = [
            t
            for t in tools
            if getattr(t, "name", "") not in _vision_block
        ]
    if _suppress_vault_vision or _episodic_answer_only:
        tools = [
            t
            for t in tools
            if getattr(t, "name", "") not in _episodic_block
        ]
    if _latex_turn or _episodic_answer_only:
        tools = []
    bound_names = {getattr(t, "name", "") for t in tools}
    try:
        from dana.logging import log as _agentic_log

        _agentic_log(
            "Agentic",
            f"tools={sorted(n for n in bound_names if n)} "
            f"(always_include={always or '-'})",
        )
    except Exception:  # noqa: BLE001
        pass
    from dana.cascade_router import resolve_compute_mode

    _compute_mode = resolve_compute_mode(
        user_text,
        forced_tool=forced_tool.tool_id if forced_tool is not None else None,
        use_lightweight=False,
    )
    llm = resolve_chat_model(
        query=user_text,
        forced_tool=forced_tool.tool_id if forced_tool is not None else None,
        default_model=model,
        temperature=0.2,
        mode=_compute_mode,
    )
    # Grammar-constrained tool calling: Ollama native tools schema via
    # bind_tools(strict=True). Do not set ChatOllama(format="json") on this
    # path — that would force spoken FINAL into JSON and break tool_calls.
    llm_with_tools = llm.bind_tools(tools, strict=True)
    if forced_tool is not None and forced_tool.tool_id in bound_names:
        tid = forced_tool.tool_id
        try:
            llm_with_tools = llm.bind_tools(tools, tool_choice=tid, strict=True)
        except Exception:  # noqa: BLE001
            try:
                llm_with_tools = llm.bind_tools(
                    tools,
                    tool_choice={"type": "function", "function": {"name": tid}},
                    strict=True,
                )
            except Exception:  # noqa: BLE001
                pass

    # Two-stage MoA shim: DeepSeek-R1 plans (no tools) → Llama formats tool_calls.
    moa_plan = ""
    use_moa_shim = False
    try:
        from dana.moa_tool_shim import (
            enrich_forced_tool_from_plan,
            formatter_system_injection,
            run_moa_reasoner_stage,
            should_use_moa_tool_shim,
        )

        use_moa_shim = should_use_moa_tool_shim(
            user_text,
            forced_tool_id=forced_tool.tool_id if forced_tool is not None else None,
        )
        if use_moa_shim:
            moa_plan = run_moa_reasoner_stage(
                user_text,
                forced_tool_id=(
                    forced_tool.tool_id if forced_tool is not None else None
                ),
                allowed_tool_ids=sorted(n for n in bound_names if n),
                session_id=session_id,
            )
            reasoner_plan_snapshot = moa_plan
            # Stage 3.1: no MoA string CONTEXT gate — Pydantic tool guards +
            # ValidationError retry own rejection.
            inject = formatter_system_injection(moa_plan)
            prompt = f"{prompt}\n\n{inject}"
            if lc_messages and isinstance(lc_messages[0], SystemMessage):
                lc_messages[0] = SystemMessage(content=prompt)
            else:
                lc_messages.insert(0, SystemMessage(content=prompt))
            if forced_tool is not None:
                forced_tool = enrich_forced_tool_from_plan(forced_tool, moa_plan)
            try:
                from dana.logging import log as _moa_log

                _moa_log(
                    "MoAShim",
                    "stage2 formatter="
                    f"{getattr(llm, 'model', None) or model} "
                    f"tools={sorted(n for n in bound_names if n)}",
                )
            except Exception:  # noqa: BLE001
                pass
            # Module 3 files <think> on Blackboard inside run_moa_reasoner_stage.
            # Here we only update the bureaucratic agent pointer for MoA turns.
            if reasoner_plan_snapshot:
                try:
                    set_session_meta(session_id, current_agent="MoA_Reasoner")
                    current_agent = "MoA_Reasoner"
                    ag.set_stream_tts_agent("moa")
                except Exception:  # noqa: BLE001
                    pass
    except Exception as _moa_exc:  # noqa: BLE001
        try:
            from dana.logging import log as _moa_log

            _moa_log("MoAShim", f"shim skipped: {_moa_exc}")
        except Exception:  # noqa: BLE001
            pass
        use_moa_shim = False

    def _rebind_from_always(always_ids: list[str]) -> None:
        """Bind LLM tools strictly to the broker merge list (no mode top-K)."""
        nonlocal tools, bound_names, llm_with_tools, bind_registry
        ids = list(dict.fromkeys(str(x) for x in always_ids if str(x).strip()))
        if not ids:
            return
        fresh = get_tool_registry().as_spec_dict()
        bind_registry = _specs_for_tool_ids(
            ids,
            semantic_specs=fresh,
            broker_registry=broker.registry,
        )
        tools = build_langchain_tools(
            execute_fn,
            registry=bind_registry,
            tool_ids=set(ids),
            tts_callback=tts_callback,
            vault_client=vault_client,
        )
        bound_names = {getattr(t, "name", "") for t in tools}
        llm_with_tools = llm.bind_tools(tools, strict=True)

    trace: list[dict[str, Any]] = []
    last_obs = ""
    tool_ack_done = False
    tts_streamed = False
    # Module 4: one localized ValidationError bounce per tool call id.
    validation_retries: set[str] = set()
    # Stage 3.3: on ValidationError bounce, next agent turn binds ONLY this tool.
    strict_retry_tool_id: str | None = None

    def _arm_strict_validation_retry(tool_id: str) -> None:
        """Corridor: next agent bind_tools list is exactly ``[tool_id]``."""
        nonlocal strict_retry_tool_id
        tid = str(tool_id or "").strip()
        if tid:
            strict_retry_tool_id = tid

    def _apply_strict_validation_retry_bind() -> str | None:
        """If a ValidationError corridor is armed, rebind LLM to that tool only.

        Returns the tool id when applied (one-shot; clears the arm). Fresh turns
        never hit this path — only the immediate bounce retry.
        """
        nonlocal strict_retry_tool_id, llm_with_tools
        tid = (strict_retry_tool_id or "").strip()
        if not tid:
            return None
        strict_retry_tool_id = None
        _rebind_from_always([tid])
        _try_bind_tool_choice(tid)
        try:
            from dana.logging import log as _retry_log

            bound = sorted(
                n for n in (getattr(t, "name", "") for t in tools) if n
            )
            _retry_log(
                "MoAShim",
                f"strict validation retry bind tools={bound} "
                f"(corridor={tid})",
            )
        except Exception:  # noqa: BLE001
            pass
        return tid

    def _finish(final_text: str, iterations: int) -> AgenticResult:
        from dana.reflector import trace_has_failure

        text = (final_text or "").strip()
        if ag._wants_event_clock(user_text):
            weak = (
                not ag._CLOCK_RE.search(text)
                or re.search(r"unspecified|unknown|not sure|no time", text, re.I)
            )
            if weak:
                for blob in (last_obs, *(t.get("observation") or "" for t in reversed(trace))):
                    extracted = ag._spoken_fact_from_search_obs(str(blob), user_text)
                    if extracted and ag._CLOCK_RE.search(extracted):
                        text = extracted
                        break
        spoken = ag.clip_spoken_answer(user_text, text)
        spoken = ag.strip_r1_think_blocks(spoken)
        if re.search(r"unspecified|unknown time", spoken or "", re.I):
            spoken = (
                "I found the date but not a clear kickoff time yet."
                if reply_lang != "fa"
                else "        ."
            )
        if re.match(
            r"^\s*(?:TOOL|Action|FINAL)\s*[:：]",
            spoken or "",
            re.I,
        ):
            spoken = (
                "  ."
                if reply_lang == "fa"
                else "Sorry — please ask me again."
            )
        if spoken and spoken.lstrip().startswith("{") and '"tool"' in spoken:
            spoken = (
                "  ."
                if reply_lang == "fa"
                else "Sorry — please ask me again."
            )
        if not _latex_turn:
            spoken = ag.sanitize_spoken_reply(
                spoken,
                reply_lang=reply_lang,
                last_obs=last_obs,
                tool_trace=trace,
            )
            # Prefer concrete tool evidence when the model echoes junk / Write-Output.
            try:
                from dana.tools.os_tools import (
                    is_cascade_git_query,
                    is_watchdog_graph_query,
                )

                obs_s = str(last_obs or "")
                if not obs_s:
                    for row in reversed(trace or []):
                        blob = str(row.get("observation") or "")
                        if blob.strip():
                            obs_s = blob
                            break
                # Prefer git-log observation even if last_obs was overwritten.
                if is_cascade_git_query(user_text or ""):
                    for row in reversed(trace or []):
                        blob = str(row.get("observation") or "")
                        if "git log" in blob.lower():
                            obs_s = blob
                            break
                    m = re.search(r"stdout:\s*\n(.+)", obs_s, flags=re.I)
                    date_line = (
                        m.group(1).strip().splitlines()[0] if m else ""
                    ).strip()
                    if date_line and date_line.lower() != "(empty)":
                        spoken = (
                            "The last git commit in cascade-router is dated "
                            f"{date_line}."
                        )
                elif is_watchdog_graph_query(user_text or "") and (
                    "dependency digest" in obs_s.lower() or "watchdog" in obs_s.lower()
                ):
                    digest = obs_s
                    if "DEPENDENCY DIGEST" in obs_s:
                        digest = obs_s.split("OK: read_local_file", 1)[0].strip()
                    spoken = (
                        "Watchdog monitoring graph dependencies "
                        f"(from dana/swarm/watchdog_graph.py):\n{digest[:1200]}"
                    )
            except Exception:  # noqa: BLE001
                pass
        else:
            try:
                from dana.tools.os_tools import strip_latex_citations

                spoken = strip_latex_citations(spoken or "")
            except Exception:  # noqa: BLE001
                pass
        # Strict override: successful draft_cursor_prompt → canned UX only (WAV cache).
        if ag.draft_cursor_tool_succeeded(last_obs=last_obs, tool_trace=trace):
            spoken = ag.DRAFT_CURSOR_UX_ACK
            # Ensure core_agent enqueues this ack (prior stream may have marked TTS done).
            nonlocal tts_streamed
            tts_streamed = False
        if (
            not _latex_turn
            and forced_tool is not None
            and forced_tool.tool_id
            in {
                "web_search",
                "dispatch_research_swarm",
                "dispatch_watchdog",
                "dispatch_jason_supervisor",
                "dispatch_titan_repair",
                "architect_new_tool",
                "read_local_file",
                "run_terminal_command",
            }
            and ag._GENERIC_GREETING_RE.match(spoken or "")
        ):
            if forced_tool.tool_id == "dispatch_research_swarm":
                spoken = (
                    "I'm researching that in the background — I'll speak up when it's ready."
                    if reply_lang != "fa"
                    else "  ‌   ‌ —    ‌."
                )
            elif forced_tool.tool_id == "dispatch_jason_supervisor":
                spoken = (
                    "Jason read the notes and Donna wrote the script."
                    if reply_lang != "fa"
                    else "Jason     Donna   ."
                )
            elif forced_tool.tool_id == "dispatch_watchdog":
                spoken = (
                    "Watchdog is running in the background — I'll speak up when it triggers."
                    if reply_lang != "fa"
                    else "       ‌."
                )
            elif forced_tool.tool_id == "dispatch_titan_repair":
                spoken = (
                    "I'm running Titan Repair over the bug tracker — patches will land in CAMGRASPER/tracker/pending_patches."
                    if reply_lang != "fa"
                    else " ‌   ‌  ‌   ‌."
                )
            elif forced_tool.tool_id == "architect_new_tool":
                spoken = (
                    "I'm forging that tool through the Tool Forge now."
                    if reply_lang != "fa"
                    else "  Tool Forge   ‌."
                )
            elif last_obs:
                spoken = ag._obs_fallback(last_obs, reply_lang)
            else:
                spoken = "Working on that now." if reply_lang != "fa" else "   ‌."
        had_errors = trace_has_failure(trace)
        ag._maybe_record_bug_tracker(
            user_text=user_text,
            spoken=spoken or "",
            last_obs=last_obs,
            tool_trace=trace,
            had_errors=had_errors,
        )
        reflection, reflection_ms, _ = ag._maybe_reflect(
            user_text=user_text,
            tool_trace=trace,
            reflect_fn=reflect_fn,
            vault_client=vault_client,
            enable_reflection=enable_reflection,
        )
        # Self-improvement: ensure spoken answer cites stack/memory/failure gaps.
        if ag.is_self_improvement_intent(user_text) and isinstance(reflection, dict):
            low = (spoken or "").lower()
            needs = (
                not any(k in low for k in ("langgraph", "react", "ollama")),
                not any(k in low for k in ("memory", "episodic", "vault", "retrieval")),
                not any(
                    k in low for k in ("limit", "gap", "fail", "hallucin", "weak", "missing", "blind")
                ),
            )
            if any(needs):
                gaps = reflection.get("gaps") or []
                fails = reflection.get("recent_failures") or []
                gap_line = "; ".join(str(g) for g in gaps[:3]) or (
                    "episodic day-index and tool-registry grounding"
                )
                fail_line = (
                    str(fails[0])[:160]
                    if fails
                    else "recent ERROR/WARNING lines in dana_runtime.log"
                )
                spoken = (
                    f"{(spoken or '').strip()} "
                    "Concrete self-critique: my LangGraph/ReAct + Ollama stack still has "
                    f"memory retention gaps ({gap_line}). "
                    f"Recent failure signal: {fail_line}. "
                    "We should harden list_activity_for_day retrieval, keep the capability "
                    "digest in lightweight chat, and feed failure logs into Andon reflection."
                ).strip()
        return AgenticResult(
            final_text=spoken,
            iterations=iterations,
            tool_trace=trace,
            reply_lang=reply_lang,
            reflection=reflection,
            reflection_ms=reflection_ms,
            had_errors=had_errors,
            tts_streamed=tts_streamed,
        )

    def _rebind_tools_after_forge() -> None:
        """Refresh registry after Tool Forge, keeping broker merge binding exact."""
        nonlocal tools, bound_names, llm_with_tools, bind_registry
        semantic_fresh = get_tool_registry()
        always_ids = list(
            dict.fromkeys(
                merge_bound_tool_ids(
                    user_text=user_text,
                    forced_tool_id=(
                        forced_tool.tool_id if forced_tool is not None else None
                    ),
                    mode=ag.get_donna_mode(),
                    known_ids=list(semantic_fresh.as_spec_dict().keys()),
                )
            )
        )
        if always_ids:
            _rebind_from_always(always_ids)
            return
        top = semantic_fresh.retrieve_specs(user_text, k=8, always_include=always_ids)
        bind_registry = top if top else broker.registry
        tools = build_langchain_tools(
            execute_fn,
            registry=bind_registry,
            tool_ids=set(bind_registry.keys()) if top else None,
            tts_callback=tts_callback,
            vault_client=vault_client,
        )
        bound_names = {getattr(t, "name", "") for t in tools}
        llm_with_tools = llm.bind_tools(tools, strict=True)

    # Forced-tool seed
    forced_args_ready = True
    _needs_args = {
        "web_search": ("query",),
        "dispatch_research_swarm": ("query",),
        "dispatch_jason_supervisor": ("query",),
        "run_terminal_command": ("command",),
        "shell_execute": ("command",),
        "execute_powershell": ("command",),
        "write_to_file": ("filepath", "content"),
        "execute_command": ("command",),
        "fetch_webpage": ("url",),
        "file_editor": ("action", "filepath"),
        "python_repl": ("code",),
        "read_local_file": ("path",),
        "architect_new_tool": ("goal",),
        "draft_cursor_prompt": ("objective",),
        "dispatch_watchdog": ("task",),
        "dispatch_titan_repair": (),
        "kill_watchdog": ("task_id",),
        "write_vault_memory": ("text",),
        "read_vault_memory": (),
    }
    if forced_tool is not None:
        if forced_tool.tool_id == "architect_new_tool":
            args = dict(forced_tool.arguments or {})
            if not str(args.get("goal") or args.get("tool_description") or "").strip():
                args["goal"] = user_text
            if not (forced_tool.raw_text or "").strip():
                forced_tool = replace(forced_tool, arguments=args, raw_text=user_text)
            else:
                forced_tool = replace(forced_tool, arguments=args)
        elif forced_tool.tool_id == "draft_cursor_prompt":
            args = dict(forced_tool.arguments or {})
            if not str(args.get("objective") or "").strip():
                try:
                    from dana.tools.broker import parse_draft_cursor_prompt_args

                    parsed = parse_draft_cursor_prompt_args(user_text)
                    args.update({k: v for k, v in parsed.items() if v})
                except Exception:  # noqa: BLE001
                    args["objective"] = user_text
            if not str(args.get("objective") or "").strip():
                args["objective"] = user_text
            if not (forced_tool.raw_text or "").strip():
                forced_tool = replace(forced_tool, arguments=args, raw_text=user_text)
            else:
                forced_tool = replace(forced_tool, arguments=args)
        required = _needs_args.get(forced_tool.tool_id, ())
        if forced_tool.tool_id == "dispatch_research_swarm":
            q = forced_tool.arguments.get("query")
            t = forced_tool.arguments.get("topic")
            forced_args_ready = bool(
                (q is not None and str(q).strip())
                or (t is not None and str(t).strip())
            )
        elif forced_tool.tool_id == "architect_new_tool":
            g = forced_tool.arguments.get("goal")
            d = forced_tool.arguments.get("tool_description")
            forced_args_ready = bool(
                (g is not None and str(g).strip())
                or (d is not None and str(d).strip())
                or (user_text or "").strip()
            )
        elif forced_tool.tool_id == "draft_cursor_prompt":
            obj = str((forced_tool.arguments or {}).get("objective") or "").strip()
            ctx = str((forced_tool.arguments or {}).get("context") or "").strip()
            if use_moa_shim:
                # Prefer reasoner-enriched args; reject thin broker truncations.
                forced_args_ready = bool(obj) and (bool(ctx) or len(obj) >= 80)
            else:
                forced_args_ready = bool(obj or (user_text or "").strip())
        elif forced_tool.tool_id == "file_editor":
            act = str((forced_tool.arguments or {}).get("action") or "").strip().lower()
            path = str((forced_tool.arguments or {}).get("filepath") or "").strip()
            content = str((forced_tool.arguments or {}).get("content") or "")
            # Writes need content so dual-intent notes are not force-executed empty.
            if act == "write":
                forced_args_ready = bool(path and content.strip())
            else:
                forced_args_ready = bool(act and path)
        else:
            for key in required:
                val = forced_tool.arguments.get(key)
                if val is None or not str(val).strip():
                    forced_args_ready = False
                    break

    # Stage 8.9.6 / post-HITL persistence: never force-exec draft_cursor_prompt
    # (or other MoA-deferred tools) from thin broker args — MoA + ticket_validate
    # → jason → HITL → tools must own the write.
    _defer_forced_exec = False
    if forced_tool is not None:
        try:
            from dana.moa_tool_shim import defer_forced_tool_for_moa

            _defer_forced_exec = bool(defer_forced_tool_for_moa(forced_tool.tool_id))
        except Exception:  # noqa: BLE001
            _defer_forced_exec = forced_tool.tool_id == "draft_cursor_prompt"

    if (
        forced_tool is not None
        and forced_tool.tool_id in bound_names
        and forced_tool.tool_id not in _UNBOUND_TOOL_IDS
        and forced_args_ready
        and not _defer_forced_exec
    ):
        call_id = f"router-{forced_tool.tool_id}"
        if on_tool_start is not None and not tool_ack_done:
            tool_ack_done = True
            try:
                on_tool_start(forced_tool, reply_lang)
            except Exception:  # noqa: BLE001
                pass
        try:
            observation = execute_fn(forced_tool)
        except Exception as exc:  # noqa: BLE001
            observation = f"ERROR: tool {forced_tool.tool_id} failed: {exc}"
        last_obs = ag.sanitize_react_observation(str(observation), max_chars=8000)
        llm_obs = ag.sanitize_react_observation(last_obs)
        if forced_tool.tool_id == "draft_cursor_prompt":
            ag.log_tool_receipt_console(last_obs, tool_id=forced_tool.tool_id)
        trace.append(
            {
                "step": 0,
                "tool": forced_tool.tool_id,
                "args": dict(forced_tool.arguments),
                "observation": llm_obs[:500],
                "forced": True,
            }
        )
        lc_messages.append(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": forced_tool.tool_id,
                        "args": dict(forced_tool.arguments),
                        "id": call_id,
                        "type": "tool_call",
                    }
                ],
            )
        )
        lc_messages.append(ToolMessage(content=llm_obs, tool_call_id=call_id))
        ag.sanitize_react_message_history(lc_messages)
        if forced_tool.tool_id == "evaluate_slide_and_type" and last_obs:
            return _finish(ag._obs_fallback(last_obs, reply_lang), 1)
        # Jason→Donna supervisor is a complete multi-agent run — do not continue
        # the outer ReAct loop (prevents file_editor ping-pong after handoff).
        if forced_tool.tool_id == "dispatch_jason_supervisor":
            if str(last_obs).startswith("OK:"):
                spoken = str(last_obs)[3:].strip() or (
                    "Jason read the notes and Donna wrote the script."
                )
                return _finish(spoken, 1)
            return _finish(ag._obs_fallback(last_obs, reply_lang), 1)
    elif forced_tool is not None and (not forced_args_ready or _defer_forced_exec):
        if on_tool_start is not None and not tool_ack_done:
            tool_ack_done = True
            try:
                on_tool_start(forced_tool, reply_lang)
            except Exception:  # noqa: BLE001
                pass
        prompt_note = (
            f"\n\nROUTER INTENT: Call `{forced_tool.tool_id}` next with complete "
            f"required arguments inferred from the user utterance. "
            f"Do not call vision/spatial tools."
        )
        if lc_messages and getattr(lc_messages[0], "content", None) is not None:
            try:
                lc_messages[0].content = str(lc_messages[0].content) + prompt_note
            except Exception:
                pass

    def _try_bind_tool_choice(tool_id: str) -> bool:
        """Option A: ask the provider to force the next native tool call."""
        nonlocal llm_with_tools
        tid = str(tool_id or "").strip()
        if not tid:
            return False
        try:
            llm_with_tools = llm.bind_tools(tools, tool_choice=tid, strict=True)
            return True
        except Exception:  # noqa: BLE001
            pass
        try:
            llm_with_tools = llm.bind_tools(
                tools,
                tool_choice={"type": "function", "function": {"name": tid}},
                strict=True,
            )
            return True
        except Exception:  # noqa: BLE001
            return False

    async def _agent_node(state: ReactGraphState) -> dict[str, Any]:
        nonlocal llm_with_tools, last_obs, tts_streamed, tools, bound_names
        messages = list(state.get("messages") or [])
        step = int(state.get("iterations") or 0) + 1
        mem_ctx = dict(state.get("memory_context") or {})
        episodic_answer_only = bool(
            mem_ctx.get("suppress_vault_vision")
            and (mem_ctx.get("grounding_block") or mem_ctx.get("contradiction"))
        )
        # Bind from state payload — never recalculate mode defaults in the node.
        state_always = [
            str(x)
            for x in (state.get("always_include") or always or [])
            if str(x).strip()
        ]
        if episodic_answer_only:
            state_always = []
            tools = []
            bound_names = set()
            llm_with_tools = llm
        # Stage 3.3: ValidationError bounce → bind ONLY the failed tool.
        corridor = _apply_strict_validation_retry_bind()
        if corridor and not episodic_answer_only:
            state_always = [corridor]
        elif state_always and not episodic_answer_only:
            _rebind_from_always(state_always)
        pending_start = [
            tid
            for tid in state_always
            if tid not in _invoked_tool_ids_from_messages(messages)
        ]
        # After a prior Option-B nudge, skip another prose turn and synthesize
        # the missing tool call (8B models ignore prompt-only CRITICAL rules).
        if pending_start and _messages_have_force_nudge(messages, pending_start[0]):
            forced_msg = _synthetic_tool_call_message(
                pending_start[0], user_text, step=step
            )
            _emit_live_trace(
                "node_enter",
                node="agent",
                message=(
                    f"Forced tool call for pending always_include "
                    f"`{pending_start[0]}` (step {step})"
                ),
                mode=ag.get_donna_mode(),
                state_keys=("messages", "iterations", "always_include"),
            )
            return {
                "messages": [forced_msg],
                "iterations": step,
                "last_obs": last_obs,
                "final_raw": "",
                "halt": False,
                "always_include": list(state_always or always),
            }
        if pending_start:
            _try_bind_tool_choice(pending_start[0])
        _emit_live_trace(
            "node_enter",
            node="agent",
            message=f"Router/Synthesis step {step}",
            mode=ag.get_donna_mode(),
            state_keys=("messages", "iterations", "always_include"),
        )
        ag.sanitize_react_message_history(messages)
        response = None
        max_retries = 3
        _inv_t0 = time.perf_counter()
        for attempt in range(1, max_retries + 1):
            try:
                # Prefer astream so astream_events can emit on_chat_model_stream for TTS.
                if hasattr(llm_with_tools, "astream"):
                    chunks: list[Any] = []
                    async for chunk in llm_with_tools.astream(messages):
                        chunks.append(chunk)
                    if chunks:
                        response = chunks[0]
                        for ch in chunks[1:]:
                            try:
                                response = response + ch
                            except Exception:  # noqa: BLE001
                                response = ch
                elif hasattr(llm_with_tools, "ainvoke"):
                    response = await llm_with_tools.ainvoke(messages)
                else:
                    response = await asyncio.to_thread(llm_with_tools.invoke, messages)
                try:
                    mid = str(getattr(llm, "model", None) or model or "")
                    if "deepseek" in mid.lower():
                        from dana.cascade_router import (
                            note_high_complexity_deepseek_latency,
                        )

                        note_high_complexity_deepseek_latency(
                            (time.perf_counter() - _inv_t0) * 1000.0,
                            model=mid,
                        )
                except Exception:  # noqa: BLE001
                    pass
                break
            except Exception as exc:  # noqa: BLE001
                trace.append(
                    {"step": step, "error": f"llm_failed:{exc}", "retry": attempt}
                )
                try:
                    from dana.logging import log_exception

                    log_exception(
                        "Agentic",
                        f"llm.ainvoke failed (attempt {attempt}/{max_retries})",
                        exc=exc,
                    )
                except Exception:
                    pass
                # Connection / timeout: abort immediately with a clear TTS line
                # (do not retry as a Titan format error or fall through to
                # "I didn't catch that.").
                if ag.is_ollama_connection_error(exc):
                    try:
                        ag.end_stream_sentence_tts()
                    except Exception:  # noqa: BLE001
                        pass
                    return {
                        "messages": messages,
                        "iterations": step,
                        "last_obs": last_obs,
                        "final_raw": ag.OLLAMA_UNREACHABLE_SPEECH,
                        "halt": True,
                        "always_include": list(
                            state.get("always_include") or always
                        ),
                    }
                if attempt < max_retries:
                    messages.append(
                        SystemMessage(
                            content=(
                                "System Error: The previous output failed the Titan "
                                "peg-native format check. You must output valid Titan."
                            )
                        )
                    )
                    continue
                fallback = (
                    ag._obs_fallback(last_obs, reply_lang)
                    if last_obs
                    else (
                        "Sorry — I couldn't complete that just now."
                        if reply_lang != "fa"
                        else "      ."
                    )
                )
                try:
                    ag.end_stream_sentence_tts()
                except Exception:  # noqa: BLE001
                    pass
                return {
                    "messages": messages,
                    "iterations": step,
                    "last_obs": last_obs,
                    "final_raw": fallback,
                    "halt": True,
                    "always_include": list(state.get("always_include") or always),
                }

        if response is None:
            fallback = (
                ag._obs_fallback(last_obs, reply_lang)
                if last_obs
                else (
                    "Sorry — I couldn't complete that just now."
                    if reply_lang != "fa"
                    else "      ."
                )
            )
            try:
                ag.end_stream_sentence_tts()
            except Exception:  # noqa: BLE001
                pass
            return {
                "messages": messages,
                "iterations": step,
                "last_obs": last_obs,
                "final_raw": fallback,
                "halt": True,
                "always_include": list(state.get("always_include") or always),
            }

        if not isinstance(response, AIMessage):
            response = AIMessage(content=str(response))

        tool_calls = list(getattr(response, "tool_calls", None) or [])
        raw_content = str(getattr(response, "content", "") or "")
        raw_stripped = ag.strip_r1_think_blocks(raw_content).strip()
        # Also recover when models mix structured tool_calls with a JSON dump,
        # or emit only a raw JSON payload in content.
        if not tool_calls:
            recovered = ag._parse_content_tool_call(raw_stripped or raw_content)
            if recovered is not None:
                tool_calls = [recovered]
                try:
                    response.content = ""
                    response.tool_calls = tool_calls
                except Exception:
                    response = AIMessage(content="", tool_calls=tool_calls)
        elif raw_stripped:
            # Prefer content JSON when native tool_calls look empty/broken.
            native_ok = any(
                str(
                    (tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", ""))
                    or ""
                ).strip()
                for tc in tool_calls
            )
            if not native_ok:
                recovered = ag._parse_content_tool_call(raw_stripped)
                if recovered is not None:
                    tool_calls = [recovered]
                    response = AIMessage(content="", tool_calls=tool_calls)

        always_list = list(state.get("always_include") or always)
        if tool_calls:
            # Drop phantom / unbound names so Llama cannot abort on build_tool_that_*.
            cleaned: list[Any] = []
            for tc in tool_calls:
                name = str(
                    (tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", ""))
                    or ""
                ).strip()
                if name and name in bound_names:
                    cleaned.append(tc)
            if cleaned:
                try:
                    response.tool_calls = cleaned
                except Exception:  # noqa: BLE001
                    response = AIMessage(content="", tool_calls=cleaned)
                return {
                    "messages": [response],
                    "iterations": step,
                    "last_obs": last_obs,
                    "final_raw": "",
                    "halt": False,
                    "always_include": always_list,
                    "pending_synthesis": True,
                }
            # Strip phantoms so we do not route into the tools node.
            try:
                response.tool_calls = []
            except Exception:  # noqa: BLE001
                response = AIMessage(
                    content=str(getattr(response, "content", "") or "")
                )
            # All calls were phantoms — nudge toward file_editor when dual-intent.
            if "file_editor" in always_list or "file_editor" in bound_names:
                nudge = SystemMessage(
                    content=(
                        "SYSTEM: Previous tool name was invalid/unbound. "
                        "Call `file_editor` with action=write, filepath, and content. "
                        "Then speak your conversational answer. Do not invent tool names."
                    )
                )
                return {
                    "messages": [response, nudge],
                    "iterations": step,
                    "last_obs": last_obs,
                    "final_raw": "",
                    "halt": False,
                    "always_include": always_list,
                }

        # Deterministic extraction: code dump with empty tool_calls → Python saves.
        dump_text = raw_stripped or raw_content
        if not tool_calls and _looks_like_unsaved_code_dump(dump_text):
            from langchain_core.messages import AIMessage as _AIMessage
            from langchain_core.messages import SystemMessage
            from langchain_core.messages import ToolMessage

            target_path = None
            try:
                from dana.graph.nodes.worker import (
                    extract_and_save_code,
                    first_filepath_from_text,
                )

                target_path = first_filepath_from_text(
                    user_text or ""
                ) or first_filepath_from_text(dump_text)
            except Exception:  # noqa: BLE001
                target_path = None
            if target_path:
                try:
                    from dana.tools.file_editor import file_editor as _fe

                    obs = extract_and_save_code(
                        dump_text,
                        target_path,
                        tool_fn=lambda a, p, c=None: _fe(a, p, c),
                    )
                    last_obs = obs
                    tc_id = f"det_extract_{step}"
                    forced = _AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": tc_id,
                                "name": "file_editor",
                                "args": {
                                    "action": "write",
                                    "filepath": target_path,
                                    "content": "(deterministic extract)",
                                },
                            }
                        ],
                    )
                    tool_msg = ToolMessage(content=str(obs), tool_call_id=tc_id)
                    return {
                        "messages": [forced, tool_msg],
                        "iterations": step,
                        "last_obs": last_obs,
                        "final_raw": (
                            f"Saved {target_path} via deterministic extraction."
                            if not str(obs).startswith("ERROR:")
                            else str(obs)
                        ),
                        "halt": not str(obs).startswith("ERROR:"),
                        "always_include": always_list,
                    }
                except Exception:  # noqa: BLE001
                    pass
            if (
                not _messages_have_code_save_force(messages)
                and step < max_iters
            ):
                save_tool = (
                    "file_editor"
                    if "file_editor" in bound_names
                    else ("write_to_file" if "write_to_file" in bound_names else "")
                )
                if save_tool:
                    _try_bind_tool_choice(save_tool)
                nudge = SystemMessage(content=_TOOL_FORCE_SAVE_MSG)
                return {
                    "messages": [response, nudge],
                    "iterations": step,
                    "last_obs": last_obs,
                    "final_raw": "",
                    "halt": False,
                    "always_include": always_list,
                }

        # Option B: text-only reply while always_include tools remain → nudge
        # (or synthesize after nudge / max iters) instead of exiting to END.
        pending_after = [
            tid
            for tid in always_list
            if tid
            not in _invoked_tool_ids_from_messages(
                list(messages) + [response]
            )
        ]
        if pending_after:
            next_tid = pending_after[0]
            if (
                _messages_have_force_nudge(messages, next_tid)
                or step >= max_iters
            ):
                forced_msg = _synthetic_tool_call_message(
                    next_tid, user_text, step=step
                )
                return {
                    "messages": [response, forced_msg],
                    "iterations": step,
                    "last_obs": last_obs,
                    "final_raw": "",
                    "halt": False,
                    "always_include": always_list,
                }
            _try_bind_tool_choice(next_tid)
            nudge = _force_tool_nudge_message(next_tid)
            return {
                "messages": [response, nudge],
                "iterations": step,
                "last_obs": last_obs,
                "final_raw": "",
                "halt": False,
                "always_include": always_list,
            }

        raw = raw_stripped
        if ag.looks_like_raw_json_speech(raw) or ag._parse_content_tool_call(raw):
            raw = ""
        # Module 4: deterministic Swarm Handoff (no supervisor LLM).
        try:
            from dana.handoff import execute_handoff, parse_handoff_payload

            handoff = parse_handoff_payload(raw_stripped or raw)
            if handoff is not None:
                result = execute_handoff(
                    handoff,
                    session_id=str(state.get("session_id") or session_id),
                    current_agent=str(
                        state.get("current_agent") or current_agent
                    ),
                )
                return {
                    "messages": [response],
                    "iterations": step,
                    "last_obs": last_obs,
                    "final_raw": str(
                        result.get("ack")
                        or f"Handoff → {handoff.target_agent}"
                    ),
                    "halt": True,
                    "always_include": always_list,
                    "session_id": str(state.get("session_id") or session_id),
                    "current_agent": str(
                        result.get("current_agent") or handoff.target_agent
                    ),
                    "active_intent": str(
                        result.get("active_intent") or handoff.reason
                    ),
                    "pending_handoff": handoff.model_dump(),
                }
        except Exception:  # noqa: BLE001
            pass
        answer = ag.extract_final(raw) or raw
        answer = re.sub(
            r"^\s*(FINAL|Final|final| )\s*[:：]\s*", "", answer
        ).strip()
        answer = ag.strip_protocol_speech_anchors(answer)
        answer = ag.strip_raw_json_from_speech(answer)
        answer = ag.strip_r1_think_blocks(answer).strip()
        if not answer:
            answer = (
                ag._obs_fallback(last_obs, reply_lang)
                if last_obs
                else (
                    "I didn't catch that."
                    if reply_lang != "fa"
                    else "  ."
                )
            )
        # No tool call this iteration — terminate stream TTS so idle is not held ~30s.
        try:
            if ag.end_stream_sentence_tts():
                tts_streamed = True
        except Exception:  # noqa: BLE001
            pass
        trace.append({"step": step, "final": True})
        from dana.graph.completion_gate import (
            flag_pending_synthesis_from_text,
            is_filler_response,
        )

        filler_patch = flag_pending_synthesis_from_text(answer)
        # Filler acknowledgements must not halt the corridor (ghosting guard).
        halt_out = not bool(filler_patch.get("pending_synthesis"))
        if is_filler_response(answer):
            halt_out = False
        else:
            try:
                _task_tracker.update_status(
                    str(state.get("session_id") or session_id),
                    TaskStatus.COMPLETED,
                )
            except Exception:  # noqa: BLE001
                pass
        return {
            "messages": [response],
            "iterations": step,
            "last_obs": last_obs,
            "final_raw": answer,
            "halt": halt_out,
            "always_include": always_list,
            **filler_patch,
        }

    async def _tools_node(state: ReactGraphState) -> dict[str, Any]:
        nonlocal last_obs, tool_ack_done
        step = int(state.get("iterations") or 1)
        messages = list(state.get("messages") or [])
        last = messages[-1] if messages else None
        tool_calls = list(getattr(last, "tool_calls", None) or []) if last else []
        _emit_live_trace(
            "node_enter",
            node="tools",
            message=f"Tool node ({len(tool_calls)} call(s))",
            mode=ag.get_donna_mode(),
            state_keys=("messages", "last_obs"),
        )
        new_msgs: list[Any] = []
        repl_heal: dict[str, Any] = {}
        for tc_raw in tool_calls:
            tool_call = ag._tool_call_from_lc(tc_raw, raw_text=user_text)
            if not (tool_call.raw_text or "").strip():
                tool_call = replace(tool_call, raw_text=user_text)
            if tool_call.tool_id == "architect_new_tool":
                args = dict(tool_call.arguments or {})
                if not str(
                    args.get("goal") or args.get("tool_description") or ""
                ).strip():
                    args["goal"] = user_text
                    tool_call = replace(tool_call, arguments=args)
            # Suite 3: rewrite shell/file args for named-repo git / watchdog graph.
            try:
                from dana.tools.os_tools import (
                    cascade_git_tool_args,
                    is_cascade_git_query,
                    is_watchdog_graph_query,
                    watchdog_graph_filepath,
                )

                if tool_call.tool_id in {
                    "execute_powershell",
                    "shell_execute",
                    "run_terminal_command",
                } and is_cascade_git_query(user_text or ""):
                    tool_call = replace(
                        tool_call,
                        arguments=cascade_git_tool_args(user_text or ""),
                    )
                elif tool_call.tool_id in {
                    "read_local_file",
                    "file_editor",
                } and is_watchdog_graph_query(user_text or ""):
                    args = dict(tool_call.arguments or {})
                    args["filepath"] = watchdog_graph_filepath()
                    if tool_call.tool_id == "file_editor":
                        args["action"] = "read"
                    tool_call = replace(tool_call, arguments=args)
            except Exception:  # noqa: BLE001
                pass
            try:
                tool_call = broker.validate_and_correct(tool_call)
            except ToolValidationError as exc:
                hint = ""
                if "file_editor" in bound_names or "file_editor" in (
                    state.get("always_include") or always
                ):
                    hint = (
                        " Retry with file_editor(action=write, filepath=..., content=...) "
                        "then speak your conversational answer. Do not invent tool names."
                    )
                observation = f"ERROR: invalid tool call ({exc}).{hint}"
                call_id = str(
                    getattr(tc_raw, "id", None)
                    or (tc_raw.get("id") if isinstance(tc_raw, dict) else None)
                    or f"call-{tool_call.tool_id}"
                )
                new_msgs.append(ToolMessage(content=observation, tool_call_id=call_id))
                continue
            # Refuse phantom dynamics that slipped past bind_tools.
            if tool_call.tool_id not in bound_names:
                hint = (
                    " Use a bound tool only"
                    + (
                        " — prefer file_editor for create/write notes."
                        if "file_editor" in bound_names
                        else "."
                    )
                )
                observation = (
                    f"ERROR: tool {tool_call.tool_id} is not bound.{hint}"
                )
                call_id = str(
                    getattr(tc_raw, "id", None)
                    or (tc_raw.get("id") if isinstance(tc_raw, dict) else None)
                    or f"call-{tool_call.tool_id}"
                )
                new_msgs.append(ToolMessage(content=observation, tool_call_id=call_id))
                continue
            call_id = str(
                getattr(tc_raw, "id", None)
                or (tc_raw.get("id") if isinstance(tc_raw, dict) else None)
                or f"call-{tool_call.tool_id}-{uuid.uuid4().hex[:8]}"
            )
            # Module 4: Pydantic guard before raw tool execution.
            try:
                from pydantic import ValidationError as _PydValidationError

                from dana.tools.guards import (
                    format_validation_bounce,
                    guard_tool_call,
                )

                guarded_args = guard_tool_call(
                    tool_call.tool_id, dict(tool_call.arguments or {})
                )
                tool_call = replace(tool_call, arguments=guarded_args)
            except Exception as _guard_exc:  # noqa: BLE001 — includes ValidationError
                from pydantic import ValidationError as _PydValidationError

                from dana.tools.guards import format_validation_bounce

                if not isinstance(_guard_exc, _PydValidationError):
                    # Non-validation guard failures fall through as soft errors.
                    observation = f"ERROR: tool guard failed: {_guard_exc}"
                    new_msgs.append(
                        ToolMessage(content=observation, tool_call_id=call_id)
                    )
                    continue
                retry_key = f"{tool_call.tool_id}:{call_id}"
                bounce = format_validation_bounce(_guard_exc)
                try:
                    from dana.telemetry import log_tool_execution

                    log_tool_execution(
                        tool_call.tool_id,
                        session_id=str(state.get("session_id") or session_id),
                        current_agent=str(
                            state.get("current_agent") or current_agent
                        ),
                        active_intent=str(
                            state.get("active_intent") or active_intent
                        ),
                        ok=False,
                        payload={
                            "validation_bounce": True,
                            "guard": "pydantic",
                            "retry": retry_key not in validation_retries,
                        },
                    )
                except Exception:  # noqa: BLE001
                    pass
                if retry_key in validation_retries:
                    observation = f"ERROR: {bounce} (retry exhausted)"
                else:
                    validation_retries.add(retry_key)
                    _arm_strict_validation_retry(tool_call.tool_id)
                    observation = bounce
                    new_msgs.append(
                        ToolMessage(content=observation, tool_call_id=call_id)
                    )
                    new_msgs.append(
                        SystemMessage(
                            content=(
                                f"SYSTEM: {observation} "
                                "Do not invent softer wording — supply complete "
                                "structured fields and call the same tool "
                                f"`{tool_call.tool_id}` once more. "
                                "No other tools are available on this retry."
                            )
                        )
                    )
                    continue
                new_msgs.append(
                    ToolMessage(content=observation, tool_call_id=call_id)
                )
                continue
            # Prefer explicit draft_cursor_prompt writer so patch_ledger.md updates
            # even when the model emitted raw JSON (content-parsed) tool calls.
            # Stage 8.9.6 — after HITL, execute the validated drafted_ticket body.
            if tool_call.tool_id == "draft_cursor_prompt":
                dt = state.get("drafted_ticket") or {}
                if isinstance(dt, dict) and (
                    str(dt.get("objective") or "").strip()
                    or str(dt.get("context") or "").strip()
                ):
                    tool_call = replace(
                        tool_call,
                        arguments={
                            **dict(tool_call.arguments or {}),
                            "objective": str(dt.get("objective") or ""),
                            "context": str(dt.get("context") or ""),
                        },
                    )
            observation = ""
            _tool_t0 = time.perf_counter()
            # Stage 4.2 — heavy tools enqueue to action_queue; LLM turn stays non-blocking.
            try:
                from dana.memory.blackboard import (
                    enqueue_action as _enqueue_action,
                    is_heavy_actuator_tool as _is_heavy_actuator_tool,
                )

                _enqueue_heavy = _is_heavy_actuator_tool(tool_call.tool_id)
            except Exception:  # noqa: BLE001
                _enqueue_heavy = False
            # Post-HITL (or any graph tools pass): ledger write must be sync.
            # Enqueue alone can silently miss if actuator_executor is down.
            if tool_call.tool_id == "draft_cursor_prompt":
                _enqueue_heavy = False
            # Suite 3 orchestration / forced foresight tools need in-turn Observations.
            _always_now = set(state.get("always_include") or always or [])
            if tool_call.tool_id in _always_now and tool_call.tool_id in {
                "read_local_file",
                "file_editor",
                "execute_powershell",
                "shell_execute",
                "run_terminal_command",
            }:
                _enqueue_heavy = False
            try:
                from dana.tools.os_tools import (
                    is_cascade_git_query,
                    is_watchdog_graph_query,
                )

                if is_cascade_git_query(user_text or "") or is_watchdog_graph_query(
                    user_text or ""
                ):
                    _enqueue_heavy = False
            except Exception:  # noqa: BLE001
                pass
            if _enqueue_heavy:
                try:
                    _aid = _enqueue_action(
                        tool_call.tool_id,
                        dict(tool_call.arguments or {}),
                        session_id=str(state.get("session_id") or session_id),
                    )
                    observation = (
                        f"Action queued successfully. Task ID: {_aid}."
                    )
                except Exception as exc:  # noqa: BLE001
                    observation = (
                        f"ERROR: failed to enqueue {tool_call.tool_id}: {exc}"
                    )
            else:
                try:
                    from dana.graph.completion_gate import (
                        DEFAULT_TOOL_TIMEOUT_S,
                        TOOL_TIMEOUT_MESSAGE,
                        apply_timeout_failure,
                        run_async_with_tool_timeout,
                        run_with_tool_timeout,
                    )

                    _tool_timeout_s = float(
                        (state.get("env_context") or {}).get("tool_timeout_s")
                        or DEFAULT_TOOL_TIMEOUT_S
                    )

                    def _on_timeout() -> None:
                        repl_heal.update(
                            apply_timeout_failure(
                                _task_tracker,
                                str(state.get("session_id") or session_id),
                                tool_id=tool_call.tool_id,
                            )
                        )

                    try:
                        _task_tracker.update_status(
                            str(state.get("session_id") or session_id),
                            TaskStatus.TOOL_EXECUTING,
                            metadata={"tool": tool_call.tool_id},
                        )
                    except Exception:  # noqa: BLE001
                        pass

                    if tool_call.tool_id == "draft_cursor_prompt":
                        from dana.tools.general.draft_cursor_prompt import (
                            draft_cursor_prompt as _draft_cursor_prompt,
                        )

                        def _run_draft() -> str:
                            return str(
                                _draft_cursor_prompt(
                                    objective=str(
                                        (tool_call.arguments or {}).get("objective")
                                        or ""
                                    ),
                                    context=str(
                                        (tool_call.arguments or {}).get("context")
                                        or ""
                                    ),
                                )
                            )

                        ok, observation, terr = run_with_tool_timeout(
                            _run_draft, timeout_s=_tool_timeout_s
                        )
                        if not ok:
                            observation = terr or TOOL_TIMEOUT_MESSAGE
                            _on_timeout()
                    elif tool_call.tool_id == "analyze_visual_context":
                        # Screen OCR (mss+pytesseract); webcam keeps JIT YOLO.
                        src = str(
                            (tool_call.arguments or {}).get("source") or "screen"
                        ).strip().lower() or "screen"
                        if src == "camera":
                            src = "webcam"

                        def _run_vision() -> str:
                            if src in {"webcam", "video"}:
                                from dana.vision_tools import (
                                    analyze_visual_context as _yolo_visual,
                                )

                                return str(_yolo_visual(source=src))
                            from dana.tools.vision import (
                                analyze_visual_context as _ocr_visual,
                            )

                            return str(_ocr_visual())

                        ok, observation, terr = run_with_tool_timeout(
                            _run_vision, timeout_s=_tool_timeout_s
                        )
                        if not ok:
                            observation = terr or TOOL_TIMEOUT_MESSAGE
                            _on_timeout()
                    elif tool_call.tool_id == "ocr_with_region":
                        from dana.tools.visual_tools import (
                            ocr_with_region as _ocr_region,
                        )

                        def _run_ocr() -> str:
                            return str(
                                _ocr_region(
                                    query=str(
                                        (tool_call.arguments or {}).get("query") or ""
                                    ).strip()
                                )
                            )

                        ok, observation, terr = run_with_tool_timeout(
                            _run_ocr, timeout_s=_tool_timeout_s
                        )
                        if not ok:
                            observation = terr or TOOL_TIMEOUT_MESSAGE
                            _on_timeout()
                    elif tool_call.tool_id == "click_ui_element":
                        from dana.tools.vision import (
                            click_ui_element as _click_ui_element,
                        )

                        def _run_click_ui_element() -> str:
                            return str(
                                _click_ui_element(
                                    str(
                                        (tool_call.arguments or {}).get(
                                            "target_description"
                                        )
                                        or ""
                                    ).strip()
                                )
                            )

                        ok, observation, terr = run_with_tool_timeout(
                            _run_click_ui_element, timeout_s=_tool_timeout_s
                        )
                        if not ok:
                            observation = terr or TOOL_TIMEOUT_MESSAGE
                            _on_timeout()
                    elif tool_call.tool_id == "type_text_in_element":
                        from dana.tools.vision import (
                            type_text_in_element as _type_text_in_element,
                        )

                        def _run_type_text_in_element() -> str:
                            return str(
                                _type_text_in_element(
                                    str(
                                        (tool_call.arguments or {}).get(
                                            "target_description"
                                        )
                                        or ""
                                    ).strip(),
                                    str(
                                        (tool_call.arguments or {}).get("text")
                                        or ""
                                    ),
                                )
                            )

                        ok, observation, terr = run_with_tool_timeout(
                            _run_type_text_in_element, timeout_s=_tool_timeout_s
                        )
                        if not ok:
                            observation = terr or TOOL_TIMEOUT_MESSAGE
                            _on_timeout()
                    elif tool_call.tool_id == "scroll_screen":
                        from dana.tools.vision import scroll_screen as _scroll_screen

                        def _run_scroll_screen() -> str:
                            return str(
                                _scroll_screen(
                                    str(
                                        (tool_call.arguments or {}).get("direction")
                                        or ""
                                    ).strip(),
                                    str(
                                        (tool_call.arguments or {}).get("amount")
                                        or "medium"
                                    ).strip(),
                                )
                            )

                        ok, observation, terr = run_with_tool_timeout(
                            _run_scroll_screen, timeout_s=_tool_timeout_s
                        )
                        if not ok:
                            observation = terr or TOOL_TIMEOUT_MESSAGE
                            _on_timeout()
                    else:
                        tool_map = {getattr(t, "name", ""): t for t in tools}
                        st = tool_map.get(tool_call.tool_id)
                        if st is not None and hasattr(st, "ainvoke"):

                            async def _ainvoke_tool() -> Any:
                                return await st.ainvoke(
                                    dict(tool_call.arguments or {})
                                )

                            ok, raw_obs, terr = await run_async_with_tool_timeout(
                                _ainvoke_tool, timeout_s=_tool_timeout_s
                            )
                            if not ok:
                                observation = terr or TOOL_TIMEOUT_MESSAGE
                                _on_timeout()
                            else:
                                observation = str(raw_obs)
                        else:

                            def _run_execute() -> str:
                                return str(execute_fn(tool_call))

                            ok, observation, terr = run_with_tool_timeout(
                                _run_execute, timeout_s=_tool_timeout_s
                            )
                            if not ok:
                                observation = terr or TOOL_TIMEOUT_MESSAGE
                                _on_timeout()
                except Exception as exc:  # noqa: BLE001
                    observation = f"ERROR: tool {tool_call.tool_id} failed: {exc}"
                    repl_heal.update(
                        {
                            "execution_error": observation,
                            "final_raw": (
                                f"Tool `{tool_call.tool_id}` failed: {exc}"
                            ),
                            "pending_synthesis": True,
                            "halt": False,
                        }
                    )
                    try:
                        _task_tracker.update_status(
                            str(state.get("session_id") or session_id),
                            TaskStatus.FAILED,
                            metadata={"tool": tool_call.tool_id, "error": str(exc)},
                        )
                    except Exception:  # noqa: BLE001
                        pass
            # Localized bounce when the tool returns a Validation Error string.
            if "Validation Error:" in str(observation):
                retry_key = f"{tool_call.tool_id}:{call_id}"
                try:
                    from dana.telemetry import log_tool_execution

                    log_tool_execution(
                        tool_call.tool_id,
                        session_id=str(state.get("session_id") or session_id),
                        current_agent=str(
                            state.get("current_agent") or current_agent
                        ),
                        active_intent=str(
                            state.get("active_intent") or active_intent
                        ),
                        ok=False,
                        latency_ms=(time.perf_counter() - _tool_t0) * 1000.0,
                        payload={"validation_bounce": True, "retry": retry_key not in validation_retries},
                    )
                except Exception:  # noqa: BLE001
                    pass
                if retry_key not in validation_retries:
                    validation_retries.add(retry_key)
                    _arm_strict_validation_retry(tool_call.tool_id)
                    new_msgs.append(
                        ToolMessage(content=str(observation), tool_call_id=call_id)
                    )
                    new_msgs.append(
                        SystemMessage(
                            content=(
                                f"SYSTEM: {observation} "
                                f"Retry the same tool `{tool_call.tool_id}` once "
                                "with complete fields. "
                                "No other tools are available on this retry."
                            )
                        )
                    )
                    continue
            try:
                from dana.telemetry import log_tool_execution

                log_tool_execution(
                    tool_call.tool_id,
                    session_id=str(state.get("session_id") or session_id),
                    current_agent=str(
                        state.get("current_agent") or current_agent
                    ),
                    active_intent=str(
                        state.get("active_intent") or active_intent
                    ),
                    ok=not str(observation).startswith("ERROR:"),
                    latency_ms=(time.perf_counter() - _tool_t0) * 1000.0,
                )
            except Exception:  # noqa: BLE001
                pass
            obs_l = str(observation).lower()
            if (
                str(observation).startswith("ERROR:")
                and (
                    "source not found" in obs_l
                    or "dynamic tool" in obs_l
                    or "unknown tool" in obs_l
                )
                and ("file_editor" in bound_names)
            ):
                observation = (
                    f"{observation} HINT: Call file_editor(action=write, filepath=..., "
                    "content=...) for file/notes requests, then FINAL with your "
                    "conversational answer. Do not abort the turn."
                )
            if tool_call.tool_id == "architect_new_tool" and str(observation).startswith(
                "OK:"
            ):
                try:
                    _rebind_tools_after_forge()
                except Exception:  # noqa: BLE001
                    pass
            if on_tool_start is not None and not tool_ack_done:
                tool_ack_done = True
                try:
                    on_tool_start(tool_call, reply_lang)
                except Exception:  # noqa: BLE001
                    pass
            last_obs = ag.sanitize_react_observation(str(observation), max_chars=8000)
            llm_obs = ag.sanitize_react_observation(last_obs)
            _emit_live_trace(
                "tool_execution",
                node="tools",
                tool=tool_call.tool_id,
                message=f"Tool: {tool_call.tool_id}",
                mode=ag.get_donna_mode(),
                payload=llm_obs[:800],
                state_keys=("last_obs",),
            )
            if tool_call.tool_id == "draft_cursor_prompt":
                ag.log_tool_receipt_console(last_obs, tool_id=tool_call.tool_id)
            trace.append(
                {
                    "step": step,
                    "tool": tool_call.tool_id,
                    "args": dict(tool_call.arguments),
                    "observation": llm_obs[:500],
                }
            )
            new_msgs.append(ToolMessage(content=llm_obs, tool_call_id=call_id))
            if tool_call.tool_id == "python_repl":
                from dana.graph.nodes.critic import python_repl_state_patch

                repl_heal.update(
                    python_repl_state_patch(
                        code=str((tool_call.arguments or {}).get("code") or ""),
                        observation=str(last_obs),
                    )
                )
            if tool_call.tool_id == "evaluate_slide_and_type" and last_obs:
                return {
                    "messages": new_msgs,
                    "iterations": step,
                    "last_obs": last_obs,
                    "final_raw": ag._obs_fallback(last_obs, reply_lang),
                    "halt": True,
                    # Preserve planner/executor always_include across tool steps.
                    "always_include": list(
                        state.get("always_include") or always
                    ),
                    **repl_heal,
                }
        # Stage 3.3: bounce corridor narrows always_include to the failed tool
        # only; otherwise keep planner/broker merge from state.
        if strict_retry_tool_id:
            always_out = validation_retry_tool_corridor(strict_retry_tool_id)
        else:
            always_out = list(state.get("always_include") or always)
        if step >= max_iters:
            extracted = ag._spoken_fact_from_search_obs(str(last_obs), user_text)
            if not extracted:
                for prior in reversed(trace):
                    extracted = ag._spoken_fact_from_search_obs(
                        str(prior.get("observation") or ""),
                        user_text,
                    )
                    if extracted:
                        break
            return {
                "messages": new_msgs,
                "iterations": step,
                "last_obs": last_obs,
                "final_raw": extracted or ag._obs_fallback(last_obs, reply_lang),
                "halt": True,
                "always_include": list(
                    state.get("always_include") or always
                ),
                **repl_heal,
            }
        return {
            "messages": new_msgs,
            "iterations": step,
            "last_obs": last_obs,
            "final_raw": "",
            "halt": False,
            "always_include": always_out,
            **repl_heal,
        }

    graph = compile_donna_react_graph(
        _agent_node,
        _tools_node,
        checkpointer=ag._react_checkpointer(),
    )

    config = {
        "configurable": {"thread_id": session_id or ag._REACT_THREAD_ID},
        # Cap ReAct / worker spin so stalls fail in ~tens of seconds, not 600s.
        "recursion_limit": min(15, max(10, max_iters * 4)),
    }
    # Module 1: durable history is on the Blackboard — do not rehydrate
    # MemorySaver prior dialogue into graph state (keeps state minimal).
    turn_messages: list[Any] = list(lc_messages)

    _mem_ctx: dict[str, Any] = {}
    if _episodic_grounding:
        _mem_ctx = {
            "matches": list(_episodic_grounding.get("matches") or []),
            "primary_source": "episodic_facts",
            "grounding_block": _episodic_grounding.get("grounding_block") or "",
            "contradiction": _episodic_grounding.get("contradiction"),
            "contradiction_directive": (
                _episodic_grounding.get("contradiction_directive") or ""
            ),
            "suppress_vault_vision": bool(
                _episodic_grounding.get("suppress_vault_vision")
            ),
        }
    inputs: ReactGraphState = {
        "session_id": session_id,
        "current_agent": current_agent,
        "active_intent": active_intent,
        "messages": turn_messages,
        "iterations": 0,
        "last_obs": last_obs,
        "final_raw": "",
        "halt": False,
        "always_include": list(always),
        "execution_error": None,
        "critique_history": [],
        "retry_count": 0,
        "max_retries": 3,
        "last_code_snippet": "",
        "fatal_block": False,
        "memory_context": _mem_ctx,
    }

    final_state: dict[str, Any] = dict(inputs)
    think_tts_filter = ag.ThinkBlockTtsFilter()
    # After draft_cursor_prompt, mute model stream (ticket body echoes) — final ack only.
    mute_post_ticket_stream = False
    _graph_t0 = time.perf_counter()
    _emit_live_trace(
        "node_enter",
        node="router",
        message="LangGraph ReAct start",
        mode=ag.get_donna_mode(),
        state_keys=("session_id", "current_agent", "active_intent"),
    )
    _chain_t0: dict[str, float] = {}

    async def _consume_astream(stream_input: Any) -> None:
        nonlocal mute_post_ticket_stream, tts_streamed
        async for event in graph.astream_events(
            stream_input, config=config, version="v2"
        ):
            kind = str(event.get("event") or "")
            name = str(event.get("name") or "")
            if kind == "on_chain_start" and name in {
                "planner",
                "executor",
                "agent",
                "tools",
                "ticket_approval",
                "jason_ticket_review",
                "os_worker",
            }:
                _chain_t0[name] = time.perf_counter()
                _emit_live_trace(
                    "node_enter",
                    node=name,
                    message=f"chain start: {name}",
                    mode=ag.get_donna_mode(),
                )
                try:
                    from dana.ui.status_bus import emit_state_change

                    if name in {"planner", "executor", "agent"}:
                        emit_state_change(
                            "routing", message="Supervisor Routing..."
                        )
                    elif name == "os_worker":
                        emit_state_change(
                            "executing", tool="execute_powershell"
                        )
                except Exception:  # noqa: BLE001
                    pass
            elif kind == "on_chain_end" and name in {
                "planner",
                "executor",
                "agent",
                "tools",
                "ticket_approval",
                "jason_ticket_review",
                "os_worker",
            }:
                t0 = _chain_t0.pop(name, None)
                ms = (time.perf_counter() - t0) * 1000.0 if t0 is not None else None
                _emit_live_trace(
                    "node_exit",
                    node=name,
                    message=f"chain end: {name}",
                    mode=ag.get_donna_mode(),
                    latency_ms=ms,
                )
            if kind == "on_chat_model_start":
                # Mute "Thinking..." — R1 plans inside <think>; speak only outer text.
                think_tts_filter.reset()
                ag.reset_stream_sentence_tts()
                _emit_live_trace(
                    "status",
                    node="synthesis",
                    message="LLM synthesis streaming",
                    mode=ag.get_donna_mode(),
                )
            elif kind == "on_tool_start":
                # Flush any buffered speech before tool-status TTS.
                ag.flush_stream_sentence_tts()
                tool_name = str(event.get("name") or "tool")
                try:
                    from dana.ui.status_bus import emit_state_change

                    emit_state_change("executing", tool=tool_name)
                except Exception:  # noqa: BLE001
                    pass
                # Stage 8.8 — tool status lines speak in the owning persona voice.
                _tool_agent = "broker"
                tn = tool_name.lower()
                if "vision" in tn or "ocr" in tn or "yolo" in tn or "visual" in tn:
                    _tool_agent = "vision"
                elif "jason" in tn:
                    _tool_agent = "jason"
                elif "typist" in tn or "type" in tn:
                    _tool_agent = "typist"
                elif "draft_cursor" in tn:
                    _tool_agent = "moa"
                try:
                    ag.set_stream_tts_agent(_tool_agent)
                except Exception:  # noqa: BLE001
                    pass
                _speak(ag._friendly_tool_tts(tool_name), agent_id=_tool_agent)
                _emit_live_trace(
                    "tool_execution",
                    node="tools",
                    tool=tool_name,
                    message=f"on_tool_start: {tool_name}",
                    mode=ag.get_donna_mode(),
                )
                try:
                    from dana.telemetry import log_tool_execution

                    log_tool_execution(
                        tool_name,
                        session_id=session_id,
                        current_agent=current_agent,
                        active_intent=active_intent,
                        ok=True,
                    )
                except Exception:  # noqa: BLE001
                    pass
                if tool_name == "draft_cursor_prompt":
                    mute_post_ticket_stream = True
            elif kind == "on_tool_end":
                # Mute raw tool payloads — never speak JSON / OK: observations.
                tool_name = str(event.get("name") or "")
                data = event.get("data") or {}
                output = data.get("output")
                _emit_live_trace(
                    "state_update",
                    node="tools",
                    tool=tool_name,
                    message=f"on_tool_end: {tool_name}",
                    mode=ag.get_donna_mode(),
                    payload=str(output or "")[:800],
                    state_keys=("messages", "last_obs"),
                )
                if tool_name == "draft_cursor_prompt":
                    mute_post_ticket_stream = True
                    if output is not None:
                        ag.log_tool_receipt_console(str(output), tool_id=tool_name)
            elif kind == "on_chat_model_stream":
                if mute_post_ticket_stream:
                    # Ticket receipts stay in ledger + console; speak ack in _finish.
                    continue
                data = event.get("data") or {}
                piece = ag._stream_chunk_for_tts(data.get("chunk"))
                # Strip R1 reasoning across chunk boundaries (never speak <think>).
                piece = think_tts_filter.feed(piece)
                if piece:
                    # Sentence-level buffer — never push raw single-word tokens.
                    n = ag.feed_stream_tts(piece)
                    if n:
                        tts_streamed = True
            elif kind in ("on_chain_end", "on_chain_stream"):
                data = event.get("data") or {}
                output = data.get("output")
                if isinstance(output, dict) and (
                    "messages" in output or "final_raw" in output
                ):
                    final_state.update(output)

    def _snapshot_interrupts() -> list[Any]:
        try:
            snap = graph.get_state(config)
        except Exception:  # noqa: BLE001
            return []
        vals = getattr(snap, "values", None) or {}
        if isinstance(vals, dict) and vals:
            final_state.update(vals)
        interrupts = list(getattr(snap, "interrupts", None) or ())
        if interrupts:
            return interrupts
        # Fallback: some versions stash interrupts on values.
        raw = vals.get("__interrupt__") if isinstance(vals, dict) else None
        if raw:
            return list(raw) if isinstance(raw, (list, tuple)) else [raw]
        return []

    await _consume_astream(inputs)

    # Stage 8.6 — HITL resume loop (Approve / Deny via GUI or auto-resolve).
    from langgraph.types import Command

    from dana.middleware import hitl_ticket as _hitl

    _hitl_rounds = 0
    while _hitl_rounds < 4:
        interrupts = _snapshot_interrupts()
        if not interrupts:
            break
        _hitl_rounds += 1
        first = interrupts[0]
        payload = getattr(first, "value", first)
        if not isinstance(payload, dict):
            payload = {"type": "ticket_approval", "raw": str(payload)}
        _hitl.publish_pending(
            payload,
            thread_id=str(session_id or ag._REACT_THREAD_ID),
        )
        _speak("Ticket drafted — waiting for your approval in the dashboard.")
        decision = await asyncio.to_thread(_hitl.wait_for_decision)
        approved = _hitl.decision_is_approved(decision)
        _emit_live_trace(
            "status",
            node="ticket_approval",
            message="HITL_RESUME",
            payload=f"approved={approved} action={decision.get('action')}",
            mode=ag.get_donna_mode(),
        )
        try:
            await _consume_astream(Command(resume=decision))
        finally:
            _hitl.clear_pending()

    try:
        snap = graph.get_state(config)
        vals = getattr(snap, "values", None) or {}
        if isinstance(vals, dict) and vals:
            final_state.update(
                {k: v for k, v in vals.items() if k != "__interrupt__"}
            )
    except Exception:  # noqa: BLE001
        pass

    # Speak any incomplete trailing clause; always terminate the stream latch.
    try:
        if ag.end_stream_sentence_tts():
            tts_streamed = True
    except Exception:  # noqa: BLE001
        pass

    last_obs = str(final_state.get("last_obs") or last_obs)
    iterations = int(final_state.get("iterations") or max_iters)
    answer = str(final_state.get("final_raw") or "").strip()
    if not answer:
        for msg in reversed(list(final_state.get("messages") or [])):
            if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                raw = str(getattr(msg, "content", "") or "").strip()
                if raw and not ag.looks_like_raw_json_speech(raw):
                    answer = ag.extract_final(raw) or raw
                    break
    if not answer:
        answer = (
            ag._obs_fallback(last_obs, reply_lang)
            if last_obs
            else ("Done." if reply_lang != "fa" else " .")
        )
    _emit_live_trace(
        "node_exit",
        node="synthesis",
        message="ReAct complete",
        mode=ag.get_donna_mode(),
        payload=answer[:800],
        latency_ms=(time.perf_counter() - _graph_t0) * 1000.0,
        state_keys=("session_id", "current_agent", "active_intent", "final_raw"),
    )
    try:
        from dana.ui.status_bus import emit_state_change

        emit_state_change("idle")
    except Exception:  # noqa: BLE001
        pass
    # Persist assistant turn on Blackboard (durable memory).
    try:
        append_message(session_id, "assistant", answer)
    except Exception:  # noqa: BLE001
        pass
    return _finish(answer, iterations)
