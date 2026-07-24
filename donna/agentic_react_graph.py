"""LangGraph async ReAct runner with MemorySaver + astream_events TTS telemetry.

Used by ``donna.agentic._run_react_loop_langchain``. Keeps strict ``bind_tools``
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

from donna.schema import AgenticResult, ReactGraphState
from donna.tools.broker import IntentBroker, ToolValidationError, get_broker
from donna.tools.schema import ToolCall

# Re-export for callers that import ReactGraphState from this module.
__all__ = (
    "ReactGraphState",
    "compile_donna_react_graph",
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
        from donna.ui.trace_bus import emit_trace_event

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
    if tid == "draft_cursor_prompt":
        try:
            from donna.tools.broker import parse_draft_cursor_prompt_args

            args = dict(parse_draft_cursor_prompt_args(raw) or {})
        except Exception:  # noqa: BLE001
            args = {}
        if not str(args.get("objective") or "").strip():
            from donna.agentic import _full_sentence_boundary

            args["objective"] = (
                _full_sentence_boundary(raw) or "Log self-improvement ticket"
            )
        if "context" not in args:
            args["context"] = ""
        return args
    if tid in {"web_search", "dispatch_research_swarm", "dispatch_jason_supervisor"}:
        return {"query": raw}
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


def _route_after_agent(state: ReactGraphState) -> str:
    """Conditional edge: agent → tools / agent (pending always_include) / END."""
    from langgraph.graph import END

    messages = state.get("messages") or []
    last = messages[-1] if messages else None
    if last is not None and getattr(last, "tool_calls", None):
        return "tools"
    # Option B: do not END while broker-merged tools remain uninvoked.
    if pending_always_include_tools(state):
        return "agent"
    if state.get("halt"):
        return END
    return END


def _route_after_tools(state: ReactGraphState) -> str:
    """Conditional edge: tools → END (halt) or agent (continue ReAct)."""
    from langgraph.graph import END

    if state.get("halt") and not pending_always_include_tools(state):
        return END
    if pending_always_include_tools(state):
        return "agent"
    return END if state.get("halt") else "agent"


def compile_donna_react_graph(
    agent_node: Callable[..., Any],
    tools_node: Callable[..., Any],
    *,
    checkpointer: Any | None = None,
) -> Any:
    """Compile the production ReAct StateGraph (same topology as live Donna).

    Topology:
      START → agent ─(tool_calls)→ tools ─(continue)→ agent
                    ╲(pending always_include / nudge)→ agent
                    ╲(halt/final)→ END          ╲(halt)→ END
    """
    from langgraph.graph import END, START, StateGraph

    from donna import agentic as ag

    workflow = StateGraph(ReactGraphState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tools_node)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        _route_after_agent,
        {"tools": "tools", "agent": "agent", END: END},
    )
    workflow.add_conditional_edges(
        "tools",
        _route_after_tools,
        {"agent": "agent", END: END},
    )
    cp = checkpointer if checkpointer is not None else ag._react_checkpointer()
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

    from donna import agentic as ag
    from donna.cascade_router import resolve_chat_model
    from donna.tools.langchain_tools import _UNBOUND_TOOL_IDS, build_langchain_tools
    from donna.tools.registry import get_tool_registry
    from donna.settings import resolve_reply_lang

    broker = broker or get_broker()
    reply_lang = resolve_reply_lang(user_text)

    def _speak(phrase: str) -> None:
        """Prefer injected TTS callback; fall back to agentic spooler helper."""
        text = (phrase or "").strip()
        if not text:
            return
        if tts_callback is not None:
            try:
                tts_callback(text)
                return
            except Exception:  # noqa: BLE001
                pass
        ag._enqueue_tts_nonblocking(text)

    prompt = system_prompt
    if ag._TOOL_EXECUTION_RULE not in prompt:
        prompt = f"{prompt}\n\n{ag._TOOL_EXECUTION_RULE}"
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
    from donna.tools.broker import merge_bound_tool_ids

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
    from donna.memory import (
        append_message,
        ensure_session,
        load_messages,
        set_session_meta,
    )
    from donna.telemetry import log_router

    current_agent = (
        "MoA_Reasoner"
        if (
            forced_tool is not None
            or "draft_cursor_prompt" in (user_text or "").lower()
        )
        else "ReAct_Agent"
    )
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
    # Stage 4.3 — piggyback unread actuator completions into this turn's prompt.
    try:
        from donna.memory.blackboard import (
            format_background_system_alert,
            get_and_clear_unread_notifications,
        )

        _unread = get_and_clear_unread_notifications(session_id)
        _alert = format_background_system_alert(_unread)
        if _alert:
            prompt = f"{prompt}\n\n{_alert}"
            try:
                from donna.telemetry import log_notification_piggyback

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
    seed = ag._build_seed_messages(
        user_text=user_text,
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
    if always:
        bind_registry = _specs_for_tool_ids(
            always,
            semantic_specs=semantic_specs,
            broker_registry=broker.registry,
        )
        tool_ids: set[str] | None = set(always)
    else:
        top_specs = semantic.retrieve_specs(user_text, k=6, always_include=always)
        bind_registry = top_specs if top_specs else broker.registry
        tool_ids = set(bind_registry.keys()) if top_specs else None
    tools = build_langchain_tools(
        execute_fn,
        registry=bind_registry,
        tool_ids=tool_ids,
        tts_callback=tts_callback,
        vault_client=vault_client,
    )
    bound_names = {getattr(t, "name", "") for t in tools}
    try:
        from donna.logging import log as _agentic_log

        _agentic_log(
            "Agentic",
            f"tools={sorted(n for n in bound_names if n)} "
            f"(always_include={always or '-'})",
        )
    except Exception:  # noqa: BLE001
        pass
    llm = resolve_chat_model(
        query=user_text,
        forced_tool=forced_tool.tool_id if forced_tool is not None else None,
        default_model=model,
        temperature=0.2,
    )
    llm_with_tools = llm.bind_tools(tools, strict=True)

    # Two-stage MoA shim: DeepSeek-R1 plans (no tools) → Llama formats tool_calls.
    moa_plan = ""
    use_moa_shim = False
    try:
        from donna.moa_tool_shim import (
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
                from donna.logging import log as _moa_log

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
                except Exception:  # noqa: BLE001
                    pass
    except Exception as _moa_exc:  # noqa: BLE001
        try:
            from donna.logging import log as _moa_log

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
            from donna.logging import log as _retry_log

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
        from donna.reflector import trace_has_failure

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
            r"^\s*(?:TOOL|Action|FINAL||| )\s*[:：]",
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
        spoken = ag.sanitize_spoken_reply(
            spoken,
            reply_lang=reply_lang,
            last_obs=last_obs,
            tool_trace=trace,
        )
        # Strict override: successful draft_cursor_prompt → canned UX only (WAV cache).
        if ag.draft_cursor_tool_succeeded(last_obs=last_obs, tool_trace=trace):
            spoken = ag.DRAFT_CURSOR_UX_ACK
            # Ensure core_agent enqueues this ack (prior stream may have marked TTS done).
            nonlocal tts_streamed
            tts_streamed = False
        if (
            forced_tool is not None
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
                    from donna.tools.broker import parse_draft_cursor_prompt_args

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

    if (
        forced_tool is not None
        and forced_tool.tool_id in bound_names
        and forced_tool.tool_id not in _UNBOUND_TOOL_IDS
        and forced_args_ready
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
    elif forced_tool is not None and not forced_args_ready:
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
        nonlocal llm_with_tools, last_obs, tts_streamed
        messages = list(state.get("messages") or [])
        step = int(state.get("iterations") or 0) + 1
        # Bind from state payload — never recalculate mode defaults in the node.
        state_always = [
            str(x)
            for x in (state.get("always_include") or always or [])
            if str(x).strip()
        ]
        # Stage 3.3: ValidationError bounce → bind ONLY the failed tool.
        corridor = _apply_strict_validation_retry_bind()
        if corridor:
            state_always = [corridor]
        elif state_always:
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
                        from donna.cascade_router import (
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
                    from donna.logging import log_exception

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
            from donna.handoff import execute_handoff, parse_handoff_payload

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
        return {
            "messages": [response],
            "iterations": step,
            "last_obs": last_obs,
            "final_raw": answer,
            "halt": True,
            "always_include": always_list,
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

                from donna.tools.guards import (
                    format_validation_bounce,
                    guard_tool_call,
                )

                guarded_args = guard_tool_call(
                    tool_call.tool_id, dict(tool_call.arguments or {})
                )
                tool_call = replace(tool_call, arguments=guarded_args)
            except Exception as _guard_exc:  # noqa: BLE001 — includes ValidationError
                from pydantic import ValidationError as _PydValidationError

                from donna.tools.guards import format_validation_bounce

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
                    from donna.telemetry import log_tool_execution

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
            observation = ""
            _tool_t0 = time.perf_counter()
            # Stage 4.2 — heavy tools enqueue to action_queue; LLM turn stays non-blocking.
            try:
                from donna.memory.blackboard import (
                    enqueue_action as _enqueue_action,
                    is_heavy_actuator_tool as _is_heavy_actuator_tool,
                )

                _enqueue_heavy = _is_heavy_actuator_tool(tool_call.tool_id)
            except Exception:  # noqa: BLE001
                _enqueue_heavy = False
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
                    if tool_call.tool_id == "draft_cursor_prompt":
                        from donna.tools.general.draft_cursor_prompt import (
                            draft_cursor_prompt as _draft_cursor_prompt,
                        )

                        observation = str(
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
                    elif tool_call.tool_id == "analyze_visual_context":
                        # Direct JIT vision path → ToolMessage content for synthesis.
                        from donna.vision_tools import (
                            analyze_visual_context as _analyze_visual,
                        )

                        src = str(
                            (tool_call.arguments or {}).get("source") or "screen"
                        ).strip().lower() or "screen"
                        if src == "camera":
                            src = "webcam"
                        observation = str(_analyze_visual(source=src))
                    elif tool_call.tool_id == "ocr_with_region":
                        from donna.tools.visual_tools import (
                            ocr_with_region as _ocr_region,
                        )

                        observation = str(
                            _ocr_region(
                                query=str(
                                    (tool_call.arguments or {}).get("query") or ""
                                ).strip()
                            )
                        )
                    else:
                        tool_map = {getattr(t, "name", ""): t for t in tools}
                        st = tool_map.get(tool_call.tool_id)
                        if st is not None and hasattr(st, "ainvoke"):
                            observation = str(
                                await st.ainvoke(dict(tool_call.arguments or {}))
                            )
                        else:
                            observation = str(execute_fn(tool_call))
                except Exception as exc:  # noqa: BLE001
                    observation = f"ERROR: tool {tool_call.tool_id} failed: {exc}"
            # Localized bounce when the tool returns a Validation Error string.
            if "Validation Error:" in str(observation):
                retry_key = f"{tool_call.tool_id}:{call_id}"
                try:
                    from donna.telemetry import log_tool_execution

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
                from donna.telemetry import log_tool_execution

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
            if tool_call.tool_id == "evaluate_slide_and_type" and last_obs:
                return {
                    "messages": new_msgs,
                    "iterations": step,
                    "last_obs": last_obs,
                    "final_raw": ag._obs_fallback(last_obs, reply_lang),
                    "halt": True,
                    "always_include": list(always),
                }
        # Stage 3.3: bounce corridor narrows always_include to the failed tool
        # only; successful / exhausted paths restore the original broker merge.
        if strict_retry_tool_id:
            always_out = validation_retry_tool_corridor(strict_retry_tool_id)
        else:
            always_out = list(always)
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
                "always_include": list(always),
            }
        return {
            "messages": new_msgs,
            "iterations": step,
            "last_obs": last_obs,
            "final_raw": "",
            "halt": False,
            "always_include": always_out,
        }

    graph = compile_donna_react_graph(
        _agent_node,
        _tools_node,
        checkpointer=ag._react_checkpointer(),
    )

    config = {
        "configurable": {"thread_id": session_id or ag._REACT_THREAD_ID},
        "recursion_limit": max(10, max_iters * 4),
    }
    # Module 1: durable history is on the Blackboard — do not rehydrate
    # MemorySaver prior dialogue into graph state (keeps state minimal).
    turn_messages: list[Any] = list(lc_messages)

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
    async for event in graph.astream_events(inputs, config=config, version="v2"):
        kind = str(event.get("event") or "")
        name = str(event.get("name") or "")
        if kind == "on_chain_start" and name in {"agent", "tools"}:
            _chain_t0[name] = time.perf_counter()
            _emit_live_trace(
                "node_enter",
                node=name,
                message=f"chain start: {name}",
                mode=ag.get_donna_mode(),
            )
        elif kind == "on_chain_end" and name in {"agent", "tools"}:
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
            _speak(ag._friendly_tool_tts(tool_name))
            _emit_live_trace(
                "tool_execution",
                node="tools",
                tool=tool_name,
                message=f"on_tool_start: {tool_name}",
                mode=ag.get_donna_mode(),
            )
            try:
                from donna.telemetry import log_tool_execution

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

    try:
        snap = graph.get_state(config)
        vals = getattr(snap, "values", None) or {}
        if isinstance(vals, dict) and vals:
            final_state.update(vals)
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
    # Persist assistant turn on Blackboard (durable memory).
    try:
        append_message(session_id, "assistant", answer)
    except Exception:  # noqa: BLE001
        pass
    return _finish(answer, iterations)
