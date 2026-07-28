"""Text-only Developer CLI Harness for Donna's LangGraph ReAct architecture.

Silent by design: no TTS, Whisper, or VAD imports. Streams graph events to
the console via ``astream_events(version=\"v2\")`` with a dedicated MemorySaver
thread (``cli-test-session``).

Usage:
    python -m dana.test_harness
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from dana import agentic as ag
from dana.agentic import (
    REACT_MAX_ITERS,
    ThinkBlockTtsFilter,
    _dicts_to_lc_messages,
    _parse_content_tool_call,
    _tool_call_from_lc,
    sanitize_react_message_history,
    sanitize_react_observation,
    strip_r1_think_blocks,
)
from dana.agentic_react_graph import (
    ReactGraphState,
    _route_after_agent,
    _route_after_tools,
    _synthetic_tool_call_message,
    _force_tool_nudge_message,
    _messages_have_force_nudge,
    _invoked_tool_ids_from_messages,
    pending_always_include_tools,
)
from dana.cascade_router import resolve_chat_model
from dana.tools.langchain_tools import build_langchain_tools
from dana.tools.schema import ToolCall

VISION_DRAFT_PROMPT = (
    "Donna, use the draft_cursor_prompt tool to log a self-improvement ticket "
    "to enhance cursor rendering capabilities."
)

# Dedicated CLI checkpointer — same MemorySaver class as dana.agentic._react_checkpointer.
def _cli_checkpointer() -> MemorySaver:
    from langchain_core.load.load import Reviver
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    import langgraph.checkpoint.serde.jsonplus as _jp_serde

    _jp_serde.LC_REVIVER = Reviver(allowed_objects="all")
    return MemorySaver(serde=JsonPlusSerializer())


CLI_CHECKPOINTER = _cli_checkpointer()
CLI_THREAD_ID = "cli-test-session"

_SYSTEM_PROMPT = (
    "You are Donna in developer CLI mode. "
    "Never emit TTS protocol markers (TOOL:/FINAL:)."
)

# Noise often leaked into streamed model tokens.
_STREAM_NOISE_RE = re.compile(
    r"</?tool\b[^>]*>|"
    r"<\|eot_id\|>|"
    r"<\|[^|>]+?\|>|"
    r"</?\|?tool_calls?\|?>",
    re.IGNORECASE,
)

# ANSI colors (Windows 10+ VT-capable consoles).
_CYAN = "\033[36m"
_YELLOW = "\033[33m"
_GREEN = "\033[32m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _enable_ansi() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:  # noqa: BLE001
        pass


def _c(color: str, text: str) -> str:
    return f"{color}{text}{_RESET}"


def _filter_stream_chunk(text: str) -> str:
    if not text:
        return ""
    return _STREAM_NOISE_RE.sub("", text)


def _cli_system_prompt() -> str:
    """Mirror production rule injection (tool-first / TPM / termination)."""
    parts = [
        _SYSTEM_PROMPT,
        ag._TOOL_EXECUTION_RULE,
        ag._STRICT_TOOL_ENFORCEMENT_RULE,
        ag._EXPLICIT_TOOL_INVOCATION_RULE,
        ag._R1_REASONING_RULE,
        ag._VOICE_SANITIZER_RULE,
        ag._INTERACTION_UX_RULE,
        ag._DRAFT_CURSOR_TPM_RULE,
        ag._DRAFT_CURSOR_TERMINATION_RULE,
    ]
    return "\n\n".join(parts)


def _harness_execute(tc: ToolCall) -> str:
    """Side-effect-free tool sink for architecture testing (no voice pipeline)."""
    args = json.dumps(dict(tc.arguments or {}), ensure_ascii=False, default=str)
    if len(args) > 800:
        args = args[:797] + "..."
    return f"OK: [harness] executed {tc.tool_id} args={args}"


def _tool_ids_for_query(
    query: str, *, vision_mode: bool = False
) -> set[str] | None:
    """Prefer a tight tool set for ticket turns (faster, fewer distractions)."""
    q = (query or "").lower()
    draft_hit = any(
        k in q
        for k in (
            "draft cursor",
            "draft_cursor",
            "self-improvement",
            "self improvement",
            "log a ticket",
            "patch ledger",
        )
    )
    if vision_mode and draft_hit:
        # Mirror broker merge in Vision: vision JIT + explicit draft request.
        return {"analyze_visual_context", "draft_cursor_prompt"}
    if draft_hit:
        return {"draft_cursor_prompt"}
    return None


def _always_include_for_query(
    query: str, *, vision_mode: bool = False
) -> list[str]:
    ids = _tool_ids_for_query(query, vision_mode=vision_mode)
    if not ids:
        return []
    # Preserve Vision order: analyze first, then explicit draft.
    ordered = ["analyze_visual_context", "draft_cursor_prompt"]
    return [tid for tid in ordered if tid in ids] or sorted(ids)


def _compile_cli_graph(
    *,
    query: str,
    max_iters: int = REACT_MAX_ITERS,
    vision_mode: bool = False,
):
    """Compile agent↔tools StateGraph with MemorySaver (mirrors production shape).

    ``query`` is the live user turn — passed into Cascade so force-high keywords
    classify complexity. ReAct ChatOllama itself uses the local tool-caller
    (DeepSeek-R1 does not emit native Ollama tool_calls).
    """
    tool_ids = _tool_ids_for_query(query, vision_mode=vision_mode)
    always_include = _always_include_for_query(query, vision_mode=vision_mode)
    tools = build_langchain_tools(
        _harness_execute,
        tool_ids=tool_ids,
        include_natives=False,
    )
    tool_map = {getattr(t, "name", ""): t for t in tools}
    llm = resolve_chat_model(
        query=query,
        forced_tool="draft_cursor_prompt" if tool_ids else None,
        default_model=None,
        temperature=0.2,
    )
    llm_with_tools = llm.bind_tools(tools, strict=True)

    async def _agent_node(state: ReactGraphState) -> dict[str, Any]:
        nonlocal llm_with_tools
        messages = list(state.get("messages") or [])
        step = int(state.get("iterations") or 0) + 1
        always_list = [
            str(x)
            for x in (state.get("always_include") or always_include or [])
            if str(x).strip()
        ]
        sanitize_react_message_history(messages)

        pending_start = [
            tid
            for tid in always_list
            if tid not in _invoked_tool_ids_from_messages(messages)
        ]
        if pending_start and _messages_have_force_nudge(messages, pending_start[0]):
            print(
                _c(
                    _YELLOW,
                    f"[System: Forcing pending always_include -> {pending_start[0]}]",
                )
            )
            return {
                "messages": [
                    _synthetic_tool_call_message(
                        pending_start[0], query, step=step
                    )
                ],
                "iterations": step,
                "last_obs": str(state.get("last_obs") or ""),
                "final_raw": "",
                "halt": False,
                "always_include": always_list,
            }

        response: Any = None
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
        if response is None:
            if hasattr(llm_with_tools, "ainvoke"):
                response = await llm_with_tools.ainvoke(messages)
            else:
                response = await asyncio.to_thread(llm_with_tools.invoke, messages)

        if not isinstance(response, AIMessage):
            response = AIMessage(content=str(response))

        tool_calls = list(getattr(response, "tool_calls", None) or [])
        raw = str(getattr(response, "content", "") or "").strip()
        raw_stripped = strip_r1_think_blocks(raw).strip()
        if not tool_calls:
            recovered = _parse_content_tool_call(raw_stripped or raw)
            if recovered is not None:
                tool_calls = [recovered]
                response = AIMessage(content="", tool_calls=tool_calls)

        if tool_calls:
            return {
                "messages": [response],
                "iterations": step,
                "last_obs": str(state.get("last_obs") or ""),
                "final_raw": "",
                "halt": False,
                "always_include": always_list,
            }

        pending_after = [
            tid
            for tid in always_list
            if tid
            not in _invoked_tool_ids_from_messages(list(messages) + [response])
        ]
        if pending_after:
            next_tid = pending_after[0]
            if _messages_have_force_nudge(messages, next_tid) or step >= max_iters:
                print(
                    _c(
                        _YELLOW,
                        f"[System: Forcing pending always_include -> {next_tid}]",
                    )
                )
                return {
                    "messages": [
                        response,
                        _synthetic_tool_call_message(next_tid, query, step=step),
                    ],
                    "iterations": step,
                    "last_obs": str(state.get("last_obs") or ""),
                    "final_raw": "",
                    "halt": False,
                    "always_include": always_list,
                }
            print(
                _c(
                    _YELLOW,
                    f"[System: Nudge pending always_include -> {next_tid}]",
                )
            )
            return {
                "messages": [response, _force_tool_nudge_message(next_tid)],
                "iterations": step,
                "last_obs": str(state.get("last_obs") or ""),
                "final_raw": "",
                "halt": False,
                "always_include": always_list,
            }

        return {
            "messages": [response],
            "iterations": step,
            "last_obs": str(state.get("last_obs") or ""),
            "final_raw": raw_stripped or raw,
            "halt": True,
            "always_include": always_list,
        }

    async def _tools_node(state: ReactGraphState) -> dict[str, Any]:
        step = int(state.get("iterations") or 1)
        messages = list(state.get("messages") or [])
        last = messages[-1] if messages else None
        tool_calls = list(getattr(last, "tool_calls", None) or []) if last else []
        new_msgs: list[Any] = []
        last_obs = str(state.get("last_obs") or "")
        always_list = [
            str(x)
            for x in (state.get("always_include") or always_include or [])
            if str(x).strip()
        ]

        for tc_raw in tool_calls:
            tool_call = _tool_call_from_lc(tc_raw, raw_text="")
            call_id = str(
                getattr(tc_raw, "id", None)
                or (tc_raw.get("id") if isinstance(tc_raw, dict) else None)
                or f"call-{tool_call.tool_id}"
            )
            try:
                st = tool_map.get(tool_call.tool_id)
                if st is not None and hasattr(st, "ainvoke"):
                    observation = str(
                        await st.ainvoke(dict(tool_call.arguments or {}))
                    )
                else:
                    observation = _harness_execute(tool_call)
            except Exception as exc:  # noqa: BLE001
                observation = f"ERROR: tool {tool_call.tool_id} failed: {exc}"
            last_obs = sanitize_react_observation(str(observation), max_chars=8000)
            new_msgs.append(ToolMessage(content=last_obs, tool_call_id=call_id))

        # Invoked ids live on prior AIMessages; keep looping while always_include
        # tools remain even when step hits max_iters.
        still_pending = pending_always_include_tools(
            {"messages": list(messages), "always_include": always_list}
        )
        halt = step >= max_iters and not still_pending
        return {
            "messages": new_msgs,
            "iterations": step,
            "last_obs": last_obs,
            "final_raw": last_obs if halt else "",
            "halt": halt,
            "always_include": always_list,
        }

    workflow = StateGraph(ReactGraphState)
    workflow.add_node("agent", _agent_node)
    workflow.add_node("tools", _tools_node)
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
    return workflow.compile(checkpointer=CLI_CHECKPOINTER)


def _turn_messages(user_text: str, *, has_history: bool) -> list[Any]:
    seed = [
        {"role": "system", "content": _cli_system_prompt()},
        {"role": "user", "content": user_text},
    ]
    lc = _dicts_to_lc_messages(seed)
    if not has_history:
        return lc
    return [m for m in lc if isinstance(m, (SystemMessage, HumanMessage))]


def _chunk_text(chunk: Any) -> str:
    if chunk is None:
        return ""
    content = getattr(chunk, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            else:
                parts.append(str(getattr(block, "text", "") or ""))
        return "".join(parts)
    return str(content or "") if content is not None else ""


async def _run_turn(
    graph: Any,
    user_text: str,
    *,
    always_include: list[str] | None = None,
    vision_mode: bool = False,
) -> list[str]:
    config = {
        "configurable": {"thread_id": CLI_THREAD_ID},
        "recursion_limit": max(10, REACT_MAX_ITERS * 4),
    }
    has_history = False
    try:
        snap = graph.get_state(config)
        vals = getattr(snap, "values", None) or {}
        has_history = bool(vals.get("messages"))
    except Exception:  # noqa: BLE001
        pass

    always = list(
        always_include
        if always_include is not None
        else _always_include_for_query(user_text, vision_mode=vision_mode)
    )
    if always:
        print(_c(_DIM, f"[System: always_include={always}]"))

    inputs: ReactGraphState = {
        "session_id": "cli-session",
        "current_agent": "ReAct_Agent",
        "active_intent": (always[0] if always else "general"),
        "messages": _turn_messages(user_text, has_history=has_history),
        "iterations": 0,
        "last_obs": "",
        "final_raw": "",
        "halt": False,
        "always_include": always,
    }

    streamed_any = False
    final_raw = ""
    think_filter = ThinkBlockTtsFilter()
    executed_tools: list[str] = []

    async for event in graph.astream_events(inputs, config=config, version="v2"):
        kind = str(event.get("event") or "")
        if kind == "on_chat_model_start":
            think_filter.reset()
            print(_c(_CYAN, "[System: Model generating...]"))
        elif kind == "on_tool_start":
            tool_name = str(event.get("name") or "tool")
            executed_tools.append(tool_name)
            print(_c(_YELLOW, f"[System: Executing Tool -> {tool_name}]"))
        elif kind == "on_tool_end":
            data = event.get("data") or {}
            output = data.get("output")
            payload = str(output or "")
            print(_c(_GREEN, f"[Tool Result]\n{payload}"))
        elif kind == "on_chat_model_stream":
            data = event.get("data") or {}
            piece = _filter_stream_chunk(_chunk_text(data.get("chunk")))
            piece = think_filter.feed(piece)
            if piece:
                if not streamed_any:
                    print(_c(_BOLD, "Donna: "), end="", flush=True)
                    streamed_any = True
                print(piece, end="", flush=True)
        elif kind == "on_chain_end":
            data = event.get("data") or {}
            output = data.get("output")
            if isinstance(output, dict) and output.get("final_raw"):
                final_raw = str(output.get("final_raw") or "")

    if streamed_any:
        print()
    answer = strip_r1_think_blocks(final_raw).strip()
    if not answer:
        try:
            snap = graph.get_state(config)
            vals = getattr(snap, "values", None) or {}
            answer = strip_r1_think_blocks(str(vals.get("final_raw") or "")).strip()
            if not answer:
                for msg in reversed(list(vals.get("messages") or [])):
                    if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                        answer = strip_r1_think_blocks(
                            str(getattr(msg, "content", "") or "")
                        ).strip()
                        if answer:
                            break
            if not answer:
                last_obs = str(vals.get("last_obs") or "").strip()
                if last_obs:
                    answer = last_obs
        except Exception as exc:  # noqa: BLE001
            print(_c(_DIM, f"[System: state read failed: {exc}]"))
            return executed_tools

    if answer and not streamed_any:
        print(_c(_BOLD, "Donna: ") + answer)
    elif not answer and not streamed_any:
        print(
            _c(
                _DIM,
                "[System: (no spoken final text this turn — model returned empty "
                "content/tool_calls)]",
            )
        )
    if executed_tools:
        print(_c(_DIM, f"[System: tools_executed={executed_tools}]"))
    return executed_tools


async def _run_vision_draft_repro() -> int:
    """Headless Vision+draft always_include repro (exit 0 iff both tools run)."""
    _enable_ansi()
    text = VISION_DRAFT_PROMPT
    print(_c(_BOLD, "Donna Vision+draft always_include repro"))
    print(_c(_DIM, f"prompt={text!r}"))
    try:
        ag.set_donna_mode("vision")
    except Exception:  # noqa: BLE001
        pass
    graph = _compile_cli_graph(query=text, vision_mode=True, max_iters=4)
    executed = await _run_turn(graph, text, vision_mode=True)
    need = {"analyze_visual_context", "draft_cursor_prompt"}
    got = set(executed)
    ok = need.issubset(got)
    print(
        _c(
            _GREEN if ok else _YELLOW,
            f"[System: PASS both tools]" if ok else f"[System: FAIL missing={need - got}]",
        )
    )
    return 0 if ok else 1


async def main() -> None:
    _enable_ansi()
    if any(a in {"--vision-draft", "--repro-vision-draft"} for a in sys.argv[1:]):
        raise SystemExit(await _run_vision_draft_repro())

    print(_c(_BOLD, "Donna LangGraph Developer CLI"))
    print(
        _c(
            _DIM,
            f"thread_id={CLI_THREAD_ID} | MemorySaver | silent (no TTS/Whisper/VAD)",
        )
    )
    print(_c(_DIM, "Commands: quit | exit | q | :vision-draft"))
    print(_c(_DIM, "Tools run in harness stub mode (OK: [harness] …)."))
    print(
        _c(
            _DIM,
            "Cascade classifies complexity from the live user string; "
            "ReAct tool-loop uses local ChatOllama (R1 reserved for MoA stages).",
        )
    )
    print(_c(_DIM, "One-shot: python -m dana.test_harness --vision-draft"))

    while True:
        try:
            user_input = input("\nUser: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        text = (user_input or "").strip()
        if not text:
            continue
        if text.lower() in {"quit", "exit", "q"}:
            break
        vision_mode = False
        if text.lower() in {":vision-draft", "vision-draft"}:
            text = VISION_DRAFT_PROMPT
            vision_mode = True
            try:
                ag.set_donna_mode("vision")
            except Exception:  # noqa: BLE001
                pass
            print(_c(_DIM, f"[System: injecting Vision draft prompt] {text}"))
        try:
            graph = _compile_cli_graph(query=text, vision_mode=vision_mode)
            await _run_turn(graph, text, vision_mode=vision_mode)
        except Exception as exc:  # noqa: BLE001
            print(_c(_YELLOW, f"[System: turn failed: {type(exc).__name__}: {exc}]"))


if __name__ == "__main__":
    asyncio.run(main())
