"""Standalone, headless CLI harness for Dānā's LIVE physical actuators.

Bypasses ``dana/core_agent.py`` (the ~13,000-line Tkinter GUI) entirely — this
compiles the same LangGraph agent<->tools ReAct topology as
``dana.test_harness`` (Dānā's existing developer CLI), but wires a fixed set
of tools to REAL execution instead of that harness's side-effect-free stub,
and forces ``DANA_OS_DRY_RUN=0`` so the Win32 actuators actually fire.

Every model token, tool call, and tool result streams to stdout via
``astream_events`` as the graph runs, so you can watch the ReAct loop's
thought process / tool usage / state transitions in real time.

Tools under test (Milestone 2-4 physical actuators):
  - focus_window            (dana.tools.window_actuator)
  - press_keyboard_shortcut (dana.tools.keyboard_actuator)
  - inject_keystrokes       (existing, already wired in core_agent.execute_tool_call)
  - read_clipboard / write_clipboard (dana.tools.clipboard_actuator)
  - send_notification       (dana.tools.notifications — Pushover)

focus_window / press_keyboard_shortcut / read_clipboard / write_clipboard /
send_notification are not yet wired into core_agent.execute_tool_call's
dispatch table (they're new, uncommitted actuators) — this script dispatches
them directly to their real actuator modules and only falls back to
``dana.core_agent.execute_tool_call`` for any other tool id the model calls.

Usage:
    python scripts/test_live_actuators.py            # REAL actuation
    python scripts/test_live_actuators.py --dry-run   # DANA_OS_DRY_RUN=1 rehearsal

Kill switch:
    F12               -> Dānā's hardware panic button (dana.middleware.kill_switch).
                          Every actuator checks GLOBAL_HALT_EVENT immediately
                          before it touches the mouse/keyboard/clipboard for
                          real, so F12 aborts an in-flight action.
    Ctrl+C (terminal)  -> stops this script / the LLM loop immediately.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
import uuid
from typing import Any

# Windows consoles often default to cp1252, which can't encode "Dānā"'s
# macron-a — reconfigure early so every print() below (banner, streamed
# tokens, tool results) is safe regardless of the launching terminal.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

# Populate PUSHOVER_USER_KEY / PUSHOVER_API_TOKEN (and anything else in the
# repo .env) before any tool runs — send_notification reads these from
# os.environ at call time. Same ENV_PATH convention as core_agent.py /
# swarm_main.py.
from dotenv import load_dotenv  # noqa: E402

try:
    from dana.paths import ENV_PATH

    load_dotenv(ENV_PATH)
except Exception:  # noqa: BLE001
    pass
load_dotenv()

TEST_PROMPT = (
    "Focus the Cursor editor window. Once focused, use a keyboard shortcut "
    "to open a new file (ctrl+n). Type the exact text 'Dānā live test "
    "successful'. Select all the text and copy it to the clipboard. "
    "Finally, read the clipboard and send a Pushover notification "
    "containing that exact text."
)

# Fixed tool set for this smoke test — bypasses the alias-based intent broker
# (built for single-intent classification, not a 6-tool chained script) so
# every actuator under test is guaranteed to be bound and callable.
ALWAYS_INCLUDE = [
    "focus_window",
    "press_keyboard_shortcut",
    "inject_keystrokes",
    "read_clipboard",
    "write_clipboard",
    "send_notification",
]
TOOL_IDS = set(ALWAYS_INCLUDE)

MAX_ITERS = 12
THREAD_ID = "live-actuator-test"

_SYSTEM_PROMPT = (
    "You are Dānā running a live physical-actuator smoke test on the host "
    "Windows machine. Execute the user's instructions as a strict, ordered "
    "sequence of tool calls — one tool call per step, waiting for each "
    "result before the next. Never claim a step succeeded without first "
    "calling its tool and reading the observation. Use press_keyboard_shortcut "
    "for key combos (ctrl+n, ctrl+a, ctrl+c) and inject_keystrokes to type "
    "literal text. Reproduce the requested text byte-for-byte, including any "
    "special characters. Never emit TTS protocol markers (TOOL:/FINAL:).\n\n"
    "CALL ONLY ONE TOOL AT A TIME. Emit a single tool call, then stop and "
    "wait for that tool's result before deciding your next call — never "
    "batch multiple tool calls into one turn, and never emit a tool call "
    "as raw text/JSON in your reply; use the native tool-calling format."
)


def _format_result(tool_id: str, result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        if result.get("ok"):
            extra = {k: v for k, v in result.items() if k != "ok"}
            return f"OK: {tool_id} {extra}"
        return f"ERROR: {tool_id} failed: {result.get('error') or result}"
    return str(result)


def live_execute_tool(tc: Any) -> str:
    """Dispatch a ToolCall to the REAL actuator — no dry-run stub, no GUI."""
    tool_id = str(tc.tool_id or "")
    args = dict(tc.arguments or {})
    try:
        if tool_id == "focus_window":
            from dana.tools.window_actuator import WindowActuator

            result = WindowActuator().focus_by_title(
                str(args.get("target_description") or "")
            )
            return _format_result(tool_id, result)
        if tool_id == "press_keyboard_shortcut":
            from dana.tools.keyboard_actuator import KeyboardActuator

            result = KeyboardActuator().execute_shortcut(str(args.get("shortcut") or ""))
            return _format_result(tool_id, result)
        if tool_id == "read_clipboard":
            from dana.tools.clipboard_actuator import ClipboardActuator

            result = ClipboardActuator().read_text()
            return _format_result(tool_id, result)
        if tool_id == "write_clipboard":
            from dana.tools.clipboard_actuator import ClipboardActuator

            result = ClipboardActuator().write_text(str(args.get("text") or ""))
            return _format_result(tool_id, result)
        if tool_id == "send_notification":
            from dana.tools.notifications import send_notification

            return send_notification(str(args.get("message") or ""))
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {tool_id} failed: {exc}"

    # Everything else (e.g. inject_keystrokes) -> real production dispatcher.
    from dana.core.agent_loop import execute_tool_call

    return execute_tool_call(tc)


# TARGET_TEXT is the literal string the test prompt asks Dānā to type/copy/
# notify with. Used as the forced-args fallback below instead of ever
# sweeping the full multi-step raw prompt into a single tool arg.
TARGET_TEXT = "Dānā live test successful"


def _forced_args_for_test_tool(tool_id: str) -> dict[str, Any]:
    """Narrow, single-purpose default args for the last-resort forced-call
    safety net (fires only when the model never spontaneously calls a
    required tool). Deliberately NOT the raw multi-step user prompt —
    dumping that whole string in as e.g. shortcut= or target_description=
    produced garbage like shortcut='Focus the Cursor editor window...'.
    """
    if tool_id == "focus_window":
        return {"target_description": "Cursor"}
    if tool_id == "press_keyboard_shortcut":
        return {"shortcut": "ctrl+n"}
    if tool_id in ("inject_keystrokes", "write_clipboard"):
        return {"text": TARGET_TEXT}
    if tool_id == "send_notification":
        return {"message": TARGET_TEXT}
    return {}


def _forced_tool_call_message(tool_id: str, *, step: int) -> Any:
    """Local stand-in for agentic_react_graph._synthetic_tool_call_message
    that uses _forced_args_for_test_tool instead of the production
    _default_args_for_forced_tool (which falls back to the raw user_text
    for tool ids it doesn't special-case)."""
    from langchain_core.messages import AIMessage

    tid = str(tool_id or "").strip()
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": tid,
                "args": _forced_args_for_test_tool(tid),
                "id": f"force-{tid}-{step}-{uuid.uuid4().hex[:8]}",
                "type": "tool_call",
            }
        ],
    )


def _compile_live_graph(checkpointer: Any):
    """Same agent<->tools LangGraph shape as dana.test_harness, real tools bound."""
    from dana.agentic import REACT_MAX_ITERS  # noqa: F401 (kept for parity/reference)
    from dana.agentic_react_graph import (
        ReactGraphState,
        _force_tool_nudge_message,
        _invoked_tool_ids_from_messages,
        _messages_have_force_nudge,
        _route_after_agent,
        _route_after_tools,
        pending_always_include_tools,
    )
    from dana.agentic import (
        _dicts_to_lc_messages,
        _parse_content_tool_call,
        sanitize_react_message_history,
        sanitize_react_observation,
        strip_r1_think_blocks,
    )
    from dana.cascade_router import resolve_chat_model
    from dana.test_harness import _c, _YELLOW
    from dana.tools.langchain_tools import build_langchain_tools
    from dana.tools.schema import ToolCall
    from langchain_core.messages import AIMessage, ToolMessage
    from langgraph.graph import END, START, StateGraph

    tools = build_langchain_tools(
        live_execute_tool,
        tool_ids=TOOL_IDS,
        include_natives=False,
    )
    tool_map = {getattr(t, "name", ""): t for t in tools}
    llm = resolve_chat_model(query=TEST_PROMPT, default_model=None, temperature=0.2)
    llm_with_tools = llm.bind_tools(tools, strict=True)

    async def _agent_node(state: ReactGraphState) -> dict[str, Any]:
        messages = list(state.get("messages") or [])
        step = int(state.get("iterations") or 0) + 1
        always_list = list(state.get("always_include") or ALWAYS_INCLUDE)
        sanitize_react_message_history(messages)

        pending_start = [
            tid
            for tid in always_list
            if tid not in _invoked_tool_ids_from_messages(messages)
        ]
        if pending_start and _messages_have_force_nudge(messages, pending_start[0]):
            print(_c(_YELLOW, f"[System: Forcing pending tool -> {pending_start[0]}]"))
            return {
                "messages": [_forced_tool_call_message(pending_start[0], step=step)],
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
            if tid not in _invoked_tool_ids_from_messages(list(messages) + [response])
        ]
        if pending_after:
            next_tid = pending_after[0]
            if _messages_have_force_nudge(messages, next_tid) or step >= MAX_ITERS:
                print(_c(_YELLOW, f"[System: Forcing pending tool -> {next_tid}]"))
                return {
                    "messages": [response, _forced_tool_call_message(next_tid, step=step)],
                    "iterations": step,
                    "last_obs": str(state.get("last_obs") or ""),
                    "final_raw": "",
                    "halt": False,
                    "always_include": always_list,
                }
            print(_c(_YELLOW, f"[System: Nudge pending tool -> {next_tid}]"))
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
        from dana.agentic import _tool_call_from_lc

        step = int(state.get("iterations") or 1)
        messages = list(state.get("messages") or [])
        last = messages[-1] if messages else None
        tool_calls = list(getattr(last, "tool_calls", None) or []) if last else []
        new_msgs: list[Any] = []
        last_obs = str(state.get("last_obs") or "")
        always_list = list(state.get("always_include") or ALWAYS_INCLUDE)

        for tc_raw in tool_calls:
            tool_call = _tool_call_from_lc(tc_raw, raw_text=TEST_PROMPT)
            call_id = str(
                getattr(tc_raw, "id", None)
                or (tc_raw.get("id") if isinstance(tc_raw, dict) else None)
                or f"call-{tool_call.tool_id}"
            )
            try:
                st = tool_map.get(tool_call.tool_id)
                if st is not None and hasattr(st, "ainvoke"):
                    observation = str(await st.ainvoke(dict(tool_call.arguments or {})))
                else:
                    observation = live_execute_tool(tool_call)
            except Exception as exc:  # noqa: BLE001
                observation = f"ERROR: tool {tool_call.tool_id} failed: {exc}"
            last_obs = sanitize_react_observation(str(observation), max_chars=8000)
            new_msgs.append(ToolMessage(content=last_obs, tool_call_id=call_id))

        still_pending = pending_always_include_tools(
            {"messages": list(messages), "always_include": always_list}
        )
        halt = step >= MAX_ITERS and not still_pending
        return {
            "messages": new_msgs,
            "iterations": step,
            "last_obs": last_obs,
            "final_raw": last_obs if halt else "",
            "halt": halt,
            "always_include": always_list,
        }

    # _route_after_agent / _route_after_tools (route_after_execution) are the
    # PRODUCTION routers — they know about nodes (critic, fail_closed,
    # verifier, ticket_validate) that this minimal 2-node agent<->tools
    # graph does not implement. route_after_execution in particular returns
    # "verifier" on a successful halt (KeyError: 'verifier' if that string
    # isn't a key in this map). Since none of those corridors exist here,
    # redirect all of them straight to END rather than adding stub nodes.
    workflow = StateGraph(ReactGraphState)
    workflow.add_node("agent", _agent_node)
    workflow.add_node("tools", _tools_node)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        _route_after_agent,
        {
            "tools": "tools",
            "agent": "agent",
            "ticket_validate": END,  # unreachable (no draft_cursor_prompt tool bound)
            END: END,
        },
    )
    workflow.add_conditional_edges(
        "tools",
        _route_after_tools,
        {
            "agent": "agent",
            "verifier": END,  # successful halt — no verifier node here
            "critic": END,  # execution_error retry corridor — no critic node here
            "fail_closed": END,  # exhausted-retry corridor — no fail_closed node here
            END: END,
        },
    )
    return workflow.compile(checkpointer=checkpointer)


async def _stream_prompt(graph: Any, user_text: str) -> None:
    """Single-shot execution: stream thought process / tool calls / results to stdout."""
    from dana.agentic import ThinkBlockTtsFilter, _dicts_to_lc_messages, strip_r1_think_blocks
    from dana.test_harness import _BOLD, _CYAN, _DIM, _GREEN, _YELLOW, _c, _chunk_text, _filter_stream_chunk
    from langchain_core.messages import AIMessage

    config = {
        "configurable": {"thread_id": THREAD_ID},
        "recursion_limit": max(10, MAX_ITERS * 4),
    }
    seed = _dicts_to_lc_messages(
        [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]
    )
    inputs = {
        "session_id": "live-actuator-session",
        "current_agent": "ReAct_Agent",
        "active_intent": ALWAYS_INCLUDE[0],
        "messages": seed,
        "iterations": 0,
        "last_obs": "",
        "final_raw": "",
        "halt": False,
        "always_include": list(ALWAYS_INCLUDE),
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
            tool_input = (event.get("data") or {}).get("input")
            executed_tools.append(tool_name)
            print(_c(_YELLOW, f"[System: Executing Tool -> {tool_name} args={tool_input}]"))
        elif kind == "on_tool_end":
            output = (event.get("data") or {}).get("output")
            print(_c(_GREEN, f"[Tool Result]\n{output}"))
        elif kind == "on_chat_model_stream":
            data = event.get("data") or {}
            piece = _filter_stream_chunk(_chunk_text(data.get("chunk")))
            piece = think_filter.feed(piece)
            if piece:
                if not streamed_any:
                    print(_c(_BOLD, "Dānā: "), end="", flush=True)
                    streamed_any = True
                print(piece, end="", flush=True)
        elif kind == "on_chain_end":
            output = (event.get("data") or {}).get("output")
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
                        answer = strip_r1_think_blocks(str(getattr(msg, "content", "") or "")).strip()
                        if answer:
                            break
        except Exception as exc:  # noqa: BLE001
            print(_c(_DIM, f"[System: state read failed: {exc}]"))

    if answer:
        print(_c(_BOLD, "Dānā: ") + answer)
    print(_c(_DIM, f"[System: tools_executed={executed_tools}]"))


def _print_banner(dry_run: bool) -> None:
    print("=" * 72)
    print("Dānā LIVE ACTUATOR TEST — headless, no GUI")
    print("=" * 72)
    print(f"DANA_OS_DRY_RUN = {os.environ['DANA_OS_DRY_RUN']!r} "
          f"({'REHEARSAL — no real OS actions' if dry_run else 'REAL Win32 actuation is LIVE'})")
    print(f"Tools under test : {ALWAYS_INCLUDE}")
    print(f"Prompt           : {TEST_PROMPT!r}")
    print("-" * 72)
    print("KILL SWITCH:")
    hotkey = os.environ.get("DANA_KILL_HOTKEY", "f12").strip() or "f12"
    print(f"  * Press {hotkey.upper()} at any time -> Dānā's hardware panic button.")
    print("    Every actuator checks this immediately before it moves your")
    print("    mouse/keyboard/clipboard for real, and aborts if it's set.")
    print("  * Press Ctrl+C in this terminal -> stops this script's LLM loop.")
    print("-" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Set DANA_OS_DRY_RUN=1 instead of 0 for a safe rehearsal pass.",
    )
    args = parser.parse_args()

    os.environ["DANA_OS_DRY_RUN"] = "1" if args.dry_run else "0"

    _print_banner(dry_run=args.dry_run)

    try:
        from dana.middleware.kill_switch import start_kill_switch_listener

        start_kill_switch_listener()
    except Exception as exc:  # noqa: BLE001
        print(f"[System: kill-switch listener unavailable: {exc}]")

    print("Hands off the mouse/keyboard — starting in:")
    for n in (3, 2, 1):
        print(f"  {n}...")
        time.sleep(1)
    print("Go.\n")

    from dana.test_harness import _cli_checkpointer, _enable_ansi

    _enable_ansi()
    checkpointer = _cli_checkpointer()
    graph = _compile_live_graph(checkpointer)
    asyncio.run(_stream_prompt(graph, TEST_PROMPT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
