#!/usr/bin/env python3
"""Headless end-to-end runner for Dana's CAD ReAct pipeline — no GUI, no
WebSocket, no Tauri frontend. Drives the exact same orchestration primitives
``dana.api.server``'s ``/ws/chat`` handler uses (``next_react_turn`` /
``dispatch_tool_call`` / ``build_assistant_tool_call_message`` /
``build_tool_result_message``), just looped directly in-process instead of
through a websocket session dict.

Every tool this is meant to exercise (create_freecad_box, insert_standard_part,
modify_freecad_parameter, perform_freecad_boolean, ...) is in
``dana.api.server._HITL_ALWAYS_APPROVED_TOOLS``, so none of them ever suspend
the loop waiting for a human approval that a headless script could never
supply — this runner does not need to (and does not) reimplement the HITL
approval gate at all.

Usage (from repo root)::

    python scripts/run_e2e_cad.py                 # runs the default master prompt
    python scripts/run_e2e_cad.py "some other prompt"
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dana.core.react_dispatch import (  # noqa: E402
    build_assistant_tool_call_message,
    build_system_prompt,
    build_tool_result_message,
    dispatch_tool_call,
    next_react_turn,
)
from dana.platform.factory import get_cad_engine, get_control_plane  # noqa: E402

_MAX_ITERATIONS = 13  # mirrors dana.api.server._MAX_REACT_ITERATIONS

_MASTER_PROMPT = (
    "Build a box 60x40x20 and insert an ISO4017 hex bolt size M8 length 30. "
    "Then move the bolt to X=30, Y=20, Z=10 and perform a boolean cut to "
    "subtract the bolt from the box."
)


def _banner(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)


async def run(prompt: str) -> int:
    engine = get_cad_engine()
    control_plane = get_control_plane()
    print(f"[runner] CAD engine driver: {type(engine).__name__}", flush=True)
    print(f"[runner] Prompt: {prompt}", flush=True)

    messages: list[dict] = [
        {"role": "system", "content": build_system_prompt(None)},
        {"role": "user", "content": prompt},
    ]

    last_failure: tuple[str, str] | None = None

    for iteration in range(_MAX_ITERATIONS):
        _banner(f"Iteration {iteration}")
        turn = await next_react_turn(messages, None, raw_text=prompt)

        if turn.kind == "error":
            print(f"[runner] LLM ERROR: {turn.content}", flush=True)
            return 1

        if turn.kind == "final":
            print(f"[runner] FINAL LLM RESPONSE:\n{turn.content}", flush=True)
            return 0

        call = turn.call
        assistant_message, tool_call_id = build_assistant_tool_call_message(call)
        messages.append(assistant_message)

        print(f"[runner] TOOL CALL -> {call.tool_id}({json.dumps(call.arguments)})", flush=True)
        result = dispatch_tool_call(call, engine, control_plane)
        status = "OK" if result.ok else "FAILED"
        payload_str = json.dumps(result.payload, default=str)
        if len(payload_str) > 1000:
            payload_str = payload_str[:1000] + "...<truncated>"
        print(f"[runner] TOOL RESULT [{status}] {call.tool_id}: {payload_str}", flush=True)
        if not result.ok:
            print(f"[runner] TOOL ERROR MESSAGE: {result.message}", flush=True)

        messages.append(build_tool_result_message(tool_call_id, result))

        current_failure = None if result.ok else (call.tool_id, result.message)
        if current_failure is not None and current_failure == last_failure:
            print(
                f"[runner] STOPPING: '{call.tool_id}' failed with the same error twice in a "
                f"row: {result.message}",
                flush=True,
            )
            return 1
        last_failure = current_failure

    print(f"[runner] STOPPING: reached max iterations ({_MAX_ITERATIONS}) without a final answer", flush=True)
    return 1


if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) or _MASTER_PROMPT
    sys.exit(asyncio.run(run(prompt)))
