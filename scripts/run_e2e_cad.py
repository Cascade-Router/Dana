#!/usr/bin/env python3
"""Headless end-to-end runner for Dana's CAD ReAct pipeline — no GUI, no
Tauri frontend, but the REAL ``dana.api.server`` orchestrator underneath.

This used to call ``dana.core.react_dispatch.dispatch_tool_call`` directly
in its own hand-rolled loop, which bypassed ``dana.api.server
._execute_and_continue`` entirely — the ONE place the auto-mesh-export
(STL/STEP export after every geometry call) and Automatic Visual
Verification (headless CAD-viewport screenshot + VLM read, merged into the
tool's own result payload) hooks actually live. A script meant to catch
regressions in either could never see one that way: it was exercising a
strictly narrower code path than a real session ever takes.

Instead, this now drives ``dana.api.server._process_user_text`` — the exact
same entry point ``ws_chat`` calls for every real chat message, which itself
chains ``_run_react_loop`` -> ``_execute_and_continue`` for each tool call —
through a minimal duck-typed ``WebSocket`` stand-in that only needs
``send_json`` (nothing here ever calls ``receive_json``; that's ``ws_chat``'s
own message loop, which this runner replaces outright rather than reusing).

Every tool this is meant to exercise (create_freecad_box, insert_standard_part,
modify_freecad_parameter, perform_freecad_boolean, ...) is in
``dana.api.server._HITL_ALWAYS_APPROVED_TOOLS``, so none of them ever suspend
the loop waiting for a human approval a headless script could never supply.
If a call DOES suspend (HITL approval or a visual-capture request — e.g. a
custom prompt reaching for a tool outside that allowlist), this runner
detects the stuck session state below and fails loudly instead of hanging.

Local Chat Session Persistence is still exercised for real (a genuine,
uniquely-named session is created and handed through the same
_process_user_text -> _finish_turn -> save_session path a live chat uses) —
full orchestrator parity means not silently skipping that either — but its
on-disk record is deleted again once the run ends, so repeated CI runs never
accumulate phantom chats in the user's real session storage.

Usage (from repo root)::

    python scripts/run_e2e_cad.py                 # runs the default master prompt
    python scripts/run_e2e_cad.py "some other prompt"
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dana.api.server import _process_user_text  # noqa: E402
from dana.api.sessions import SESSIONS_DIR, new_session_id  # noqa: E402
from dana.platform.factory import get_cad_engine  # noqa: E402
from dana.plugins.freecad.call_log import CadCallLog  # noqa: E402

_MASTER_PROMPT = (
    "Build a box 60x40x20 and insert an ISO4017 hex bolt size M8 length 30. "
    "Then move the bolt to X=30, Y=20, Z=10 and perform a boolean cut to "
    "subtract the bolt from the box."
)

# Substrings of the specific terminal messages dana.api.server._run_react_loop/
# _execute_and_continue hand to _finish_turn on every NON-success ending
# (max iterations, a repeated identical failure, an LLM/proxy error, or an
# explicit abort) — matched against the run's final "assistant_message" event
# to decide this process's exit code, the same distinction the old hand-rolled
# loop drew from turn.kind/repeated dispatch failures itself. A genuine final
# answer never contains any of these.
_FAILURE_MARKERS: tuple[str, ...] = (
    "Reached the maximum number of reasoning steps",
    "failed with the same error twice in a row",
    "I ran into a problem talking to the model",
    "Generation aborted by user.",
)


def _banner(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)


class _FakeWebSocket:
    """Duck-typed stand-in for ``fastapi.WebSocket``. ``_process_user_text``'s
    whole call chain only ever calls ``send_json`` on the websocket it's
    given (``receive_json`` belongs to ``ws_chat``'s own message loop, which
    this runner replaces rather than reuses) — so that's the only method
    this needs, recording every event and printing the ones a human watching
    this run would actually care about.
    """

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def send_json(self, data: dict[str, Any]) -> None:
        self.events.append(data)
        event_type = data.get("type")

        if event_type == "dag_node_start" and data.get("label") == "Parse intent":
            _banner(f"Iteration {data.get('inputs', {}).get('step')}")
        elif event_type == "tool_call":
            print(f"[runner] TOOL CALL -> {data['tool_id']}({json.dumps(data.get('arguments'))})", flush=True)
        elif event_type == "tool_result":
            payload_str = json.dumps(data.get("payload"), default=str)
            if len(payload_str) > 1000:
                payload_str = payload_str[:1000] + "...<truncated>"
            status = "OK" if data.get("ok") else "FAILED"
            print(f"[runner] TOOL RESULT [{status}] {data['tool_id']}: {payload_str}", flush=True)
            if data.get("mesh_url"):
                print(f"[runner]   mesh_url (auto-export hook fired): {data['mesh_url']}", flush=True)
            if not data.get("ok"):
                print(f"[runner] TOOL ERROR MESSAGE: {data.get('message')}", flush=True)
        elif event_type == "assistant_message":
            print(f"[runner] FINAL LLM RESPONSE:\n{data.get('content')}", flush=True)
        elif event_type == "hitl_approval_required":
            action_name = (data.get("payload") or {}).get("action_name")
            print(
                f"[runner] STUCK: '{action_name}' needs HITL approval — a headless run can't supply that.",
                flush=True,
            )
        elif event_type == "visual_capture_request":
            print(
                "[runner] STUCK: a visual-inspection tool needs a live canvas capture — "
                "a headless run can't supply that.",
                flush=True,
            )


async def run(prompt: str) -> int:
    engine = get_cad_engine()
    print(f"[runner] CAD engine driver: {type(engine).__name__}", flush=True)
    print(f"[runner] Prompt: {prompt}", flush=True)

    websocket = _FakeWebSocket()
    session_id = f"e2e-{new_session_id()}"
    # Mirrors dana.api.server.ws_chat's own session dict shape verbatim
    # (minus its websocket-bootstrap-only fields) so _process_user_text/
    # _run_react_loop/_execute_and_continue see exactly what they'd see from
    # a real connection — nothing here special-cases "this is a test run".
    session: dict[str, Any] = {
        "active_selection": None,
        "react_state": None,
        "visual_state": None,
        "call_log": CadCallLog(),
        "session_id": session_id,
        "chat_history": [],
        "session_title": None,
        "session_created_at": None,
        "api_keys": {},
        # "freecad" is the full raw CAD tool domain (create_freecad_*,
        # perform_freecad_boolean, generate_urdf_assembly, export_freecad_model,
        # ...) — what a real session gets with the CAD tab active, and the
        # closest equivalent to this script's old build_system_prompt(None)
        # "everything" fallback now that _run_react_loop always resolves an
        # explicit active_plugins frozenset (never None) via
        # _effective_capabilities.
        "active_plugins": frozenset({"freecad"}),
        "capability_unlocked_at_turn": {},
        "working_memory": {"summary": "", "turn": 0},
        "turn_counter": 0,
        "abort_requested": False,
        "hitl_approved_tools": set(),
    }

    try:
        await _process_user_text(websocket, session, prompt)
    finally:
        try:
            (SESSIONS_DIR / f"{session_id}.json").unlink(missing_ok=True)
        except OSError:
            pass

    if session.get("react_state") is not None or session.get("visual_state") is not None:
        print(
            "[runner] STOPPING: turn suspended waiting for a human/frontend response "
            "this headless runner cannot supply.",
            flush=True,
        )
        return 1

    final_events = [e for e in websocket.events if e.get("type") == "assistant_message"]
    if not final_events:
        print("[runner] STOPPING: no final assistant_message was ever sent.", flush=True)
        return 1

    final_content = final_events[-1].get("content") or ""
    if any(marker in final_content for marker in _FAILURE_MARKERS):
        return 1
    return 0


if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) or _MASTER_PROMPT
    sys.exit(asyncio.run(run(prompt)))
