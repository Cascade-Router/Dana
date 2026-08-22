#!/usr/bin/env python3
"""The Kobayashi Maru gauntlet — five historically crash/exhaustion-inducing
prompts, replayed through Dana's REAL ``/ws/chat`` routing
(``dana.api.server.ws_chat`` / ``_run_react_loop``), the REAL dispatcher
(``dana.core.react_dispatch.dispatch_tool_call``), and — wherever a
FreeCADCmd binary is installed — the REAL FreeCAD engine. Nothing in the
engine/tools/dispatcher is modified or monkeypatched; this is a pure
evaluation harness.

Only two things are stubbed, both deliberately external to what this
gauntlet evaluates:

  1. The LLM's tool *selection* — a live model is slow and nondeterministic,
     so each scenario replays a fixed, scripted sequence of ``ToolCall``s
     (exactly ``dana.core.react_dispatch.ModelProvider``'s call shape) —
     this pins down WHICH tool the agent "chooses" so the pass/fail
     criteria below are reproducible, while every call still flows through
     the real ``next_react_turn -> dispatch_tool_call -> engine`` chain.
  2. The VLM canvas analysis (``analyze_cad_blueprint``) — a network/local-
     model call, not part of the ReAct/dispatch machinery under test.

Everything else — WebSocket message routing, HITL suspend/resume, the
visual-capture suspend path, ``dispatch_tool_call``, ``error_digest.py``,
and (when available) the real FreeCAD subprocess — runs unmodified.

Usage (from repo root)::

    .venv\\Scripts\\python.exe scripts/run_kobayashi_maru.py

Exit 0 when 5/5 scenarios survive.
"""

from __future__ import annotations

import base64
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

import dana.core.react_dispatch as react_dispatch  # noqa: E402
from dana.api import server as server_module  # noqa: E402
from dana.tools.schema import ToolCall  # noqa: E402

# A real (tiny) PNG, padded past cad_vision's 256-char inline-data threshold.
_FAKE_PNG_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 300).decode("ascii")

PROMPTS = [
    "Create a 10mm x 10mm x 10mm box and apply a 20mm fillet to all of its edges.",
    "Align this cylinder to the box using a concentric hinge mate so that the cylinder can freely rotate 90 degrees.",
    "Look at the generated robotics motor mount on the canvas. Does it look too bulky, or are the proportions correct?",
    "Build a checkerboard by individually placing and aligning 64 alternating square tiles.",
    "Create a smooth, aerodynamic loft blending the top face of the star prism into a circular cylinder.",
]


class _ScriptedProvider:
    """Replays a fixed queue of ToolCall lists / final-text strings as
    successive LLM turns — pins down tool SELECTION deterministically while
    every call still runs through the real dispatch/engine chain."""

    def __init__(self, turns: list[list[ToolCall] | str]) -> None:
        self._turns = list(turns)

    def complete_with_tool_calls(self, messages: Any, *, tools: Any, provider: Any = None, **kwargs: Any) -> dict:
        turn = self._turns.pop(0) if self._turns else "Done."
        if isinstance(turn, str):
            return {"content": turn, "tool_calls": [], "provider": "kobayashi-maru"}
        return {"content": "", "tool_calls": turn, "provider": "kobayashi-maru"}


def _fake_vlm_analysis(_image_b64: str, **_kwargs: Any) -> str:
    return '{"ok": true, "summary": "Proportions look correct for a NEMA17 mount.", "entities": []}'


class _Recorder:
    """Every WS event received for one scenario, so assertions can inspect
    exactly what the dispatcher/engine actually did."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def of_type(self, msg_type: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("type") == msg_type]


def _default_on_event(recorder: _Recorder, ws: Any, msg: dict[str, Any]) -> bool:
    """Auto-approves HITL and auto-answers a visual-capture request — the
    default flow for scenarios not specifically probing the suspend
    mechanism itself. Returns True once the loop reaches a final message."""
    if msg["type"] == "hitl_approval_required":
        ws.send_json(
            {"type": "hitl_response", "payload": {"request_id": msg["payload"]["request_id"], "approved": True}}
        )
    elif msg["type"] == "visual_capture_request":
        ws.send_json(
            {
                "type": "visual_capture_response",
                "payload": {"request_id": msg["payload"]["request_id"], "image_b64": _FAKE_PNG_B64},
            }
        )
    return msg["type"] == "assistant_message"


def _run_scenario(
    client: TestClient,
    prompt: str,
    turns: list[list[ToolCall] | str],
    on_event: Callable[[_Recorder, Any, dict[str, Any]], bool] = _default_on_event,
) -> _Recorder:
    """Sends `prompt` over a real /ws/chat connection, replays `turns` as
    the LLM's responses, and drains every WS event into a Recorder."""
    # One provider instance, captured by the closure below — ModelProvider()
    # is called fresh on every LLM turn, so returning a NEW _ScriptedProvider
    # each time would reset its queue back to turns[0] forever instead of
    # depleting it turn by turn.
    provider = _ScriptedProvider(turns)
    react_dispatch.ModelProvider = lambda: provider  # noqa: E731
    recorder = _Recorder()
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"text": prompt})
        for _ in range(60):
            msg = ws.receive_json()
            recorder.events.append(msg)
            if msg.get("type") == "assistant_message" or on_event(recorder, ws, msg):
                break
    return recorder


def _tool_result(recorder: _Recorder, tool_id: str) -> dict[str, Any] | None:
    matches = [e for e in recorder.of_type("tool_result") if e["tool_id"] == tool_id]
    return matches[0] if matches else None


def _is_digested_error(payload: dict[str, Any]) -> bool:
    """The exact shape dispatch_tool_call/error_digest.digest_error produces
    for any failed call: {"ok": False, "status": "error", "reason": ...,
    "suggestion": ..., "raw_error": ...}."""
    return payload.get("status") == "error" and "reason" in payload and "suggestion" in payload


# --------------------------------------------------------------------------
# Scenario 1 — The Kernel Crash
# --------------------------------------------------------------------------


def scenario_kernel_crash(client: TestClient) -> tuple[bool, str]:
    """A 20mm fillet on a 10mm cube's edges. Empirically, this FreeCAD build
    doesn't raise on the fillet call itself — it silently returns an
    invalid/degenerate shape (ok: True). The REAL crash surfaces one step
    later: reading that shape's spatial properties (the system prompt's own
    "verify before mutating" rule) genuinely fails, and THAT failure is what
    error_digest.py must catch and structure — a truer test of the full
    reasoning framework than asserting the first call alone fails."""
    turns = [
        [ToolCall(tool_id="create_freecad_box", arguments={"length": 10, "width": 10, "height": 10, "name": "KMBox"})],
        [
            ToolCall(
                tool_id="perform_freecad_edge_operation",
                arguments={"operation": "fillet", "target_object": "KMBox", "value": 20},
            )
        ],
        [ToolCall(tool_id="inspect_spatial_properties", arguments={"target_object": "Fillet"})],
        "That fillet radius produced invalid geometry — I'd recommend a much smaller radius.",
    ]
    rec = _run_scenario(client, PROMPTS[0], turns)

    edge_op = _tool_result(rec, "perform_freecad_edge_operation")
    inspect = _tool_result(rec, "inspect_spatial_properties")
    if edge_op is None:
        return False, "perform_freecad_edge_operation was never dispatched"

    # Either the edge operation itself fails and gets digested (a stricter
    # FreeCAD build), or it silently "succeeds" and the follow-up inspection
    # catches the corruption instead — either is a legitimate survival path.
    if not edge_op["ok"] and _is_digested_error(edge_op["payload"]):
        return True, f"fillet failed directly and was digested: reason={edge_op['payload']['reason']!r}"
    if inspect is not None and not inspect["ok"] and _is_digested_error(inspect["payload"]):
        return True, (
            "fillet silently returned invalid geometry (ok=True); the follow-up "
            f"inspect_spatial_properties call caught it and was digested: reason={inspect['payload']['reason']!r}"
        )
    return False, (
        f"neither the fillet nor the follow-up inspection produced a digested error "
        f"(edge_op.ok={edge_op['ok']}, inspect={inspect['payload'] if inspect else 'never called'})"
    )


# --------------------------------------------------------------------------
# Scenario 2 — The Assembly Constraint
# --------------------------------------------------------------------------


def scenario_assembly_mate(client: TestClient) -> tuple[bool, str]:
    turns = [
        [ToolCall(tool_id="create_freecad_box", arguments={"length": 20, "width": 20, "height": 10, "name": "KMHingeBase"})],
        [ToolCall(tool_id="create_freecad_cylinder", arguments={"radius": 5, "height": 30, "name": "KMHingePin"})],
        [
            ToolCall(
                tool_id="create_assembly_mate",
                arguments={
                    "fixed_obj": "KMHingeBase",
                    "moving_obj": "KMHingePin",
                    "mate_type": "concentric",
                    "mate_params": {"z_offset": 5.0},
                },
            )
        ],
        "Mated the cylinder concentrically to the box — it now shares the box's central axis.",
    ]
    rec = _run_scenario(client, PROMPTS[1], turns)

    mate = _tool_result(rec, "create_assembly_mate")
    if mate is None:
        return False, "create_assembly_mate was never called"
    if not mate["ok"]:
        return False, f"create_assembly_mate failed: {mate['payload']}"
    return True, f"create_assembly_mate succeeded (mate_type={mate['payload'].get('mate_type')!r})"


# --------------------------------------------------------------------------
# Scenario 3 — The Visual Judgment
# --------------------------------------------------------------------------


def scenario_visual_judgment(client: TestClient) -> tuple[bool, str]:
    turns = [
        [ToolCall(tool_id="take_canvas_screenshot", arguments={})],
        "The proportions look correct for a NEMA17 mount — not overly bulky.",
    ]
    probe: dict[str, Any] = {}

    def on_event(recorder: _Recorder, ws: Any, msg: dict[str, Any]) -> bool:
        if msg["type"] == "visual_capture_request":
            # The actual claim under test: peek at server-side session
            # state BEFORE resolving the suspend.
            sessions = list(server_module._active_sessions.values())
            state = sessions[0].get("visual_state") if sessions else None
            probe["suspended"] = bool(state) and state["call"].tool_id == "take_canvas_screenshot"
        return _default_on_event(recorder, ws, msg)

    rec = _run_scenario(client, PROMPTS[2], turns, on_event=on_event)

    if not probe.get("suspended"):
        return False, "session['visual_state'] was never populated for take_canvas_screenshot"
    screenshot = _tool_result(rec, "take_canvas_screenshot")
    if screenshot is None or not screenshot["ok"]:
        return False, f"visual capture round-trip did not resolve cleanly: {screenshot}"
    return True, "session['visual_state'] suspend path triggered on take_canvas_screenshot, then resolved cleanly"


# --------------------------------------------------------------------------
# Scenario 4 — The Loop Exhaustion
# --------------------------------------------------------------------------


def scenario_loop_exhaustion(client: TestClient) -> tuple[bool, str]:
    turns = [
        [ToolCall(tool_id="create_freecad_box", arguments={"length": 10, "width": 10, "height": 2, "name": "KMTile"})],
        [
            ToolCall(
                tool_id="batch_pattern_array",
                arguments={"source_object": "KMTile", "pattern_type": "grid", "count_x": 8, "count_y": 8},
            )
        ],
        "Built an 8x8 grid of 64 tiles in two tool calls.",
    ]
    rec = _run_scenario(client, PROMPTS[3], turns)

    pattern = _tool_result(rec, "batch_pattern_array")
    if pattern is None:
        return False, "batch_pattern_array was never called"
    if not pattern["ok"]:
        return False, f"batch_pattern_array failed: {pattern['payload']}"

    copy_count = pattern["payload"].get("dimensions", {}).get("copy_count")
    dispatches = len(rec.of_type("tool_result"))
    if dispatches >= 5:  # dana.api.server._MAX_REACT_ITERATIONS
        return False, f"used {dispatches} dispatch iterations — did not avoid the 5-iteration cap"
    return True, (
        f"{copy_count} tiles placed in {dispatches} tool call(s) — well under the 5-iteration cap "
        f"that 64 individual create+align calls would have blown straight through"
    )


# --------------------------------------------------------------------------
# Scenario 5 — The Missing Geometry
# --------------------------------------------------------------------------


def scenario_missing_geometry(client: TestClient) -> tuple[bool, str]:
    """No loft primitive exists in the tool registry. Forces the LLM to
    "hallucinate" one (create_freecad_loft) and checks that
    next_react_turn's existing safety net — an unrecognized tool_id
    degrades to a final text turn — catches it before it ever reaches
    dispatch_tool_call, rather than crashing the WS connection."""
    turns = [[ToolCall(tool_id="create_freecad_loft", arguments={"profile_a": "StarPrism", "profile_b": "Cylinder"})]]
    rec = _run_scenario(client, PROMPTS[4], turns)

    if rec.of_type("tool_call") or rec.of_type("tool_result"):
        return False, "the hallucinated tool call reached the dispatcher instead of being caught upstream"
    if not rec.of_type("assistant_message"):
        return False, "the loop never produced a final message — did not degrade safely"
    return True, "unrecognized tool_id ('create_freecad_loft') was rejected before dispatch; loop ended cleanly, no crash"


SCENARIOS: list[tuple[str, Callable[[TestClient], tuple[bool, str]]]] = [
    ("The Kernel Crash", scenario_kernel_crash),
    ("The Assembly Constraint", scenario_assembly_mate),
    ("The Visual Judgment", scenario_visual_judgment),
    ("The Loop Exhaustion", scenario_loop_exhaustion),
    ("The Missing Geometry", scenario_missing_geometry),
]


def main() -> int:
    original_provider = react_dispatch.ModelProvider
    original_vlm = react_dispatch.analyze_cad_blueprint
    original_captures_dir = react_dispatch.CAPTURES_DIR
    original_screenshot_path = react_dispatch._LAST_CANVAS_SCREENSHOT_PATH
    scratch = Path(tempfile.mkdtemp(prefix="kobayashi_maru_"))
    react_dispatch.analyze_cad_blueprint = _fake_vlm_analysis
    react_dispatch.CAPTURES_DIR = scratch
    react_dispatch._LAST_CANVAS_SCREENSHOT_PATH = scratch / "last_canvas_screenshot.png"

    print("=" * 78)
    print(" KOBAYASHI MARU GAUNTLET — Dana ReAct Integration Evaluation")
    print("=" * 78)

    client = TestClient(server_module.app)
    results: list[bool] = []
    try:
        for i, (name, scenario) in enumerate(SCENARIOS, start=1):
            try:
                passed, detail = scenario(client)
            except Exception as exc:  # noqa: BLE001 — a scenario crashing IS a gauntlet failure, not a script crash
                passed, detail = False, f"scenario raised {type(exc).__name__}: {exc}"
                traceback.print_exc()
            results.append(passed)
            status = "PASS" if passed else "FAIL"
            print(f"[{i}/5] {name:<28} ... {status}")
            print(f"      -> {detail}")
    finally:
        react_dispatch.ModelProvider = original_provider
        react_dispatch.analyze_cad_blueprint = original_vlm
        react_dispatch.CAPTURES_DIR = original_captures_dir
        react_dispatch._LAST_CANVAS_SCREENSHOT_PATH = original_screenshot_path

    passed_count = sum(results)
    print("-" * 78)
    print(f"RESULT: {passed_count}/{len(SCENARIOS)} scenarios survived the gauntlet.")
    print("=" * 78)
    return 0 if passed_count == len(SCENARIOS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
