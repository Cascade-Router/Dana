#!/usr/bin/env python3
"""The Kobayashi Maru gauntlet — five historically crash/exhaustion-inducing
prompts, replayed through Dana's REAL ``/ws/chat`` routing
(``dana.api.server.ws_chat`` / ``_run_react_loop``), the REAL plan-gated
FSM and dispatcher (``dana.core.react_dispatch``), and either the REAL
FreeCAD engine or the trimesh mock engine.

Only two things are stubbed, both deliberately external to what this
gauntlet evaluates:

  1. The LLM's tool *selection* — a live model is slow and nondeterministic,
     so each scenario replays a fixed, scripted sequence of ``ToolCall``s
     (exactly ``dana.core.react_dispatch.ModelProvider``'s call shape) —
     this pins down WHICH tool the agent "chooses" so the pass/fail
     criteria below are reproducible, while every call still flows through
     the real ``next_react_turn -> dispatch_tool_call -> engine`` chain.
     Every geometry scenario opens with ``create_plan``, one task per
     operation, because the FSM refuses geometry before a plan exists.
  2. The VLM canvas analysis (``analyze_cad_blueprint``) — a network/local-
     model call, not part of the ReAct/dispatch machinery under test.

Everything else — WebSocket message routing, HITL suspend/resume, the
visual-capture suspend path, ``dispatch_tool_call``, ``error_digest.py`` —
runs unmodified.

Usage (from repo root)::

    .venv\\Scripts\\python.exe scripts/run_kobayashi_maru.py [--mock | --freecad] [--keep-artifacts]

``--mock`` runs on the trimesh mock engine (no FreeCAD needed, what CI and
``tests/test_kobayashi_maru.py`` use); ``--freecad`` requires FreeCADCmd.
Without either, FreeCAD is used when FreeCADCmd is found, else the mock.

Runs in a scratch workspace (``scripts/scratch_workspace.py``): the
gauntlet's chats, documents and exports never reach the user's real ones.

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


def _plan(objective: str, *tasks: tuple[str, list[str]]) -> list[ToolCall]:
    """The create_plan turn every geometry scenario now opens with: one task
    per operation, each declaring the one tool it runs (the FSM offers and
    dispatches exactly the active task's tools)."""
    return [
        ToolCall(
            tool_id="create_plan",
            arguments={
                "objective": objective,
                "tasks": [{"description": d, "expected_tools": tools} for d, tools in tasks],
            },
        )
    ]


class _Recorder:
    """Every WS event received for one scenario, so assertions can inspect
    exactly what the dispatcher/engine actually did."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def of_type(self, msg_type: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("type") == msg_type]

    def dispatched(self) -> list[str]:
        return [e["tool_name"] for e in self.of_type("tool_dispatch_start")]


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
    # is called fresh (with api_keys=...) on every LLM turn, so returning a
    # NEW _ScriptedProvider each time would reset its queue back to
    # turns[0] forever instead of depleting it turn by turn.
    provider = _ScriptedProvider(turns)
    react_dispatch.ModelProvider = lambda **_kwargs: provider  # noqa: E731
    recorder = _Recorder()
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"type": "update_context", "active_plugins": ["cad"]})
        ws.send_json({"text": prompt})
        for _ in range(200):
            msg = ws.receive_json()
            recorder.events.append(msg)
            if msg.get("type") == "assistant_message" or on_event(recorder, ws, msg):
                break
    return recorder


def _tool_result(recorder: _Recorder, tool_id: str) -> dict[str, Any] | None:
    """The first ``tool_dispatch_end`` for ``tool_id`` as {"ok", "payload"}."""
    for event in recorder.of_type("tool_dispatch_end"):
        if event["tool_id"] == tool_id:
            return {"ok": event["status"] == "success", "payload": event.get("output") or {}}
    return None


def _is_digested_error(payload: dict[str, Any]) -> bool:
    """The exact shape dispatch_tool_call/error_digest.digest_error produces
    for any failed call: {"ok": False, "status": "error", "reason": ...,
    "suggestion": ..., "raw_error": ...}."""
    return payload.get("status") == "error" and "reason" in payload and "suggestion" in payload


# --------------------------------------------------------------------------
# Scenario 1 — The Kernel Crash
# --------------------------------------------------------------------------


def scenario_kernel_crash(client: TestClient) -> tuple[bool, str]:
    """A 20mm fillet on a 10mm cube's edges. Depending on the kernel, the
    fillet call itself fails, or it silently returns a degenerate shape
    (ok: True) and the follow-up read of its spatial properties fails. Either
    failure must come back as a structured error_digest payload the model can
    act on, never an unhandled exception."""
    turns = [
        _plan(
            PROMPTS[0],
            ("Create the 10 mm cube", ["create_freecad_box"]),
            ("Fillet the cube's edges with a 20 mm radius", ["perform_freecad_edge_operation"]),
            ("Inspect the filleted solid", ["inspect_spatial_properties"]),
        ),
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
        return False, f"perform_freecad_edge_operation was never dispatched (dispatched: {rec.dispatched()})"
    if not edge_op["ok"] and _is_digested_error(edge_op["payload"]):
        return True, f"fillet failed directly and was digested: reason={edge_op['payload']['reason']!r}"
    if inspect is not None and not inspect["ok"] and _is_digested_error(inspect["payload"]):
        return True, (
            "fillet silently returned invalid geometry (ok=True); the follow-up "
            f"inspect_spatial_properties call caught it and was digested: reason={inspect['payload']['reason']!r}"
        )
    return False, (
        f"neither the fillet nor the follow-up inspection produced a digested error "
        f"(edge_op={edge_op['payload']}, inspect={inspect['payload'] if inspect else 'never called'})"
    )


# --------------------------------------------------------------------------
# Scenario 2 — The Assembly Constraint
# --------------------------------------------------------------------------


def scenario_assembly_mate(client: TestClient) -> tuple[bool, str]:
    """A concentric hinge mate between two freshly made parts must succeed
    through the real assembly path, not just be accepted."""
    turns = [
        _plan(
            PROMPTS[1],
            ("Create the hinge base block", ["create_freecad_box"]),
            ("Create the hinge pin cylinder", ["create_freecad_cylinder"]),
            ("Mate the pin concentrically to the base", ["create_assembly_mate"]),
        ),
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
        return False, f"create_assembly_mate was never called (dispatched: {rec.dispatched()})"
    if not mate["ok"]:
        return False, f"create_assembly_mate failed: {mate['payload']}"
    return True, f"create_assembly_mate succeeded (mate_type={mate['payload'].get('mate_type')!r})"


# --------------------------------------------------------------------------
# Scenario 3 — The Visual Judgment
# --------------------------------------------------------------------------


def scenario_visual_judgment(client: TestClient) -> tuple[bool, str]:
    """take_canvas_screenshot needs the frontend's canvas: the loop must park
    on session['visual_state'], ask for a capture, and resume once one
    arrives. No plan needed: it is a core tool, offered in every phase."""
    turns = [
        [ToolCall(tool_id="take_canvas_screenshot", arguments={})],
        "The proportions look correct for a NEMA17 mount — not overly bulky.",
    ]
    probe: dict[str, Any] = {}

    def on_event(recorder: _Recorder, ws: Any, msg: dict[str, Any]) -> bool:
        if msg["type"] == "visual_capture_request":
            # The actual claim under test: peek at server-side session
            # state BEFORE resolving the suspend.
            states = [s.get("visual_state") for s in server_module._active_sessions.values()]
            probe["suspended"] = any(s and s["call"].tool_id == "take_canvas_screenshot" for s in states)
        return _default_on_event(recorder, ws, msg)

    # Headless mode (CI, the local .env) answers take_canvas_screenshot with
    # a "skipped" result instead of suspending, since there is no canvas.
    # Turn off only that pruning, as tests/conftest.py's
    # interactive_tool_schema fixture does: DANA_HEADLESS itself stays set,
    # so the FreeCAD GUI-launch lock is untouched.
    real_is_headless = react_dispatch.is_headless
    react_dispatch.is_headless = lambda: False
    try:
        rec = _run_scenario(client, PROMPTS[2], turns, on_event=on_event)
    finally:
        react_dispatch.is_headless = real_is_headless

    if not probe.get("suspended"):
        return False, "session['visual_state'] was never populated for take_canvas_screenshot"
    screenshot = _tool_result(rec, "take_canvas_screenshot")
    if screenshot is None or not screenshot["ok"]:
        return False, f"visual capture round-trip did not resolve cleanly: {screenshot}"
    if not rec.of_type("assistant_message"):
        return False, "the loop never resumed to a final answer after the capture"
    return True, "session['visual_state'] suspend path triggered on take_canvas_screenshot, then resolved cleanly"


# --------------------------------------------------------------------------
# Scenario 4 — The Loop Exhaustion
# --------------------------------------------------------------------------


def scenario_loop_exhaustion(client: TestClient) -> tuple[bool, str]:
    """64 tiles one call at a time would blow through the iteration cap; a
    single batch_pattern_array call must place all of them."""
    turns = [
        _plan(
            PROMPTS[3],
            ("Create one square tile", ["create_freecad_box"]),
            ("Pattern the tile into an 8 by 8 grid", ["batch_pattern_array"]),
        ),
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
        return False, f"batch_pattern_array was never called (dispatched: {rec.dispatched()})"
    if not pattern["ok"]:
        return False, f"batch_pattern_array failed: {pattern['payload']}"

    copy_count = pattern["payload"].get("dimensions", {}).get("copy_count")
    geometry_calls = [t for t in rec.dispatched() if t != "create_plan"]
    if copy_count != 64:
        return False, f"batch_pattern_array placed {copy_count} tiles, not 64"
    if len(geometry_calls) > 2:
        return False, f"used {len(geometry_calls)} geometry calls ({geometry_calls}) for what one pattern call covers"
    return True, (
        f"{copy_count} tiles placed in {len(geometry_calls)} geometry calls — 64 individual "
        f"create+align calls would have blown through the iteration cap"
    )


# --------------------------------------------------------------------------
# Scenario 5 — The Missing Geometry
# --------------------------------------------------------------------------


def scenario_missing_geometry(client: TestClient) -> tuple[bool, str]:
    """No loft primitive exists in the tool registry. Forces the LLM to
    "hallucinate" one (create_freecad_loft) and checks that the router
    rejects the unknown tool_id before dispatch_tool_call ever sees it, and
    the loop still ends in a final message instead of crashing the socket."""
    turns = [
        [ToolCall(tool_id="create_freecad_loft", arguments={"profile_a": "StarPrism", "profile_b": "Cylinder"})],
        "There is no loft tool; I can approximate it with a boolean of simpler solids instead.",
    ]
    rec = _run_scenario(client, PROMPTS[4], turns)

    if "create_freecad_loft" in rec.dispatched() or _tool_result(rec, "create_freecad_loft") is not None:
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


def resolve_engine_mode(mock: bool, freecad: bool) -> str:
    """"mock" or "freecad": an explicit flag wins, else FreeCAD when
    FreeCADCmd is found."""
    if mock:
        return "mock"
    if freecad:
        return "freecad"
    from dana.plugins.freecad.engine import detect_freecadcmd

    return "freecad" if detect_freecadcmd() else "mock"


def run_gauntlet(engine_mode: str) -> list[tuple[str, bool, str]]:
    """Runs every scenario on ``engine_mode`` and returns (name, passed,
    detail) rows. Restores every module attribute it patches."""
    from dana.platform import factory

    patched: list[tuple[Any, str, Any]] = []

    def patch(target: Any, attr: str, value: Any) -> None:
        patched.append((target, attr, getattr(target, attr)))
        setattr(target, attr, value)

    scratch = Path(tempfile.mkdtemp(prefix="kobayashi_maru_"))
    patch(react_dispatch, "analyze_cad_blueprint", _fake_vlm_analysis)
    patch(react_dispatch, "CAPTURES_DIR", scratch)
    patch(react_dispatch, "_LAST_CANVAS_SCREENSHOT_PATH", scratch / "last_canvas_screenshot.png")
    patch(react_dispatch, "ModelProvider", react_dispatch.ModelProvider)  # restored after the scenarios
    if engine_mode == "mock":
        from dana.platform.mock import MockControlPlane, MockFreeCADEngine

        # The ReAct loop and plugin tools reach the engine through two
        # different module names.
        patch(server_module, "get_cad_engine", lambda: MockFreeCADEngine())
        patch(server_module, "get_control_plane", lambda: MockControlPlane())
        patch(factory, "get_cad_engine", lambda: MockFreeCADEngine())

    # Not `with TestClient(...)`: that runs the server lifespan (speech
    # models, printer watchdog, stdout broadcast), none of which this needs.
    client = TestClient(server_module.app)
    rows: list[tuple[str, bool, str]] = []
    try:
        for name, scenario in SCENARIOS:
            try:
                passed, detail = scenario(client)
            except Exception as exc:  # noqa: BLE001 — a scenario crashing IS a gauntlet failure, not a script crash
                passed, detail = False, f"scenario raised {type(exc).__name__}: {exc}"
                traceback.print_exc()
            rows.append((name, passed, detail))
    finally:
        for target, attr, value in reversed(patched):
            setattr(target, attr, value)
    return rows


def main(engine_mode: str) -> int:
    print("=" * 78)
    print(f" KOBAYASHI MARU GAUNTLET — Dana ReAct Integration Evaluation ({engine_mode} engine)")
    print("=" * 78)
    rows = run_gauntlet(engine_mode)
    for i, (name, passed, detail) in enumerate(rows, start=1):
        print(f"[{i}/{len(rows)}] {name:<28} ... {'PASS' if passed else 'FAIL'}")
        print(f"      -> {detail}")
    passed_count = sum(passed for _, passed, _ in rows)
    print("-" * 78)
    print(f"RESULT: {passed_count}/{len(rows)} scenarios survived the gauntlet.")
    print("=" * 78)
    return 0 if passed_count == len(rows) else 1


if __name__ == "__main__":
    import argparse

    import scratch_workspace  # scripts/ is sys.path[0] when run as a script

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    engine = parser.add_mutually_exclusive_group()
    engine.add_argument("--mock", action="store_true", help="run on the trimesh mock engine")
    engine.add_argument("--freecad", action="store_true", help="run on real FreeCAD (needs FreeCADCmd)")
    scratch_workspace.add_arguments(parser, session_id=False)
    args = parser.parse_args()
    mode = resolve_engine_mode(args.mock, args.freecad)
    with scratch_workspace.scratch_workspace("kobayashi-maru", keep=args.keep_artifacts):
        code = main(mode)
    raise SystemExit(code)
