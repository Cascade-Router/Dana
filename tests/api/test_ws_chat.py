"""Integration tests for the ``/ws/chat`` protocol: multi-step ReAct loop
DAG event streaming, canvas-selection context injection, camera automation,
and the HITL approval gate — all layered on dana.core.react_dispatch's
dispatch core.

The LLM is driven now, so every test mocks the one LLM call site
(``dana.core.react_dispatch.ModelProvider``) with a queue of canned
per-iteration responses — these are protocol/wiring tests, not LLM-quality
tests, and must not require a real Ollama daemon to run. ``_FakeProvider``
returns one queued response per call and falls back to a plain "Done."
final turn once the queue is exhausted, so a single queued tool call still
terminates the loop cleanly on the next iteration instead of looping until
the safety-counter cap.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from dana.api import server as server_module
from dana.platform.mock import MockControlPlane, MockFreeCADEngine
from dana.tools.schema import ToolCall


class _FakeProvider:
    def __init__(self, turns: list[list[ToolCall] | str]) -> None:
        self._turns = list(turns)

    def complete_with_tool_calls(self, messages: Any, *, tools: Any, provider: Any = None, **kwargs: Any) -> dict:
        turn = self._turns.pop(0) if self._turns else "Done."
        if isinstance(turn, str):
            return {"content": turn, "tool_calls": [], "provider": "test"}
        return {"content": "", "tool_calls": turn, "provider": "test"}


def _mock_llm(monkeypatch: pytest.MonkeyPatch, *turns: list[ToolCall] | str) -> None:
    """Queue ``turns`` as successive LLM responses for the ReAct loop.

    Each turn is either a ``list[ToolCall]`` (that iteration proposes those
    tool calls) or a plain ``str`` (that iteration's final assistant text,
    no tool calls — the loop stops there). Once the queue is exhausted,
    further calls return a plain "Done." final turn.
    """
    import dana.core.react_dispatch as react_dispatch

    fake = _FakeProvider(list(turns))
    monkeypatch.setattr(react_dispatch, "ModelProvider", lambda **_kwargs: fake)


@pytest.fixture(autouse=True)
def _plan_gate_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """This module's tests are about HITL approval, DAG event streaming, and
    canvas-selection injection — not about dana.core.react_dispatch's
    Plan-and-Execute Gatekeeper (which requires a create_plan call before
    any geometry-mutating tool dispatches; see
    tests/core/test_react_dispatch.py's own dedicated gatekeeper tests for
    that). Each test here opens a fresh WebSocket with its own
    server-generated session_id, so there's no fixed session_id to
    pre-seed _set_has_plan for ahead of time the way the unit tests do —
    patching _get_has_plan itself to always report "already planned" is the
    simplest way to keep this module's mocked tool-call sequences (which
    predate the gatekeeper) dispatching exactly as before.
    """
    import dana.core.react_dispatch as react_dispatch

    monkeypatch.setattr(react_dispatch, "_get_has_plan", lambda *_a, **_k: True)


@pytest.fixture(autouse=True)
def _mock_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server_module, "get_cad_engine", lambda: MockFreeCADEngine())
    monkeypatch.setattr(server_module, "get_control_plane", lambda: MockControlPlane())
    # _execute_and_continue's Automatic Visual Verification step reaches the
    # real OS (a screen capture) and a real Ollama daemon (VLM analysis) on
    # every successful CAD-mutating tool call, gated only by DANA_OS_DRY_RUN
    # — without this, every test here that completes a create/modify/boolean
    # tool would silently screenshot this machine's actual screen and make a
    # real HTTP call, which is exactly what this file's own docstring says
    # these protocol/wiring tests must never require.
    monkeypatch.setenv("DANA_OS_DRY_RUN", "1")


# Captured once, before any test clears the live attribute below — the real
# production set, for the dedicated whitelist tests to restore explicitly.
_REAL_HITL_ALWAYS_APPROVED_TOOLS = frozenset(server_module._HITL_ALWAYS_APPROVED_TOOLS)


@pytest.fixture(autouse=True)
def _disable_permanent_hitl_whitelist(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every existing test in this file uses create_freecad_box as its
    representative "a mutating tool" fixture — written before
    dana.api.server._HITL_ALWAYS_APPROVED_TOOLS permanently exempted
    FreeCAD's geometry-CRUD tools (create_freecad_box included) from HITL
    approval. Cleared here so those tests keep exercising generic HITL
    protocol mechanics (approve/reject/modify/timeout/DAG lifecycle)
    unaffected by that later, unrelated feature. The tests that actually
    verify the whitelist restore _REAL_HITL_ALWAYS_APPROVED_TOOLS
    explicitly instead of relying on this fixture's default.
    """
    monkeypatch.setattr(server_module, "_HITL_ALWAYS_APPROVED_TOOLS", frozenset())


@pytest.fixture
def client() -> TestClient:
    return TestClient(server_module.app)


def _activate_freecad(ws: Any) -> None:
    """Simulates the frontend having the CAD plugin tab active. Capability
    routing (dana.core.react_dispatch's _tool_ids_for_plugins) gates
    FreeCAD-specific tools (create_freecad_box, manipulate_camera, ...)
    behind session["active_plugins"] containing "freecad" — a fresh test
    session defaults to an EMPTY active-plugin set, same as a real frontend
    connection before the user opens the CAD tab (see dana.api.server's
    ws_chat), so any test that dispatches one of those tools must call this
    first or next_react_turn's routing guard will silently downgrade the
    turn to "final" instead of a tool_call.
    """
    ws.send_json({"type": "update_context", "active_plugins": ["freecad"]})


def _drain_until(ws: Any, msg_type: str, limit: int = 20) -> dict[str, Any]:
    for _ in range(limit):
        msg = ws.receive_json()
        if msg.get("type") == msg_type:
            return msg
    raise AssertionError(f"never received a {msg_type!r} message")


def test_ready_message_on_connect(client: TestClient) -> None:
    with client.websocket_connect("/ws/chat") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "ready"
        assert "driver_state" in msg


def test_safe_tool_streams_dag_events_then_loops_to_final_text(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-mutating tool executes immediately, then the loop asks the LLM
    again with the tool's result appended — this second iteration is what
    actually distinguishes the multi-step loop from the old single-shot
    dispatch, so the test supplies a distinct final-text second turn rather
    than relying on the fallback "Done." to prove the loop-back happened."""
    _mock_llm(
        monkeypatch,
        [ToolCall(tool_id="system_state", arguments={})],
        "The system is healthy and ready.",
    )
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"text": "system status"})

        parse_start = _drain_until(ws, "dag_node_start")
        assert parse_start["node_id"] == "parse-0"
        parse_complete = _drain_until(ws, "dag_node_complete")
        assert parse_complete["node_id"] == "parse-0" and parse_complete["status"] == "success"

        dispatch_start = _drain_until(ws, "tool_dispatch_start")
        assert dispatch_start["node_id"] == "dispatch-0"
        assert dispatch_start["tool_name"] == "system_state"

        dispatch_end = _drain_until(ws, "tool_dispatch_end")
        assert dispatch_end["node_id"] == "dispatch-0"
        assert dispatch_end["status"] == "success"

        # Second loop iteration: the LLM sees the tool result and produces
        # its own final text — no synthesized "control_plane=..." summary.
        second_parse_start = _drain_until(ws, "dag_node_start")
        assert second_parse_start["node_id"] == "parse-1"

        assistant = _drain_until(ws, "assistant_message")
        assert assistant["content"] == "The system is healthy and ready."


def test_no_tool_call_yields_plain_fallback_message(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_llm(monkeypatch, [])
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"text": "thanks!"})

        parse_complete = _drain_until(ws, "dag_node_complete")
        # A tool-less final turn is a normal (successful) loop termination
        # now, not a parse failure — the old single-shot design had no
        # other reason to return final text besides "couldn't parse".
        assert parse_complete["status"] == "success"

        assistant = _drain_until(ws, "assistant_message")
        assert "tool call" in assistant["content"] or "action" in assistant["content"]


def test_llm_proxy_error_replies_gracefully_without_leaking_the_raw_failure(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A cloud provider 502/400 (dana.core.openai_tool_bridge already turns
    this into a RuntimeError, never a crash) must still end the turn with a
    generic apology — the raw failure text (provider internals: status
    codes, endpoint URLs, sometimes response bodies) has no business
    reaching the chat bubble, so it's logged server-side (stderr) instead,
    for whoever's actually debugging the outage."""
    import dana.core.react_dispatch as react_dispatch

    class _FailingProvider:
        def complete_with_tool_calls(self, *_a: Any, **_k: Any) -> dict:
            raise RuntimeError("cloud HTTP 502: Bad Gateway -- <html>upstream unavailable</html>")

    monkeypatch.setattr(react_dispatch, "ModelProvider", lambda **_kwargs: _FailingProvider())

    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"text": "hello"})

        assistant = _drain_until(ws, "assistant_message")
        assert "cloud HTTP 502" not in assistant["content"]
        assert "Bad Gateway" not in assistant["content"]
        assert "problem talking to the model" in assistant["content"]

    captured = capsys.readouterr()
    assert "cloud HTTP 502" in captured.err
    assert "Bad Gateway" in captured.err


def test_mutating_tool_requires_hitl_approval_then_proceeds(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_llm(monkeypatch, [ToolCall(tool_id="create_freecad_box", arguments={"length": 60, "width": 40, "height": 20})])
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        _activate_freecad(ws)
        ws.send_json({"text": "build a box 60x40x20"})

        approval = _drain_until(ws, "hitl_approval_required")
        request_id = approval["payload"]["request_id"]
        assert approval["payload"]["action_name"] == "create_freecad_box"
        assert "60" in approval["payload"]["description"]

        ws.send_json({"type": "hitl_response", "payload": {"request_id": request_id, "approved": True}})

        dispatch_start = _drain_until(ws, "tool_dispatch_start")
        assert dispatch_start["tool_name"] == "create_freecad_box"

        dispatch_end = _drain_until(ws, "tool_dispatch_end")
        assert dispatch_end["status"] == "success"
        assert dispatch_end["mesh_url"] is not None

        # Loop continues after approval+execution — falls back to "Done."
        # since only one turn was queued, and terminates cleanly.
        assistant = _drain_until(ws, "assistant_message")
        assert assistant["content"] == "Done."


def test_hitl_always_approved_tools_skip_approval_entirely(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real _HITL_ALWAYS_APPROVED_TOOLS set (restored here — see
    _disable_permanent_hitl_whitelist's own docstring for why it's cleared
    by default in this file) must let create_freecad_box dispatch with NO
    hitl_approval_required at all, on the very first call this session —
    distinct from the session allowlist feature, which only skips a SECOND
    call after an explicit first approval."""
    monkeypatch.setattr(server_module, "_HITL_ALWAYS_APPROVED_TOOLS", _REAL_HITL_ALWAYS_APPROVED_TOOLS)
    _mock_llm(monkeypatch, [ToolCall(tool_id="create_freecad_box", arguments={"length": 10, "width": 10, "height": 10})])
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        _activate_freecad(ws)
        ws.send_json({"text": "build a small box"})

        seen_types = []
        for _ in range(6):
            msg = ws.receive_json()
            seen_types.append(msg["type"])
            if msg["type"] == "tool_dispatch_start":
                assert msg["tool_name"] == "create_freecad_box"
                break
        assert "hitl_approval_required" not in seen_types, f"expected no approval prompt, got: {seen_types}"

        dispatch_end = _drain_until(ws, "tool_dispatch_end")
        assert dispatch_end["status"] == "success"


def test_cad_mutation_auto_injects_screenshot_and_visual_verification(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Automatic Visual Verification: a successful CAD-mutating tool call
    must come back with screenshot_path/visual_verification merged into its
    OWN result payload, with no separate tool call — dana.tools.cad_vision's
    real capture/VLM functions are mocked here (never the real OS/network,
    matching this whole file's design) and DANA_OS_DRY_RUN is explicitly
    overridden to false for just this test so _execute_and_continue's own
    gate doesn't skip the code path being tested.

    Mocks ``verify_visual_operation`` (a plain string return, never a JSON
    envelope — see its own docstring), not ``analyze_cad_blueprint``: the
    "outsource visual verification to a cloud VLM" refactor moved
    _execute_and_continue's automatic per-tool-call hook onto the former,
    cloud-only path — analyze_cad_blueprint is local-Ollama-first
    and is only ever called by the separate, on-demand verify_cad_rendering
    tool now, not this automatic hook.
    """
    monkeypatch.setattr(server_module, "_HITL_ALWAYS_APPROVED_TOOLS", _REAL_HITL_ALWAYS_APPROVED_TOOLS)
    monkeypatch.setenv("DANA_OS_DRY_RUN", "0")
    # DANA_HEADLESS (this machine's own .env sets it true) takes the hook's
    # headless branch instead, which never sets screenshot_path at all and
    # never calls capture_cad_viewport/verify_visual_operation below — this
    # test is specifically about the non-headless capture+VLM path, so it
    # must override that ambient setting the same way it already does for
    # DANA_OS_DRY_RUN.
    monkeypatch.setenv("DANA_HEADLESS", "false")
    monkeypatch.setattr(
        "dana.tools.cad_vision.capture_cad_viewport",
        lambda: {"ok": True, "path": "/fake/last_cad_viewport.png", "window_found": True},
    )
    monkeypatch.setattr(
        "dana.tools.cad_vision.verify_visual_operation",
        lambda *_a, **_k: "a single rectangular box, no visible defects",
    )
    _mock_llm(monkeypatch, [ToolCall(tool_id="create_freecad_box", arguments={"length": 10, "width": 10, "height": 10})])
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        _activate_freecad(ws)
        ws.send_json({"text": "build a small box"})

        dispatch_end = _drain_until(ws, "tool_dispatch_end")
        assert dispatch_end["status"] == "success"
        assert dispatch_end["output"]["screenshot_path"] == "/fake/last_cad_viewport.png"
        assert dispatch_end["output"]["visual_verification"] == "a single rectangular box, no visible defects"


def test_cad_mutation_visual_verification_failure_does_not_fail_the_tool(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A capture/VLM failure is a convenience miss, not a tool failure — the
    underlying create_freecad_box result must still report ok: true."""
    monkeypatch.setattr(server_module, "_HITL_ALWAYS_APPROVED_TOOLS", _REAL_HITL_ALWAYS_APPROVED_TOOLS)
    monkeypatch.setenv("DANA_OS_DRY_RUN", "0")
    monkeypatch.setattr(
        "dana.tools.cad_vision.capture_cad_viewport",
        lambda: (_ for _ in ()).throw(RuntimeError("no display attached")),
    )
    _mock_llm(monkeypatch, [ToolCall(tool_id="create_freecad_box", arguments={"length": 10, "width": 10, "height": 10})])
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        _activate_freecad(ws)
        ws.send_json({"text": "build a small box"})

        dispatch_end = _drain_until(ws, "tool_dispatch_end")
        assert dispatch_end["status"] == "success"
        assert "screenshot_path" not in dispatch_end["output"]


def test_hitl_whitelist_never_covers_script_execution_tools(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Security-critical regression guard: execute_freecad_script and
    modify_existing_freecad_document run an ARBITRARY caller-supplied
    script — dana.api.server._HITL_ALWAYS_APPROVED_TOOLS deliberately
    never exempts either, no matter how that set is edited in the future.
    Rejects rather than approves once the prompt is confirmed, so this
    stays hermetic (no real FreeCADCmd/file I/O needed)."""
    monkeypatch.setattr(server_module, "_HITL_ALWAYS_APPROVED_TOOLS", _REAL_HITL_ALWAYS_APPROVED_TOOLS)
    for tool_id, arguments in (
        ("execute_freecad_script", {"python_script_str": "print('hi')"}),
        ("modify_existing_freecad_document", {"filepath": "C:/tmp/whatever.FCStd", "modification_script": "pass"}),
    ):
        _mock_llm(monkeypatch, [ToolCall(tool_id=tool_id, arguments=arguments)])
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # ready
            _activate_freecad(ws)
            ws.send_json({"text": "run it"})

            approval = _drain_until(ws, "hitl_approval_required")
            assert approval["payload"]["action_name"] == tool_id
            ws.send_json(
                {"type": "hitl_response", "payload": {"request_id": approval["payload"]["request_id"], "approved": False}}
            )
            assistant = _drain_until(ws, "assistant_message")
            assert "Cancelled" in assistant["content"]


def test_hitl_session_allowlist_skips_approval_on_repeat_tool(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Approving create_freecad_box once must let a LATER, SEPARATE turn's
    call to that same tool_id skip hitl_approval_required entirely — the
    session["hitl_approved_tools"] allowlist populated by _resolve_react_hitl
    on approval, checked in _run_react_loop's is_mutating_tool gate.

    Note the LLM queue has THREE entries, not two: after the first box is
    approved and dispatched, the ReAct loop calls the LLM again WITHIN THE
    SAME turn to decide the next step (_execute_and_continue recurses back
    into _run_react_loop) — so a plain "Done." has to be queued there to
    end turn 1, or the second ToolCall would be consumed by that same-turn
    continuation instead of by the second user message below.
    """
    _mock_llm(
        monkeypatch,
        [ToolCall(tool_id="create_freecad_box", arguments={"length": 60, "width": 40, "height": 20})],
        "Done.",
        [ToolCall(tool_id="create_freecad_box", arguments={"length": 10, "width": 10, "height": 10})],
    )
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        _activate_freecad(ws)
        ws.send_json({"text": "build a box 60x40x20"})

        approval = _drain_until(ws, "hitl_approval_required")
        request_id = approval["payload"]["request_id"]
        ws.send_json({"type": "hitl_response", "payload": {"request_id": request_id, "approved": True}})

        _drain_until(ws, "tool_dispatch_start")
        _drain_until(ws, "tool_dispatch_end")
        assistant = _drain_until(ws, "assistant_message")
        assert assistant["content"] == "Done."

        # Second, separate turn — same tool_id. No hitl_approval_required
        # this time: it must go straight to tool_dispatch_start/tool_dispatch_end.
        ws.send_json({"text": "now build a small 10x10x10 box"})

        # limit=20 (not a small hardcoded count): incidental events between
        # here and the tool_dispatch_start (e.g. "usage_update", Cost Tracking's
        # per-iteration broadcast) must never make this loop fall through
        # before it ever sees a tool_dispatch_start.
        seen_types = []
        for _ in range(20):
            msg = ws.receive_json()
            seen_types.append(msg["type"])
            if msg["type"] == "tool_dispatch_start":
                assert msg["tool_name"] == "create_freecad_box"
                break
        assert "hitl_approval_required" not in seen_types, (
            f"expected no second approval prompt, got message sequence: {seen_types}"
        )
        assert "tool_dispatch_start" in seen_types, f"never saw a tool_dispatch_start, got: {seen_types}"

        dispatch_end = _drain_until(ws, "tool_dispatch_end")
        assert dispatch_end["status"] == "success"


def test_mutating_tool_cancelled_when_not_approved(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_llm(monkeypatch, [ToolCall(tool_id="create_freecad_box", arguments={"length": 60, "width": 40, "height": 20})])
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        _activate_freecad(ws)
        ws.send_json({"text": "build a box 60x40x20"})

        approval = _drain_until(ws, "hitl_approval_required")
        request_id = approval["payload"]["request_id"]

        ws.send_json({"type": "hitl_response", "payload": {"request_id": request_id, "approved": False}})

        assistant = _drain_until(ws, "assistant_message")
        assert "Cancelled" in assistant["content"]


def test_hitl_modify_overrides_parameters_before_dispatch(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_llm(monkeypatch, [ToolCall(tool_id="create_freecad_box", arguments={"length": 60, "width": 40, "height": 20})])
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        _activate_freecad(ws)
        ws.send_json({"text": "build a box 60x40x20"})

        approval = _drain_until(ws, "hitl_approval_required")
        request_id = approval["payload"]["request_id"]

        ws.send_json(
            {
                "type": "hitl_response",
                "payload": {"request_id": request_id, "approved": True, "parameters": {"length": "99"}},
            }
        )

        dispatch_start = _drain_until(ws, "tool_dispatch_start")
        assert dispatch_start["arguments"]["length"] == "99"


def test_canvas_selection_feeds_camera_target(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        _activate_freecad(ws)
        ws.send_json(
            {
                "type": "canvas_selection",
                "payload": {"mesh_id": "current_mesh", "centroid": [5.0, 6.0, 7.0], "normal": [0, 1, 0]},
            }
        )
        _mock_llm(monkeypatch, [ToolCall(tool_id="manipulate_camera", arguments={"preset": "top"})])
        ws.send_json({"text": "look at it from the top"})

        dispatch_start = _drain_until(ws, "tool_dispatch_start")
        assert dispatch_start["tool_name"] == "manipulate_camera"
        assert dispatch_start["arguments"]["target"] == [5.0, 6.0, 7.0]

        camera_animate = _drain_until(ws, "camera_animate")
        assert camera_animate["target"] == [5.0, 6.0, 7.0]


def test_canvas_selection_injects_target_position_on_mutating_tool(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        _activate_freecad(ws)
        ws.send_json(
            {
                "type": "canvas_selection",
                "payload": {"mesh_id": "current_mesh", "centroid": [1.0, 2.0, 3.0], "normal": [0, 0, 1]},
            }
        )
        # The LLM proposes the box with no anchor of its own — the
        # deterministic fallback in react_dispatch._finalize_call_arguments
        # must inject it since the user said "here".
        _mock_llm(monkeypatch, [ToolCall(tool_id="create_freecad_box", arguments={})])
        ws.send_json({"text": "add a box here"})

        approval = _drain_until(ws, "hitl_approval_required")
        assert approval["payload"]["parameters"]["target_position"] == [1.0, 2.0, 3.0]

        ws.send_json(
            {
                "type": "hitl_response",
                "payload": {"request_id": approval["payload"]["request_id"], "approved": True},
            }
        )
        dispatch_end = _drain_until(ws, "tool_dispatch_end")
        assert dispatch_end["status"] == "success"


# --------------------------------------------------------------------------
# Multi-step ReAct loop: the actual new behavior this directive adds.
# --------------------------------------------------------------------------


def test_multi_step_loop_chains_two_tool_calls_before_final_text(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The core new capability: after a non-mutating tool executes, the
    loop asks the LLM again — and the LLM can decide to call ANOTHER tool
    (not just stop), all within one user turn."""
    _mock_llm(
        monkeypatch,
        [ToolCall(tool_id="system_state", arguments={})],
        [ToolCall(tool_id="check_plugin_registry", arguments={})],
        "Checked both system state and the plugin registry.",
    )
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"text": "check everything"})

        first_call = _drain_until(ws, "tool_dispatch_start")
        assert first_call["tool_name"] == "system_state"
        _drain_until(ws, "tool_dispatch_end")

        second_parse_start = _drain_until(ws, "dag_node_start")
        assert second_parse_start["node_id"] == "parse-1"

        second_call = _drain_until(ws, "tool_dispatch_start")
        assert second_call["tool_name"] == "check_plugin_registry"
        _drain_until(ws, "tool_dispatch_end")

        assistant = _drain_until(ws, "assistant_message")
        assert assistant["content"] == "Checked both system state and the plugin registry."


def test_multi_step_loop_suspends_for_hitl_mid_chain_and_resumes(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-mutating query (get_freecad_bounding_box-style) followed by a
    mutating create call — the loop must pause for approval on the SECOND
    tool, not just the first, and resume with the right messages/loop_count."""
    _mock_llm(
        monkeypatch,
        [ToolCall(tool_id="system_state", arguments={})],
        [ToolCall(tool_id="create_freecad_box", arguments={"length": 10, "width": 10, "height": 10})],
        "Built the box after checking system state.",
    )
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        _activate_freecad(ws)
        ws.send_json({"text": "check status then build a box"})

        _drain_until(ws, "tool_dispatch_end")  # system_state, no HITL

        approval = _drain_until(ws, "hitl_approval_required")
        assert approval["payload"]["action_name"] == "create_freecad_box"

        ws.send_json(
            {
                "type": "hitl_response",
                "payload": {"request_id": approval["payload"]["request_id"], "approved": True},
            }
        )

        dispatch_start = _drain_until(ws, "tool_dispatch_start")
        assert dispatch_start["tool_name"] == "create_freecad_box"
        dispatch_end = _drain_until(ws, "tool_dispatch_end")
        assert dispatch_end["status"] == "success"

        assistant = _drain_until(ws, "assistant_message")
        assert assistant["content"] == "Built the box after checking system state."


def test_react_loop_stops_on_repeated_identical_call(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """A model stuck proposing the exact same non-mutating tool call (same
    tool_id, same arguments) must be stopped after the third repeat, not
    left to cycle all the way to _MAX_REACT_ITERATIONS — the live bug this
    guards against: local-Ollama fallback calling
    load_capability({"domain": "freecad_full"}) over 20 times in a row,
    never itself a "failure" (it succeeds every time), so the older
    same-failure-twice guard never caught it. system_state stands in here
    as any cheap, always-succeeding, argument-less non-mutating tool."""
    endless_tool_call = [ToolCall(tool_id="system_state", arguments={})]
    _mock_llm(monkeypatch, *([endless_tool_call] * 10))  # far more than the 3-repeat guard allows
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"text": "loop forever"})

        assistant = _drain_until(ws, "assistant_message", limit=100)
        assert "same arguments" in assistant["content"]
        assert "3 times in a row" in assistant["content"]


def test_react_loop_stops_at_max_iterations(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """The raw _MAX_REACT_ITERATIONS cap must still fire and report a clear
    fallback message when a model keeps proposing tool calls indefinitely
    but VARIES them enough (alternating tool_id here) that the repeated-
    identical-call guard (see test_react_loop_stops_on_repeated_identical_call
    above) never trips first."""
    alternating_calls = [
        [ToolCall(tool_id="system_state", arguments={})],
        [ToolCall(tool_id="check_plugin_registry", arguments={})],
    ]
    # dana.api.server._MAX_REACT_ITERATIONS is 30 — queue more canned turns
    # than that so the loop actually reaches the cap instead of running out
    # of mock responses first (which would fall back to a "Done." final turn).
    _mock_llm(monkeypatch, *(alternating_calls * 16))  # 32 turns, far more than the cap allows
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"text": "loop forever"})

        # 30 full iterations (dag_start/dag_complete/tool_dispatch_start/
        # tool_dispatch_end per iteration, plus any tee'd server_log lines)
        # produce well more than 200 websocket messages before the cap's
        # own assistant_message —
        # generous headroom here rather than hand-computing an exact count.
        assistant = _drain_until(ws, "assistant_message", limit=500)
        assert "maximum number of reasoning steps" in assistant["content"]


def test_second_message_while_hitl_pending_is_bounced(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_llm(monkeypatch, [ToolCall(tool_id="create_freecad_box", arguments={"length": 10, "width": 10, "height": 10})])
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        _activate_freecad(ws)
        ws.send_json({"text": "build a box"})
        _drain_until(ws, "hitl_approval_required")

        ws.send_json({"text": "build another box"})
        assistant = _drain_until(ws, "assistant_message")
        assert "pending action" in assistant["content"]


# --------------------------------------------------------------------------
# BYOK — "update_secrets" populates session["api_keys"], and next_react_turn
# threads it into ModelProvider(...) unchanged. See dana.core.model_provider
# (session key preferred over env) and dana.core.react_dispatch for the
# rest of the chain — this is the one end-to-end wire-protocol test.
# --------------------------------------------------------------------------


def test_update_secrets_reaches_model_provider_constructor(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dana.core.react_dispatch as react_dispatch

    captured: dict[str, Any] = {}

    class _CapturingProvider:
        def __init__(self, **kwargs: Any) -> None:
            captured["constructor_kwargs"] = kwargs

        def complete_with_tool_calls(self, messages: Any, *, tools: Any, provider: Any = None, **kwargs: Any) -> dict:
            return {"content": "done", "tool_calls": [], "provider": "test"}

    monkeypatch.setattr(react_dispatch, "ModelProvider", _CapturingProvider)

    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json(
            {"type": "update_secrets", "keys": {"openai": "sk-test-123", "anthropic": "sk-ant-456"}}
        )
        ws.send_json({"text": "hello"})
        _drain_until(ws, "assistant_message")

    assert captured["constructor_kwargs"] == {
        "api_keys": {"openai": "sk-test-123", "anthropic": "sk-ant-456"}
    }


def test_update_secrets_with_non_dict_keys_is_ignored_not_fatal(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed payload must not corrupt session state or kill the
    connection — the ReAct loop must keep working normally afterward."""
    _mock_llm(monkeypatch, "Done.")
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"type": "update_secrets", "keys": "not-a-dict"})
        ws.send_json({"text": "hello"})
        assistant = _drain_until(ws, "assistant_message")
        assert assistant["content"] == "Done."


# --------------------------------------------------------------------------
# Capability routing — "update_context" populates session["active_plugins"]
# (normalized, e.g. the frontend's "cad" -> "freecad"), and it actually
# narrows the tools= schema the real (mocked-transport-only) ModelProvider
# sees for that session's ReAct turns. See dana.core.react_dispatch's
# _tool_ids_for_plugins/build_system_prompt for the rest of the chain.
# --------------------------------------------------------------------------


def _capture_tool_names(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    import dana.core.react_dispatch as react_dispatch

    captured: dict[str, Any] = {}

    class _CapturingProvider:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def complete_with_tool_calls(self, messages: Any, *, tools: Any, provider: Any = None, **kwargs: Any) -> dict:
            captured["tool_names"] = {t["function"]["name"] for t in tools}
            return {"content": "done", "tool_calls": [], "provider": "test"}

    monkeypatch.setattr(react_dispatch, "ModelProvider", _CapturingProvider)
    return captured


def test_update_context_normalizes_cad_to_freecad_essential_tool_set(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """"cad" (the frontend's plugin id) -> "freecad_essential" (NOT the raw
    "freecad" domain's full ~24-tool set) — dana.api.server._PLUGIN_ID_TO_CAPABILITY
    routes the CAD tab's activation to the same trimmed 7-tool default the
    agent's own autonomous load_capability("freecad") call already used, so
    a real CAD-panel session doesn't blow a free-tier Groq model's 8000 TPM
    ceiling on every turn (previously the one unmitigated path to the full
    schema — see _FREECAD_ESSENTIAL_TOOL_IDS's docstring). The 2 manifest-only
    extension tools (modify_existing_freecad_document/execute_freecad_script)
    only ever unioned into the raw "freecad" domain, never into
    "freecad_essential", so they're correctly absent here too — heavier,
    load-on-demand tools, exactly what trimming the default is for. The
    agent can still reach the full set (these two included) via
    load_capability(domain="freecad") or "freecad_full"."""
    import dana.core.react_dispatch as react_dispatch

    captured = _capture_tool_names(monkeypatch)
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"type": "update_context", "active_plugins": ["cad"]})
        ws.send_json({"text": "hello"})
        _drain_until(ws, "assistant_message")

    assert captured["tool_names"] == react_dispatch._CORE_TOOL_IDS | react_dispatch._FREECAD_ESSENTIAL_TOOL_IDS


def test_no_active_plugins_yields_core_tools_only(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """A fresh connection that never sends update_context (or one that sends
    an empty active_plugins list) must only ever offer the LLM the 3 core
    tools — not the full legacy set, and not a crash."""
    import dana.core.react_dispatch as react_dispatch

    captured = _capture_tool_names(monkeypatch)
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"text": "hello"})
        _drain_until(ws, "assistant_message")

    assert captured["tool_names"] == react_dispatch._CORE_TOOL_IDS


def test_update_context_with_non_list_active_plugins_is_ignored_not_fatal(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _capture_tool_names(monkeypatch)
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"type": "update_context", "active_plugins": "cad"})  # malformed: not a list
        ws.send_json({"text": "hello"})
        assistant = _drain_until(ws, "assistant_message")
        assert assistant["content"] == "done"

    import dana.core.react_dispatch as react_dispatch

    assert captured["tool_names"] == react_dispatch._CORE_TOOL_IDS


# --------------------------------------------------------------------------
# Agent Activity feed — "tool_dispatch_start"/"tool_dispatch_end". Plugin-
# agnostic transparency events the ChatPanel renders inline (unlike
# "dag_node_start"/"dag_node_complete" for a "parse-N" node, which only the
# CadPlugin's DAG Monitor ever renders) — so a user doing OS/web work with
# no plugin tab open can still see what the agent is doing turn-by-turn,
# not just a silent wait until the final assistant_message.
# --------------------------------------------------------------------------


def test_tool_dispatch_start_and_end_emitted_for_non_mutating_tool(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_llm(monkeypatch, [ToolCall(tool_id="system_state", arguments={})], "The system is healthy.")
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"text": "system status"})

        dispatch_start = _drain_until(ws, "tool_dispatch_start")
        assert dispatch_start["tool_name"] == "system_state"
        assert dispatch_start["args_summary"] == ""

        dispatch_end = _drain_until(ws, "tool_dispatch_end")
        assert dispatch_end["tool_id"] == "system_state"
        assert dispatch_end["status"] == "success"

        # Both fire strictly before the assistant_message that closes out
        # the turn — a live consumer sees them WHILE the turn is still in
        # flight, not only after the fact.
        _drain_until(ws, "assistant_message")


def test_tool_dispatch_start_args_summary_surfaces_the_identifying_argument(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_llm(monkeypatch, [ToolCall(tool_id="search_web", arguments={"query": "gearbox torque rating", "max_results": 3})])
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"type": "update_context", "active_plugins": ["web_tools"]})
        ws.send_json({"text": "look up gearbox torque ratings"})

        dispatch_start = _drain_until(ws, "tool_dispatch_start")
        assert dispatch_start["tool_name"] == "search_web"
        assert dispatch_start["args_summary"] == "gearbox torque rating"


def test_tool_dispatch_start_args_summary_is_truncated_for_long_values(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    long_content = "x" * 500
    _mock_llm(monkeypatch, [ToolCall(tool_id="write_file", arguments={"path": "notes.txt", "content": long_content})])
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"type": "update_context", "active_plugins": ["os_tools"]})
        ws.send_json({"text": "save these notes"})

        approval = _drain_until(ws, "hitl_approval_required")
        ws.send_json(
            {"type": "hitl_response", "payload": {"request_id": approval["payload"]["request_id"], "approved": True}}
        )

        dispatch_start = _drain_until(ws, "tool_dispatch_start")
        # write_file's args_summary key is "path" (short + identifying) —
        # never the (here, 500-char) "content" value.
        assert dispatch_start["args_summary"] == "notes.txt"
        assert len(dispatch_start["args_summary"]) <= 80


def test_tool_dispatch_start_is_deferred_until_after_hitl_approval(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mutating tool must not be reported as "started" while it's still
    only a pending HITL proposal — tool_dispatch_start belongs to actual
    dispatch, which for a mutating tool only happens post-approval."""
    _mock_llm(monkeypatch, [ToolCall(tool_id="create_freecad_box", arguments={"length": 60, "width": 40, "height": 20})])
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        _activate_freecad(ws)
        ws.send_json({"text": "build a box 60x40x20"})

        approval = _drain_until(ws, "hitl_approval_required")

        ws.send_json(
            {"type": "hitl_response", "payload": {"request_id": approval["payload"]["request_id"], "approved": True}}
        )

        dispatch_start = _drain_until(ws, "tool_dispatch_start")
        assert dispatch_start["tool_name"] == "create_freecad_box"

        dispatch_end = _drain_until(ws, "tool_dispatch_end")
        assert dispatch_end["tool_id"] == "create_freecad_box"
        assert dispatch_end["status"] == "success"


def test_tool_dispatch_end_status_is_error_when_the_tool_fails(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_llm(monkeypatch, [ToolCall(tool_id="read_file", arguments={"path": "does_not_exist.txt"})], "No such file.")
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"type": "update_context", "active_plugins": ["os_tools"]})
        ws.send_json({"text": "read does_not_exist.txt"})

        _drain_until(ws, "tool_dispatch_start")
        dispatch_end = _drain_until(ws, "tool_dispatch_end")
        assert dispatch_end["tool_id"] == "read_file"
        assert dispatch_end["status"] == "error"


def test_tool_failure_injects_system_override_directive_for_next_turn(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_execute_and_continue must inject a blunt SYSTEM OVERRIDE reinforcement
    into the very next LLM call after an actual tool failure — on top of the
    standing system-prompt rule, this is what the next next_react_turn call
    literally sees, guarding against the model hallucinating success instead
    of noticing `ok: false` and retrying.

    A fresh ``_RecordingProvider()`` is constructed for every turn (the
    ``ModelProvider`` patch below), so its ``_turns`` script always resets
    and the SAME read_file failure repeats every call — this doubles as the
    identical-failure circuit breaker's own regression test: occurrence #2
    must get the stronger "ABORT this exact approach" nudge (not the plain
    fix-and-retry one), and occurrence #3 must hard-stop the turn instead of
    calling the LLM a 4th time.
    """
    import dana.core.react_dispatch as react_dispatch

    captured_messages: list[list[dict[str, Any]]] = []

    class _RecordingProvider:
        def __init__(self) -> None:
            self._turns: list[Any] = [
                [ToolCall(tool_id="read_file", arguments={"path": "does_not_exist.txt"})],
                "Understood, stopping here.",
            ]

        def complete_with_tool_calls(self, messages: Any, *, tools: Any, provider: Any = None, **kwargs: Any) -> dict:
            captured_messages.append(messages)
            turn = self._turns.pop(0) if self._turns else "Done."
            if isinstance(turn, str):
                return {"content": turn, "tool_calls": [], "provider": "test"}
            return {"content": "", "tool_calls": turn, "provider": "test"}

    monkeypatch.setattr(react_dispatch, "ModelProvider", lambda **_kwargs: _RecordingProvider())

    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"type": "update_context", "active_plugins": ["os_tools"]})
        ws.send_json({"text": "read does_not_exist.txt"})

        _drain_until(ws, "assistant_message")

    assert len(captured_messages) == 3
    second_call_messages = captured_messages[1]
    assert any(
        m.get("role") == "system" and "SYSTEM OVERRIDE" in m.get("content", "")
        for m in second_call_messages
    )
    third_call_messages = captured_messages[2]
    assert any(
        m.get("role") == "system" and "ABORT this exact approach" in m.get("content", "")
        for m in third_call_messages
    )


# --------------------------------------------------------------------------
# Global Abort — "abort_turn". This server processes one websocket frame
# at a time on a single task per connection (ws_chat's own `while True:
# await websocket.receive_json()` loop) — a client CANNOT interleave a new
# message into the MIDDLE of an uninterrupted chain of non-mutating tool
# calls, since the server isn't back at receive_json() until that whole
# chain finishes. The one point where control genuinely returns to the
# receive loop mid-turn is a HITL/visual-capture suspension — so that's
# the realistic, concurrency-correct scenario these tests exercise for
# "abort mid-loop, prevents the next tool call from ever dispatching".
# --------------------------------------------------------------------------


def test_abort_turn_during_pending_hitl_cancels_and_prevents_dispatch(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 2-step chain: system_state runs immediately (non-mutating), then
    the LLM proposes create_freecad_box, which suspends for HITL approval —
    the loop hands control back to ws_chat's receive loop right there.
    Sending "abort_turn" instead of a hitl_response there must cancel the
    pending call outright: it must never dispatch (no tool_dispatch_start
    for it anywhere before the abort's own assistant_message), and the
    loop must not continue to a third iteration's final text.
    """
    _mock_llm(
        monkeypatch,
        [ToolCall(tool_id="system_state", arguments={})],
        [ToolCall(tool_id="create_freecad_box", arguments={"length": 10, "width": 10, "height": 10})],
        "This should never be reached.",
    )
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        _activate_freecad(ws)
        ws.send_json({"text": "check status then build a box"})

        _drain_until(ws, "tool_dispatch_end")  # system_state, non-mutating, runs immediately

        approval = _drain_until(ws, "hitl_approval_required")
        assert approval["payload"]["action_name"] == "create_freecad_box"

        ws.send_json({"type": "abort_turn"})

        seen_before_final: list[dict[str, Any]] = []
        assistant = None
        for _ in range(20):
            msg = ws.receive_json()
            if msg["type"] == "assistant_message":
                assistant = msg
                break
            seen_before_final.append(msg)
        assert assistant is not None, "assistant_message never arrived after abort_turn"
        assert assistant["content"] == "Generation aborted by user."
        assert not any(m.get("tool_id") == "create_freecad_box" for m in seen_before_final)
        assert not any(m["type"] == "tool_dispatch_start" and m.get("tool_name") == "create_freecad_box" for m in seen_before_final)


def test_hitl_response_after_aborted_turn_is_ignored_as_stale(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once a pending call has been aborted, react_state is cleared — a
    hitl_response that still names its (now-stale) request_id must be
    silently ignored, never re-dispatching the aborted call, and the
    connection must stay healthy for the NEXT, unrelated turn."""
    _mock_llm(monkeypatch, [ToolCall(tool_id="create_freecad_box", arguments={"length": 10, "width": 10, "height": 10})])
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        _activate_freecad(ws)
        ws.send_json({"text": "build a box"})

        approval = _drain_until(ws, "hitl_approval_required")
        request_id = approval["payload"]["request_id"]

        ws.send_json({"type": "abort_turn"})
        _drain_until(ws, "assistant_message")

        ws.send_json({"type": "hitl_response", "payload": {"request_id": request_id, "approved": True}})

        _mock_llm(monkeypatch, "All good.")
        ws.send_json({"text": "hi again"})
        assistant = _drain_until(ws, "assistant_message")
        assert assistant["content"] == "All good."


def _capture_provider_messages(monkeypatch: pytest.MonkeyPatch, *, tool_calls: list[ToolCall] | None = None) -> dict[str, Any]:
    """Mocks ModelProvider so the exact ``messages`` array next_react_turn
    hands to the LLM is inspectable — mirrors test_chat_attachments.py's
    helper of the same shape."""
    import dana.core.react_dispatch as react_dispatch

    captured: dict[str, Any] = {}

    class _CapturingProvider:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def complete_with_tool_calls(self, messages: Any, *, tools: Any, provider: Any = None, **_kwargs: Any) -> dict:
            captured["messages"] = messages
            if tool_calls:
                return {"content": "", "tool_calls": tool_calls, "provider": "test"}
            return {"content": "done", "tool_calls": [], "provider": "test"}

    monkeypatch.setattr(react_dispatch, "ModelProvider", _CapturingProvider)
    return captured


# --------------------------------------------------------------------------
# Implicit Screen Awareness — "include_desktop_context" on a plain chat
# message synchronously captures the desktop (dana.plugins.os.desktop_vision's
# _capture_primary_monitor_jpeg_b64, imported as-is into dana.api.server) and
# attaches it BEFORE the ReAct loop's first LLM call, with no dispatched
# analyze_desktop_screen tool call and no HITL approval gate — the whole
# point being to skip the latency of a mid-loop tool round trip.
# --------------------------------------------------------------------------


def test_include_desktop_context_attaches_capture_as_multimodal_content(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server_module, "_capture_primary_monitor_jpeg_b64", lambda: "FAKEBASE64==")
    captured = _capture_provider_messages(monkeypatch)

    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"text": "what's wrong with this code?", "include_desktop_context": True})
        _drain_until(ws, "assistant_message")

    user_message = next(m for m in captured["messages"] if m["role"] == "user")
    assert isinstance(user_message["content"], list)
    assert user_message["content"][0] == {"type": "text", "text": "what's wrong with this code?"}
    image_parts = [p for p in user_message["content"] if p["type"] == "image_url"]
    assert len(image_parts) == 1
    assert image_parts[0]["image_url"]["url"] == "data:image/jpeg;base64,FAKEBASE64=="


def test_include_desktop_context_combines_with_explicit_attachments(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server_module, "_capture_primary_monitor_jpeg_b64", lambda: "FAKEBASE64==")
    captured = _capture_provider_messages(monkeypatch)
    png = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="

    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"text": "compare these", "attachments": [png], "include_desktop_context": True})
        _drain_until(ws, "assistant_message")

    user_message = next(m for m in captured["messages"] if m["role"] == "user")
    image_urls = [p["image_url"]["url"] for p in user_message["content"] if p["type"] == "image_url"]
    assert image_urls == [png, "data:image/jpeg;base64,FAKEBASE64=="]


def test_include_desktop_context_false_or_absent_never_captures(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The capture function must not even be called unless the flag is
    explicitly True — an ordinary text turn must never trigger a desktop
    grab."""
    calls: list[bool] = []
    monkeypatch.setattr(server_module, "_capture_primary_monitor_jpeg_b64", lambda: calls.append(True) or "unused")
    _mock_llm(monkeypatch, "Hello!")

    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"text": "hi"})
        assistant = _drain_until(ws, "assistant_message")
        assert assistant["content"] == "Hello!"

    assert calls == []


def test_include_desktop_context_capture_failure_degrades_to_text_only(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A capture failure (no display, mss/Pillow error, permission denial)
    must never crash the turn — it degrades to the plain-text turn, same as
    the flag being off."""

    def _boom() -> str:
        raise RuntimeError("no display available")

    monkeypatch.setattr(server_module, "_capture_primary_monitor_jpeg_b64", _boom)
    captured = _capture_provider_messages(monkeypatch)

    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"text": "hi", "include_desktop_context": True})
        assistant = _drain_until(ws, "assistant_message")
        assert assistant["content"] == "done"

    user_message = next(m for m in captured["messages"] if m["role"] == "user")
    assert user_message["content"] == "hi"


def test_include_desktop_context_does_not_trigger_hitl_approval(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The toggle is a standing user opt-in, not an agent-initiated
    analyze_desktop_screen tool call — no hitl_approval_required must ever
    be raised just because a screenshot was attached this way."""
    monkeypatch.setattr(server_module, "_capture_primary_monitor_jpeg_b64", lambda: "FAKEBASE64==")
    _mock_llm(monkeypatch, "Looks fine to me.")

    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"text": "what's wrong with this code?", "include_desktop_context": True})

        seen_before_final: list[dict[str, Any]] = []
        assistant = None
        for _ in range(20):
            msg = ws.receive_json()
            if msg["type"] == "assistant_message":
                assistant = msg
                break
            seen_before_final.append(msg)

    assert assistant is not None
    assert assistant["content"] == "Looks fine to me."
    assert not any(m["type"] == "hitl_approval_required" for m in seen_before_final)


def test_abort_turn_with_nothing_pending_does_not_affect_the_next_turn(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An abort_turn sent with no turn actually in flight (e.g. the click
    landed just after the previous turn had already finished) must not
    bleed into the NEXT turn and wrongly cancel it."""
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"type": "abort_turn"})  # no-op: nothing pending, no active loop

        _mock_llm(monkeypatch, "Hello!")
        ws.send_json({"text": "hi"})
        assistant = _drain_until(ws, "assistant_message")
        assert assistant["content"] == "Hello!"


def test_execute_code_task_requires_hitl_approval_with_files_then_dispatches(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """execute_code_task (dana/plugins/coder_plugin/manifest.json, domain
    "software_engineering") is registered generically via
    react_dispatch.refresh_plugin_tools() rather than a hardcoded handler,
    but must still be HITL-gated exactly like a native mutating tool
    (create_freecad_box) and must still surface `files`/`task_description`
    in the approval payload's `parameters` for the frontend's
    CodeTaskApprovalDetails card. The real handler shells out to `aider`, so
    it's swapped for a fake in TOOL_HANDLERS rather than actually invoked.
    """
    import dana.core.react_dispatch as react_dispatch

    task_args = {
        "task_description": "Fix the off-by-one in the paginator",
        "files": ["dana/api/sessions.py"],
    }
    _mock_llm(monkeypatch, [ToolCall(tool_id="execute_code_task", arguments=task_args)])

    def _fake_execute_code_task(args: dict[str, Any], _engine: Any, _cp: Any) -> dict[str, Any]:
        assert args == task_args
        return {"ok": True, "stdout": "Commit abcdef1: fix pagination", "stderr": "", "returncode": 0}

    monkeypatch.setitem(react_dispatch.TOOL_HANDLERS, "execute_code_task", _fake_execute_code_task)

    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"type": "update_context", "active_plugins": ["coder"]})
        ws.send_json({"text": "fix the paginator bug"})

        approval = _drain_until(ws, "hitl_approval_required")
        request_id = approval["payload"]["request_id"]
        assert approval["payload"]["action_name"] == "execute_code_task"
        assert approval["payload"]["parameters"]["files"] == ["dana/api/sessions.py"]
        assert approval["payload"]["parameters"]["task_description"] == task_args["task_description"]

        ws.send_json({"type": "hitl_response", "payload": {"request_id": request_id, "approved": True}})

        dispatch_start = _drain_until(ws, "tool_dispatch_start")
        assert dispatch_start["tool_name"] == "execute_code_task"
        assert dispatch_start["arguments"] == task_args

        dispatch_end = _drain_until(ws, "tool_dispatch_end")
        assert dispatch_end["status"] == "success"
        assert "abcdef1" in dispatch_end["output"]["stdout"]

        assistant = _drain_until(ws, "assistant_message")
        assert assistant["content"] == "Done."


def test_verification_gate_rejects_unacknowledged_failure_then_accepts_admission(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test for the hallucinated-success bug a live E2E run hit:
    perform_freecad_boolean failed (no such object), yet the model's
    proposed "final" answer confidently reported a fused bounding box as if
    the operation had succeeded. _run_react_loop's verification gate
    (last_failure + _acknowledges_failure) must reject that first "final"
    turn and loop back with a SYSTEM OVERRIDE message instead of ending the
    turn on it — then accept the SECOND "final" turn once it actually
    admits the failure in plain language.

    perform_freecad_boolean is (in production) in _HITL_ALWAYS_APPROVED_TOOLS
    so this never needs an approval round-trip — but this file's own
    _disable_permanent_hitl_whitelist autouse fixture clears that set for
    every test here by default, so it must be restored explicitly (same
    pattern every other test in this file that relies on the real whitelist
    already uses) or the tool call suspends on hitl_approval_required
    forever instead of dispatching. MockFreeCADEngine.apply_boolean fails
    deterministically for an object name that was never created.
    """
    monkeypatch.setattr(server_module, "_HITL_ALWAYS_APPROVED_TOOLS", _REAL_HITL_ALWAYS_APPROVED_TOOLS)
    _mock_llm(
        monkeypatch,
        [ToolCall(tool_id="perform_freecad_boolean", arguments={"operation": "union", "base_object": "AI_Part", "tool_object": "BaseBox"})],
        "The bounding box of the fused result is (0, 0, 0, 50, 50, 50).",  # hallucinated success — must be rejected
        "I wasn't able to complete this — the boolean union failed because 'AI_Part' doesn't exist yet.",
    )
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        _activate_freecad(ws)
        ws.send_json({"text": "fuse AI_Part and BaseBox"})

        dispatch_end = _drain_until(ws, "tool_dispatch_end")
        assert dispatch_end["status"] == "error"

        assistant = _drain_until(ws, "assistant_message")
        assert "wasn't able to complete" in assistant["content"]
        assert "fused result is (0, 0, 0, 50, 50, 50)" not in assistant["content"]


def test_unload_capability_tool_id_hides_tool_from_the_very_next_turn(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dynamic Domain Locking end-to-end: unload_capability(tool_id=...)
    must actually remove that tool_id from the VERY NEXT turn's offered
    tools= schema, not just report success — dana.api.server's
    _execute_and_continue is what writes session["hidden_tool_ids"], and
    _run_react_loop's next_react_turn call is what reads it back every
    iteration. Uses "freecad_essential" (not "freecad") — it's in
    _NARROWING_EXEMPT_DOMAINS, so create_freecad_cylinder is unconditionally
    offered whenever active regardless of Pillar 1's separate query-relevance
    narrowing, isolating this test from that unrelated behavior.
    """
    import dana.core.react_dispatch as react_dispatch

    captured_tools: list[list[dict[str, Any]]] = []

    class _RecordingProvider:
        def __init__(self, turns: list[Any]) -> None:
            self._turns = list(turns)

        def complete_with_tool_calls(self, messages: Any, *, tools: Any, provider: Any = None, **kwargs: Any) -> dict:
            captured_tools.append(tools)
            turn = self._turns.pop(0) if self._turns else "Done."
            if isinstance(turn, str):
                return {"content": turn, "tool_calls": [], "provider": "test"}
            return {"content": "", "tool_calls": turn, "provider": "test"}

    fake = _RecordingProvider(
        [
            [ToolCall(tool_id="unload_capability", arguments={"tool_id": "create_freecad_cylinder"})],
            "Done.",
        ]
    )
    monkeypatch.setattr(react_dispatch, "ModelProvider", lambda **_kwargs: fake)

    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"type": "update_context", "active_plugins": ["freecad_essential"]})
        ws.send_json({"text": "stop letting yourself reach for create_freecad_cylinder directly"})

        _drain_until(ws, "assistant_message")

    assert len(captured_tools) == 2
    first_names = {t["function"]["name"] for t in captured_tools[0]}
    second_names = {t["function"]["name"] for t in captured_tools[1]}
    assert "create_freecad_cylinder" in first_names
    assert "create_freecad_cylinder" not in second_names
