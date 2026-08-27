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
def _mock_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server_module, "get_cad_engine", lambda: MockFreeCADEngine())
    monkeypatch.setattr(server_module, "get_control_plane", lambda: MockControlPlane())


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

        dispatch_start = _drain_until(ws, "dag_node_start")
        assert dispatch_start["node_id"] == "dispatch-0"

        tool_call = _drain_until(ws, "tool_call")
        assert tool_call["tool_id"] == "system_state"

        dispatch_complete = _drain_until(ws, "dag_node_complete")
        assert dispatch_complete["node_id"] == "dispatch-0" and dispatch_complete["status"] == "success"

        tool_result = _drain_until(ws, "tool_result")
        assert tool_result["ok"] is True

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

        tool_call = _drain_until(ws, "tool_call")
        assert tool_call["tool_id"] == "create_freecad_box"

        tool_result = _drain_until(ws, "tool_result")
        assert tool_result["ok"] is True
        assert tool_result["mesh_url"] is not None

        # Loop continues after approval+execution — falls back to "Done."
        # since only one turn was queued, and terminates cleanly.
        assistant = _drain_until(ws, "assistant_message")
        assert assistant["content"] == "Done."


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

        _drain_until(ws, "tool_call")
        _drain_until(ws, "tool_result")
        assistant = _drain_until(ws, "assistant_message")
        assert assistant["content"] == "Done."

        # Second, separate turn — same tool_id. No hitl_approval_required
        # this time: it must go straight to tool_call/tool_result.
        ws.send_json({"text": "now build a small 10x10x10 box"})

        seen_types = []
        for _ in range(6):
            msg = ws.receive_json()
            seen_types.append(msg["type"])
            if msg["type"] == "tool_call":
                assert msg["tool_id"] == "create_freecad_box"
                break
        assert "hitl_approval_required" not in seen_types, (
            f"expected no second approval prompt, got message sequence: {seen_types}"
        )
        assert "tool_call" in seen_types, f"never saw a tool_call, got: {seen_types}"

        tool_result = _drain_until(ws, "tool_result")
        assert tool_result["ok"] is True


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

        tool_call = _drain_until(ws, "tool_call")
        assert tool_call["arguments"]["length"] == "99"


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

        tool_call = _drain_until(ws, "tool_call")
        assert tool_call["tool_id"] == "manipulate_camera"
        assert tool_call["arguments"]["target"] == [5.0, 6.0, 7.0]

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
        tool_result = _drain_until(ws, "tool_result")
        assert tool_result["ok"] is True


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

        first_call = _drain_until(ws, "tool_call")
        assert first_call["tool_id"] == "system_state"
        _drain_until(ws, "tool_result")

        second_parse_start = _drain_until(ws, "dag_node_start")
        assert second_parse_start["node_id"] == "parse-1"

        second_call = _drain_until(ws, "tool_call")
        assert second_call["tool_id"] == "check_plugin_registry"
        _drain_until(ws, "tool_result")

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

        _drain_until(ws, "tool_result")  # system_state, no HITL

        approval = _drain_until(ws, "hitl_approval_required")
        assert approval["payload"]["action_name"] == "create_freecad_box"

        ws.send_json(
            {
                "type": "hitl_response",
                "payload": {"request_id": approval["payload"]["request_id"], "approved": True},
            }
        )

        tool_call = _drain_until(ws, "tool_call")
        assert tool_call["tool_id"] == "create_freecad_box"
        tool_result = _drain_until(ws, "tool_result")
        assert tool_result["ok"] is True

        assistant = _drain_until(ws, "assistant_message")
        assert assistant["content"] == "Built the box after checking system state."


def test_react_loop_stops_at_max_iterations(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """A model stuck proposing the same non-mutating tool call forever must
    be forcefully stopped, not left to run indefinitely."""
    endless_tool_call = [ToolCall(tool_id="system_state", arguments={})]
    # dana.api.server._MAX_REACT_ITERATIONS is 13 — queue more canned turns
    # than that so the loop actually reaches the cap instead of running out
    # of mock responses first (which would fall back to a "Done." final turn).
    _mock_llm(monkeypatch, *([endless_tool_call] * 15))  # far more turns than the cap allows
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"text": "loop forever"})

        # 13 full iterations (dag_start/dag_complete/tool_call/tool_result per
        # iteration, plus any tee'd server_log lines) produce well more than
        # 100 websocket messages before the cap's own assistant_message —
        # generous headroom here rather than hand-computing an exact count.
        assistant = _drain_until(ws, "assistant_message", limit=250)
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
# Agent Activity feed — "tool_start"/"tool_complete". Plugin-agnostic
# transparency events the ChatPanel renders inline (unlike "dag_node_start"/
# "tool_call"/"dag_node_complete", which only the CadPlugin's DAG Monitor
# ever renders) — so a user doing OS/web work with no plugin tab open can
# still see what the agent is doing turn-by-turn, not just a silent wait
# until the final assistant_message.
# --------------------------------------------------------------------------


def test_tool_start_and_tool_complete_emitted_for_non_mutating_tool(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_llm(monkeypatch, [ToolCall(tool_id="system_state", arguments={})], "The system is healthy.")
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"text": "system status"})

        tool_start = _drain_until(ws, "tool_start")
        assert tool_start == {"type": "tool_start", "tool_name": "system_state", "args_summary": ""}

        tool_complete = _drain_until(ws, "tool_complete")
        assert tool_complete == {"type": "tool_complete", "tool_name": "system_state", "status": "success"}

        # Both fire strictly before the tool_result/assistant_message that
        # close out the turn — a live consumer sees them WHILE the turn is
        # still in flight, not only after the fact.
        _drain_until(ws, "tool_result")
        _drain_until(ws, "assistant_message")


def test_tool_start_args_summary_surfaces_the_identifying_argument(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_llm(monkeypatch, [ToolCall(tool_id="search_web", arguments={"query": "gearbox torque rating", "max_results": 3})])
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"type": "update_context", "active_plugins": ["web_tools"]})
        ws.send_json({"text": "look up gearbox torque ratings"})

        tool_start = _drain_until(ws, "tool_start")
        assert tool_start["tool_name"] == "search_web"
        assert tool_start["args_summary"] == "gearbox torque rating"


def test_tool_start_args_summary_is_truncated_for_long_values(
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

        tool_start = _drain_until(ws, "tool_start")
        # write_file's args_summary key is "path" (short + identifying) —
        # never the (here, 500-char) "content" value.
        assert tool_start["args_summary"] == "notes.txt"
        assert len(tool_start["args_summary"]) <= 80


def test_tool_start_is_deferred_until_after_hitl_approval(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mutating tool must not be reported as "started" while it's still
    only a pending HITL proposal — tool_start belongs to actual dispatch,
    which for a mutating tool only happens post-approval."""
    _mock_llm(monkeypatch, [ToolCall(tool_id="create_freecad_box", arguments={"length": 60, "width": 40, "height": 20})])
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        _activate_freecad(ws)
        ws.send_json({"text": "build a box 60x40x20"})

        approval = _drain_until(ws, "hitl_approval_required")

        ws.send_json(
            {"type": "hitl_response", "payload": {"request_id": approval["payload"]["request_id"], "approved": True}}
        )

        tool_start = _drain_until(ws, "tool_start")
        assert tool_start["tool_name"] == "create_freecad_box"

        tool_complete = _drain_until(ws, "tool_complete")
        assert tool_complete == {"type": "tool_complete", "tool_name": "create_freecad_box", "status": "success"}


def test_tool_complete_status_is_error_when_the_tool_fails(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_llm(monkeypatch, [ToolCall(tool_id="read_file", arguments={"path": "does_not_exist.txt"})], "No such file.")
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"type": "update_context", "active_plugins": ["os_tools"]})
        ws.send_json({"text": "read does_not_exist.txt"})

        _drain_until(ws, "tool_start")
        tool_complete = _drain_until(ws, "tool_complete")
        assert tool_complete == {"type": "tool_complete", "tool_name": "read_file", "status": "error"}


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
    pending call outright: it must never dispatch (no tool_start/tool_call
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

        _drain_until(ws, "tool_result")  # system_state, non-mutating, runs immediately

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
        assert not any(m["type"] in ("tool_start", "tool_call") and m.get("tool_name", m.get("tool_id")) == "create_freecad_box" for m in seen_before_final)


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

        tool_call = _drain_until(ws, "tool_call")
        assert tool_call["tool_id"] == "execute_code_task"
        assert tool_call["arguments"] == task_args

        tool_result = _drain_until(ws, "tool_result")
        assert tool_result["ok"] is True
        assert "abcdef1" in tool_result["payload"]["stdout"]

        assistant = _drain_until(ws, "assistant_message")
        assert assistant["content"] == "Done."
