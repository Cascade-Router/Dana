"""Unit tests for dana.core.react_dispatch's LLM-driven ReAct parsing step
and its (unchanged) tool execution / HITL-gating layer.

``parse_utterance`` now calls a real LLM (via ``ModelProvider.
complete_with_tool_calls``) instead of matching regex, so every test here
mocks that one call site (``rd.ModelProvider``) rather than starting a real
Ollama daemon — these tests are about the ReAct wiring (system prompt
construction, tool-call translation, HITL/DAG-safe fallbacks), not about
LLM quality.

No async test functions/plugin needed: each test drives the coroutine with
a plain ``asyncio.run(...)`` call from an ordinary sync ``def test_...()``.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import Any
from unittest.mock import patch

import pytest

from dana.core import react_dispatch as rd
from dana.session_context import set_session_id
from dana.tools.schema import ToolCall


@pytest.fixture(autouse=True)
def _plan_gate_open() -> None:
    """Every dispatch test in this module exercises tool MECHANICS, not the
    Plan-and-Execute Gatekeeper itself (see the dedicated
    test_plan_gatekeeper_* tests below for that) — so the gate is
    pre-opened for the (default, ambient) session every test here runs
    under, exactly as if create_plan had already been called.

    Safe to leave open with no explicit teardown here: tests/conftest.py's
    own ``_reset_plan_gate_state`` (a global safety net, same rationale as
    its ``_reset_task_board_plan``/``_reset_user_skills_registry``
    neighbors) clears ``_PLAN_STATE_REGISTRY`` after EVERY test in the
    whole suite, so this can never leak a permanently-open gate into some
    other test module. The dedicated test_plan_gatekeeper_* tests below run
    under their OWN isolated session_id (never touched here) instead of
    relying on toggling this one.
    """
    rd._set_has_plan(True, "test-harness plan")


class _FakeProvider:
    """Stands in for ``dana.core.model_provider.ModelProvider``."""

    def __init__(
        self,
        tool_calls: list[ToolCall] | None = None,
        content: str = "",
        raises: Exception | None = None,
    ) -> None:
        self._tool_calls = tool_calls or []
        self._content = content
        self._raises = raises
        self.calls: list[dict[str, Any]] = []
        # Every dict of kwargs the (mocked) ModelProvider(...) constructor
        # was called with — lets BYOK tests assert api_keys actually reached
        # the constructor without needing a real ModelProvider/HTTP call.
        self.constructor_kwargs: list[dict[str, Any]] = []

    def complete_with_tool_calls(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        provider: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.calls.append({"messages": messages, "tools": tools, "provider": provider})
        if self._raises is not None:
            raise self._raises
        return {"content": self._content, "tool_calls": self._tool_calls, "provider": "test"}


def _mock_llm(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tool_calls: list[ToolCall] | None = None,
    content: str = "",
    raises: Exception | None = None,
) -> _FakeProvider:
    fake = _FakeProvider(tool_calls=tool_calls, content=content, raises=raises)

    def _fake_constructor(**kwargs: Any) -> _FakeProvider:
        fake.constructor_kwargs.append(kwargs)
        return fake

    monkeypatch.setattr(rd, "ModelProvider", _fake_constructor)
    return fake


def _parse(text: str, active_selection: dict[str, Any] | None = None) -> ToolCall | None:
    return asyncio.run(rd.parse_utterance(text, active_selection))


# --------------------------------------------------------------------------
# build_system_prompt
# --------------------------------------------------------------------------


def test_build_system_prompt_without_selection_omits_selection_text() -> None:
    prompt = rd.build_system_prompt(None)
    assert "canvas selection" not in prompt.lower()


def test_build_system_prompt_includes_active_selection() -> None:
    selection = {"centroid": [1.0, 2.0, 3.0], "normal": [0.0, 1.0, 0.0]}
    prompt = rd.build_system_prompt(selection)
    assert "[1.0, 2.0, 3.0]" in prompt
    assert "[0.0, 1.0, 0.0]" in prompt


# --------------------------------------------------------------------------
# parse_utterance — control flow
# --------------------------------------------------------------------------


def test_parse_utterance_empty_text_short_circuits_without_calling_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom() -> None:
        raise AssertionError("LLM should not be called for empty text")

    monkeypatch.setattr(rd, "ModelProvider", _boom)
    assert _parse("   ") is None


def test_parse_utterance_no_tool_calls_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_llm(monkeypatch, tool_calls=[], content="Sure, happy to chat!")
    assert _parse("thanks!") is None


def test_parse_utterance_llm_exception_returns_none_gracefully(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_llm(monkeypatch, raises=RuntimeError("ollama unreachable"))
    assert _parse("build a box") is None


def test_parse_utterance_unknown_tool_id_from_llm_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_llm(monkeypatch, tool_calls=[ToolCall(tool_id="not_a_real_tool", arguments={})])
    assert _parse("do something weird") is None


def test_parse_utterance_returns_first_proposed_tool_call(monkeypatch: pytest.MonkeyPatch) -> None:
    # Forces the local-Ollama default deterministically: tool_calling_provider()
    # otherwise resolves whatever this machine's own .env has configured
    # (DANA_CLOUD_PRIMARY/DANA_CLOUD_PROVIDER), which is an ambient dev-machine
    # setting, not part of this test's actual contract — same monkeypatch
    # pattern the groq-timeout test below uses to pin the OTHER provider.
    monkeypatch.setattr(rd, "tool_calling_provider", lambda: "ollama")
    fake = _mock_llm(
        monkeypatch,
        tool_calls=[ToolCall(tool_id="create_freecad_box", arguments={"length": 60, "width": 40, "height": 20})],
    )
    call = _parse("Create a parametric 60x40x20mm box")
    assert call is not None
    assert call.tool_id == "create_freecad_box"
    assert call.arguments == {"length": 60, "width": 40, "height": 20}
    assert call.raw_text == "Create a parametric 60x40x20mm box"
    # The tools handed to the LLM are exactly the wired subset, not the
    # full tools.json registry (which also serves the legacy regex broker).
    tool_names = {t["function"]["name"] for t in fake.calls[0]["tools"]}
    assert tool_names == rd._LLM_TOOL_IDS
    assert fake.calls[0]["provider"] == "ollama"


# --------------------------------------------------------------------------
# parse_utterance — camera preset resolution
# --------------------------------------------------------------------------


def test_parse_utterance_camera_preset_resolves_to_position_target(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_llm(monkeypatch, tool_calls=[ToolCall(tool_id="manipulate_camera", arguments={"preset": "iso"})])
    call = _parse("Show me the isometric view")
    assert call is not None
    assert call.tool_id == "manipulate_camera"
    assert call.arguments["position"] == list(rd._CAMERA_PRESETS["iso"])
    assert call.arguments["target"] == [0.0, 0.0, 0.0]


def test_parse_utterance_camera_preset_uses_active_selection_as_target(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_llm(monkeypatch, tool_calls=[ToolCall(tool_id="manipulate_camera", arguments={"preset": "top"})])
    selection = {"centroid": [5.0, 6.0, 7.0], "normal": [0.0, 0.0, 1.0]}
    call = _parse("orbit to the top view", selection)
    assert call is not None
    assert call.arguments["position"] == list(rd._CAMERA_PRESETS["top"])
    assert call.arguments["target"] == [5.0, 6.0, 7.0]


def test_parse_utterance_camera_preset_invalid_value_defaults_to_iso(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_llm(monkeypatch, tool_calls=[ToolCall(tool_id="manipulate_camera", arguments={"preset": "bottom"})])
    call = _parse("show me the bottom")
    assert call is not None
    assert call.arguments["position"] == list(rd._CAMERA_PRESETS["iso"])


# --------------------------------------------------------------------------
# parse_utterance — selection-injection fallback
# --------------------------------------------------------------------------


def test_parse_utterance_injects_selection_when_llm_omits_it_but_user_said_here(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_llm(monkeypatch, tool_calls=[ToolCall(tool_id="create_freecad_box", arguments={})])
    selection = {"centroid": [1.0, 2.0, 3.0], "normal": [0.0, 1.0, 0.0]}
    call = _parse("add a box here", selection)
    assert call is not None
    assert call.arguments["target_position"] == [1.0, 2.0, 3.0]
    assert call.arguments["target_normal"] == [0.0, 1.0, 0.0]


def test_parse_utterance_no_fallback_injection_without_reference_word(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_llm(monkeypatch, tool_calls=[ToolCall(tool_id="create_freecad_box", arguments={})])
    selection = {"centroid": [1.0, 2.0, 3.0], "normal": [0.0, 1.0, 0.0]}
    call = _parse("build a cube", selection)
    assert call is not None
    assert "target_position" not in call.arguments


def test_parse_utterance_respects_llm_provided_target_over_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_llm(
        monkeypatch,
        tool_calls=[
            ToolCall(
                tool_id="create_freecad_extrusion",
                arguments={"height": 25, "target_position": [9.0, 9.0, 9.0], "target_normal": [0.0, 0.0, 1.0]},
            )
        ],
    )
    selection = {"centroid": [1.0, 2.0, 3.0], "normal": [0.0, 1.0, 0.0]}
    call = _parse("extrude this by 25mm", selection)
    assert call is not None
    # The LLM's own values win — the fallback never overwrites a real answer.
    assert call.arguments["target_position"] == [9.0, 9.0, 9.0]


# --------------------------------------------------------------------------
# next_react_turn / ReactTurn — the multi-step loop's single-iteration
# building block. parse_utterance (tested above) is now a thin single-turn
# wrapper around this; dana.api.server drives it directly, iteration after
# iteration, with an evolving `messages` history.
# --------------------------------------------------------------------------


def test_next_react_turn_returns_final_when_no_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_llm(monkeypatch, content="All done here.")
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    turn = asyncio.run(rd.next_react_turn(messages))
    assert turn.kind == "final"
    assert turn.content == "All done here."


def test_next_react_turn_returns_tool_call_and_finalizes_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_llm(monkeypatch, tool_calls=[ToolCall(tool_id="create_freecad_box", arguments={})])
    selection = {"centroid": [1.0, 2.0, 3.0], "normal": [0.0, 1.0, 0.0]}
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "add a box here"}]
    turn = asyncio.run(rd.next_react_turn(messages, selection, raw_text="add a box here"))
    assert turn.kind == "tool_call"
    assert turn.call.tool_id == "create_freecad_box"
    # _finalize_call_arguments still runs — same selection-injection fallback
    # parse_utterance relies on, now reachable at any loop iteration.
    assert turn.call.arguments["target_position"] == [1.0, 2.0, 3.0]


def test_next_react_turn_uses_explicit_raw_text_not_latest_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """raw_text must be the ORIGINAL user utterance, not whatever the last
    message in a multi-turn history happens to be — a later loop iteration's
    last message is a tool result, which has no bearing on whether the user
    said "here" several tool calls ago."""
    _mock_llm(monkeypatch, tool_calls=[ToolCall(tool_id="create_freecad_box", arguments={})])
    selection = {"centroid": [4.0, 5.0, 6.0], "normal": [0.0, 0.0, 1.0]}
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "add a box here"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "system_state", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": '{"ok": true}'},
    ]
    turn = asyncio.run(rd.next_react_turn(messages, selection, raw_text="add a box here"))
    assert turn.call.arguments["target_position"] == [4.0, 5.0, 6.0]


def test_next_react_turn_prunes_stale_tool_output_before_calling_the_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Integration proof for the Groq-429 fix: next_react_turn -> _call_llm_once
    must actually apply dana.core.context_manager.prune_tool_output_history to
    the payload it hands the model, not just have the pure function exist.
    A long-ago tool result gets truncated; message count/order (the strict
    tool_calls/tool-result pairing OpenAI/Groq require) stays identical.

    Forces the resolved provider to a non-Ollama one — ``_call_llm_once``
    routes Ollama through ``compress_tool_output_history`` instead (a
    different keep_recent default and compression strategy; see
    ``test_next_react_turn_compresses_stale_tool_output_for_ollama`` below)
    — so this test's cloud-path assertions stay deterministic regardless of
    the ambient DANA_CLOUD_PRIMARY/.env state."""
    monkeypatch.setattr(rd, "tool_calling_provider", lambda: "openrouter")
    fake = _mock_llm(monkeypatch, content="done")
    stale_output = "OLDSTUFF-" * 200  # 1800 chars, well past the truncation threshold
    fresh_output = "NEWSTUFF-" * 200

    def _tool_cycle(call_id: str, content: str) -> list[dict]:
        return [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "search_codebase", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": call_id, "content": content},
        ]

    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]
    messages += _tool_cycle("call_1", stale_output)
    messages += _tool_cycle("call_2", stale_output)
    messages += _tool_cycle("call_3", stale_output)
    messages += _tool_cycle("call_4", fresh_output)

    asyncio.run(rd.next_react_turn(messages))

    sent_messages = fake.calls[0]["messages"]
    assert len(sent_messages) == len(messages)  # count never changes
    sent_tool_contents = [m["content"] for m in sent_messages if m["role"] == "tool"]
    # Default keep_recent=3 -> only call_1 (everything but the 3 most recent
    # tool results) is stale; call_2/call_3/call_4 stay intact.
    assert sent_tool_contents[0] != stale_output
    assert sent_tool_contents[0].startswith("[Pruned to save context]")
    assert sent_tool_contents[1] == stale_output
    assert sent_tool_contents[2] == stale_output
    assert sent_tool_contents[3] == fresh_output  # most recent tool result untouched

    # The caller's own messages list must never be mutated in place.
    assert messages[3]["content"] == stale_output


def test_next_react_turn_compresses_stale_tool_output_for_ollama(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """For the local Ollama provider specifically, _call_llm_once routes
    through dana.core.context_manager.compress_tool_output_history instead
    of prune_tool_output_history — JSON-structure-aware compression
    (keep_recent=4) rather than a blind character slice. A stale JSON tool
    result keeps its name/bounding_box fields verbatim and drops a large
    low-signal field; an unresolved error stays untouched even though it's
    the oldest message in the chain."""
    monkeypatch.setattr(rd, "tool_calling_provider", lambda: "ollama")
    fake = _mock_llm(monkeypatch, content="done")

    def _box_result(name: str) -> str:
        return json.dumps(
            {
                "ok": True,
                "name": name,
                "bounding_box": [0.0, 0.0, 0.0, 60.0, 40.0, 20.0],
                "unlocked_tools": [f"tool_{i}_with_a_fairly_long_descriptive_name" for i in range(28)],
            }
        )

    error_result = json.dumps({"ok": False, "status": "error", "reason": "length must be positive"})

    def _tool_cycle(call_id: str, content: str) -> list[dict]:
        return [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "create_freecad_box", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": call_id, "content": content},
        ]

    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]
    messages += _tool_cycle("call_0", error_result)
    for i in range(1, 6):
        messages += _tool_cycle(f"call_{i}", _box_result(f"Box{i}"))

    asyncio.run(rd.next_react_turn(messages))

    sent_messages = fake.calls[0]["messages"]
    assert len(sent_messages) == len(messages)  # count never changes
    sent_tool_contents = [m["content"] for m in sent_messages if m["role"] == "tool"]

    # 6 tool messages total, keep_recent=4 -> the 2 OLDEST (call_0's error,
    # call_1's box result) are stale; call_0 still stays verbatim (unresolved
    # error), only call_1 actually gets compressed.
    assert sent_tool_contents[0] == error_result
    stale = json.loads(sent_tool_contents[1])
    assert stale["name"] == "Box1"
    assert stale["bounding_box"] == [0.0, 0.0, 0.0, 60.0, 40.0, 20.0]
    assert "unlocked_tools" not in stale
    # The 4 most recent box results stay byte-for-byte intact.
    for i in range(2, 6):
        assert sent_tool_contents[i] == _box_result(f"Box{i}")


def test_next_react_turn_unknown_tool_id_yields_final(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_llm(monkeypatch, tool_calls=[ToolCall(tool_id="not_a_real_tool", arguments={})], content="")
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "do something weird"}]
    turn = asyncio.run(rd.next_react_turn(messages))
    assert turn.kind == "final"


def test_next_react_turn_llm_exception_yields_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_llm(monkeypatch, raises=RuntimeError("ollama unreachable"))
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "build a box"}]
    turn = asyncio.run(rd.next_react_turn(messages))
    assert turn.kind == "error"


# --------------------------------------------------------------------------
# next_react_turn — BYOK: dana.api.server's session["api_keys"] must reach
# ModelProvider's constructor unchanged, so a session-provided OpenAI/
# Anthropic key takes precedence over the environment (see
# dana.core.model_provider._resolve_openai_endpoint/_complete_openai_compatible).
# --------------------------------------------------------------------------


def test_next_react_turn_threads_api_keys_into_model_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _mock_llm(monkeypatch, content="done")
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    api_keys = {"openai": "sk-session-key"}

    asyncio.run(rd.next_react_turn(messages, api_keys=api_keys))

    assert fake.constructor_kwargs == [{"api_keys": api_keys}]


def test_next_react_turn_without_api_keys_passes_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """No session["api_keys"] (e.g. the frontend hasn't sent update_secrets
    yet) must not crash the constructor call — ModelProvider(api_keys=None)
    is the documented no-BYOK case, same as omitting the kwarg entirely."""
    fake = _mock_llm(monkeypatch, content="done")
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]

    asyncio.run(rd.next_react_turn(messages))

    assert fake.constructor_kwargs == [{"api_keys": None}]


def test_build_assistant_tool_call_message_shape() -> None:
    call = ToolCall(tool_id="create_freecad_box", arguments={"length": 30, "width": 30, "height": 30})
    message, call_id = rd.build_assistant_tool_call_message(call)
    assert message["role"] == "assistant"
    assert message["tool_calls"][0]["id"] == call_id
    assert message["tool_calls"][0]["function"]["name"] == "create_freecad_box"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"length": 30, "width": 30, "height": 30}


def test_build_assistant_tool_call_message_ids_are_unique() -> None:
    call = ToolCall(tool_id="system_state", arguments={})
    _msg1, id1 = rd.build_assistant_tool_call_message(call)
    _msg2, id2 = rd.build_assistant_tool_call_message(call)
    assert id1 != id2


def test_build_tool_result_message_success() -> None:
    """create_freecad_box is a _GEOMETRY_RESULT_TOOL_IDS tool, so its result
    is slimmed to _GEOMETRY_RESULT_KEEP_KEYS before entering the LLM-facing
    message — the absolute .FCStd path is dropped (the model only ever needs
    the object's `name`, resolved server-side via _OBJECT_PATH_REGISTRY for
    any later tool call, never a raw path)."""
    result = rd.ToolResult("create_freecad_box", True, {"ok": True, "name": "Box", "path": "x.FCStd"}, "ok", 5)
    message = rd.build_tool_result_message("call_abc", result)
    assert message == {
        "role": "tool",
        "tool_call_id": "call_abc",
        "content": json.dumps({"ok": True, "name": "Box"}),
    }


def test_build_tool_result_message_non_geometry_tool_is_never_slimmed() -> None:
    """A tool outside _GEOMETRY_RESULT_TOOL_IDS keeps its full payload shape
    untouched — the geometry allowlist would otherwise gut e.g. a bounding-box
    query's z_max, a search result's matches, or a skill's own traceback."""
    result = rd.ToolResult("get_freecad_bounding_box", True, {"ok": True, "z_max": 30.0, "path": "x.FCStd"}, "ok", 5)
    message = rd.build_tool_result_message("call_abc", result)
    assert json.loads(message["content"]) == {"ok": True, "z_max": 30.0, "path": "x.FCStd"}


def test_build_tool_result_message_geometry_failure_keeps_diagnostic_fields() -> None:
    """A failed geometry-tool call goes through digest_error's structured
    shape (status/reason/suggestion/raw_error/tool_id, no top-level "error"
    key at all) — the slim-down must preserve these so the model can still
    tell WHY it failed, not just that it did."""
    digested_payload = {
        "ok": False,
        "status": "error",
        "tool_id": "perform_freecad_boolean",
        "reason": "Boolean operation produced a non-manifold result",
        "suggestion": "Check that the tool object actually overlaps the base",
        "raw_error": "OCC kernel: BRepAlgoAPI_Cut failed",
    }
    result = rd.ToolResult("perform_freecad_boolean", False, digested_payload, "reason text", 5)
    message = rd.build_tool_result_message("call_abc", result)
    assert json.loads(message["content"]) == digested_payload


def test_build_tool_result_message_failure_reports_error() -> None:
    result = rd.ToolResult("create_freecad_box", False, {}, "boom", 5)
    message = rd.build_tool_result_message("call_abc", result)
    assert json.loads(message["content"]) == {"ok": False, "error": "boom"}


def test_multi_iteration_message_history_round_trips_through_next_react_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A small end-to-end check of the actual pattern dana.api.server drives:
    build the assistant+tool messages for one tool call, append them, then
    ask next_react_turn again — proving the shapes these helpers produce
    are exactly what next_react_turn's own LLM call site can consume."""
    fake = _mock_llm(
        monkeypatch,
        tool_calls=[ToolCall(tool_id="get_freecad_bounding_box", arguments={"target_object": "Box"})],
    )
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "check the box then stop"}]
    turn1 = asyncio.run(rd.next_react_turn(messages, raw_text="check the box then stop"))
    assert turn1.kind == "tool_call"

    assistant_message, call_id = rd.build_assistant_tool_call_message(turn1.call)
    messages.append(assistant_message)
    result = rd.ToolResult("get_freecad_bounding_box", True, {"ok": True, "z_max": 30.0}, "ok", 5)
    messages.append(rd.build_tool_result_message(call_id, result))

    fake._tool_calls = []
    fake._content = "The box is 30mm tall."
    turn2 = asyncio.run(rd.next_react_turn(messages, raw_text="check the box then stop"))
    assert turn2.kind == "final"
    assert turn2.content == "The box is 30mm tall."
    # The second LLM call actually received the full history, tool result included.
    assert fake.calls[-1]["messages"] == messages


# --------------------------------------------------------------------------
# Tool execution / HITL layer — unchanged by the LLM swap, so ToolCall is
# constructed directly here rather than routed through parse_utterance.
# --------------------------------------------------------------------------


def test_is_mutating_tool_classification() -> None:
    assert rd.is_mutating_tool("create_freecad_box") is True
    assert rd.is_mutating_tool("create_freecad_cylinder") is True
    assert rd.is_mutating_tool("create_freecad_extrusion") is True
    assert rd.is_mutating_tool("create_freecad_pyramid") is True
    assert rd.is_mutating_tool("create_freecad_star_prism") is True
    assert rd.is_mutating_tool("perform_freecad_boolean") is True
    assert rd.is_mutating_tool("perform_freecad_edge_operation") is True
    assert rd.is_mutating_tool("modify_freecad_parameter") is True
    assert rd.is_mutating_tool("create_freecad_pipe") is True
    assert rd.is_mutating_tool("align_freecad_objects") is True
    assert rd.is_mutating_tool("resync_workspace") is True
    assert rd.is_mutating_tool("system_state") is False
    assert rd.is_mutating_tool("execute_vision_analysis") is False
    assert rd.is_mutating_tool("manipulate_camera") is False
    # CRITICAL: read-only/file-IO, must never require HITL approval.
    assert rd.is_mutating_tool("get_freecad_bounding_box") is False
    assert rd.is_mutating_tool("export_freecad_model") is False


def test_describe_tool_call_box() -> None:
    call = ToolCall(tool_id="create_freecad_box", arguments={"length": 60, "width": 40, "height": 20})
    description = rd.describe_tool_call(call)
    assert "60" in description and "40" in description and "20" in description


def test_manipulate_camera_tool_handler_requires_vectors() -> None:
    payload = rd._tool_manipulate_camera({"position": [1, 2, 3]}, None, None)
    assert payload["ok"] is False

    payload = rd._tool_manipulate_camera({"position": [1, 2, 3], "target": [0, 0, 0]}, None, None)
    assert payload == {"ok": True, "position": [1.0, 2.0, 3.0], "target": [0.0, 0.0, 0.0]}


def test_dispatch_manipulate_camera_via_registry() -> None:
    call = ToolCall(tool_id="manipulate_camera", arguments={"position": [200, 0, 0], "target": [0, 0, 0]})
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is True
    assert result.payload["position"] == [200.0, 0.0, 0.0]


def test_dispatch_extrusion_without_profile_or_selection_fails_cleanly() -> None:
    call = ToolCall(tool_id="create_freecad_extrusion", arguments={"height": 25})
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "profile points" in result.message or "selected face" in result.message


def test_dispatch_extrusion_rejects_non_z_normal() -> None:
    call = ToolCall(
        tool_id="create_freecad_extrusion",
        arguments={"height": 25, "target_position": [0, 0, 50], "target_normal": [1, 0, 0]},
    )
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "Z axis" in result.message


def test_dispatch_extrusion_with_selection_builds_default_footprint() -> None:
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    call = ToolCall(
        tool_id="create_freecad_extrusion",
        arguments={"height": 25, "target_position": [0, 0, 50], "target_normal": [0, 0, 1]},
    )
    result = rd.dispatch_tool_call(call, MockFreeCADEngine(), MockControlPlane())
    assert result.ok is True
    assert result.payload["dimensions"]["height"] == 25.0
    assert result.payload["dimensions"]["profile_points"] == 4


# --------------------------------------------------------------------------
# Sharp-edged primitives: pyramid + star prism
# --------------------------------------------------------------------------


def test_describe_tool_call_pyramid() -> None:
    call = ToolCall(tool_id="create_freecad_pyramid", arguments={"length": 50, "width": 50, "height": 75})
    description = rd.describe_tool_call(call)
    assert "50" in description and "75" in description


def test_describe_tool_call_star_prism() -> None:
    call = ToolCall(
        tool_id="create_freecad_star_prism",
        arguments={"points": 8, "outer_radius": 60, "inner_radius": 20, "height": 5},
    )
    description = rd.describe_tool_call(call)
    assert "8" in description and "60" in description and "20" in description and "5" in description


def test_dispatch_pyramid_via_mock_engine() -> None:
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    call = ToolCall(tool_id="create_freecad_pyramid", arguments={"length": 50, "width": 50, "height": 75})
    result = rd.dispatch_tool_call(call, MockFreeCADEngine(), MockControlPlane())
    assert result.ok is True
    assert result.payload["dimensions"] == {"length": 50.0, "width": 50.0, "height": 75.0}
    assert result.payload["bounding_box"] == [-25.0, -25.0, 0.0, 25.0, 25.0, 75.0]


def test_dispatch_star_prism_via_mock_engine() -> None:
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    call = ToolCall(
        tool_id="create_freecad_star_prism",
        arguments={"points": 8, "outer_radius": 60, "inner_radius": 20, "height": 5},
    )
    result = rd.dispatch_tool_call(call, MockFreeCADEngine(), MockControlPlane())
    assert result.ok is True
    assert result.payload["dimensions"] == {
        "points": 8,
        "outer_radius": 60.0,
        "inner_radius": 20.0,
        "height": 5.0,
    }
    assert result.payload["bounding_box"] == [-60.0, -60.0, 0.0, 60.0, 60.0, 5.0]


def test_dispatch_star_prism_rejects_too_few_points() -> None:
    call = ToolCall(tool_id="create_freecad_star_prism", arguments={"points": 2, "outer_radius": 60, "inner_radius": 20})
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "at least 3 points" in result.message


# --------------------------------------------------------------------------
# Regular polygon primitive: create_freecad_polygon
# --------------------------------------------------------------------------


def test_describe_tool_call_polygon() -> None:
    call = ToolCall(tool_id="create_freecad_polygon", arguments={"sides": 6, "radius": 50, "height": 10})
    description = rd.describe_tool_call(call)
    assert "6" in description and "50" in description and "10" in description


def test_dispatch_polygon_via_mock_engine() -> None:
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    call = ToolCall(tool_id="create_freecad_polygon", arguments={"sides": 6, "radius": 50, "height": 10})
    result = rd.dispatch_tool_call(call, MockFreeCADEngine(), MockControlPlane())
    assert result.ok is True
    assert result.payload["dimensions"] == {"sides": 6, "radius": 50.0, "height": 10.0}
    bbox = result.payload["bounding_box"]
    assert bbox[2] == 0.0 and bbox[5] == 10.0  # Z spans exactly the extrusion height


def test_dispatch_polygon_rejects_too_few_sides() -> None:
    call = ToolCall(tool_id="create_freecad_polygon", arguments={"sides": 2, "radius": 50, "height": 10})
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "at least 3 sides" in result.message


def test_polygon_is_mutating_and_registered_in_freecad_domain() -> None:
    assert rd.is_mutating_tool("create_freecad_polygon") is True
    assert "create_freecad_polygon" in rd._FREECAD_TOOL_IDS


def test_extract_placement_defaults_to_origin() -> None:
    assert rd._extract_placement({}) == (0.0, 0.0, 0.0)


def test_extract_placement_reads_xyz_args() -> None:
    assert rd._extract_placement({"placement_x": 1, "placement_y": 2, "placement_z": 3}) == (1.0, 2.0, 3.0)


def test_dispatch_box_with_placement_passes_through_to_engine() -> None:
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    call = ToolCall(
        tool_id="create_freecad_box",
        arguments={
            "length": 20,
            "width": 20,
            "height": 20,
            "placement_x": 0,
            "placement_y": 0,
            "placement_z": 25,
        },
    )
    result = rd.dispatch_tool_call(call, MockFreeCADEngine(), MockControlPlane())
    assert result.ok is True
    assert result.payload["placement"] == [0.0, 0.0, 25.0]


def test_dispatch_cylinder_without_placement_defaults_to_origin() -> None:
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    call = ToolCall(tool_id="create_freecad_cylinder", arguments={"radius": 10, "height": 30})
    result = rd.dispatch_tool_call(call, MockFreeCADEngine(), MockControlPlane())
    assert result.ok is True
    assert result.payload["placement"] == [0.0, 0.0, 0.0]


def test_dispatch_pyramid_and_star_prism_with_placement() -> None:
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    pyramid_call = ToolCall(
        tool_id="create_freecad_pyramid",
        arguments={"length": 50, "width": 50, "height": 75, "placement_x": 10, "placement_y": -5},
    )
    result = rd.dispatch_tool_call(pyramid_call, MockFreeCADEngine(), MockControlPlane())
    assert result.ok is True
    assert result.payload["placement"] == [10.0, -5.0, 0.0]

    star_call = ToolCall(
        tool_id="create_freecad_star_prism",
        arguments={"points": 8, "outer_radius": 60, "inner_radius": 20, "height": 5, "placement_z": 12},
    )
    result = rd.dispatch_tool_call(star_call, MockFreeCADEngine(), MockControlPlane())
    assert result.ok is True
    assert result.payload["placement"] == [0.0, 0.0, 12.0]


def test_parse_utterance_pyramid_and_star_prism_pass_through(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_llm(
        monkeypatch,
        tool_calls=[
            ToolCall(tool_id="create_freecad_pyramid", arguments={"length": 50, "width": 50, "height": 75})
        ],
    )
    call = _parse("Build me a pyramid with a 50x50 base and a height of 75.")
    assert call is not None
    assert call.tool_id == "create_freecad_pyramid"

    _mock_llm(
        monkeypatch,
        tool_calls=[
            ToolCall(
                tool_id="create_freecad_star_prism",
                arguments={"points": 8, "outer_radius": 60, "inner_radius": 20, "height": 5},
            )
        ],
    )
    call = _parse("Create a sharp-edged ninja star with 8 points, outer radius 60mm, inner radius 20mm, thickness 5mm.")
    assert call is not None
    assert call.tool_id == "create_freecad_star_prism"


# --------------------------------------------------------------------------
# Boolean CSG operations: perform_freecad_boolean
# --------------------------------------------------------------------------


def test_describe_tool_call_boolean_cut() -> None:
    call = ToolCall(
        tool_id="perform_freecad_boolean",
        arguments={"operation": "cut", "base_object": "Box", "tool_object": "Cylinder"},
    )
    description = rd.describe_tool_call(call)
    assert "Cylinder" in description and "Box" in description


def test_dispatch_boolean_rejects_unknown_operation() -> None:
    call = ToolCall(
        tool_id="perform_freecad_boolean",
        arguments={"operation": "bogus", "base_object": "Box", "tool_object": "Cylinder"},
    )
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "cut, union, intersect" in result.message


def test_dispatch_boolean_requires_base_and_tool_object() -> None:
    call = ToolCall(tool_id="perform_freecad_boolean", arguments={"operation": "cut"})
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "base_object" in result.message


def test_dispatch_boolean_rejects_unknown_object_names() -> None:
    call = ToolCall(
        tool_id="perform_freecad_boolean",
        arguments={"operation": "cut", "base_object": "NeverCreated1", "tool_object": "NeverCreated2"},
    )
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "NeverCreated1" in result.message


def test_dispatch_boolean_end_to_end_via_object_name_registry() -> None:
    """create_freecad_box/cylinder register their (name -> path) in
    rd._OBJECT_PATH_REGISTRY as a side effect of dispatch_tool_call; a later
    perform_freecad_boolean call resolves base_object/tool_object against
    that registry instead of needing a persistent FreeCAD ActiveDocument."""
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    engine = MockFreeCADEngine()
    control_plane = MockControlPlane()

    box_result = rd.dispatch_tool_call(
        ToolCall(tool_id="create_freecad_box", arguments={"length": 50, "width": 50, "height": 50, "name": "CsgBox"}),
        engine,
        control_plane,
    )
    assert box_result.ok is True

    cyl_result = rd.dispatch_tool_call(
        ToolCall(
            tool_id="create_freecad_cylinder",
            arguments={
                "radius": 15,
                "height": 50,
                "name": "CsgCylinder",
                "placement_x": 25,
                "placement_y": 25,
            },
        ),
        engine,
        control_plane,
    )
    assert cyl_result.ok is True

    cut_result = rd.dispatch_tool_call(
        ToolCall(
            tool_id="perform_freecad_boolean",
            arguments={"operation": "cut", "base_object": "CsgBox", "tool_object": "CsgCylinder"},
        ),
        engine,
        control_plane,
    )
    assert cut_result.ok is True
    assert cut_result.payload["type"] == "Part::Cut"
    assert cut_result.payload["name"] == "Cut"

    # The boolean result itself registers too, so it can chain into a
    # further boolean op as someone else's base_object/tool_object.
    assert rd._object_registry()["Cut"] == cut_result.payload["path"]


def test_dispatch_boolean_union_and_intersect_use_default_names() -> None:
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    engine = MockFreeCADEngine()
    control_plane = MockControlPlane()
    rd.dispatch_tool_call(
        ToolCall(tool_id="create_freecad_box", arguments={"length": 20, "width": 20, "height": 20, "name": "UBoxA"}),
        engine,
        control_plane,
    )
    rd.dispatch_tool_call(
        ToolCall(tool_id="create_freecad_cylinder", arguments={"radius": 5, "height": 20, "name": "UCylA"}),
        engine,
        control_plane,
    )

    union_result = rd.dispatch_tool_call(
        ToolCall(
            tool_id="perform_freecad_boolean",
            arguments={"operation": "union", "base_object": "UBoxA", "tool_object": "UCylA"},
        ),
        engine,
        control_plane,
    )
    assert union_result.ok is True
    assert union_result.payload["name"] == "Fusion"
    assert union_result.payload["type"] == "Part::MultiFuse"

    intersect_result = rd.dispatch_tool_call(
        ToolCall(
            tool_id="perform_freecad_boolean",
            arguments={"operation": "intersect", "base_object": "UBoxA", "tool_object": "UCylA"},
        ),
        engine,
        control_plane,
    )
    assert intersect_result.ok is True
    assert intersect_result.payload["name"] == "Common"
    assert intersect_result.payload["type"] == "Part::MultiCommon"


def test_parse_utterance_boolean_pass_through(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_llm(
        monkeypatch,
        tool_calls=[
            ToolCall(
                tool_id="perform_freecad_boolean",
                arguments={"operation": "cut", "base_object": "Box", "tool_object": "Cylinder"},
            )
        ],
    )
    call = _parse("Drill a hole through the box using the cylinder.")
    assert call is not None
    assert call.tool_id == "perform_freecad_boolean"
    assert call.arguments["operation"] == "cut"


# --------------------------------------------------------------------------
# Edge manipulation: perform_freecad_edge_operation
# --------------------------------------------------------------------------


def test_describe_tool_call_edge_operation_whole_object() -> None:
    call = ToolCall(
        tool_id="perform_freecad_edge_operation",
        arguments={"operation": "fillet", "target_object": "Box", "value": 5},
    )
    description = rd.describe_tool_call(call)
    assert "Fillet" in description and "Box" in description and "every edge" in description


def test_describe_tool_call_edge_operation_face_targeted() -> None:
    call = ToolCall(
        tool_id="perform_freecad_edge_operation",
        arguments={"operation": "chamfer", "target_object": "Box", "value": 3, "face_centroid": [25, 25, 50]},
    )
    description = rd.describe_tool_call(call)
    assert "Chamfer" in description and "selected face" in description


def test_dispatch_edge_operation_rejects_unknown_operation() -> None:
    call = ToolCall(
        tool_id="perform_freecad_edge_operation",
        arguments={"operation": "bogus", "target_object": "Box", "value": 5},
    )
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "fillet, chamfer" in result.message


def test_dispatch_edge_operation_requires_target_object() -> None:
    call = ToolCall(tool_id="perform_freecad_edge_operation", arguments={"operation": "fillet", "value": 5})
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "target_object" in result.message


def test_dispatch_edge_operation_requires_numeric_value() -> None:
    call = ToolCall(
        tool_id="perform_freecad_edge_operation",
        arguments={"operation": "fillet", "target_object": "Box", "value": "not-a-number"},
    )
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "numeric value" in result.message


def test_dispatch_edge_operation_rejects_unknown_object_name() -> None:
    call = ToolCall(
        tool_id="perform_freecad_edge_operation",
        arguments={"operation": "fillet", "target_object": "NeverCreated", "value": 5},
    )
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "NeverCreated" in result.message


def test_dispatch_edge_operation_whole_object_via_registry() -> None:
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    engine = MockFreeCADEngine()
    control_plane = MockControlPlane()
    rd.dispatch_tool_call(
        ToolCall(tool_id="create_freecad_box", arguments={"length": 20, "width": 20, "height": 20, "name": "EdgeBoxA"}),
        engine,
        control_plane,
    )

    result = rd.dispatch_tool_call(
        ToolCall(
            tool_id="perform_freecad_edge_operation",
            arguments={"operation": "fillet", "target_object": "EdgeBoxA", "value": 3},
        ),
        engine,
        control_plane,
    )
    assert result.ok is True
    assert result.payload["type"] == "Part::Fillet"
    assert result.payload["face_targeted"] is False
    # The edge-op result itself registers too, so it can chain further.
    assert rd._object_registry()["Fillet"] == result.payload["path"]


def test_dispatch_edge_operation_face_targeted_via_explicit_centroid() -> None:
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    engine = MockFreeCADEngine()
    control_plane = MockControlPlane()
    rd.dispatch_tool_call(
        ToolCall(tool_id="create_freecad_box", arguments={"length": 20, "width": 20, "height": 20, "name": "EdgeBoxB"}),
        engine,
        control_plane,
    )

    result = rd.dispatch_tool_call(
        ToolCall(
            tool_id="perform_freecad_edge_operation",
            arguments={
                "operation": "chamfer",
                "target_object": "EdgeBoxB",
                "value": 2,
                "face_centroid": [10, 10, 20],
            },
        ),
        engine,
        control_plane,
    )
    assert result.ok is True
    assert result.payload["type"] == "Part::Chamfer"
    assert result.payload["face_targeted"] is True


def test_finalize_call_arguments_injects_face_centroid_for_edge_operation() -> None:
    call = ToolCall(
        tool_id="perform_freecad_edge_operation",
        arguments={"operation": "fillet", "target_object": "Box", "value": 5},
        raw_text="Fillet the edges of this face by 5mm.",
    )
    selection = {"centroid": [25.0, 25.0, 50.0], "normal": [0.0, 0.0, 1.0]}
    rd._finalize_call_arguments(call, selection)
    assert call.arguments["face_centroid"] == [25.0, 25.0, 50.0]


def test_finalize_call_arguments_edge_operation_no_selection_leaves_whole_object() -> None:
    call = ToolCall(
        tool_id="perform_freecad_edge_operation",
        arguments={"operation": "fillet", "target_object": "Box", "value": 5},
    )
    rd._finalize_call_arguments(call, None)
    assert "face_centroid" not in call.arguments


def test_finalize_call_arguments_edge_operation_respects_llm_provided_centroid() -> None:
    call = ToolCall(
        tool_id="perform_freecad_edge_operation",
        arguments={
            "operation": "fillet",
            "target_object": "Box",
            "value": 5,
            "face_centroid": [1.0, 2.0, 3.0],
        },
    )
    selection = {"centroid": [9.0, 9.0, 9.0], "normal": [0.0, 0.0, 1.0]}
    rd._finalize_call_arguments(call, selection)
    assert call.arguments["face_centroid"] == [1.0, 2.0, 3.0]


def test_parse_utterance_edge_operation_pass_through_with_active_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_llm(
        monkeypatch,
        tool_calls=[
            ToolCall(
                tool_id="perform_freecad_edge_operation",
                arguments={"operation": "fillet", "target_object": "Box", "value": 5},
            )
        ],
    )
    selection = {"centroid": [25.0, 25.0, 50.0], "normal": [0.0, 0.0, 1.0]}
    call = _parse("Fillet the edges of this face by 5mm.", selection)
    assert call is not None
    assert call.tool_id == "perform_freecad_edge_operation"
    assert call.arguments["face_centroid"] == [25.0, 25.0, 50.0]


# --------------------------------------------------------------------------
# Parametric modification: modify_freecad_parameter
# --------------------------------------------------------------------------


def test_describe_tool_call_modify_parameter() -> None:
    call = ToolCall(
        tool_id="modify_freecad_parameter",
        arguments={"target_object": "Box", "parameter_name": "Height", "new_value": 100},
    )
    description = rd.describe_tool_call(call)
    assert "Box" in description and "Height" in description and "100" in description


def test_dispatch_modify_parameter_requires_target_object() -> None:
    call = ToolCall(
        tool_id="modify_freecad_parameter",
        arguments={"parameter_name": "Height", "new_value": 100},
    )
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "target_object" in result.message


def test_dispatch_modify_parameter_rejects_unknown_object_name() -> None:
    call = ToolCall(
        tool_id="modify_freecad_parameter",
        arguments={"target_object": "NeverCreated", "parameter_name": "Height", "new_value": 100},
    )
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "NeverCreated" in result.message


def test_dispatch_modify_parameter_requires_numeric_new_value() -> None:
    call = ToolCall(
        tool_id="modify_freecad_parameter",
        arguments={"target_object": "Box", "parameter_name": "Height", "new_value": "not-a-number"},
    )
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "numeric new_value" in result.message


def test_dispatch_modify_parameter_end_to_end_via_registry() -> None:
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    engine = MockFreeCADEngine()
    control_plane = MockControlPlane()
    rd.dispatch_tool_call(
        ToolCall(tool_id="create_freecad_box", arguments={"length": 20, "width": 20, "height": 20, "name": "ModBoxA"}),
        engine,
        control_plane,
    )

    result = rd.dispatch_tool_call(
        ToolCall(
            tool_id="modify_freecad_parameter",
            arguments={"target_object": "ModBoxA", "parameter_name": "Height", "new_value": 99},
        ),
        engine,
        control_plane,
    )
    assert result.ok is True
    assert result.payload["parameter_name"] == "Height"
    assert result.payload["new_value"] == 99.0


def test_dispatch_modify_parameter_placement_accepts_bracket_string_vector() -> None:
    """The LLM naturally emits new_value as the string "[30, 20, 10]" for a
    move — this must parse into a 3-float vector, not be forced through the
    scalar float() path used by dimensional properties."""
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    engine = MockFreeCADEngine()
    control_plane = MockControlPlane()
    rd.dispatch_tool_call(
        ToolCall(tool_id="create_freecad_box", arguments={"length": 20, "width": 20, "height": 20, "name": "MoveBoxA"}),
        engine,
        control_plane,
    )

    result = rd.dispatch_tool_call(
        ToolCall(
            tool_id="modify_freecad_parameter",
            arguments={"target_object": "MoveBoxA", "parameter_name": "Placement", "new_value": "[30, 20, 10]"},
        ),
        engine,
        control_plane,
    )
    assert result.ok is True
    assert result.payload["new_value"] == [30.0, 20.0, 10.0]


def test_dispatch_modify_parameter_placement_base_alias_and_compact_string() -> None:
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    engine = MockFreeCADEngine()
    control_plane = MockControlPlane()
    rd.dispatch_tool_call(
        ToolCall(tool_id="create_freecad_box", arguments={"length": 20, "width": 20, "height": 20, "name": "MoveBoxB"}),
        engine,
        control_plane,
    )

    result = rd.dispatch_tool_call(
        ToolCall(
            tool_id="modify_freecad_parameter",
            arguments={"target_object": "MoveBoxB", "parameter_name": "Placement.Base", "new_value": "[30,20,10]"},
        ),
        engine,
        control_plane,
    )
    assert result.ok is True
    assert result.payload["new_value"] == [30.0, 20.0, 10.0]


def test_dispatch_modify_parameter_placement_rejects_bad_vector() -> None:
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    engine = MockFreeCADEngine()
    control_plane = MockControlPlane()
    rd.dispatch_tool_call(
        ToolCall(tool_id="create_freecad_box", arguments={"length": 20, "width": 20, "height": 20, "name": "MoveBoxC"}),
        engine,
        control_plane,
    )

    call = ToolCall(
        tool_id="modify_freecad_parameter",
        arguments={"target_object": "MoveBoxC", "parameter_name": "Placement", "new_value": "[30, 20]"},
    )
    result = rd.dispatch_tool_call(call, engine, control_plane)
    assert result.ok is False
    assert "3-number" in result.message


# --------------------------------------------------------------------------
# Non-mutating spatial query: get_freecad_bounding_box
# --------------------------------------------------------------------------


def test_describe_tool_call_get_bounding_box() -> None:
    call = ToolCall(tool_id="get_freecad_bounding_box", arguments={"target_object": "Box"})
    description = rd.describe_tool_call(call)
    assert "Box" in description


def test_dispatch_get_bounding_box_requires_target_object() -> None:
    call = ToolCall(tool_id="get_freecad_bounding_box", arguments={})
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "target_object" in result.message


def test_dispatch_get_bounding_box_rejects_unknown_object_name() -> None:
    call = ToolCall(tool_id="get_freecad_bounding_box", arguments={"target_object": "NeverCreated"})
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "NeverCreated" in result.message


def test_dispatch_get_bounding_box_end_to_end_via_registry() -> None:
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    engine = MockFreeCADEngine()
    control_plane = MockControlPlane()
    rd.dispatch_tool_call(
        ToolCall(tool_id="create_freecad_box", arguments={"length": 30, "width": 30, "height": 30, "name": "BBoxBoxA"}),
        engine,
        control_plane,
    )

    result = rd.dispatch_tool_call(
        ToolCall(tool_id="get_freecad_bounding_box", arguments={"target_object": "BBoxBoxA"}),
        engine,
        control_plane,
    )
    assert result.ok is True
    for key in ("x_min", "y_min", "z_min", "x_max", "y_max", "z_max"):
        assert key in result.payload


def test_get_bounding_box_never_registers_a_new_object() -> None:
    """A read shouldn't mutate the object registry — get_bounding_box's
    payload has no "name" key at all, so dispatch_tool_call's generic
    post-success registration (keyed on payload["name"]) is a no-op here."""
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    engine = MockFreeCADEngine()
    control_plane = MockControlPlane()
    rd.dispatch_tool_call(
        ToolCall(tool_id="create_freecad_box", arguments={"length": 10, "width": 10, "height": 10, "name": "BBoxBoxB"}),
        engine,
        control_plane,
    )
    before = dict(rd._object_registry())
    rd.dispatch_tool_call(
        ToolCall(tool_id="get_freecad_bounding_box", arguments={"target_object": "BBoxBoxB"}),
        engine,
        control_plane,
    )
    assert rd._object_registry() == before


# --------------------------------------------------------------------------
# 2D-to-3D sweeps: create_freecad_pipe
# --------------------------------------------------------------------------


def test_describe_tool_call_pipe_straight() -> None:
    call = ToolCall(
        tool_id="create_freecad_pipe",
        arguments={"pipe_radius": 8, "path_type": "straight", "length_or_angle": 60},
    )
    description = rd.describe_tool_call(call)
    assert "straight" in description.lower() and "8" in description and "60" in description


def test_describe_tool_call_pipe_arc() -> None:
    call = ToolCall(
        tool_id="create_freecad_pipe",
        arguments={"pipe_radius": 10, "path_type": "arc", "length_or_angle": 90},
    )
    description = rd.describe_tool_call(call)
    assert "curved" in description.lower() and "90" in description


def test_dispatch_pipe_rejects_unknown_path_type() -> None:
    call = ToolCall(
        tool_id="create_freecad_pipe",
        arguments={"pipe_radius": 10, "path_type": "bogus", "length_or_angle": 90},
    )
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "straight, arc" in result.message


def test_dispatch_pipe_requires_numeric_fields() -> None:
    call = ToolCall(
        tool_id="create_freecad_pipe",
        arguments={"pipe_radius": "nope", "path_type": "straight", "length_or_angle": 60},
    )
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "numeric" in result.message


def test_dispatch_pipe_straight_via_mock_engine() -> None:
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    call = ToolCall(
        tool_id="create_freecad_pipe",
        arguments={"pipe_radius": 5, "path_type": "straight", "length_or_angle": 40, "name": "PipeA"},
    )
    result = rd.dispatch_tool_call(call, MockFreeCADEngine(), MockControlPlane())
    assert result.ok is True
    assert result.payload["type"] == "Part::Sweep"
    assert result.payload["dimensions"]["path_type"] == "straight"


def test_dispatch_pipe_arc_with_placement_via_mock_engine() -> None:
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    call = ToolCall(
        tool_id="create_freecad_pipe",
        arguments={
            "pipe_radius": 10,
            "path_type": "arc",
            "length_or_angle": 90,
            "name": "PipeB",
            "placement_x": 5,
        },
    )
    result = rd.dispatch_tool_call(call, MockFreeCADEngine(), MockControlPlane())
    assert result.ok is True
    assert result.payload["placement"] == [5.0, 0.0, 0.0]
    assert result.payload["dimensions"]["path_type"] == "arc"


def test_parse_utterance_pipe_pass_through(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_llm(
        monkeypatch,
        tool_calls=[
            ToolCall(
                tool_id="create_freecad_pipe",
                arguments={"pipe_radius": 10, "path_type": "arc", "length_or_angle": 90},
            )
        ],
    )
    call = _parse("Create a curved pipe with a 10mm radius that bends at a 90-degree angle.")
    assert call is not None
    assert call.tool_id == "create_freecad_pipe"
    assert call.arguments["path_type"] == "arc"
    assert call.arguments["length_or_angle"] == 90


# --------------------------------------------------------------------------
# Assembly alignment: align_freecad_objects
# --------------------------------------------------------------------------


def test_describe_tool_call_align() -> None:
    call = ToolCall(
        tool_id="align_freecad_objects",
        arguments={"source_object": "Cylinder", "target_object": "Box", "alignment_type": "top_center"},
    )
    description = rd.describe_tool_call(call)
    assert "Cylinder" in description and "Box" in description and "top_center" in description


def test_dispatch_align_rejects_unknown_alignment_type() -> None:
    call = ToolCall(
        tool_id="align_freecad_objects",
        arguments={"source_object": "Cylinder", "target_object": "Box", "alignment_type": "sideways"},
    )
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "top_center" in result.message


def test_dispatch_align_requires_source_and_target_object() -> None:
    call = ToolCall(tool_id="align_freecad_objects", arguments={"alignment_type": "top_center"})
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "source_object" in result.message


def test_dispatch_align_rejects_unknown_object_names() -> None:
    call = ToolCall(
        tool_id="align_freecad_objects",
        arguments={"source_object": "NeverCreated1", "target_object": "NeverCreated2", "alignment_type": "top_center"},
    )
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "NeverCreated1" in result.message


def test_dispatch_align_end_to_end_via_registry() -> None:
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    engine = MockFreeCADEngine()
    control_plane = MockControlPlane()
    rd.dispatch_tool_call(
        ToolCall(tool_id="create_freecad_box", arguments={"length": 60, "width": 60, "height": 5, "name": "AlignBaseA"}),
        engine,
        control_plane,
    )
    rd.dispatch_tool_call(
        ToolCall(tool_id="create_freecad_cylinder", arguments={"radius": 20, "height": 40, "name": "AlignCylA"}),
        engine,
        control_plane,
    )

    result = rd.dispatch_tool_call(
        ToolCall(
            tool_id="align_freecad_objects",
            arguments={"source_object": "AlignCylA", "target_object": "AlignBaseA", "alignment_type": "top_center"},
        ),
        engine,
        control_plane,
    )
    assert result.ok is True
    assert result.payload["alignment_type"] == "top_center"
    assert len(result.payload["placement"]) == 3
    # The aligned object re-registers under the same name -> same path
    # (it moved in place, it didn't get a new identity or file).
    assert rd._object_registry()["AlignCylA"] == result.payload["path"]


# --------------------------------------------------------------------------
# Export pipelines: export_freecad_model
# --------------------------------------------------------------------------


def test_describe_tool_call_export() -> None:
    call = ToolCall(
        tool_id="export_freecad_model",
        arguments={"format": "stl", "target_objects": ["Box", "Cylinder"], "filename": "assembly"},
    )
    description = rd.describe_tool_call(call)
    assert "Box" in description and "Cylinder" in description and "STL" in description and "assembly" in description


def test_dispatch_export_rejects_unknown_format() -> None:
    call = ToolCall(
        tool_id="export_freecad_model",
        arguments={"format": "obj", "target_objects": ["Box"], "filename": "x"},
    )
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "stl, step" in result.message


def test_dispatch_export_requires_non_empty_target_objects() -> None:
    call = ToolCall(tool_id="export_freecad_model", arguments={"format": "stl", "target_objects": [], "filename": "x"})
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "target_objects" in result.message


def test_dispatch_export_requires_filename() -> None:
    call = ToolCall(tool_id="export_freecad_model", arguments={"format": "stl", "target_objects": ["Box"]})
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "filename" in result.message


def test_dispatch_export_rejects_unknown_object_name() -> None:
    call = ToolCall(
        tool_id="export_freecad_model",
        arguments={"format": "stl", "target_objects": ["NeverCreated"], "filename": "x"},
    )
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "NeverCreated" in result.message


def test_dispatch_export_end_to_end_via_registry() -> None:
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    engine = MockFreeCADEngine()
    control_plane = MockControlPlane()
    rd.dispatch_tool_call(
        ToolCall(tool_id="create_freecad_box", arguments={"length": 60, "width": 60, "height": 5, "name": "ExportBaseA"}),
        engine,
        control_plane,
    )
    rd.dispatch_tool_call(
        ToolCall(tool_id="create_freecad_cylinder", arguments={"radius": 20, "height": 40, "name": "ExportCylA"}),
        engine,
        control_plane,
    )

    result = rd.dispatch_tool_call(
        ToolCall(
            tool_id="export_freecad_model",
            arguments={"format": "stl", "target_objects": ["ExportBaseA", "ExportCylA"], "filename": "motor_mount_assembly"},
        ),
        engine,
        control_plane,
    )
    assert result.ok is True
    assert result.payload["target_count"] == 2
    # An export result has no "name" of its own — it must NOT register as
    # a fresh object in the name->path registry.
    assert "motor_mount_assembly" not in rd._object_registry()


def test_dispatch_export_step_reports_mock_limitation_not_a_crash() -> None:
    """The mock engine can't produce real STEP (B-rep) data via trimesh —
    this must surface as a clean ok:False, not an exception."""
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    engine = MockFreeCADEngine()
    control_plane = MockControlPlane()
    rd.dispatch_tool_call(
        ToolCall(tool_id="create_freecad_box", arguments={"length": 10, "width": 10, "height": 10, "name": "StepBoxA"}),
        engine,
        control_plane,
    )
    result = rd.dispatch_tool_call(
        ToolCall(
            tool_id="export_freecad_model",
            arguments={"format": "step", "target_objects": ["StepBoxA"], "filename": "x"},
        ),
        engine,
        control_plane,
    )
    assert result.ok is False
    assert "STEP" in result.message


def test_parse_utterance_align_and_export_pass_through(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_llm(
        monkeypatch,
        tool_calls=[
            ToolCall(
                tool_id="align_freecad_objects",
                arguments={"source_object": "Cylinder", "target_object": "Box", "alignment_type": "top_center"},
            )
        ],
    )
    call = _parse("Snap the cylinder to the top center of the box.")
    assert call is not None
    assert call.tool_id == "align_freecad_objects"

    _mock_llm(
        monkeypatch,
        tool_calls=[
            ToolCall(
                tool_id="export_freecad_model",
                arguments={"format": "stl", "target_objects": ["Box", "Cylinder"], "filename": "motor_mount_assembly"},
            )
        ],
    )
    call = _parse("Export both objects as an STL file named motor_mount_assembly.")
    assert call is not None
    assert call.tool_id == "export_freecad_model"


# --------------------------------------------------------------------------
# Phase A: schema registry unification — the LLM tool subset must actually
# resolve against tools.json (dana/tools/tools.json), and must line up
# exactly with the dispatch-side TOOL_HANDLERS set and is_mutating_tool's
# fail-closed schema check.
# --------------------------------------------------------------------------


def test_llm_tool_ids_all_have_handlers() -> None:
    for tool_id in rd._LLM_TOOL_IDS:
        assert tool_id in rd.TOOL_HANDLERS, tool_id


def test_llm_tools_schema_resolves_against_tools_json() -> None:
    schema = rd._llm_tools_schema()
    names = {t["function"]["name"] for t in schema}
    assert names == rd._LLM_TOOL_IDS


def test_llm_tools_schema_has_no_duplicate_or_missing_parameter_names() -> None:
    for entry in rd._llm_tools_schema():
        fn = entry["function"]
        properties = fn["parameters"]["properties"]
        for required_name in fn["parameters"]["required"]:
            assert required_name in properties, f"{fn['name']}: required {required_name!r} not in properties"


# --------------------------------------------------------------------------
# build_visual_inspection_result — BYOK: the take_canvas_screenshot
# suspend/resume path's api_key must reach analyze_cad_blueprint unchanged,
# same session["api_keys"]["openai"] dana.api.server._resolve_visual_capture
# threads through.
# --------------------------------------------------------------------------


def test_build_visual_inspection_result_threads_api_key_to_vlm(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_analyze(image_b64: str, *, api_key: str | None = None) -> str:
        captured["api_key"] = api_key
        return json.dumps({"ok": True, "entities": [], "summary": "empty"})

    monkeypatch.setattr(rd, "analyze_cad_blueprint", fake_analyze)
    image_b64 = base64.b64encode(b"fake-png-bytes").decode("ascii")

    result = rd.build_visual_inspection_result(image_b64, api_key="sk-session-key")

    assert result["ok"] is True
    assert captured["api_key"] == "sk-session-key"


def test_build_visual_inspection_result_without_api_key_passes_none(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_analyze(image_b64: str, *, api_key: str | None = None) -> str:
        captured["api_key"] = api_key
        return json.dumps({"ok": True, "entities": [], "summary": "empty"})

    monkeypatch.setattr(rd, "analyze_cad_blueprint", fake_analyze)
    image_b64 = base64.b64encode(b"fake-png-bytes").decode("ascii")

    rd.build_visual_inspection_result(image_b64)

    assert captured["api_key"] is None


# --------------------------------------------------------------------------
# Capability routing — dana.api.server's session["active_plugins"] should
# dictate which tools/system-prompt sections a turn gets, with "freecad" the
# domain name for the CAD plugin (see dana.api.server._PLUGIN_ID_TO_CAPABILITY
# for the frontend "cad" -> "freecad" normalization this module never sees).
# --------------------------------------------------------------------------


def test_llm_tools_schema_with_no_plugins_active_is_core_only() -> None:
    schema = rd._llm_tools_schema(frozenset())
    names = {t["function"]["name"] for t in schema}
    assert names == rd._CORE_TOOL_IDS


def test_llm_tools_schema_with_freecad_active_matches_core_plus_freecad() -> None:
    """"freecad" active is core + a TOKEN-BUDGET-CAPPED subset of FreeCAD's
    own (native + manifest-extended) tools — no longer the full ~26-tool set
    unconditionally. "freecad" was deliberately dropped from
    _NARROWING_EXEMPT_DOMAINS (same TPM-overflow reasoning as
    "freecad_full"), and _cap_schemas_by_token_budget is a hard backstop on
    top of that even when narrowing itself is a no-op (empty query text, as
    here). "list_directory" (os_tools) staying absent still proves domain
    gating itself is untouched — capping only ever trims WITHIN what's
    already eligible, never adds a different domain's tools."""
    import json

    import tiktoken

    schema = rd._llm_tools_schema(frozenset({"freecad"}))
    names = {t["function"]["name"] for t in schema}
    assert rd._CORE_TOOL_IDS <= names  # core always survives capping
    assert names < (rd._CORE_TOOL_IDS | rd._FREECAD_TOOL_IDS | {
        "modify_existing_freecad_document", "execute_freecad_script",
    })  # strictly fewer than the full set — capping actually did something
    assert "list_directory" not in names
    enc = tiktoken.get_encoding("cl100k_base")
    assert len(enc.encode(json.dumps(schema))) <= rd._TOOL_TOKEN_BUDGET


# --------------------------------------------------------------------------
# freecad manifest.json extension — regression for the domain-collision fix
# that used to skip dana/plugins/freecad/manifest.json's tools wholesale.
# --------------------------------------------------------------------------


def test_freecad_manifest_new_tool_ids_are_dispatchable() -> None:
    """modify_existing_freecad_document/execute_freecad_script have no
    native handler collision — they must actually be in TOOL_HANDLERS and
    in the "freecad" domain's tool-id set, not silently dropped the way the
    whole manifest used to be just because its domain name ("freecad")
    already existed as a hardcoded capability."""
    assert "modify_existing_freecad_document" in rd.TOOL_HANDLERS
    assert "execute_freecad_script" in rd.TOOL_HANDLERS
    assert "modify_existing_freecad_document" in rd._CAPABILITY_TOOL_IDS["freecad"]
    assert "execute_freecad_script" in rd._CAPABILITY_TOOL_IDS["freecad"]


def test_freecad_manifest_new_tools_are_mutating_by_default() -> None:
    """Neither declares "read_only": true in the manifest (one edits an
    existing .FCStd document, the other runs arbitrary Python inside
    FreeCADCmd) — both must be HITL-gated, same fail-closed default every
    other unannotated tool gets."""
    assert rd.is_mutating_tool("modify_existing_freecad_document") is True
    assert rd.is_mutating_tool("execute_freecad_script") is True


# --------------------------------------------------------------------------
# _wrap_plugin_handler — regression for the "'dict' object has no attribute
# 'strip'" bug: execute_freecad_script(python_script_str: str) was getting
# the WHOLE arguments dict bound to its one parameter (fn(args) called a
# freecad-style, individually-named-parameters function exactly like it
# would call a coder_plugin-style single-dict-parameter one), so
# `(python_script_str or "").strip()` ran .strip() on a dict, not a string.
# --------------------------------------------------------------------------


def test_wrap_plugin_handler_passes_whole_dict_for_coder_plugin_style_single_args_param() -> None:
    """A function whose one parameter is literally named 'args' (coder_
    plugin's convention — search_codebase/analyze_codebase/execute_code_task/
    run_verification_command all do their own args.get(...) extraction
    internally) must receive the dict itself, unpacked as nothing."""

    def fake_coder_tool(args: dict[str, Any]) -> dict[str, Any]:
        return {"received": args}

    handler = rd._wrap_plugin_handler(fake_coder_tool)
    result = handler({"regex_pattern": "def foo"}, None, None)
    assert result == {"received": {"regex_pattern": "def foo"}}


def test_wrap_plugin_handler_unpacks_dict_as_kwargs_for_named_parameters() -> None:
    """The actual bug: a function with individually-named parameters (the
    freecad manifest convention, matching each manifest 'parameters' entry
    by name) must get the dict unpacked as **kwargs, NOT passed as one
    positional dict — passing the dict positionally would silently bind the
    whole dict to the first parameter instead of raising, exactly how
    execute_freecad_script ended up calling .strip() on a dict."""

    def fake_freecad_tool(python_script_str: str) -> str:
        # JSON-encoded, mirroring the real FreeCAD-engine convention
        # (engine._ok/_error) that _wrap_plugin_handler now parses back
        # into a dict (see its own docstring/comment) — wraps whatever was
        # actually bound to python_script_str so the assertion below can
        # still tell "bound correctly as the string" apart from the
        # regression this guards against.
        return json.dumps({"bound_value": python_script_str})

    handler = rd._wrap_plugin_handler(fake_freecad_tool)
    result = handler({"python_script_str": "import FreeCAD"}, None, None)
    # regression: used to silently bind the WHOLE args dict positionally to
    # python_script_str instead of unpacking as a kwarg — python_script_str
    # would then BE {"python_script_str": "import FreeCAD"}, not the string.
    assert result == {"bound_value": "import FreeCAD"}
    assert isinstance(result["bound_value"], str)


def test_wrap_plugin_handler_unpacks_multiple_named_parameters() -> None:
    """modify_existing_document(filepath, modification_script) — the OTHER
    genuinely-new freecad manifest tool id with the same multi-named-
    parameter convention as execute_freecad_script."""

    def fake_modify_tool(filepath: str, modification_script: str) -> dict[str, str]:
        return {"filepath": filepath, "modification_script": modification_script}

    handler = rd._wrap_plugin_handler(fake_modify_tool)
    result = handler(
        {"filepath": "/tmp/model.FCStd", "modification_script": "doc.recompute()"}, None, None
    )
    assert result == {"filepath": "/tmp/model.FCStd", "modification_script": "doc.recompute()"}


def test_execute_freecad_script_end_to_end_through_tool_handlers_dispatch() -> None:
    """Full-stack regression: dispatching execute_freecad_script through the
    REAL TOOL_HANDLERS entry (as dispatch_tool_call would) with a realistic
    tool-call arguments dict must reach FreeCAD's own engine.execute_freecad_
    script with the script as a plain string, not crash on `.strip()`.
    FreeCADCmd itself is mocked out (dana.plugins.freecad.engine.
    _run_freecad_script) so this test doesn't depend on a real FreeCAD
    install."""
    from dana.plugins.freecad import engine as freecad_engine

    with patch.object(
        freecad_engine, "_run_freecad_script", return_value={"ok": True, "stdout": "done", "stderr": ""}
    ) as mock_run:
        handler = rd.TOOL_HANDLERS["execute_freecad_script"]
        result = handler({"python_script_str": "import FreeCAD as App"}, None, None)

    # engine.execute_freecad_script itself returns a JSON-encoded string
    # (engine._ok) — _wrap_plugin_handler now parses that into a real dict
    # before it reaches dispatch_tool_call (see its own docstring/comment:
    # dispatch_tool_call's payload.get("ok", True) used to crash with
    # AttributeError on the raw string, confirmed live against
    # modify_existing_freecad_document, the other tool sharing this exact
    # convention). The earlier, unrelated regression this test also guards
    # against ("'dict' object has no attribute 'strip'") crashed BEFORE ever
    # reaching this return, so a well-formed ok:true dict proves both fixes.
    assert result["ok"] is True
    # The script text actually reached FreeCADCmd as a string, not a dict.
    script_arg = mock_run.call_args.args[0]
    assert script_arg == "import FreeCAD as App"


def test_freecad_manifest_colliding_tool_ids_still_use_native_handler() -> None:
    """create_freecad_box/cylinder/extrusion are declared in BOTH the
    manifest (under a different underlying function name) and as native
    handlers — the native, tested implementation must remain authoritative;
    the manifest's version of these three specific ids is never installed
    into TOOL_HANDLERS."""
    for tool_id in ("create_freecad_box", "create_freecad_cylinder", "create_freecad_extrusion"):
        assert tool_id not in rd._PLUGIN_TOOL_SCHEMAS, f"{tool_id} should stay native, not plugin-wrapped"


def test_llm_tools_schema_default_still_matches_full_legacy_set() -> None:
    """No active_plugins arg at all (e.g. a caller not yet plugin-aware)
    must keep behaving exactly like the pre-capability-routing code —
    already covered by test_llm_tools_schema_resolves_against_tools_json,
    reasserted here for locality with the rest of this section."""
    schema = rd._llm_tools_schema()
    names = {t["function"]["name"] for t in schema}
    assert names == rd._LLM_TOOL_IDS


def test_llm_tools_schema_unknown_plugin_name_yields_core_only() -> None:
    """An active plugin id with no matching tool set (e.g. a future plugin
    react_dispatch doesn't know about yet) must degrade to core tools, not
    raise — _PLUGIN_TOOL_IDS.get(..., frozenset()) is the guard."""
    schema = rd._llm_tools_schema(frozenset({"some_future_plugin"}))
    names = {t["function"]["name"] for t in schema}
    assert names == rd._CORE_TOOL_IDS


def test_tool_ids_for_plugins_is_cached_per_combination() -> None:
    """Same active-plugin combination -> the exact same cached frozenset
    object (functools.lru_cache), not a freshly recomputed one — this is
    the O(1)-lookup-per-combination behavior the routing design relies on."""
    first = rd._tool_ids_for_plugins(frozenset({"freecad"}))
    second = rd._tool_ids_for_plugins(frozenset({"freecad"}))
    assert first is second


# --------------------------------------------------------------------------
# software_engineering domain (dana/plugins/coder_plugin) — the agent must
# be able to explicitly request this domain by name via load_capability,
# same as any other plugin-contributed capability, and its OpenAI-facing
# schema must list it too (regression: the "domain" parameter's tools.json
# enum previously omitted "software_engineering" entirely, so a model that
# tried to call load_capability(domain="software_engineering") verbatim had
# no valid enum value for it and fell back to an unrelated domain instead).
# --------------------------------------------------------------------------


def test_load_capability_unlocks_software_engineering_tools() -> None:
    result = rd._tool_load_capability({"domain": "software_engineering"}, None, None)
    assert result["ok"] is True
    assert result["domain"] == "software_engineering"
    assert set(result["unlocked_tools"]) == {
        "search_codebase", "analyze_codebase", "run_verification_command", "execute_code_task",
    }


def test_software_engineering_domain_tool_ids_include_coder_plugin_tools() -> None:
    tool_ids = rd._tool_ids_for_plugins(frozenset({"software_engineering"}))
    assert "search_codebase" in tool_ids
    assert "analyze_codebase" in tool_ids
    assert "run_verification_command" in tool_ids
    assert "execute_code_task" in tool_ids


def test_load_capability_schema_enum_lists_software_engineering() -> None:
    """The LLM-facing schema (dana/tools/tools.json, not just the internal
    _CAPABILITY_TOOL_IDS routing table) must expose "software_engineering"
    as a valid enum value — an enum a strict tool-calling provider (e.g.
    Groq) validates against, so omitting it here silently makes the domain
    uncallable by name even though the routing table already knows it."""
    from dana.tools.schema import load_tool_registry, to_openai_function_schema

    spec = load_tool_registry()["load_capability"]
    schema = to_openai_function_schema(spec)
    domain_enum = schema["function"]["parameters"]["properties"]["domain"]["enum"]
    assert "software_engineering" in domain_enum
    assert "freecad" in domain_enum


def test_core_system_prompt_routes_coding_requests_to_software_engineering() -> None:
    prompt = rd.build_system_prompt(None, active_plugins=frozenset())
    assert 'load_capability(domain="software_engineering")' in prompt


# --------------------------------------------------------------------------
# Dynamic Domain Locking — unload_capability(tool_id=...)/
# load_capability(tool_id=...) hide/unhide one SPECIFIC tool_id, independent
# of any domain boundary (create_freecad_cylinder and batch_pattern_array
# are both members of the same "freecad" domain, so no domain-level
# unload_capability call could ever suppress just one of them).
# --------------------------------------------------------------------------


def test_unload_capability_with_tool_id_hides_one_specific_tool() -> None:
    result = rd._tool_unload_capability({"tool_id": "create_freecad_cylinder"}, None, None)
    assert result["ok"] is True
    assert result["hidden_tool_id"] == "create_freecad_cylinder"
    assert "domain" not in result


def test_unload_capability_with_unknown_tool_id_errors() -> None:
    result = rd._tool_unload_capability({"tool_id": "not_a_real_tool"}, None, None)
    assert result["ok"] is False


def test_load_capability_with_tool_id_unhides_one_specific_tool() -> None:
    result = rd._tool_load_capability({"tool_id": "create_freecad_cylinder"}, None, None)
    assert result["ok"] is True
    assert result["unhidden_tool_id"] == "create_freecad_cylinder"


def test_load_capability_with_unknown_tool_id_errors() -> None:
    result = rd._tool_load_capability({"tool_id": "not_a_real_tool"}, None, None)
    assert result["ok"] is False


def test_unload_capability_tool_id_takes_priority_over_domain() -> None:
    # tool_id branch must short-circuit before the domain branch even
    # inspects args.get("domain") — passing both should still hide the tool.
    result = rd._tool_unload_capability({"tool_id": "create_freecad_cylinder", "domain": "freecad"}, None, None)
    assert result["hidden_tool_id"] == "create_freecad_cylinder"


def test_llm_tools_schema_hidden_tool_ids_removes_tool_even_though_domain_active() -> None:
    # create_freecad_cylinder and batch_pattern_array are both in the
    # "freecad" domain — hiding one must not affect the other, and the
    # domain itself must stay otherwise fully offered. active_plugins=None
    # bypasses Pillar 1's separate semantic-narrowing/token-budget behavior
    # (this module's own "legacy, not capability-aware caller" contract),
    # isolating the hidden_tool_ids subtraction this test actually targets.
    schema = rd._llm_tools_schema(None, hidden_tool_ids=frozenset({"create_freecad_cylinder"}))
    names = {s["function"]["name"] for s in schema}
    assert "create_freecad_cylinder" not in names
    assert "batch_pattern_array" in names
    assert "create_freecad_box" in names


def test_llm_tools_schema_force_include_overrides_a_hide() -> None:
    # An explicit force_include (e.g. load_specific_tool) is a deliberate
    # override and must win even over a standing hide.
    schema = rd._llm_tools_schema(
        None,
        hidden_tool_ids=frozenset({"create_freecad_cylinder"}),
        force_include=frozenset({"create_freecad_cylinder"}),
    )
    names = {s["function"]["name"] for s in schema}
    assert "create_freecad_cylinder" in names


# --------------------------------------------------------------------------
# Semantic RAG "context drop" regression — a tool load_capability just
# unlocked (e.g. execute_code_task right after
# load_capability(domain="software_engineering")) must survive Pillar 1's
# narrowing into the VERY NEXT turn's tools= schema, not just the tool
# that was actually invoked (load_capability itself).
# --------------------------------------------------------------------------


def test_sticky_tool_ids_includes_already_invoked_tool() -> None:
    messages = [
        {"role": "user", "content": "create a box then check its bbox"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "create_freecad_box", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": json.dumps({"ok": True})},
    ]
    assert rd._sticky_tool_ids_from_messages(messages) == frozenset({"create_freecad_box"})


def test_sticky_tool_ids_includes_tools_unlocked_by_load_capability() -> None:
    """Regression: load_capability itself is the only tool actually
    INVOKED in this chain — without reading its own "unlocked_tools" result
    field, the tools it unlocked (e.g. execute_code_task) would never be
    sticky and could be dropped by Pillar 1's narrowing on the very next
    turn, the one turn they're overwhelmingly likely to actually get
    called."""
    messages = [
        {"role": "user", "content": "refactor foo.py to use snake_case"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "load_capability", "arguments": '{"domain": "software_engineering"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": json.dumps(
                {
                    "ok": True,
                    "domain": "software_engineering",
                    "unlocked_tools": ["analyze_codebase", "execute_code_task"],
                    "message": "Loaded 'software_engineering' -- newly available tools: ...",
                }
            ),
        },
    ]
    sticky = rd._sticky_tool_ids_from_messages(messages)
    assert sticky == frozenset({"load_capability", "analyze_codebase", "execute_code_task"})


def test_sticky_tool_ids_ignores_unlocked_tools_from_a_different_tool_call() -> None:
    """An unrelated tool's result that happens to contain an
    "unlocked_tools" key must NOT be mistaken for a load_capability result
    — matched strictly by tool_call_id -> the assistant call's own function
    name, never by payload shape alone."""
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "check_plugin_registry", "arguments": "{}"}}
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": json.dumps({"ok": True, "unlocked_tools": ["execute_code_task"]}),
        },
    ]
    sticky = rd._sticky_tool_ids_from_messages(messages)
    assert sticky == frozenset({"check_plugin_registry"})


def test_llm_tools_schema_keeps_newly_unlocked_tools_sticky_across_narrowing() -> None:
    """End-to-end: once software_engineering is active and load_capability
    has reported unlocking execute_code_task/analyze_codebase, those two
    tool ids must survive into the tools= schema on the VERY NEXT turn even
    when that turn's query text has nothing to do with code — this is
    exactly Turn N+1 immediately after Turn N's load_capability call, where
    the bug reproduced live (the model saw no tools and exited "final")."""
    messages = [
        {"role": "user", "content": "refactor foo.py to use snake_case"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "load_capability", "arguments": '{"domain": "software_engineering"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": json.dumps(
                {
                    "ok": True,
                    "domain": "software_engineering",
                    "unlocked_tools": ["analyze_codebase", "execute_code_task"],
                }
            ),
        },
    ]
    sticky_ids = rd._sticky_tool_ids_from_messages(messages)
    schema = rd._llm_tools_schema(
        frozenset({"software_engineering"}), query="completely unrelated filler text", sticky_ids=sticky_ids
    )
    names = {t["function"]["name"] for t in schema}
    assert "analyze_codebase" in names
    assert "execute_code_task" in names


def test_load_capability_freecad_full_unlock_stays_under_budget() -> None:
    """Regression for the reported 8,966-token 413: load_capability(domain=
    "freecad_full") unlocks ~24 tools via its own unlocked_tools payload —
    _sticky_tool_ids_from_messages used to fold ALL of them into must_keep,
    which _cap_schemas_by_token_budget never trims, so the very next turn's
    schema blew far past _TOOL_TOKEN_BUDGET on its own. The unlocked set must
    now be ranked/capped the same way a large keyword-suggested domain is."""
    raw_text = "create a box, cut a circular hole through it, export as STEP"
    unlock_result = rd._tool_load_capability({"domain": "freecad_full"}, None, None)
    assert unlock_result["ok"] is True
    assert len(unlock_result["unlocked_tools"]) > rd._KEYWORD_DOMAIN_NARROW_MIN

    messages = [
        {"role": "user", "content": raw_text},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "load_capability", "arguments": '{"domain": "freecad_full"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(unlock_result)},
    ]
    sticky = rd._sticky_tool_ids_from_messages(messages, raw_text)
    # load_capability itself (already-invoked) plus a capped top-K slice of
    # the unlocked domain — never the whole ~24-tool set.
    assert len(sticky) <= rd._KEYWORD_DOMAIN_TOP_K + 1

    schema = rd._llm_tools_schema(
        frozenset({"freecad_full"}), query=raw_text, sticky_ids=sticky, force_include=sticky
    )
    schema_tokens = rd._count_tokens(json.dumps(schema))
    # Well under Groq's 8000 TPM ceiling once system prompt + conversation
    # history are added on top (~800-1000 tokens) — the old behavior alone
    # measured at 6,605+ tokens for this same scenario.
    assert schema_tokens < 4000


# --- Layer 3: Lazy Loading (search_tool_catalog / load_specific_tool) ------


def test_search_tool_catalog_returns_lightweight_matches_only() -> None:
    result = rd._tool_search_tool_catalog({"query": "read a webpage"}, None, None)
    assert result["ok"] is True
    assert result["matches"], "expected at least one match for a webpage-reading query"
    for match in result["matches"]:
        assert set(match) == {"tool_id", "description"}
        assert isinstance(match["tool_id"], str) and match["tool_id"]
        assert isinstance(match["description"], str) and match["description"]


def test_search_tool_catalog_requires_a_query() -> None:
    result = rd._tool_search_tool_catalog({}, None, None)
    assert result["ok"] is False


def test_load_specific_tool_unlocks_a_known_tool_id() -> None:
    result = rd._tool_load_specific_tool({"tool_id": "read_webpage"}, None, None)
    assert result["ok"] is True
    assert result["unlocked_tools"] == ["read_webpage"]


def test_load_specific_tool_rejects_an_unknown_tool_id() -> None:
    result = rd._tool_load_specific_tool({"tool_id": "not_a_real_tool"}, None, None)
    assert result["ok"] is False
    assert "error" in result


def test_search_tool_catalog_and_load_specific_tool_are_core_and_have_handlers() -> None:
    assert {"search_tool_catalog", "load_specific_tool"} <= rd._CORE_TOOL_IDS
    assert rd.TOOL_HANDLERS["search_tool_catalog"] is rd._tool_search_tool_catalog
    assert rd.TOOL_HANDLERS["load_specific_tool"] is rd._tool_load_specific_tool


def test_load_specific_tool_bypasses_the_capability_gate_next_turn() -> None:
    """End-to-end Layer 3 regression: load_specific_tool's own tool-result
    message must make its tool_id reachable on the VERY NEXT turn even when
    NO active domain covers it at all — "read_webpage" only otherwise lives
    under "web_tools", which is not active here."""
    messages = [
        {"role": "user", "content": "grab that one tool for me"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "load_specific_tool", "arguments": '{"tool_id": "read_webpage"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": json.dumps({"ok": True, "tool_id": "read_webpage", "unlocked_tools": ["read_webpage"]}),
        },
    ]
    force_include_ids = rd._sticky_tool_ids_from_messages(messages)
    assert "read_webpage" in force_include_ids

    # No domain active at all — "web_tools" (the only domain that owns
    # read_webpage) is absent, so without force_include this must be a
    # core-only schema and read_webpage must be unreachable.
    baseline = rd._llm_tools_schema(frozenset(), query="grab that one tool for me")
    assert "read_webpage" not in {t["function"]["name"] for t in baseline}

    schema = rd._llm_tools_schema(
        frozenset(),
        query="grab that one tool for me",
        sticky_ids=force_include_ids,
        force_include=force_include_ids,
    )
    assert "read_webpage" in {t["function"]["name"] for t in schema}


def test_keyword_suggested_domains_covers_os_tools() -> None:
    assert "os_tools" in rd._keyword_suggested_domains("please read file config.json for me")


def test_keyword_suggested_tool_ids_narrows_large_domains_but_not_small_ones() -> None:
    """Regression for the must_keep-defeats-the-budget bug: a CAD prompt used
    to make _keyword_suggested_tool_ids return the ENTIRE ~24-tool "freecad"
    domain, which _cap_schemas_by_token_budget's must_keep NEVER trims even
    when it alone exceeds _TOOL_TOKEN_BUDGET — so a keyword-matched large
    domain silently defeated the whole budget instead of just surviving
    narrowing within it. The suggested set must now be a narrowed (small)
    subset of the full domain, while a small curated domain (software_engineering,
    at/under _MIN_TOOLS_TO_NARROW) is still returned whole, unchanged."""
    cad_prompt = (
        "Use the FreeCAD tools to create a box, then cut a circular hole through "
        "it, then export the result as a STEP file."
    )
    freecad_suggested = rd._keyword_suggested_tool_ids(cad_prompt)
    assert freecad_suggested, "expected at least some freecad tools suggested"
    assert freecad_suggested < rd._CAPABILITY_TOOL_IDS["freecad"]  # proper subset, not the whole domain
    assert len(freecad_suggested) <= 8

    coder_prompt = "refactor foo.py to fix the bug in the function signature"
    coder_suggested = rd._keyword_suggested_tool_ids(coder_prompt)
    # software_engineering has 4 tools, at/under _MIN_TOOLS_TO_NARROW (8) —
    # narrowing is a no-op for it, so the whole small domain still survives.
    assert coder_suggested == rd._CAPABILITY_TOOL_IDS["software_engineering"]


def test_build_system_prompt_with_no_plugins_active_omits_engineering_rules() -> None:
    prompt = rd.build_system_prompt(None, active_plugins=frozenset())
    assert "Engineering Rules" not in prompt
    assert "CAD co-pilot for FreeCAD" not in prompt
    assert "general-purpose AI desktop assistant" in prompt


def test_build_system_prompt_with_freecad_active_includes_engineering_rules() -> None:
    prompt = rd.build_system_prompt(None, active_plugins=frozenset({"freecad"}))
    assert "## Engineering Rules" in prompt
    assert "CAD co-pilot for FreeCAD" in prompt


def test_build_system_prompt_default_matches_freecad_active_exactly() -> None:
    """No active_plugins arg (legacy callers, e.g. parse_utterance) must
    produce the IDENTICAL prompt "freecad" active does — same selection
    argument, only the plugin-set argument differs."""
    selection = {"centroid": [1.0, 2.0, 3.0], "normal": [0.0, 1.0, 0.0]}
    default_prompt = rd.build_system_prompt(selection)
    freecad_prompt = rd.build_system_prompt(selection, active_plugins=frozenset({"freecad"}))
    assert default_prompt == freecad_prompt


def test_next_react_turn_blocks_freecad_tool_when_plugin_not_active(monkeypatch: pytest.MonkeyPatch) -> None:
    """Defensive guard: even if the (mocked) model proposes a FreeCAD tool,
    next_react_turn must not hand back a "tool_call" turn for it when
    "freecad" isn't in the active-plugin set — same fallback as an unknown
    tool_id, so a stale/hallucinated call from earlier history can't dispatch
    a plugin the frontend doesn't even have open anymore."""
    fake = _mock_llm(
        monkeypatch,
        tool_calls=[ToolCall(tool_id="create_freecad_box", arguments={"length": 40})],
        content="I'll build that.",
    )
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "build a box"}]

    turn = asyncio.run(rd.next_react_turn(messages, active_plugins=frozenset()))

    assert turn.kind == "final"
    assert turn.content == "I'll build that."
    # And the model was never even offered the tool in the first place:
    assert "create_freecad_box" not in {t["function"]["name"] for t in fake.calls[0]["tools"]}


def test_next_react_turn_allows_freecad_tool_when_plugin_active(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_llm(monkeypatch, tool_calls=[ToolCall(tool_id="create_freecad_box", arguments={"length": 40})])
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "build a box"}]

    turn = asyncio.run(rd.next_react_turn(messages, active_plugins=frozenset({"freecad"})))

    assert turn.kind == "tool_call"
    assert turn.call.tool_id == "create_freecad_box"


def test_next_react_turn_core_tool_allowed_with_no_plugins_active(monkeypatch: pytest.MonkeyPatch) -> None:
    """system_state (core) must still dispatch normally even with an empty
    active-plugin set — capability routing must never block core tools."""
    _mock_llm(monkeypatch, tool_calls=[ToolCall(tool_id="system_state", arguments={})])
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "status?"}]

    turn = asyncio.run(rd.next_react_turn(messages, active_plugins=frozenset()))

    assert turn.kind == "tool_call"
    assert turn.call.tool_id == "system_state"


# --------------------------------------------------------------------------
# _keyword_suggested_domains — Turn-0 pre-warming so a plain-chat session
# with NO plugin active doesn't have to waste a whole turn on
# load_capability before search_codebase/analyze_codebase/create_freecad_*/
# search_web/analyze_workspace_image are even offered.
# --------------------------------------------------------------------------


def test_keyword_suggested_domains_detects_software_engineering_intent() -> None:
    assert rd._keyword_suggested_domains("please refactor this function in utils.py") == frozenset(
        {"software_engineering"}
    )
    assert rd._keyword_suggested_domains("can you run pytest on this?") == frozenset({"software_engineering"})
    assert rd._keyword_suggested_domains("there's a traceback I need help with") == frozenset(
        {"software_engineering"}
    )


def test_keyword_suggested_domains_detects_cad_intent() -> None:
    assert rd._keyword_suggested_domains("create a cylinder with radius 10") == frozenset({"freecad"})


def test_keyword_suggested_domains_detects_web_intent() -> None:
    assert rd._keyword_suggested_domains("please search the web for this") == frozenset({"web_tools"})


def test_keyword_suggested_domains_detects_vision_intent() -> None:
    assert rd._keyword_suggested_domains("take a screenshot and tell me what's there") == frozenset(
        {"vision_tools"}
    )


def test_keyword_suggested_domains_can_suggest_multiple_domains_at_once() -> None:
    result = rd._keyword_suggested_domains("take a screenshot of my code and fix the bug in it")
    assert result == frozenset({"vision_tools", "software_engineering"})


def test_keyword_suggested_domains_empty_for_ordinary_chat() -> None:
    assert rd._keyword_suggested_domains("how's the weather today?") == frozenset()
    assert rd._keyword_suggested_domains("") == frozenset()


def test_next_react_turn_offers_coder_tools_on_turn_zero_without_load_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact Turn-0 overhead this pre-warming eliminates: a code-related
    prompt must already offer search_codebase/analyze_codebase in THIS
    turn's schema — active_plugins starts EMPTY (nothing manually activated,
    no prior load_capability call in this messages history) — so the model
    can call search_codebase directly instead of spending Turn 0 on
    load_capability(domain="software_engineering") first."""
    fake = _mock_llm(monkeypatch, tool_calls=[ToolCall(tool_id="search_codebase", arguments={"regex_pattern": "def foo"})])
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "refactor the foo function to fix the bug"},
    ]

    turn = asyncio.run(
        rd.next_react_turn(
            messages, raw_text="refactor the foo function to fix the bug", active_plugins=frozenset()
        )
    )

    assert turn.kind == "tool_call"
    assert turn.call.tool_id == "search_codebase"
    offered = {t["function"]["name"] for t in fake.calls[0]["tools"]}
    assert "search_codebase" in offered
    assert "analyze_codebase" in offered
    # load_capability must still be present too — a backwards-compatible
    # fallback for whatever the keyword heuristic doesn't catch, never
    # replaced by pre-warming.
    assert "load_capability" in offered


def test_next_react_turn_does_not_prewarm_software_engineering_for_unrelated_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _mock_llm(monkeypatch, content="Partly cloudy, 72 degrees.")
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "how's the weather today?"}]

    turn = asyncio.run(
        rd.next_react_turn(messages, raw_text="how's the weather today?", active_plugins=frozenset())
    )

    assert turn.kind == "final"
    offered = {t["function"]["name"] for t in fake.calls[0]["tools"]}
    assert "search_codebase" not in offered
    assert "analyze_codebase" not in offered


# --------------------------------------------------------------------------
# Turn-level timeout ceiling / provider attribution. Cloud tool-calling now
# calls a single resolved provider directly (OpenRouter by default), whose
# own server-side ``models`` fallback array retries a 429/5xx against the
# next model upstream in milliseconds, so openai_tool_bridge no longer
# sleeps out a Groq TPM retry-after hint — a 429/5xx fails this call fast
# instead. This timeout only needs to bound
# a genuinely stalling connection now.
# --------------------------------------------------------------------------


def test_local_tool_call_timeout_is_positive_and_bounded() -> None:
    # Raised to 600s to accommodate a heavier local model (e.g. a
    # 14B-parameter qwen2.5-coder) run for exact-artifact accuracy over
    # speed, which can legitimately take 2-3 minutes per turn — this is now
    # a genuine "the connection is dead" backstop, not a UX-latency budget.
    assert 0 < rd._LOCAL_TOOL_CALL_TIMEOUT_SEC <= 900.0


def test_timeout_apology_text_attributes_local_provider_correctly() -> None:
    text = rd._timeout_apology_text("ollama")
    assert "local model" in text
    assert "Ollama" in text


def test_timeout_apology_text_attributes_cloud_provider_correctly() -> None:
    text = rd._timeout_apology_text("groq")
    assert "cloud model" in text
    assert "groq" in text
    assert "local" not in text.lower()


def test_call_llm_once_timeout_attributes_apology_to_the_actual_cloud_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end regression: a primary call that hangs past the (shortened,
    for this test) timeout must fall back to an apology correctly blaming
    the ACTUAL provider that timed out (groq), never unconditionally "the
    local model" — the exact user-facing bug this task fixes."""
    monkeypatch.setattr(rd, "_LOCAL_TOOL_CALL_TIMEOUT_SEC", 0.05)
    monkeypatch.setattr(rd, "tool_calling_provider", lambda: "groq")

    class _HangingProvider:
        """No .complete method on purpose: the fallback's own tiny local-
        apology attempt (ModelProvider(local_model=..., api_keys=...).
        complete(...)) hits AttributeError immediately, caught by its own
        broad except, deterministically reaching the final hardcoded
        _timeout_apology_text(...) path this test actually asserts on."""

        def __init__(self, **_kwargs: Any) -> None:
            pass

        def complete_with_tool_calls(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            time.sleep(0.3)
            return {"content": "too late", "tool_calls": [], "provider": "groq"}

    monkeypatch.setattr(rd, "ModelProvider", _HangingProvider)

    result = asyncio.run(rd._call_llm_once([{"role": "user", "content": "hi"}]))
    assert "cloud model (groq)" in result["content"]
    assert "local" not in result["content"].lower()


# Plan-and-Execute Gatekeeper (Phase 6). Each test below runs under its OWN
# isolated session_id — never the shared (default) ambient one the
# `_plan_gate_open` autouse fixture pre-opens for every other test in this
# module — so these can freely exercise the CLOSED-gate state without
# fighting that fixture or leaking a closed gate into later tests.


def test_plan_gatekeeper_blocks_geometry_tool_without_plan() -> None:
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    set_session_id("gatekeeper-blocks-without-plan")
    engine = MockFreeCADEngine()
    control_plane = MockControlPlane()

    result = rd.dispatch_tool_call(
        ToolCall(tool_id="create_freecad_box", arguments={"name": "GatekeeperBoxA"}),
        engine,
        control_plane,
    )
    assert result.ok is False
    assert "create_plan" in result.message
    # Never reached the engine at all — no object was registered.
    assert "GatekeeperBoxA" not in rd._object_registry()


def test_plan_gatekeeper_unblocks_after_create_plan() -> None:
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    set_session_id("gatekeeper-unblocks-after-plan")
    engine = MockFreeCADEngine()
    control_plane = MockControlPlane()

    assert rd._get_has_plan() is False
    plan_result = rd.dispatch_tool_call(
        ToolCall(
            tool_id="create_plan",
            arguments={"objective": "Bracket with a mounting hole", "tasks": ["Create box", "Create cylinder", "Cut"]},
        ),
        engine,
        control_plane,
    )
    assert plan_result.ok is True
    assert rd._get_has_plan() is True

    result = rd.dispatch_tool_call(
        ToolCall(tool_id="create_freecad_box", arguments={"name": "GatekeeperBoxB"}),
        engine,
        control_plane,
    )
    assert result.ok is True
    assert result.payload["name"] == "GatekeeperBoxB"


def test_plan_gatekeeper_does_not_block_non_restricted_tools() -> None:
    """The gate only covers _RESTRICTED_GEOMETRY_TOOLS — an ordinary
    non-geometry tool (here, a read-only introspection call) must dispatch
    normally even with no plan at all."""
    set_session_id("gatekeeper-non-restricted-tool")
    result = rd.dispatch_tool_call(
        ToolCall(tool_id="check_plugin_registry", arguments={}),
        engine=None,
        control_plane=None,
    )
    assert result.ok is True


def test_build_system_prompt_omits_anchor_when_no_plan() -> None:
    set_session_id("gatekeeper-prompt-no-plan")
    prompt = rd.build_system_prompt(None, session_id="gatekeeper-prompt-no-plan")
    assert "ACTIVE PLAN" not in prompt


def test_build_system_prompt_includes_active_plan_anchor_after_create_plan() -> None:
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    set_session_id("gatekeeper-prompt-with-plan")
    engine = MockFreeCADEngine()
    control_plane = MockControlPlane()
    rd.dispatch_tool_call(
        ToolCall(
            tool_id="create_plan",
            arguments={"objective": "Motor mount", "tasks": ["Create MotorBracket", "Cut Mounting_Hole"]},
        ),
        engine,
        control_plane,
    )

    prompt = rd.build_system_prompt(None, session_id="gatekeeper-prompt-with-plan")
    assert "=== ACTIVE PLAN ===" in prompt
    assert "Motor mount" in prompt
    assert "Create MotorBracket" in prompt
    # The anchor is the literal LAST thing appended to the prompt.
    assert prompt.rstrip().endswith("Stick to this naming convention and topological strategy.")


# ---------------------------------------------------------------------------
# Plan-and-Execute Finite State Machine (Phases 1-4)
# ---------------------------------------------------------------------------


def test_fsm_transition_table_matches_spec() -> None:
    assert rd._fsm_transition("planning", "create_plan_succeeded") == "executing"
    assert rd._fsm_transition("executing", "expected_tool_dispatched") == "validating"
    assert rd._fsm_transition("validating", "tool_ok") == "executing"
    assert rd._fsm_transition("validating", "tool_failed") == "executing"
    assert rd._fsm_transition("done", "create_plan_succeeded") == "executing"


def test_fsm_transition_unknown_pair_is_a_noop() -> None:
    """An out-of-sequence or already-resolved (state, event) pair must never
    raise -- same defensive posture task_board.mark_task_completed already
    takes for an unknown task_id."""
    assert rd._fsm_transition("executing", "some_unknown_event") == "executing"
    assert rd._fsm_transition("planning", "tool_ok") == "planning"


def test_create_plan_seeds_expected_tool_ids_and_executing_state() -> None:
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    set_session_id("fsm-create-plan-seeds-state")
    engine = MockFreeCADEngine()
    control_plane = MockControlPlane()
    rd.dispatch_tool_call(
        ToolCall(
            tool_id="create_plan",
            arguments={"objective": "Bracket", "tasks": ["Create a box", "Cut a hole"]},
        ),
        engine,
        control_plane,
    )
    assert rd._get_fsm_state("fsm-create-plan-seeds-state") == "executing"
    active = rd._active_task("fsm-create-plan-seeds-state")
    assert active is not None and active["id"] == 1
    entry = rd._PLAN_STATE_REGISTRY["fsm-create-plan-seeds-state"]
    assert len(entry["tasks"]) == 2
    # Every task gets a real (possibly empty) expected_tool_ids frozenset --
    # the FIELD always exists, regardless of whether the semantic mapping
    # found a confident match for this particular description.
    for task in entry["tasks"]:
        assert isinstance(task["expected_tool_ids"], frozenset)


def _seed_two_task_plan(
    session_id: str, tool_for_task_1: frozenset[str], tool_for_task_2: frozenset[str]
) -> None:
    """Deterministic FSM-mechanics test setup, bypassing the real semantic
    narrowing call (_tool_create_plan's own narrow_tool_ids_by_query) so
    these tests exercise ONLY the FSM's own transition/gating logic against
    a KNOWN mapping, not the embedding backend's actual ranking."""
    set_session_id(session_id)
    rd._set_has_plan(True, "Objective: test\n1. task one\n2. task two")
    rd._set_session_plan_tasks(
        "test objective",
        [
            {"id": 1, "description": "task one", "status": "active", "expected_tool_ids": tool_for_task_1},
            {"id": 2, "description": "task two", "status": "pending", "expected_tool_ids": tool_for_task_2},
        ],
        session_id=session_id,
    )
    rd._PLAN_STATE_REGISTRY[session_id]["fsm_state"] = "executing"


def test_dispatch_auto_advances_on_expected_tool_success() -> None:
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    sid = "fsm-auto-advance"
    _seed_two_task_plan(sid, frozenset({"create_freecad_box"}), frozenset({"perform_freecad_boolean"}))
    engine = MockFreeCADEngine()
    control_plane = MockControlPlane()

    result = rd.dispatch_tool_call(
        ToolCall(tool_id="create_freecad_box", arguments={"name": "AutoAdvanceBox"}),
        engine,
        control_plane,
    )
    assert result.ok is True
    assert "Task 1 complete" in result.payload.get("message", "")
    entry = rd._PLAN_STATE_REGISTRY[sid]
    assert entry["fsm_state"] == "executing"  # resolved straight back to executing (next task active)
    tasks_by_id = {t["id"]: t for t in entry["tasks"]}
    assert tasks_by_id[1]["status"] == "completed"
    assert tasks_by_id[2]["status"] == "active"


def test_dispatch_advances_to_done_after_last_task() -> None:
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    sid = "fsm-advance-to-done"
    set_session_id(sid)
    rd._set_has_plan(True, "Objective: test\n1. only task")
    rd._set_session_plan_tasks(
        "test objective",
        [{"id": 1, "description": "only task", "status": "active", "expected_tool_ids": frozenset({"create_freecad_box"})}],
        session_id=sid,
    )
    rd._PLAN_STATE_REGISTRY[sid]["fsm_state"] = "executing"
    engine = MockFreeCADEngine()
    control_plane = MockControlPlane()

    result = rd.dispatch_tool_call(
        ToolCall(tool_id="create_freecad_box", arguments={"name": "LastBox"}),
        engine,
        control_plane,
    )
    assert result.ok is True
    assert rd._PLAN_STATE_REGISTRY[sid]["fsm_state"] == "done"


def test_dispatch_does_not_advance_on_failed_expected_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    sid = "fsm-no-advance-on-failure"
    _seed_two_task_plan(sid, frozenset({"create_freecad_box"}), frozenset({"perform_freecad_boolean"}))
    monkeypatch.setitem(
        rd.TOOL_HANDLERS, "create_freecad_box", lambda args, engine, cp: {"ok": False, "error": "boom"}
    )
    engine = MockFreeCADEngine()
    control_plane = MockControlPlane()

    result = rd.dispatch_tool_call(
        ToolCall(tool_id="create_freecad_box", arguments={"name": "WillFail"}),
        engine,
        control_plane,
    )
    assert result.ok is False
    entry = rd._PLAN_STATE_REGISTRY[sid]
    assert entry["fsm_state"] == "executing"  # never left "executing" for the failed dispatch
    tasks_by_id = {t["id"]: t for t in entry["tasks"]}
    assert tasks_by_id[1]["status"] == "active"  # unchanged -- the model retries the SAME task


def test_dispatch_hard_blocks_tool_belonging_to_a_different_pending_task() -> None:
    """Hard-Blocking Policy, case 2: `perform_freecad_boolean` is task 2's
    own expected tool, not task 1's (the active one) -- positive evidence
    of skipping ahead, so this must be refused with an actionable reason."""
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    sid = "fsm-hard-block-out-of-order"
    _seed_two_task_plan(sid, frozenset({"create_freecad_box"}), frozenset({"perform_freecad_boolean"}))
    engine = MockFreeCADEngine()
    control_plane = MockControlPlane()

    result = rd.dispatch_tool_call(
        ToolCall(tool_id="perform_freecad_boolean", arguments={"operation": "cut", "base_object": "A", "tool_object": "B"}),
        engine,
        control_plane,
    )
    assert result.ok is False
    assert "out of order" in result.message.lower()
    assert "task 2" in result.message.lower()
    # Never reached the engine -- the active task's own status is untouched.
    entry = rd._PLAN_STATE_REGISTRY[sid]
    tasks_by_id = {t["id"]: t for t in entry["tasks"]}
    assert tasks_by_id[1]["status"] == "active"
    assert tasks_by_id[2]["status"] == "pending"


def test_dispatch_allows_unmapped_tool_without_advancing() -> None:
    """Hard-Blocking Policy, case 3: a geometry tool that belongs to NO task
    in the plan (a k=3 mapping gap, not evidence of skipping ahead) must be
    ALLOWED through -- refusing on pure absence of evidence is exactly how
    an unrecoverable stall happens."""
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    sid = "fsm-allow-unmapped-tool"
    _seed_two_task_plan(sid, frozenset({"create_freecad_box"}), frozenset())
    engine = MockFreeCADEngine()
    control_plane = MockControlPlane()

    result = rd.dispatch_tool_call(
        ToolCall(tool_id="create_freecad_cylinder", arguments={"name": "UnmappedCylinder"}),
        engine,
        control_plane,
    )
    assert result.ok is True
    # Allowed, but NOT auto-advanced -- create_freecad_cylinder isn't task 1's
    # expected tool, so there's nothing to confirm the task actually finished.
    entry = rd._PLAN_STATE_REGISTRY[sid]
    tasks_by_id = {t["id"]: t for t in entry["tasks"]}
    assert tasks_by_id[1]["status"] == "active"


def test_dispatch_parks_validating_for_task_with_no_expected_tools() -> None:
    """A task the k=3 mapping found nothing tool-shaped for (empty
    expected_tool_ids) cannot auto-advance -- ANY successful geometry tool
    while it's active parks the FSM in "validating" instead, awaiting an
    explicit mark_task_completed (see _build_validator_prompt)."""
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    sid = "fsm-park-validating"
    set_session_id(sid)
    # mark_task_completed's manual-override path (exercised below) validates
    # against task_board's own GLOBAL plan, not this session-scoped registry
    # -- seed both, mirroring exactly what the real _tool_create_plan does.
    rd._tb_create_plan("test objective", ["inspect the assembly"])
    rd._set_has_plan(True, "Objective: test\n1. inspect the assembly")
    rd._set_session_plan_tasks(
        "test objective",
        [{"id": 1, "description": "inspect the assembly", "status": "active", "expected_tool_ids": frozenset()}],
        session_id=sid,
    )
    rd._PLAN_STATE_REGISTRY[sid]["fsm_state"] = "executing"
    engine = MockFreeCADEngine()
    control_plane = MockControlPlane()

    result = rd.dispatch_tool_call(
        ToolCall(tool_id="create_freecad_box", arguments={"name": "InspectionAid"}),
        engine,
        control_plane,
    )
    assert result.ok is True
    assert "no fixed tool signature" in result.payload.get("message", "")
    entry = rd._PLAN_STATE_REGISTRY[sid]
    assert entry["fsm_state"] == "validating"
    assert entry["last_validation"] == {"task_id": 1, "tool_id": "create_freecad_box", "ok": True}

    # mark_task_completed is the manual override that resolves it.
    resolve_result = rd.dispatch_tool_call(
        ToolCall(tool_id="mark_task_completed", arguments={"task_id": 1}), engine, control_plane
    )
    assert resolve_result.ok is True
    assert rd._PLAN_STATE_REGISTRY[sid]["fsm_state"] == "done"


def test_build_system_prompt_planning_phase_excludes_geometry_rulebook() -> None:
    set_session_id("fsm-prompt-planning")
    prompt = rd.build_system_prompt(None, active_plugins=frozenset({"freecad_essential"}), session_id="fsm-prompt-planning")
    assert "=== PLANNING PHASE ===" in prompt
    assert "## Engineering Rules" not in prompt
    assert "create_freecad_box" not in prompt
    assert "CAD co-pilot for FreeCAD" not in prompt
    assert "create_plan" in prompt


def test_build_system_prompt_executing_phase_shows_only_current_task() -> None:
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    sid = "fsm-prompt-executing"
    set_session_id(sid)
    engine = MockFreeCADEngine()
    control_plane = MockControlPlane()
    rd.dispatch_tool_call(
        ToolCall(
            tool_id="create_plan",
            arguments={"objective": "Widget", "tasks": ["Create a box", "Create a cylinder", "Cut a hole"]},
        ),
        engine,
        control_plane,
    )
    prompt = rd.build_system_prompt(None, active_plugins=frozenset({"freecad_essential"}), session_id=sid)
    assert "=== EXECUTING PHASE ===" in prompt
    assert "CURRENT ACTIVE TASK 1" in prompt
    assert "2 task(s) remain after this one" in prompt
    assert "=== PLANNING PHASE ===" not in prompt


def test_next_react_turn_hard_restricts_tools_during_planning(monkeypatch: pytest.MonkeyPatch) -> None:
    """Phase 2/3: with no plan yet, the schema offered to the model must
    contain ONLY create_plan/search_tool_catalog/check_plugin_registry/
    core tools -- never a single geometry tool, structurally."""
    sid = "fsm-hard-restrict-planning"
    set_session_id(sid)
    fake = _mock_llm(
        monkeypatch,
        tool_calls=[ToolCall(tool_id="create_plan", arguments={"objective": "x", "tasks": ["y"]})],
    )
    asyncio.run(
        rd.next_react_turn(
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "build a box"}],
            None,
            raw_text="build a box",
            active_plugins=frozenset({"freecad_essential"}),
            session_id=sid,
        )
    )
    offered_tool_ids = {t["function"]["name"] for t in fake.calls[-1]["tools"]}
    assert "create_freecad_box" not in offered_tool_ids
    assert "create_plan" in offered_tool_ids
    assert "search_tool_catalog" in offered_tool_ids
    assert "check_plugin_registry" in offered_tool_ids
