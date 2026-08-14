"""Tests for dana.core.openai_tool_bridge's OpenAI-wire tool-calling bridge,
including its fallback parser for a real, observed Ollama quirk: some local
models (verified live against qwen2.5-coder:7b) emit a tool call as plain
JSON text in ``message.content`` instead of populating the OpenAI
``message.tool_calls`` array.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from dana.core import openai_tool_bridge as bridge


class _FakeResponse:
    def __init__(self, body: dict[str, Any]) -> None:
        self._body = json.dumps(body).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


def _mock_response(monkeypatch: pytest.MonkeyPatch, message: dict[str, Any]) -> None:
    body = {"choices": [{"message": message}]}
    monkeypatch.setattr(bridge.urllib.request, "urlopen", lambda *_a, **_k: _FakeResponse(body))


# --------------------------------------------------------------------------
# _fallback_tool_calls_from_content — the helper itself
# --------------------------------------------------------------------------


def test_fallback_parses_plain_name_arguments_object() -> None:
    content = '{"name": "create_freecad_box", "arguments": {"length": 60, "width": 40, "height": 20}}'
    calls = bridge._fallback_tool_calls_from_content(content)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "create_freecad_box"
    assert calls[0]["function"]["arguments"] == {"length": 60, "width": 40, "height": 20}


def test_fallback_parses_nested_function_shape() -> None:
    content = '{"function": {"name": "system_state", "arguments": {}}}'
    calls = bridge._fallback_tool_calls_from_content(content)
    assert calls[0]["function"]["name"] == "system_state"


def test_fallback_strips_markdown_json_fence() -> None:
    content = '```json\n{"name": "manipulate_camera", "arguments": {"preset": "iso"}}\n```'
    calls = bridge._fallback_tool_calls_from_content(content)
    assert calls[0]["function"]["name"] == "manipulate_camera"


def test_fallback_ignores_plain_prose() -> None:
    assert bridge._fallback_tool_calls_from_content("Sure, happy to help!") == []


def test_fallback_ignores_empty_or_none() -> None:
    assert bridge._fallback_tool_calls_from_content("") == []
    assert bridge._fallback_tool_calls_from_content(None) == []


def test_fallback_ignores_json_without_a_name() -> None:
    assert bridge._fallback_tool_calls_from_content('{"arguments": {"a": 1}}') == []


# --------------------------------------------------------------------------
# complete_openai_with_tools — end-to-end through the fake HTTP layer
# --------------------------------------------------------------------------


def test_native_tool_calls_pass_through_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    native_calls = [{"type": "function", "function": {"name": "system_state", "arguments": "{}"}}]
    _mock_response(monkeypatch, {"content": None, "tool_calls": native_calls})

    result = bridge.complete_openai_with_tools(
        [{"role": "user", "content": "status"}],
        api_key="k",
        base_url="http://127.0.0.1:11434/v1",
        model="qwen2.5-coder:7b",
        tools=[{"type": "function", "function": {"name": "system_state", "parameters": {}}}],
    )
    assert result["tool_calls"] == native_calls


def test_falls_back_to_content_json_when_tool_calls_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression test for the exact bug found live-testing the LLM ReAct
    unification: qwen2.5-coder:7b via Ollama's OpenAI-compat surface
    returned the tool call as JSON text in `content` with an empty
    `tool_calls` array, silently discarding a correct tool decision."""
    content = '{"name": "create_freecad_box", "arguments": {"height": 50, "length": 100, "width": 100}}'
    _mock_response(monkeypatch, {"content": content, "tool_calls": []})

    result = bridge.complete_openai_with_tools(
        [{"role": "user", "content": "Create a parametric 100x100x50mm box"}],
        api_key="ollama",
        base_url="http://127.0.0.1:11434/v1",
        model="qwen2.5-coder:7b",
        tools=[{"type": "function", "function": {"name": "create_freecad_box", "parameters": {}}}],
    )
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["function"]["name"] == "create_freecad_box"
    assert result["content"] == ""  # it was a function call, not a reply meant for the user


def test_no_fallback_attempted_when_tools_not_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    """A plain-text completion (no tools=...) must never have its content
    reinterpreted as a tool call, even if it happens to look JSON-shaped."""
    content = '{"name": "coincidence", "arguments": {}}'
    _mock_response(monkeypatch, {"content": content, "tool_calls": []})

    result = bridge.complete_openai_with_tools(
        [{"role": "user", "content": "hi"}],
        api_key="k",
        base_url="http://127.0.0.1:11434/v1",
        model="qwen2.5-coder:7b",
    )
    assert result["tool_calls"] == []
    assert result["content"] == content


def test_plain_prose_content_with_tools_requested_stays_empty_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_response(monkeypatch, {"content": "Sure, I can help with that!", "tool_calls": []})

    result = bridge.complete_openai_with_tools(
        [{"role": "user", "content": "thanks"}],
        api_key="k",
        base_url="http://127.0.0.1:11434/v1",
        model="qwen2.5-coder:7b",
        tools=[{"type": "function", "function": {"name": "system_state", "parameters": {}}}],
    )
    assert result["tool_calls"] == []
    assert result["content"] == "Sure, I can help with that!"
