"""Tests for dana.core.openai_tool_bridge's OpenAI-wire tool-calling bridge,
including its fallback parser for a real, observed Ollama quirk: some local
models (verified live against qwen2.5-coder:7b) emit a tool call as plain
JSON text in ``message.content`` instead of populating the OpenAI
``message.tool_calls`` array.
"""

from __future__ import annotations

import io
import json
import urllib.error
from typing import Any

import pytest

from dana.core import openai_tool_bridge as bridge


class _FakeResponse:
    """Mimics the SSE-chunked ``requests``/``urllib`` streaming response
    ``complete_openai_with_tools`` now iterates line-by-line (it used to
    read one blocking JSON body — see the module docstring's note on the
    P2 rescue-plan streaming rewrite)."""

    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines

    def __iter__(self):
        return iter(self._lines)

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


def _sse_chunk(delta: dict[str, Any]) -> bytes:
    return ("data: " + json.dumps({"choices": [{"delta": delta}]}) + "\n").encode("utf-8")


def _sse_lines_for_message(message: dict[str, Any]) -> list[bytes]:
    """Reshapes a one-shot ``{"content", "tool_calls"}`` message (the old
    non-streaming fixture shape every test below already uses) into the
    streamed ``delta`` chunks a real OpenAI-compatible server would emit —
    one content chunk (if any) plus one indexed tool-call chunk per call,
    terminated by the standard ``[DONE]`` sentinel.
    """
    lines: list[bytes] = []
    content = message.get("content")
    if content:
        lines.append(_sse_chunk({"content": content}))
    for i, call in enumerate(message.get("tool_calls") or []):
        fn = (call or {}).get("function") or {}
        lines.append(
            _sse_chunk({"tool_calls": [{"index": i, "function": {"name": fn.get("name"), "arguments": fn.get("arguments")}}]})
        )
    lines.append(b"data: [DONE]\n")
    return lines


def _mock_response(monkeypatch: pytest.MonkeyPatch, message: dict[str, Any]) -> None:
    lines = _sse_lines_for_message(message)
    monkeypatch.setattr(bridge.urllib.request, "urlopen", lambda *_a, **_k: _FakeResponse(lines))


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


def test_fallback_recovers_json_embedded_after_prose() -> None:
    """Real, observed live: qwen2.5-coder:7b answering a spatial-math
    question with its own explanation THEN a JSON tool call, instead of
    emitting only JSON — a further-degraded variant of the plain-JSON quirk
    the other fallback tests cover."""
    content = (
        "The top-center coordinate of the box is at (15, 15, 30). Now, let's "
        "create a 15mm radius cylinder perfectly resting on top of it.\n\n"
        '{"name": "create_freecad_cylinder", "arguments": '
        '{"height": 1, "name": "Cylinder", "placement_x": 15, "placement_y": 15, '
        '"placement_z": 30, "radius": 15}}'
    )
    calls = bridge._fallback_tool_calls_from_content(content)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "create_freecad_cylinder"
    assert calls[0]["function"]["arguments"]["placement_z"] == 30


def test_fallback_recovers_first_of_multiple_embedded_json_blobs() -> None:
    """A model that plans several steps at once and dumps each as its own
    JSON blob — callers only ever take the first (see next_react_turn's
    "one tool per turn" convention), but the fallback itself should recover
    all of them rather than silently dropping everything after the first."""
    content = (
        '{"name": "create_freecad_box", "arguments": {"height": 30}}\n\n'
        '{"name": "get_freecad_bounding_box", "arguments": {"target_object": "Box"}}'
    )
    calls = bridge._fallback_tool_calls_from_content(content)
    assert [c["function"]["name"] for c in calls] == ["create_freecad_box", "get_freecad_bounding_box"]


def test_fallback_prose_with_parenthesized_numbers_yields_nothing_spurious() -> None:
    """"(15, 15, 30)"-style prose must not be mistaken for a JSON array —
    only real {..}/[..] JSON literals should ever be scanned."""
    assert bridge._fallback_tool_calls_from_content("The coordinate is (15, 15, 30).") == []


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
    # Compares just the function name/arguments, not the whole dict — a
    # streamed reconstruction adds its own "id"/"type" bookkeeping fields
    # (see complete_openai_with_tools' tool_call_parts accumulation) that a
    # single non-streamed native_calls fixture never carried in the first
    # place; openai_tool_calls_to_ir only ever reads "function" anyway.
    assert result["tool_calls"][0]["function"] == native_calls[0]["function"]


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


# --------------------------------------------------------------------------
# State-aware pause/throttle for Groq's tokens-per-minute 429 — no more
# hardcoded fallback model, just wait out the provider's own retry-after
# hint and retry the identical request against the identical model.
# --------------------------------------------------------------------------


def _tpm_http_error(retry_after_text: str) -> urllib.error.HTTPError:
    """A live-shaped Groq TPM 429 body, e.g. '...Please try again in 19.0725s...'."""
    body = json.dumps(
        {
            "error": {
                "message": (
                    "Rate limit reached for model `openai/gpt-oss-120b` in organization "
                    "`org_123` on tokens per minute (TPM): Limit 8000, Used 6800, Requested "
                    f"3200. Please try again in {retry_after_text}. Visit "
                    "https://console.groq.com/docs/rate-limits for more information."
                ),
                "type": "tokens",
                "code": "rate_limit_exceeded",
            }
        }
    ).encode("utf-8")
    return urllib.error.HTTPError(
        "https://api.groq.com/openai/v1/chat/completions", 429, "Too Many Requests", {}, io.BytesIO(body)
    )


def test_tpm_429_raises_immediately_without_sleeping(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cloud tool-calling now calls a single resolved provider directly
    (OpenRouter by default), whose own server-side ``models`` fallback array
    already retries a 429/5xx against the next model upstream in
    milliseconds. A 429 reaching this bridge means that was already
    exhausted, so it must raise immediately as a standard failure — no
    client-side sleep/retry loop."""
    call_count = {"n": 0}

    def fake_urlopen(*_a: Any, **_k: Any) -> Any:
        call_count["n"] += 1
        raise _tpm_http_error("19.0725s")

    monkeypatch.setattr(bridge.urllib.request, "urlopen", fake_urlopen)
    sleeps: list[float] = []
    monkeypatch.setattr(bridge.time, "sleep", lambda s: sleeps.append(s))

    with pytest.raises(RuntimeError):
        bridge.complete_openai_with_tools(
            [{"role": "user", "content": "hi"}],
            api_key="k",
            base_url="https://api.groq.com/openai/v1",
            model="openai/gpt-oss-120b",
        )
    assert call_count["n"] == 1
    assert sleeps == []


def test_non_tpm_429_is_not_retried_or_throttled(monkeypatch: pytest.MonkeyPatch) -> None:
    """A requests-per-minute 429 (type != 'tokens') must raise immediately —
    waiting out a TPM window wouldn't fix a requests-per-minute cap, and
    this shape must still reach whatever caller-side fallback (e.g. local
    Ollama) exists for a real failure, not be swallowed by this loop."""
    body = json.dumps(
        {"error": {"message": "too many requests", "type": "requests", "code": "rate_limit_exceeded"}}
    ).encode("utf-8")

    def fake_urlopen(*_a: Any, **_k: Any) -> Any:
        raise urllib.error.HTTPError("url", 429, "Too Many Requests", {}, io.BytesIO(body))

    monkeypatch.setattr(bridge.urllib.request, "urlopen", fake_urlopen)
    sleeps: list[float] = []
    monkeypatch.setattr(bridge.time, "sleep", lambda s: sleeps.append(s))

    with pytest.raises(RuntimeError):
        bridge.complete_openai_with_tools(
            [{"role": "user", "content": "hi"}],
            api_key="k",
            base_url="https://api.groq.com/openai/v1",
            model="openai/gpt-oss-120b",
        )
    assert sleeps == []


def test_5xx_error_is_not_retried_or_throttled(monkeypatch: pytest.MonkeyPatch) -> None:
    """A real server-side 5xx must also raise immediately, unretried by
    this TPM-specific loop, so it still reaches whatever caller-side
    fallback exists for an actual outage."""

    def fake_urlopen(*_a: Any, **_k: Any) -> Any:
        raise urllib.error.HTTPError("url", 503, "Service Unavailable", {}, io.BytesIO(b"{}"))

    monkeypatch.setattr(bridge.urllib.request, "urlopen", fake_urlopen)
    sleeps: list[float] = []
    monkeypatch.setattr(bridge.time, "sleep", lambda s: sleeps.append(s))

    with pytest.raises(RuntimeError):
        bridge.complete_openai_with_tools(
            [{"role": "user", "content": "hi"}],
            api_key="k",
            base_url="https://api.groq.com/openai/v1",
            model="openai/gpt-oss-120b",
        )
    assert sleeps == []


# --------------------------------------------------------------------------
# extra_headers — provider-specific headers (e.g. OpenRouter's HTTP-Referer/
# X-Title attribution); this module's only knowledge of any specific
# provider is accepting whatever dict the caller (dana.core.model_provider)
# hands it.
# --------------------------------------------------------------------------


def test_extra_headers_are_merged_into_the_request(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_response(monkeypatch, {"content": "hi", "tool_calls": []})
    captured: dict[str, Any] = {}
    real_request_cls = bridge.urllib.request.Request

    def capturing_request(url, *a, **kw):
        req = real_request_cls(url, *a, **kw)
        captured["headers"] = dict(req.headers)
        return req

    monkeypatch.setattr(bridge.urllib.request, "Request", capturing_request)

    bridge.complete_openai_with_tools(
        [{"role": "user", "content": "hi"}],
        api_key="k",
        base_url="https://openrouter.ai/api/v1",
        model="meta-llama/llama-3.3-70b-instruct:free",
        extra_headers={"Http-referer": "https://my-space.hf.space", "X-title": "Dana CAD Agent"},
    )
    # urllib.request.Request title-cases header names it's given verbatim
    # keys through, so this asserts on the exact keys this bridge sets them
    # with (see _complete_openai_with_tools_once) rather than re-deriving
    # urllib's own casing convention here.
    assert captured["headers"]["Http-referer"] == "https://my-space.hf.space"
    assert captured["headers"]["X-title"] == "Dana CAD Agent"
    # The bridge's own standard headers must still be present alongside them.
    assert captured["headers"]["Authorization"] == "Bearer k"


def test_no_extra_headers_does_not_add_anything_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_response(monkeypatch, {"content": "hi", "tool_calls": []})
    captured: dict[str, Any] = {}
    real_request_cls = bridge.urllib.request.Request

    def capturing_request(url, *a, **kw):
        req = real_request_cls(url, *a, **kw)
        captured["headers"] = dict(req.headers)
        return req

    monkeypatch.setattr(bridge.urllib.request, "Request", capturing_request)

    bridge.complete_openai_with_tools(
        [{"role": "user", "content": "hi"}],
        api_key="k",
        base_url="http://127.0.0.1:11434/v1",
        model="qwen2.5-coder:7b",
    )
    assert set(captured["headers"]) == {"Content-type", "Authorization", "User-agent"}


def test_fallback_models_are_added_as_models_array_in_request_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenRouter's native server-side cascade: passing ``fallback_models``
    must land in the JSON body as ``"models"`` alongside the primary
    ``"model"`` — the raw-HTTP equivalent of the OpenAI SDK's
    ``extra_body={"models": [...]}``."""
    _mock_response(monkeypatch, {"content": "hi", "tool_calls": []})
    captured: dict[str, Any] = {}
    real_request_cls = bridge.urllib.request.Request

    def capturing_request(url, data=None, *a, **kw):
        captured["payload"] = json.loads(data.decode("utf-8"))
        return real_request_cls(url, data, *a, **kw)

    monkeypatch.setattr(bridge.urllib.request, "Request", capturing_request)

    bridge.complete_openai_with_tools(
        [{"role": "user", "content": "hi"}],
        api_key="k",
        base_url="https://openrouter.ai/api/v1",
        model="google/gemma-4-26b-a4b-it:free",
        fallback_models=["openai/gpt-oss-120b:free", "openrouter/free"],
    )
    assert captured["payload"]["model"] == "google/gemma-4-26b-a4b-it:free"
    assert captured["payload"]["models"] == ["openai/gpt-oss-120b:free", "openrouter/free"]


def test_no_fallback_models_omits_models_key_from_request_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_response(monkeypatch, {"content": "hi", "tool_calls": []})
    captured: dict[str, Any] = {}
    real_request_cls = bridge.urllib.request.Request

    def capturing_request(url, data=None, *a, **kw):
        captured["payload"] = json.loads(data.decode("utf-8"))
        return real_request_cls(url, data, *a, **kw)

    monkeypatch.setattr(bridge.urllib.request, "Request", capturing_request)

    bridge.complete_openai_with_tools(
        [{"role": "user", "content": "hi"}],
        api_key="k",
        base_url="http://127.0.0.1:11434/v1",
        model="qwen2.5-coder:7b",
    )
    assert "models" not in captured["payload"]
