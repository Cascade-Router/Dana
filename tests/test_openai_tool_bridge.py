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


def test_parse_retry_after_seconds_extracts_groq_hint() -> None:
    body = json.dumps({"error": {"message": "Requested 3200. Please try again in 19.0725s. Visit ..."}})
    assert bridge._parse_retry_after_seconds(body) == pytest.approx(19.0725)


def test_parse_retry_after_seconds_returns_none_when_absent_or_unparseable() -> None:
    assert bridge._parse_retry_after_seconds(json.dumps({"error": {"message": "no hint here"}})) is None
    assert bridge._parse_retry_after_seconds("not json at all") is None
    assert bridge._parse_retry_after_seconds(json.dumps({"error": {}})) is None


def test_tpm_429_sleeps_per_groq_hint_then_retries_same_model_and_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    success_lines = _sse_lines_for_message({"content": "ok", "tool_calls": []})

    def fake_urlopen(request: Any, *_a: Any, **_k: Any) -> Any:
        calls.append(request.full_url if hasattr(request, "full_url") else "?")
        if len(calls) == 1:
            raise _tpm_http_error("19.0725s")
        return _FakeResponse(success_lines)

    monkeypatch.setattr(bridge.urllib.request, "urlopen", fake_urlopen)
    sleeps: list[float] = []
    monkeypatch.setattr(bridge.time, "sleep", lambda s: sleeps.append(s))

    result = bridge.complete_openai_with_tools(
        [{"role": "user", "content": "hi"}],
        api_key="k",
        base_url="https://api.groq.com/openai/v1",
        model="openai/gpt-oss-120b",
    )
    assert result["content"] == "ok"
    assert len(calls) == 2  # first attempt (429) + one retry against the SAME model
    assert sleeps == [pytest.approx(20.0725)]  # retry_after (19.0725) + 1


def test_tpm_429_exhausts_max_retries_then_returns_degraded_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = {"n": 0}

    def fake_urlopen(*_a: Any, **_k: Any) -> Any:
        call_count["n"] += 1
        raise _tpm_http_error("5s")

    monkeypatch.setattr(bridge.urllib.request, "urlopen", fake_urlopen)
    sleeps: list[float] = []
    monkeypatch.setattr(bridge.time, "sleep", lambda s: sleeps.append(s))

    result = bridge.complete_openai_with_tools(
        [{"role": "user", "content": "hi"}],
        api_key="k",
        base_url="https://api.groq.com/openai/v1",
        model="openai/gpt-oss-120b",
    )
    assert result == bridge._degraded_summary_response()
    # Initial attempt + _MAX_TPM_RETRIES retries, never more.
    assert call_count["n"] == bridge._MAX_TPM_RETRIES + 1
    assert len(sleeps) == bridge._MAX_TPM_RETRIES


def test_tpm_429_with_unreasonably_long_retry_after_degrades_without_sleeping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = {"n": 0}

    def fake_urlopen(*_a: Any, **_k: Any) -> Any:
        call_count["n"] += 1
        raise _tpm_http_error("120s")  # >= _MAX_REASONABLE_RETRY_AFTER_SEC

    monkeypatch.setattr(bridge.urllib.request, "urlopen", fake_urlopen)
    sleeps: list[float] = []
    monkeypatch.setattr(bridge.time, "sleep", lambda s: sleeps.append(s))

    result = bridge.complete_openai_with_tools(
        [{"role": "user", "content": "hi"}],
        api_key="k",
        base_url="https://api.groq.com/openai/v1",
        model="openai/gpt-oss-120b",
    )
    assert result == bridge._degraded_summary_response()
    assert call_count["n"] == 1  # never retries on an unreasonable hint
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
