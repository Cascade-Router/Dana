"""Live token streaming (assistant_delta): chunks flow from the provider
bridges through ModelProvider and the ReAct loop to the WebSocket, and every
path that voids already-streamed text sends exactly one reset."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from dana.api import server as server_module
from dana.core import model_provider as mp
from dana.core import openai_tool_bridge as bridge
from dana.core import react_dispatch as rd
from dana.platform.mock import MockControlPlane, MockFreeCADEngine
from dana.tools.schema import ToolCall


class _Sink:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def __call__(self, text: str) -> None:
        self.events.append(("delta", text))

    def reset(self) -> None:
        self.events.append(("reset", ""))

    @property
    def text(self) -> str:
        out = ""
        for kind, text in self.events:
            out = "" if kind == "reset" else out + text
        return out


class _Resp:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines

    def __iter__(self):
        return iter(self._lines)

    def __enter__(self) -> "_Resp":
        return self

    def __exit__(self, *_exc: Any) -> None:
        return None


def _sse(delta: dict[str, Any]) -> bytes:
    return ("data: " + json.dumps({"choices": [{"delta": delta}]}) + "\n").encode()


def _serve(monkeypatch: pytest.MonkeyPatch, lines: list[bytes]) -> None:
    monkeypatch.setattr(bridge.urllib.request, "urlopen", lambda *_a, **_k: _Resp(lines))


def _complete(sink: _Sink | None, **kw: Any) -> dict[str, Any]:
    return bridge.complete_openai_with_tools(
        [{"role": "user", "content": "hi"}], api_key="k", base_url="http://x", model="m", on_delta=sink, **kw
    )


# --- bridges ------------------------------------------------------------------


def test_openai_bridge_streams_content_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, [_sse({"content": "Hel"}), _sse({"content": "lo"}), b"data: [DONE]\n"])
    sink = _Sink()
    assert _complete(sink)["content"] == "Hello"
    assert sink.events == [("delta", "Hel"), ("delta", "lo")]


def test_openai_bridge_resets_when_tool_call_deltas_begin(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = {"tool_calls": [{"index": 0, "function": {"name": "create_plan", "arguments": "{}"}}]}
    _serve(monkeypatch, [_sse({"content": "Let me plan."}), _sse(tool), _sse({"content": " more"}), b"data: [DONE]\n"])
    sink = _Sink()
    result = _complete(sink, tools=[{"type": "function", "function": {"name": "create_plan"}}])
    assert result["tool_calls"]
    assert sink.events == [("delta", "Let me plan."), ("reset", "")]  # nothing streamed after the tool call


def test_openai_bridge_resets_when_content_is_coerced_into_a_tool_call(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = '{"name": "create_freecad_box", "arguments": {"length": 1}}'
    _serve(monkeypatch, [_sse({"content": raw}), b"data: [DONE]\n"])
    sink = _Sink()
    result = _complete(sink, tools=[{"type": "function", "function": {"name": "create_freecad_box"}}])
    assert result["tool_calls"] and result["content"] == ""
    assert sink.events[-1] == ("reset", "") and sink.text == ""


def test_openai_bridge_without_a_callback_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, [_sse({"content": "ok"}), b"data: [DONE]\n"])
    assert _complete(None)["content"] == "ok"


def test_ollama_native_bridge_streams_and_resets(monkeypatch: pytest.MonkeyPatch) -> None:
    lines = [
        json.dumps({"message": {"content": "Thinking"}}).encode() + b"\n",
        json.dumps({"message": {"tool_calls": [{"function": {"name": "t", "arguments": {}}}]}}).encode() + b"\n",
        json.dumps({"done": True}).encode() + b"\n",
    ]
    _serve(monkeypatch, lines)
    sink = _Sink()
    bridge.complete_ollama_native_with_tools(
        [{"role": "user", "content": "hi"}], base_url="http://x", model="m", tools=[{}], on_delta=sink
    )
    assert sink.events == [("delta", "Thinking"), ("reset", "")]


# --- provider fallbacks -------------------------------------------------------


def test_cloud_failure_resets_the_stream_before_the_ollama_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def cloud(*_a: Any, on_delta: Any = None, **_k: Any) -> dict[str, Any]:
        on_delta("partial cloud text")
        raise RuntimeError("cloud HTTP 429")

    def local(*_a: Any, on_delta: Any = None, **_k: Any) -> dict[str, Any]:
        on_delta("local answer")
        return {"content": "local answer", "tool_calls": []}

    monkeypatch.setattr(mp, "complete_openai_with_tools", cloud)
    monkeypatch.setattr(mp, "complete_ollama_native_with_tools", local)
    monkeypatch.setattr(mp, "ollama_fallback_enabled", lambda: True)
    sink = _Sink()
    provider = mp.ModelProvider(api_keys={"openrouter": "k"})
    out = provider.complete_with_tool_calls([{"role": "user", "content": "hi"}], tools=[], provider="openrouter", on_delta=sink)
    assert out["content"] == "local answer"
    assert sink.events == [("delta", "partial cloud text"), ("reset", ""), ("delta", "local answer")]


def test_router_resets_when_advancing_to_the_next_fleet_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    from dana.core import llm_router

    class _Entry:
        def __init__(self, entry_id: str) -> None:
            self.id, self.provider, self.model = entry_id, "openrouter", entry_id
            self.cost_per_1m_prompt = self.cost_per_1m_completion = 0.0

    calls = iter([RuntimeError("down"), "second answer"])

    def fake(*_a: Any, on_delta: Any = None, **_k: Any) -> dict[str, Any]:
        outcome = next(calls)
        on_delta("streamed")
        if isinstance(outcome, Exception):
            raise outcome
        return {"content": outcome, "tool_calls": []}

    monkeypatch.setattr(mp, "complete_openai_with_tools", fake)
    monkeypatch.setattr(llm_router, "api_key_for", lambda *_a, **_k: "k")
    monkeypatch.setattr(llm_router, "report_fleet_entry_failure", lambda *_a, **_k: None)
    sink = _Sink()
    out = mp.ModelProvider(api_keys={"openrouter": "k"})._complete_with_tool_calls_via_router(
        [{"role": "user", "content": "hi"}],
        tools=[],
        tool_choice=None,
        num_predict=16,
        temperature=0.1,
        chain=[_Entry("a"), _Entry("b")],
        on_delta=sink,
    )
    assert out["content"] == "second answer"
    assert sink.events == [("delta", "streamed"), ("reset", ""), ("delta", "streamed")]


def test_guarded_stream_drops_chunks_after_close() -> None:
    sink = _Sink()
    guarded = rd._GuardedStream(sink)
    guarded("before")
    guarded.close()  # what a timeout does
    guarded("late chunk from the abandoned thread")
    guarded.reset()
    assert sink.events == [("delta", "before"), ("reset", "")]


# --- ReAct loop + WebSocket ---------------------------------------------------


class _StreamingProvider:
    """Streams each scripted reply through on_delta in small chunks."""

    def __init__(self, turns: list[Any]) -> None:
        self._turns = list(turns)

    def complete_with_tool_calls(self, messages: Any, *, tools: Any, provider: Any = None, **kwargs: Any) -> dict:
        turn = self._turns.pop(0) if self._turns else "Done."
        on_delta = kwargs.get("on_delta")
        if isinstance(turn, str):
            for i in range(0, len(turn), 3):
                if on_delta:
                    on_delta(turn[i : i + 3])
            return {"content": turn, "tool_calls": [], "provider": "test"}
        if on_delta:
            on_delta("Calling a tool")
            bridge.reset_stream(on_delta)
        return {"content": "", "tool_calls": turn, "provider": "test"}


@pytest.fixture
def ws_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(server_module, "get_cad_engine", lambda: MockFreeCADEngine())
    monkeypatch.setattr(server_module, "get_control_plane", lambda: MockControlPlane())
    monkeypatch.setattr(rd, "_get_has_plan", lambda *_a, **_k: True)
    monkeypatch.setenv("DANA_OS_DRY_RUN", "1")
    return TestClient(server_module.app)


def _use(monkeypatch: pytest.MonkeyPatch, *turns: Any) -> None:
    provider = _StreamingProvider(list(turns))
    monkeypatch.setattr(rd, "ModelProvider", lambda **_k: provider)


def _events_until_reply(ws: Any) -> list[dict[str, Any]]:
    events = []
    while True:
        event = ws.receive_json()
        events.append(event)
        if event["type"] == "assistant_message":
            return events


def test_final_reply_streams_batched_deltas_then_the_authoritative_message(
    monkeypatch: pytest.MonkeyPatch, ws_client: TestClient
) -> None:
    reply = "A plate is a flat box; I can build one 60 x 40 x 5 mm if you like. " * 3
    _use(monkeypatch, reply)
    with ws_client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"text": "what is a plate?"})
        events = _events_until_reply(ws)

    deltas = [e for e in events if e["type"] == "assistant_delta"]
    assert "".join(d["text"] for d in deltas) == reply
    assert len(deltas) < len(reply) / 3  # batched, not one frame per chunk
    assert {d["node_id"] for d in deltas} == {"parse-0"}
    assert events[-1] == {"type": "assistant_message", "content": reply}
    assert not any(e["type"] == "assistant_delta_reset" for e in events)


def test_tool_call_step_resets_its_streamed_preamble(monkeypatch: pytest.MonkeyPatch, ws_client: TestClient) -> None:
    _use(monkeypatch, [ToolCall("system_state", {})], "All good.")
    with ws_client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()
        ws.send_json({"text": "how are you?"})
        events = _events_until_reply(ws)
    kinds = [(e["type"], e.get("node_id")) for e in events if e["type"].startswith("assistant_delta")]
    assert ("assistant_delta_reset", "parse-0") in kinds
    assert ("assistant_delta", "parse-1") in kinds


def test_self_correction_retry_resets_the_rejected_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    """A premature "I'm done" while a task is still active is retried; its
    streamed text must be discarded before the retry streams."""
    monkeypatch.setattr(rd, "_get_fsm_state", lambda *_a, **_k: "executing")
    monkeypatch.setattr(rd, "_active_task", lambda *_a, **_k: {"id": 1, "description": "build plate"})
    _use(monkeypatch, "I can't do that.", "Fine, done.")
    sink = _Sink()
    turn = asyncio.run(rd.next_react_turn([{"role": "user", "content": "go"}], raw_text="go", on_delta=sink))
    assert turn.content == "Fine, done."
    assert ("reset", "") in sink.events
    assert sink.text == "Fine, done."


def test_abort_resets_a_stream_still_in_flight(ws_client: TestClient) -> None:
    closed: list[bool] = []

    class _Stream:
        def close(self, *, discard: bool) -> None:
            closed.append(discard)

    with ws_client.websocket_connect("/ws/chat") as ws:
        ready = ws.receive_json()
        session = next(s for s in server_module._active_sessions.values() if s["session_id"] == ready["session_id"])
        session["active_stream"] = _Stream()
        ws.send_json({"type": "abort_turn"})
        ws.send_json({"type": "export_python_script"})  # replies, so abort_turn was handled before it
        ws.receive_json()
    assert closed == [True]
    assert session["abort_requested"] is True
