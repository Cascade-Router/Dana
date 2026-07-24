"""Module 1: Blackboard + tagged telemetry + minimal graph state."""

from __future__ import annotations

import json
from pathlib import Path

from donna.memory import (
    append_message,
    append_reasoning_trace,
    ensure_session,
    get_session_meta,
    init_blackboard,
    load_messages,
    load_reasoning_traces,
    set_session_meta,
)
from donna.schema import ReactGraphState
from donna.telemetry import (
    TELEMETRY_JSONL_PATH,
    emit_tagged,
    log_handoff,
    log_reasoning_trace,
    log_router,
    log_tool_execution,
    log_voice_asr,
)


def test_blackboard_sqlite_standing(tmp_path: Path) -> None:
    db = tmp_path / "bb.db"
    path = init_blackboard(db)
    assert path.is_file()

    sid = ensure_session(
        "sess-demo",
        current_agent="ReAct_Agent",
        active_intent="general",
        db_path=db,
    )
    assert sid == "sess-demo"
    append_message(sid, "user", "hello blackboard", db_path=db)
    append_message(sid, "assistant", "hi from donna", db_path=db)
    msgs = load_messages(sid, db_path=db)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["content"] == "hi from donna"

    set_session_meta(sid, current_agent="MoA_Reasoner", active_intent="draft_cursor_prompt", db_path=db)
    meta = get_session_meta(sid, db_path=db)
    assert meta["current_agent"] == "MoA_Reasoner"
    assert meta["active_intent"] == "draft_cursor_prompt"

    rid = append_reasoning_trace(
        sid,
        "<think>plan steps</think>",
        clean_text="do the thing",
        db_path=db,
    )
    assert rid > 0
    traces = load_reasoning_traces(sid, db_path=db)
    assert len(traces) == 1
    assert "plan steps" in traces[0]["think_text"]


def test_tagged_telemetry_jsonl(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    out = tmp_path / "donna_telemetry.jsonl"
    monkeypatch.setattr("donna.telemetry.TELEMETRY_JSONL_PATH", out)

    log_voice_asr("hello donna", session_id="s1")
    log_router("route=local", session_id="s1", current_agent="ReAct_Agent", active_intent="general")
    log_reasoning_trace("because reasons", session_id="s1")
    log_tool_execution("draft_cursor_prompt", session_id="s1", ok=True, latency_ms=12.5)
    log_handoff("MoA_Reasoner", session_id="s1", current_agent="Chat_Agent")

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5
    tags = [json.loads(line)["tag"] for line in lines]
    assert tags == [
        "[VOICE_ASR]",
        "[ROUTER]",
        "[REASONING_TRACE]",
        "[TOOL_EXECUTION]",
        "[HANDOFF]",
    ]
    tool_rec = json.loads(lines[3])
    assert tool_rec["latency_ms"] == 12.5
    assert TELEMETRY_JSONL_PATH.name == "donna_telemetry.jsonl"


def test_emit_tagged_rejects_unknown() -> None:
    try:
        emit_tagged("NOT_A_TAG", "x")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_react_graph_state_is_minimal() -> None:
    required = {"session_id", "current_agent", "active_intent"}
    annotations = set(ReactGraphState.__annotations__.keys())
    assert required.issubset(annotations)
    # Ephemeral turn scratch may exist, but bureaucratic triad is present.
    state: ReactGraphState = {
        "session_id": "abc",
        "current_agent": "ReAct_Agent",
        "active_intent": "general",
        "messages": [],
        "iterations": 0,
        "last_obs": "",
        "final_raw": "",
        "halt": False,
        "always_include": [],
    }
    assert state["session_id"] == "abc"
    assert state["current_agent"] == "ReAct_Agent"
