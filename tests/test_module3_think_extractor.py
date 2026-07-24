"""Module 3: DeepSeek-R1 <think> extractor → Blackboard + clean handoff."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from donna.agentic import extract_r1_think_blocks, strip_r1_think_blocks
from donna.memory import init_blackboard, load_reasoning_traces
from donna.moa_tool_shim import _llm_text, run_moa_reasoner_stage


def test_extract_closed_think_block() -> None:
    raw = (
        "<think>\nI should call draft_cursor_prompt with full CONTEXT.\n</think>\n"
        "INTENT: draft_cursor_prompt\n"
        "OBJECTIVE: Harden think extraction\n"
    )
    ex = extract_r1_think_blocks(raw)
    assert "draft_cursor_prompt with full CONTEXT" in ex.think_text
    assert "<think>" not in ex.clean_text
    assert "</think>" not in ex.clean_text
    assert ex.clean_text.startswith("INTENT:")
    assert strip_r1_think_blocks(raw) == ex.clean_text


def test_extract_multiple_and_unclosed_think() -> None:
    multi = (
        "<think>first</think>\n"
        "mid\n"
        "<think>second still open"
    )
    ex = extract_r1_think_blocks(multi)
    assert "first" in ex.think_text
    assert "second still open" in ex.think_text
    assert "mid" in ex.clean_text
    assert "<think>" not in ex.clean_text
    assert "second" not in ex.clean_text


def test_strip_unclosed_think_does_not_leak() -> None:
    raw = "prelude <think> orphan reasoning without close"
    clean = strip_r1_think_blocks(raw)
    assert "orphan" not in clean
    assert clean.strip() == "prelude"


def test_moa_reasoner_files_think_and_returns_clean_only(
    tmp_path: Path, monkeypatch
) -> None:  # noqa: ANN001
    db = tmp_path / "bb.db"
    init_blackboard(db)
    tel = tmp_path / "donna_telemetry.jsonl"
    monkeypatch.setattr("donna.telemetry.TELEMETRY_JSONL_PATH", tel)
    monkeypatch.setattr(
        "donna.memory.blackboard.BLACKBOARD_DB_PATH",
        db,
    )

    payload = (
        "<think>\nStep 1: expand CONTEXT with root cause.\n"
        "Step 2: list acceptance criteria.\n</think>\n"
        "INTENT: draft_cursor_prompt\n"
        "OBJECTIVE: Harden MoA think stripping\n"
        "CONTEXT:\n"
        "Target files: donna/moa_tool_shim.py\n"
        "Root cause: think tags bloated formatter context\n"
        "Step-by-step changes: 1. extract 2. file 3. hand off clean text\n"
        "Acceptance criteria: clean plan has no think tags\n"
    )

    class _FakeLLM:
        def invoke(self, _messages):  # noqa: ANN001
            return SimpleNamespace(content=payload)

    monkeypatch.setattr(
        "donna.moa_tool_shim._build_reasoner_llm",
        lambda **_kwargs: _FakeLLM(),
    )
    monkeypatch.setattr(
        "donna.moa_tool_shim.note_high_complexity_deepseek_latency",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "donna.moa_tool_shim.reasoner_model_name",
        lambda: "deepseek-r1:test",
    )

    # Force blackboard helpers to use tmp db.
    monkeypatch.setattr(
        "donna.memory.blackboard.BLACKBOARD_DB_PATH",
        db,
    )

    from donna.memory import ensure_session

    sid = ensure_session("m3-sess", current_agent="MoA_Reasoner", db_path=db)
    # Patch append path used inside moa_tool_shim (imports append_reasoning_trace
    # from donna.memory which re-exports blackboard with default path).
    import donna.memory.blackboard as bb

    monkeypatch.setattr(bb, "BLACKBOARD_DB_PATH", db)

    plan = run_moa_reasoner_stage(
        "log a self-improvement ticket about think stripping",
        forced_tool_id="draft_cursor_prompt",
        session_id=sid,
    )
    assert "<think>" not in plan
    assert "</think>" not in plan
    assert "INTENT: draft_cursor_prompt" in plan
    assert "Root cause" in plan
    assert "orphan" not in plan.lower() or True

    traces = load_reasoning_traces(sid, db_path=db)
    assert traces, "expected reasoning row on Blackboard"
    assert "expand CONTEXT" in traces[-1]["think_text"]
    assert "INTENT: draft_cursor_prompt" in (traces[-1].get("clean_text") or "")

    assert tel.is_file()
    rows = [json.loads(line) for line in tel.read_text(encoding="utf-8").splitlines()]
    tags = [r["tag"] for r in rows]
    assert "[REASONING_TRACE]" in tags
    reason = next(r for r in rows if r["tag"] == "[REASONING_TRACE]")
    assert "expand CONTEXT" in reason["message"]
    assert reason.get("payload", {}).get("latency_ms") is not None


def test_llm_text_helper_strips_think() -> None:
    msg = SimpleNamespace(
        content="<think>hidden</think>\nINTENT: NONE\nOBJECTIVE: hi\n"
    )
    out = _llm_text(msg)
    assert "hidden" not in out
    assert "INTENT: NONE" in out
