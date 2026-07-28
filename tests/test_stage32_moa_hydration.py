"""Stage 3.2 — MoA Blackboard context hydration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from dana.memory import append_message, ensure_session, init_blackboard
from dana.moa_tool_shim import (
    _REASONER_SYSTEM,
    format_blackboard_history_block,
    run_moa_reasoner_stage,
)


def test_format_blackboard_history_block_includes_entities(tmp_path: Path) -> None:
    db = tmp_path / "bb.db"
    init_blackboard(db)
    sid = ensure_session("hydrate-1", current_agent="Chat_Node", db_path=db)
    append_message(
        sid,
        "user",
        "I am working on my YC project, Cascade Router, with latency spikes.",
        db_path=db,
    )
    append_message(
        sid,
        "assistant",
        "Noted — Cascade Router latency spikes.",
        db_path=db,
    )
    append_message(
        sid,
        "user",
        "fix the latency spikes in the project I mentioned earlier",
        db_path=db,
    )

    # Point load_messages at tmp db.
    import dana.memory.blackboard as bb

    prev = bb.BLACKBOARD_DB_PATH
    bb.BLACKBOARD_DB_PATH = db
    try:
        block = format_blackboard_history_block(
            sid,
            user_text="fix the latency spikes in the project I mentioned earlier",
            max_turns=5,
        )
    finally:
        bb.BLACKBOARD_DB_PATH = prev

    assert "[RECENT CONVERSATION HISTORY]" in block
    assert "[END RECENT CONVERSATION HISTORY]" in block
    assert "Cascade Router" in block
    # Current duplicate user turn should be stripped.
    assert block.count("fix the latency spikes in the project I mentioned earlier") == 0
    # Core reasoner instructions must remain a separate constant.
    assert "You must enclose your internal chain of thought" in _REASONER_SYSTEM
    assert "[RECENT CONVERSATION HISTORY]" not in _REASONER_SYSTEM


def test_run_moa_hydrates_system_without_leaking_into_plan(
    tmp_path: Path, monkeypatch
) -> None:  # noqa: ANN001
    """History is appended to SystemMessage only; returned plan stays clean."""
    db = tmp_path / "bb.db"
    init_blackboard(db)
    sid = ensure_session("hydrate-2", current_agent="MoA_Reasoner", db_path=db)
    append_message(
        sid,
        "user",
        "YC project Cascade Router has message-passing latency spikes.",
        db_path=db,
    )
    append_message(
        sid,
        "assistant",
        "Got it about Cascade Router.",
        db_path=db,
    )

    import dana.memory.blackboard as bb

    monkeypatch.setattr(bb, "BLACKBOARD_DB_PATH", db)
    monkeypatch.setattr(
        "dana.memory.blackboard.BLACKBOARD_DB_PATH",
        db,
    )

    captured: dict[str, str] = {}

    class _FakeLLM:
        def invoke(self, messages):
            system = messages[0].content
            human = messages[1].content
            captured["system"] = system
            captured["human"] = human
            return SimpleNamespace(
                content=(
                    "<think>\nResolve project name from history: Cascade Router.\n"
                    "</think>\n"
                    "INTENT: draft_cursor_prompt\n"
                    "OBJECTIVE: Fix Cascade Router latency spikes\n"
                    "CONTEXT:\n"
                    "  Target files: dana/cascade_router.py\n"
                    "  Root cause: message passing spikes\n"
                    "  Step-by-step changes: 1. profile 2. patch\n"
                    "  Acceptance criteria: RapidFuzz mailroom still works\n"
                )
            )

    monkeypatch.setattr(
        "dana.moa_tool_shim._build_reasoner_llm",
        lambda **_k: _FakeLLM(),
    )
    monkeypatch.setattr(
        "dana.moa_tool_shim.note_high_complexity_deepseek_latency",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "dana.moa_tool_shim.reasoner_model_name",
        lambda: "deepseek-r1:test",
    )

    plan = run_moa_reasoner_stage(
        "log a ticket to fix the latency spikes in the project I mentioned earlier",
        forced_tool_id="draft_cursor_prompt",
        session_id=sid,
    )

    assert "[RECENT CONVERSATION HISTORY]" in captured["system"]
    assert "Cascade Router" in captured["system"]
    assert captured["system"].startswith(_REASONER_SYSTEM[:40])
    # History must not be the human payload (keeps extractor/guards on plan only).
    assert "[RECENT CONVERSATION HISTORY]" not in captured["human"]
    # Returned plan is post-think clean text — no history wrapper, no think tags.
    assert "<think>" not in plan
    assert "[RECENT CONVERSATION HISTORY]" not in plan
    assert "INTENT: draft_cursor_prompt" in plan
    assert "Cascade Router" in plan
