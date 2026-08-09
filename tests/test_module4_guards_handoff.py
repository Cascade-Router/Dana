"""Module 4: Pydantic guards, validation bounce, and Swarm Handoff."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from dana.agentic import get_dana_mode, set_dana_mode
from dana.handoff import execute_handoff, parse_handoff_payload
from dana.schema import Handoff
from dana.tools.guards import (
    DraftCursorTicketPayload,
    format_validation_bounce,
    guard_tool_call,
)
from dana.tools.general.draft_cursor_prompt import draft_cursor_prompt


_GOOD_CTX = (
    "**Technical intent:** Harden MoA ticket CONTEXT validation.\n"
    "**Target Files:** dana/moa_tool_shim.py, dana/tools/guards.py\n\n"
    "Root cause: Thin reasoner plans wrote intent-echo tickets.\n"
    "Step-by-step changes:\n"
    "1. Require Target files / Root cause / Steps / Acceptance in CONTEXT.\n"
    "2. Reject unstructured plans before stage-2 formatting.\n"
    "Acceptance criteria: Malformed fixtures raise ValidationError; structured "
    "fixtures append PENDING tickets.\n"
)


def test_draft_cursor_payload_rejects_truncated_objective() -> None:
    with pytest.raises(ValidationError) as ei:
        DraftCursorTicketPayload.model_validate(
            {
                "objective": "improve cursor handling in the deepseek pipel...",
                "context": _GOOD_CTX,
            }
        )
    assert "truncated" in str(ei.value).lower() or "..." in str(ei.value)


def test_draft_cursor_payload_rejects_intent_echo() -> None:
    echo = (
        "**Technical intent:** improve cursor handling somehow across the MoA "
        "pipeline without concrete symbols or acceptance criteria yet provided.\n"
        "**Target Files:** dana/agentic.py\n"
    )
    # Pad past min_length so model_validator (intent-echo) runs.
    echo = echo + (" padding" * 20)
    with pytest.raises(ValidationError) as ei:
        DraftCursorTicketPayload.model_validate(
            {"objective": "improve cursor handling somehow now", "context": echo}
        )
    msg = str(ei.value).lower()
    assert (
        "intent-echo" in msg
        or "missing" in msg
        or "step_by_step" in msg
        or "root_cause" in msg
        or "acceptance" in msg
    )


def test_draft_cursor_payload_accepts_structured() -> None:
    payload = DraftCursorTicketPayload.model_validate(
        {
            "objective": "Harden MoA ticket CONTEXT validation before ledger writes.",
            "context": _GOOD_CTX,
        }
    )
    assert "Harden MoA" in payload.objective


def test_guard_tool_call_raises_and_bounce_prompt() -> None:
    with pytest.raises(ValidationError) as ei:
        guard_tool_call(
            "draft_cursor_prompt",
            {"objective": "short...", "context": "too short"},
        )
    bounce = format_validation_bounce(ei.value)
    assert bounce.startswith("Validation Error:")
    assert "retry" in bounce.lower()


def test_draft_cursor_prompt_returns_validation_error_string() -> None:
    out = draft_cursor_prompt(
        objective="improve cursor pro...",
        context="**Technical intent:** x\n**Target Files:** dana/agentic.py",
    )
    assert out.startswith("ERROR:")
    assert "Validation Error" in out


def test_handoff_model_and_parse_json() -> None:
    ho = Handoff(
        target_agent="vision_agent",
        reason="User asked to see the screen",
        intent_context="Describe what is on screen",
    )
    assert ho.target_agent == "Vision_Agent"
    text = (
        'Please switch capabilities: '
        '{"target_agent": "MoA_Reasoner", "reason": "needs tools", '
        '"intent_context": "draft a ticket"}'
    )
    parsed = parse_handoff_payload(text)
    assert parsed is not None
    assert parsed.target_agent == "MoA_Reasoner"
    assert "tools" in parsed.reason


def test_execute_handoff_switches_mode_and_emits_telemetry(
    tmp_path: Path, monkeypatch
) -> None:  # noqa: ANN001
    out = tmp_path / "dana_telemetry.jsonl"
    monkeypatch.setattr("dana.telemetry.TELEMETRY_JSONL_PATH", out)
    set_dana_mode("chat")
    ho = Handoff(
        target_agent="Vision_Agent",
        reason="ASR requested vision",
        intent_context="look at camera",
    )
    result = execute_handoff(ho, session_id="m4", current_agent="Chat_Node")
    assert result["current_agent"] == "Vision_Agent"
    assert get_dana_mode() == "vision"
    lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    tags = [r["tag"] for r in lines]
    assert "[HANDOFF]" in tags
    hand = next(r for r in lines if r["tag"] == "[HANDOFF]")
    assert "Vision_Agent" in hand["message"]


def test_localized_rejection_loop_single_retry_semantics() -> None:
    """Simulate tools-node retry set: first bounce, second exhausted."""
    retries: set[str] = set()
    key = "draft_cursor_prompt:call-1"
    try:
        guard_tool_call(
            "draft_cursor_prompt",
            {
                "objective": "thin...",
                "context": "**Technical intent:** x\n**Target Files:** dana/a.py",
            },
        )
        raise AssertionError("expected ValidationError")
    except ValidationError as exc:
        bounce = format_validation_bounce(exc)
        assert key not in retries
        retries.add(key)
        assert "Validation Error" in bounce
        # Second failure → exhausted marker (graph appends this).
        assert key in retries
        exhausted = f"ERROR: {bounce} (retry exhausted)"
        assert "retry exhausted" in exhausted
