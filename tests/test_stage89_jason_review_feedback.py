"""Stage 8.9 — Jason ticket review + local feedback JSONL logging."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dana.agentic_react_graph import (
    _heuristic_jason_critique,
    generate_jason_ticket_critique,
    jason_ticket_review_node,
)
from dana.memory.feedback_log import (
    clear_feedback_logs,
    feedback_log_path,
    log_human_feedback,
)
from dana.middleware import hitl_ticket as hitl
from dana.schema import ReactGraphState


@pytest.fixture(autouse=True)
def _clean_hitl() -> None:
    hitl.clear_pending()
    yield
    hitl.clear_pending()


def test_heuristic_critique_flags_missing_visual() -> None:
    c = _heuristic_jason_critique(
        "Please log a ticket about OCR visual bounds on the screen",
        objective="Harden cascade router for tool selection",
        context="Focus on broker foresight only",
    )
    assert "visual" in c.lower() or "deny" in c.lower()


def test_generate_critique_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a, **_k):  # noqa: ANN001
        raise ConnectionError("offline")

    monkeypatch.setattr("dana.core_agent.ask_ollama_messages", _boom)
    c = generate_jason_ticket_critique(
        "Fix the API schema validation",
        objective="Harden draft_cursor_prompt against empty objective",
        context="Include API schema checks in the ticket body",
    )
    assert isinstance(c, str) and len(c) > 10


def test_jason_review_node_speaks_and_stores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spoken: list[tuple[str, str]] = []

    monkeypatch.setattr(
        "dana.agentic_react_graph.generate_jason_ticket_critique",
        lambda *_a, **_k: "This ticket accurately captures the API constraints.",
    )

    def _speak(text: str, *, agent_id: str | None = None, **_kw) -> None:  # noqa: ANN001
        spoken.append((text, str(agent_id or "")))

    monkeypatch.setattr("dana.core_agent.enqueue_speech", _speak)

    from langchain_core.messages import AIMessage, HumanMessage

    state: ReactGraphState = {
        "messages": [
            HumanMessage(content="Log a ticket about API constraints"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "draft_cursor_prompt",
                        "args": {
                            "objective": "Capture API constraints",
                            "context": "Schema validation gaps",
                        },
                        "id": "c1",
                        "type": "tool_call",
                    }
                ],
            ),
        ],
        "session_id": "s89",
    }
    out = jason_ticket_review_node(state)
    assert "accurately" in str(out.get("jason_critique") or "")
    assert spoken and spoken[0][1] == "jason"


def test_log_human_feedback_appends_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "feedback_logs.jsonl"
    monkeypatch.setattr(
        "dana.memory.feedback_log.feedback_log_path",
        lambda: target,
    )
    path = log_human_feedback(
        "task-1",
        "approve",
        "Looks good.",
        {"objective": "O", "context": "C", "tool": "draft_cursor_prompt"},
        session_id="sess",
    )
    assert path == target
    assert target.is_file()
    row = json.loads(target.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert row["human_decision"] == "approve"
    assert row["jason_critique"] == "Looks good."
    assert row["ticket_content"]["objective"] == "O"
    assert row["task_id"] == "task-1"


def test_hitl_format_shows_jason_critique_first() -> None:
    text = hitl.format_ticket_payload(
        {
            "objective": "Obj",
            "context": "Ctx",
            "jason_critique": "Deny — missing visual bounds.",
        }
    )
    assert "JASON REVIEW" in text
    assert text.index("JASON REVIEW") < text.index("Objective:")
    assert "Deny — missing visual bounds." in text
    assert "Files:" in text


def test_submit_decision_writes_feedback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "feedback_logs.jsonl"
    monkeypatch.setattr(
        "dana.memory.feedback_log.feedback_log_path",
        lambda: target,
    )
    monkeypatch.setenv("DONNA_HITL_REQUIRE_GUI", "1")
    hitl.publish_pending(
        {
            "objective": "O",
            "context": "C",
            "jason_critique": "Approve if you agree.",
            "tool_call_id": "tid-89",
            "session_id": "s",
        },
        thread_id="t",
    )
    assert hitl.submit_decision(False, action="deny")
    rows = [json.loads(x) for x in target.read_text(encoding="utf-8").splitlines() if x]
    assert rows[-1]["human_decision"] == "deny"
    assert "Approve if you agree" in rows[-1]["jason_critique"]


def test_clear_feedback_logs_truncates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "feedback_logs.jsonl"
    target.write_text('{"human_decision":"approve"}\n', encoding="utf-8")
    assert target.stat().st_size > 0
    monkeypatch.setattr(
        "dana.memory.feedback_log.feedback_log_path",
        lambda: target,
    )
    result = clear_feedback_logs()
    assert result["ok"] is True
    assert result["bytes"] == 0
    assert "Logs Cleared (0 B)" in result["message"]
    assert target.read_text(encoding="utf-8") == ""

    missing = tmp_path / "missing" / "feedback_logs.jsonl"
    monkeypatch.setattr(
        "dana.memory.feedback_log.feedback_log_path",
        lambda: missing,
    )
    result2 = clear_feedback_logs()
    assert result2["ok"] is True
    assert result2["bytes"] == 0


def test_dev_tools_clear_logs_button() -> None:
    import customtkinter as ctk

    from dana.ui.trace_window import LiveTracePanel

    root = ctk.CTk()
    root.withdraw()
    panel = LiveTracePanel(root)
    root.update_idletasks()
    assert hasattr(panel, "_on_clear_feedback_logs")
    assert hasattr(panel, "_clear_logs_status")
    panel._on_clear_feedback_logs()
    root.update_idletasks()
    assert "Cleared" in str(panel._clear_logs_status.cget("text"))
    root.destroy()


def test_gitignore_mentions_feedback_jsonl() -> None:
    root = Path(__file__).resolve().parents[1]
    gi = (root / ".gitignore").read_text(encoding="utf-8")
    assert "feedback_logs.jsonl" in gi
    assert "memory/*.jsonl" in gi
    # Default path stays under repo memory/ (gitignored).
    assert "feedback_logs" in str(feedback_log_path())
