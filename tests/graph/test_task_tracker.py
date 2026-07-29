"""TaskTracker + completion-gate reliability tests."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from langgraph.graph import END

from dana.agentic_react_graph import _route_after_agent, route_after_execution
from dana.graph.completion_gate import (
    TOOL_TIMEOUT_MESSAGE,
    format_tool_failure_message,
    gate_route_after_execution,
    is_filler_response,
    run_with_tool_timeout,
    should_block_end,
    tool_failure_state_patch,
)
from dana.graph.task_tracker import TaskStatus, TaskTracker
from dana.schema import ReactGraphState


def test_tool_failure_emits_explicit_error_not_silent(tmp_path: Path) -> None:
    """Simulated tool failure → explicit user-facing message; never silent END."""
    tracker = TaskTracker(
        dropped_log_path=tmp_path / "dropped_tasks.log",
        ledger_path=tmp_path / "patch_ledger.md",
    )
    tracker.start_task("t-fail-1", "run the spreadsheet tool")
    tracker.update_status("t-fail-1", TaskStatus.TOOL_EXECUTING)

    patch = tool_failure_state_patch(
        tool_id="spreadsheet_query",
        detail="connection refused",
    )
    spoken = str(patch["final_raw"])
    assert spoken
    assert "failed" in spoken.lower()
    assert "spreadsheet_query" in spoken
    assert patch["pending_synthesis"] is True
    assert patch["halt"] is False

    # Explicit timeout path also surfaces the required fallback phrase.
    timed = tool_failure_state_patch(timed_out=True)
    assert timed["final_raw"] == TOOL_TIMEOUT_MESSAGE
    assert TOOL_TIMEOUT_MESSAGE in format_tool_failure_message(timed_out=True)

    tracker.update_status("t-fail-1", TaskStatus.FAILED, metadata={"error": spoken})
    rec = tracker.get_task("t-fail-1")
    assert rec is not None
    assert rec.status == TaskStatus.FAILED

    # Routing must not silent-END on the failure patch.
    state: ReactGraphState = {
        "session_id": "t-fail-1",
        "messages": [],
        "halt": False,
        **patch,
        "retry_count": 0,
        "max_retries": 3,
    }
    route = route_after_execution(state)
    assert route == "critic"
    assert gate_route_after_execution(state, proposed=END) == "critic"
    assert route != END


def test_dropped_trajectory_writes_log(tmp_path: Path) -> None:
    """Incomplete/dropped trajectory → entry in dropped_tasks.log (temp path)."""
    log_path = tmp_path / "logs" / "dropped_tasks.log"
    ledger = tmp_path / "dana_security" / "patch_ledger.md"
    tracker = TaskTracker(dropped_log_path=log_path, ledger_path=ledger)

    tracker.start_task("t-drop-1", "what is on my screen?")
    tracker.update_status("t-drop-1", TaskStatus.IN_PROGRESS)
    result = tracker.log_dropped_task(
        "t-drop-1",
        "agent replied 'let me check' then corridor ended",
        last_state_buffer={"pending_synthesis": True, "final_raw": "Let me check"},
    )
    assert result["ok"] is True
    assert log_path.is_file()
    text = log_path.read_text(encoding="utf-8")
    assert "t-drop-1" in text
    assert "what is on my screen?" in text
    assert "let me check" in text.lower() or "corridor ended" in text

    rec = tracker.get_task("t-drop-1")
    assert rec is not None
    assert rec.status == TaskStatus.DROPPED

    # Injectable ledger must receive a PENDING ticket (never the real ledger).
    assert ledger.is_file()
    ledger_text = ledger.read_text(encoding="utf-8")
    assert "[PENDING]" in ledger_text
    assert "DROPPED" in ledger_text


def test_status_transitions_received_to_completed(tmp_path: Path) -> None:
    """RECEIVED → IN_PROGRESS → TOOL_EXECUTING → COMPLETED."""
    tracker = TaskTracker(
        dropped_log_path=tmp_path / "dropped.log",
        ledger_path=tmp_path / "ledger.md",
    )
    tracker.start_task("t-ok-1", "summarize the PDF")
    assert tracker.get_task("t-ok-1").status == TaskStatus.RECEIVED

    tracker.update_status("t-ok-1", TaskStatus.IN_PROGRESS)
    assert tracker.get_task("t-ok-1").status == TaskStatus.IN_PROGRESS

    tracker.update_status("t-ok-1", "TOOL_EXECUTING")
    assert tracker.get_task("t-ok-1").status == TaskStatus.TOOL_EXECUTING

    tracker.update_status("t-ok-1", TaskStatus.COMPLETED, metadata={"ok": True})
    rec = tracker.get_task("t-ok-1")
    assert rec is not None
    assert rec.status == TaskStatus.COMPLETED
    assert rec.metadata.get("ok") is True


def test_filler_blocks_end_via_completion_gate() -> None:
    """'let me check' filler → pending_synthesis blocks END."""
    assert is_filler_response("Let me check.")
    assert is_filler_response("Looking into that now")
    assert not is_filler_response(
        "Here is the full answer with the spreadsheet totals you asked for."
    )

    state: dict[str, Any] = {
        "pending_synthesis": True,
        "halt": True,
        "messages": [],
        "always_include": [],
    }
    assert should_block_end(state) is True
    assert _route_after_agent(state) == "agent"
    assert _route_after_agent(state) != END


def test_tool_timeout_helper_returns_fallback_message() -> None:
    """Watchdog returns the spoken timeout message and never raises silently."""

    def _hang() -> str:
        time.sleep(2.0)
        return "done"

    ok, result, err = run_with_tool_timeout(_hang, timeout_s=0.1)
    assert ok is False
    assert result is None
    assert err == TOOL_TIMEOUT_MESSAGE
