"""Escalation tickets must flush ``[PENDING]`` into an injectable ledger path."""

from __future__ import annotations

from pathlib import Path

from dana.graph.buffer import store_raw_trace
from dana.graph.nodes.critic import FATAL_OS_BLOCK_MSG, fail_closed_node
from dana_security.ledger_writer import (
    append_pending_ticket,
    format_escalation_ticket,
    resolve_ledger_path,
    set_ledger_path_override,
    write_escalation_ticket,
)


def test_format_escalation_ticket_has_pending_metadata() -> None:
    block = format_escalation_ticket(
        task_id="esc_test_1",
        error_trace="PermissionError: denied",
        recommended_fix="Fix host permissions.",
        objective="Fatal OS Block: Missing dependency or permission denied",
        timestamp="2026-07-28 00:00:00 UTC",
    )
    assert "**Status:** `[PENDING]`" in block
    assert "**Task ID:** `esc_test_1`" in block
    assert "**Timestamp:** 2026-07-28 00:00:00 UTC" in block
    assert "PermissionError: denied" in block
    assert "Fix host permissions." in block


def test_append_pending_uses_injectable_path(tmp_path: Path) -> None:
    ledger = tmp_path / "nested" / "patch_ledger.md"
    ticket = format_escalation_ticket(
        task_id="esc_inject",
        error_trace="OSError: boom",
        recommended_fix="Retry after fix.",
        objective="Inject path test",
    )
    dest = append_pending_ticket(ticket, ledger_path=ledger)
    assert dest == ledger
    body = ledger.read_text(encoding="utf-8")
    assert body.startswith("# Donna Patch Ledger")
    assert "[PENDING]" in body
    assert "OSError: boom" in body


def test_set_ledger_path_override(tmp_path: Path) -> None:
    ledger = tmp_path / "override_ledger.md"
    set_ledger_path_override(ledger)
    try:
        assert resolve_ledger_path() == ledger
        meta = write_escalation_ticket(
            {
                "execution_error": "ModuleNotFoundError: No module named 'x'",
                "session_id": "ov-1",
            },
            reason="unit_override",
            objective="Override path write",
        )
        assert meta["ok"] is True
        assert ledger.is_file()
        assert "[PENDING]" in ledger.read_text(encoding="utf-8")
    finally:
        set_ledger_path_override(None)


def test_fail_closed_fatal_os_writes_pending_ledger(tmp_path: Path) -> None:
    """Simulate fatal OS error → fail_closed flushes ``[PENDING]`` to mock ledger."""
    ledger = tmp_path / "dana_security" / "patch_ledger.md"
    obs = (
        "exit_code=1\nstderr:\n"
        "PermissionError: [Errno 13] Permission denied: '/etc/shadow'"
    )
    closed = fail_closed_node(
        {
            "execution_error": obs,
            "fatal_block": True,
            "session_id": "fatal-ledger-1",
            "critique_history": [],
            "retry_count": 0,
            "patch_ledger_path": str(ledger),
            **store_raw_trace({}, PermissionError(obs), {"node": "tools"}),
        }
    )
    assert closed.get("fatal_block") is True
    assert closed.get("halt") is True
    assert closed.get("final_raw") == FATAL_OS_BLOCK_MSG
    assert ledger.is_file()
    body = ledger.read_text(encoding="utf-8")
    assert "[PENDING]" in body
    assert "PermissionError" in body
    assert "Recommended Fix" in body


def test_fail_closed_exhausted_retries_writes_pending(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.md"
    closed = fail_closed_node(
        {
            "execution_error": "ZeroDivisionError: division by zero",
            "fatal_block": False,
            "session_id": "exhausted-1",
            "critique_history": ["fix divisor"],
            "retry_count": 3,
            "patch_ledger_path": str(ledger),
        }
    )
    assert closed.get("halt") is True
    assert "FAIL_CLOSED" in str(closed.get("final_raw") or "")
    body = ledger.read_text(encoding="utf-8")
    assert "[PENDING]" in body
    assert "ZeroDivisionError" in body
