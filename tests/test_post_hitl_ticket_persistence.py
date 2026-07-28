"""Post-HITL ticket persistence — defer force-exec + sync ledger write."""

from __future__ import annotations

from pathlib import Path


def test_validation_bounce_not_logged_to_bug_tracker(tmp_path: Path, monkeypatch) -> None:
    from dana import agentic as ag
    from dana import bug_tracker as bt

    target = tmp_path / "bug_tracker.json"
    monkeypatch.setattr(bt, "BUG_TRACKER_PATH", target)
    monkeypatch.setattr(bt, "TRACKER_DIR", tmp_path)
    monkeypatch.setattr(bt, "PENDING_PATCHES_DIR", tmp_path / "pending")

    ag._maybe_record_bug_tracker(
        user_text="log a ticket about drama servers",
        spoken="I couldn't finish that cleanly — please ask me again.",
        last_obs=(
            "ERROR: Validation Error: Value error, context is an intent-echo "
            "payload; require root cause, step-by-step changes, and acceptance "
            "criteria."
        ),
        tool_trace=[
            {
                "tool": "draft_cursor_prompt",
                "observation": "ERROR: Validation Error: intent-echo",
                "forced": True,
            }
        ],
        had_errors=True,
    )
    assert not target.exists() or bt.load_bug_tracker(target) == []


def test_draft_cursor_writes_ledger_with_makedirs(tmp_path: Path, monkeypatch) -> None:
    from dana.tools.general import draft_cursor_prompt as mod

    ledger_dir = tmp_path / "dana_security"
    ledger = ledger_dir / "patch_ledger.md"
    # Parent missing until write — _append_ticket_to_ledger must mkdir.
    monkeypatch.setattr(mod, "_ledger_path", lambda: ledger)

    # Already-enriched shape so enrich_draft_cursor_args does not flatten paths.
    ctx = (
        "**Technical intent:** Persist approved HITL tickets to the patch ledger\n"
        "**Target Files:** dana/agentic_react_graph.py, dana/tools/guards.py\n"
        "\n"
        "Root cause: Forced broker exec skipped HITL and wrote nowhere useful.\n"
        "Step-by-step changes: 1) defer draft_cursor_prompt force-exec "
        "2) after Approve write ledger inline (no actuator enqueue).\n"
        "Acceptance criteria: Approved tickets appear in patch_ledger.md with "
        "[PENDING] status."
    )
    result = mod.draft_cursor_prompt(
        objective="Persist approved HITL tickets to the patch ledger",
        context=ctx,
    )
    assert "Ticket added to patch_ledger.md" in result
    assert ledger.is_file()
    body = ledger.read_text(encoding="utf-8")
    assert "[PENDING]" in body
    assert "Persist approved HITL" in body


def test_defer_forced_tool_covers_draft_cursor() -> None:
    from dana.moa_tool_shim import defer_forced_tool_for_moa

    assert defer_forced_tool_for_moa("draft_cursor_prompt") is True
