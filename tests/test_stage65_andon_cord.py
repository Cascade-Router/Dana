"""Stage 6.5 — Andon Cord: operator failure → error_context → wake Jason."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from dana.management.jason_supervisor import (
    format_wake_cto_ticket,
    recovery_mode,
)
from dana.memory.blackboard import (
    claim_next_pending,
    enqueue_action,
    get_action,
    get_sensor_state,
    init_blackboard,
    set_sensor_state,
)
from dana.middleware.actuator_executor import process_action


def test_error_context_column_exists(tmp_path: Path) -> None:
    db = tmp_path / "bb.db"
    init_blackboard(db)
    with sqlite3.connect(str(db)) as conn:
        cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(action_queue)")}
    assert "error_context" in cols


def test_navigate_missing_target_pulls_andon(
    tmp_path: Path, monkeypatch
) -> None:  # noqa: ANN001
    """Dry-run: navigate_and_click for absent target → failed + Jason wake."""
    monkeypatch.setenv("DANA_OS_DRY_RUN", "1")
    monkeypatch.setenv("DANA_DISABLE_TOAST", "1")
    db = tmp_path / "bb.db"
    init_blackboard(db)
    # Screen has no matching Target box (typed OCR topic).
    from dana.memory.blackboard import publish_perception_ocr

    publish_perception_ocr(
        "Florence-2 OCR: empty desktop, no Target, no Enter Comments.",
        producer="test",
        db_path=db,
    )

    aid = enqueue_action(
        "navigate_and_click",
        {
            "query": "Target",
            "visual_context": "Florence-2 OCR: empty desktop, no Target.",
        },
        session_id="andon-test",
        db_path=db,
    )
    claimed = claim_next_pending(db_path=db)
    assert claimed is not None
    assert claimed["action_id"] == aid

    stats = process_action(claimed, db_path=db)
    assert stats["status"] == "failed"
    assert stats.get("andon"), "expected Andon Cord to fire"

    row = get_action(aid, db_path=db)
    assert row is not None
    assert row["status"] == "failed"
    assert (row.get("error_context") or "").strip()
    assert "not found" in (row.get("error_context") or "").lower() or "failed" in (
        row.get("error_context") or ""
    ).lower()

    andon = stats["andon"]
    ticket = andon.get("ticket") or ""
    assert "Operator failed on task navigate_and_click" in ticket
    assert "Review perception.ocr" in ticket
    wake_id = int(andon.get("wake_cto_action_id") or 0)
    assert wake_id > 0
    wake = get_action(wake_id, db_path=db)
    assert wake is not None
    assert wake["tool_name"] == "wake_cto"
    assert wake["status"] == "completed"

    recovery = andon.get("recovery") or {}
    enqueued = list(recovery.get("enqueued") or [])
    tools = [str(e.get("tool_name") or "") for e in enqueued]
    assert "click_close_button" in tools
    assert "navigate_and_click" in tools

    # Pending recovery actions exist on the queue.
    pending_tools: list[str] = []
    while True:
        nxt = claim_next_pending(db_path=db)
        if nxt is None:
            break
        pending_tools.append(str(nxt.get("tool_name") or ""))
        # Resolve so we can drain without re-andon loops on retry failure.
        from dana.memory.blackboard import resolve_action

        resolve_action(
            int(nxt["action_id"]),
            status="completed",
            result="OK: test drain",
            db_path=db,
        )
    assert "click_close_button" in pending_tools
    assert "navigate_and_click" in pending_tools

    sensor = get_sensor_state("jason_andon_last_recovery", db_path=db)
    assert sensor is not None
    assert "navigate_and_click" in str(sensor.get("value") or "")


def test_recovery_mode_direct(tmp_path: Path) -> None:
    db = tmp_path / "bb.db"
    init_blackboard(db)
    from dana.memory.blackboard import publish_perception_ocr

    publish_perception_ocr(
        "modal popup Close button [10,10,40,40]",
        producer="test",
        db_path=db,
    )
    result = recovery_mode(
        failed_action_id=42,
        failed_tool="navigate_and_click",
        error_context="target box not found for query='Target'",
        failed_arguments={"query": "Target"},
        session_id="andon-direct",
        db_path=db,
    )
    assert result["ok"] is True
    assert len(result["enqueued"]) >= 2
    ticket = format_wake_cto_ticket("navigate_and_click", "boom")
    assert "Operator failed on task navigate_and_click" in ticket
