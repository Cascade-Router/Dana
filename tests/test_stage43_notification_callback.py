"""Stage 4.3 — unread action receipts, toast telemetry, chat piggyback."""

from __future__ import annotations

import json
from pathlib import Path

from dana.agentic import run_lightweight_chat
from dana.memory.blackboard import (
    enqueue_action,
    format_background_system_alert,
    get_action,
    get_and_clear_unread_notifications,
    init_blackboard,
    resolve_action,
)
from dana.middleware.actuator_executor import process_action
from dana.middleware.toast_notify import format_actuator_toast


def test_get_and_clear_unread_notifications(tmp_path: Path) -> None:
    db = tmp_path / "bb.db"
    init_blackboard(db)
    aid = enqueue_action(
        "draft_cursor_prompt",
        {"objective": "x", "context": "y"},
        session_id="sess-a",
        db_path=db,
    )
    resolve_action(aid, status="completed", result="OK", db_path=db)
    # Still pending notify.
    assert get_action(aid, db_path=db)["is_notified"] is False

    unread = get_and_clear_unread_notifications("sess-a", db_path=db)
    assert len(unread) == 1
    assert unread[0]["action_id"] == aid
    assert unread[0]["tool_name"] == "draft_cursor_prompt"
    assert get_action(aid, db_path=db)["is_notified"] is True
    # Second call is empty (read-receipt cleared).
    assert get_and_clear_unread_notifications("sess-a", db_path=db) == []


def test_format_background_system_alert() -> None:
    text = format_background_system_alert(
        [
            {
                "tool_name": "draft_cursor_prompt",
                "status": "completed",
            }
        ]
    )
    assert text.startswith("[BACKGROUND SYSTEM ALERT:")
    assert "draft_cursor_prompt" in text
    assert "successfully finished" in text
    assert format_background_system_alert([]) == ""


def test_toast_copy() -> None:
    title, body = format_actuator_toast("draft_cursor_prompt", "completed")
    assert title == "Donna Task"
    assert body == "Donna Task: draft_cursor_prompt completed."


def test_actuator_emits_notification_toast(
    tmp_path: Path, monkeypatch
) -> None:  # noqa: ANN001
    db = tmp_path / "bb.db"
    out = tmp_path / "donna_telemetry.jsonl"
    monkeypatch.setattr("dana.telemetry.TELEMETRY_JSONL_PATH", out)
    monkeypatch.setenv("DONNA_DISABLE_TOAST", "1")
    monkeypatch.setattr(
        "dana.middleware.actuator_executor.execute_tool_payload",
        lambda tool_name, arguments, **_kw: "OK: done",
    )
    monkeypatch.setattr(
        "dana.middleware.actuator_executor.resolve_action",
        lambda action_id, status, result="", db_path=None, **kw: resolve_action(
            action_id,
            status=status,
            result=result,
            error_context=kw.get("error_context", ""),
            db_path=db,
        ),
    )
    shown: list[tuple[str, str]] = []

    def _fake_toast(title: str, message: str, **kwargs):  # noqa: ANN003
        shown.append((title, message))
        return True

    import dana.middleware.toast_notify as tn

    monkeypatch.setattr(tn, "show_silent_toast", _fake_toast)

    init_blackboard(db)
    from dana.memory.blackboard import claim_next_pending

    aid = enqueue_action("draft_cursor_prompt", {"objective": "t"}, db_path=db)
    claimed = claim_next_pending(db_path=db)
    assert claimed is not None

    process_action(claimed, db_path=db)
    assert shown, "toast helper should have been invoked"
    assert "draft_cursor_prompt" in shown[0][1]
    tags = [
        json.loads(line)["tag"]
        for line in out.read_text(encoding="utf-8").strip().splitlines()
    ]
    assert "[NOTIFICATION_TOAST]" in tags
    assert get_action(aid, db_path=db)["status"] == "completed"


def test_chat_piggyback_injects_alert(
    tmp_path: Path, monkeypatch
) -> None:  # noqa: ANN001
    db = tmp_path / "bb.db"
    out = tmp_path / "donna_telemetry.jsonl"
    monkeypatch.setattr("dana.telemetry.TELEMETRY_JSONL_PATH", out)
    monkeypatch.setattr(
        "dana.memory.blackboard.BLACKBOARD_DB_PATH",
        db,
    )
    # Also patch the imported default path used by init via module attribute.
    import dana.memory.blackboard as bb

    monkeypatch.setattr(bb, "BLACKBOARD_DB_PATH", db)
    init_blackboard(db)
    aid = enqueue_action(
        "draft_cursor_prompt",
        {"objective": "x", "context": "y"},
        session_id="chat-sess",
        db_path=db,
    )
    resolve_action(aid, status="completed", result="OK", db_path=db)

    captured: list[list[dict[str, str]]] = []

    def _ask(messages, model=None):  # noqa: ANN001
        captured.append(messages)
        return "Got it — that ticket is done."

    result = run_lightweight_chat(
        user_text="hey donna",
        system_prompt="You are Donna.",
        ask_fn=_ask,
        use_chat_memory=False,
        session_id="chat-sess",
    )
    assert result.final_text
    assert captured
    system = captured[0][0]["content"]
    assert "[BACKGROUND SYSTEM ALERT:" in system
    assert "draft_cursor_prompt" in system
    # Cleared after piggyback.
    assert get_and_clear_unread_notifications("chat-sess", db_path=db) == []
    tags = [
        json.loads(line)["tag"]
        for line in out.read_text(encoding="utf-8").strip().splitlines()
    ]
    assert "[NOTIFICATION_PIGGYBACK]" in tags
