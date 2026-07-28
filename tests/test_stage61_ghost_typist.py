"""Stage 6.1 — Ghost Typist Operator closed-loop tests."""

from __future__ import annotations

import os
from pathlib import Path

from dana.memory.blackboard import (
    enqueue_action,
    get_action,
    init_blackboard,
    is_heavy_actuator_tool,
)
from dana.middleware.actuator_executor import process_action
from dana.operators.ghost_typist import (
    GhostTypistOperator,
    evaluate_visual_guard,
    type_stealth_text,
)


def test_type_stealth_text_is_heavy() -> None:
    assert is_heavy_actuator_tool("type_stealth_text")


def test_evaluate_visual_guard_pause_on_popup() -> None:
    safe, reason = evaluate_visual_guard(
        baseline="Notepad focused editing notes.txt caret visible",
        current="A modal popup dialog appeared requesting permissions",
    )
    assert safe is False
    assert "unsafe" in reason or "drastic" in reason


def test_evaluate_visual_guard_ok_stable() -> None:
    base = "Notepad focused editing notes.txt with visible caret in document body"
    safe, reason = evaluate_visual_guard(baseline=base, current=base + " still typing")
    assert safe is True
    assert reason == "ok"


def test_ghost_typist_completes_dry_run(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("DONNA_OS_DRY_RUN", "1")
    monkeypatch.setenv("DONNA_GHOST_SKIP_HOTKEY", "1")
    visual = {"v": "Notepad focused document.txt caret blinking in editor pane"}

    typed: list[str] = []

    op = GhostTypistOperator(
        chunk_size=16,
        read_visual=lambda: visual["v"],
        type_char=lambda ch: (typed.append(ch) or True),
    )
    paragraph = "Hello from Ghost Typist Operator closed loop test."
    result = op.run(paragraph, wait_hotkey=True)
    assert result["ok"] is True
    assert result["paused"] is False
    assert "".join(typed) == paragraph
    assert result["chunks_done"] >= 2


def test_ghost_typist_pauses_when_focus_lost(monkeypatch) -> None:  # noqa: ANN001
    """Simulate clicking away mid-paragraph → Operator pauses."""
    monkeypatch.setenv("DONNA_OS_DRY_RUN", "1")
    monkeypatch.setenv("DONNA_GHOST_SKIP_HOTKEY", "1")

    states = {
        "n": 0,
        "baseline": (
            "Notepad window focused title notes.txt caret visible in text area"
        ),
    }

    def _visual() -> str:
        # First sense = baseline; after first chunk, user clicked Chrome.
        states["n"] += 1
        if states["n"] <= 1:
            return states["baseline"]
        return (
            "Google Chrome browser window active — different window, "
            "search bar focused, Notepad lost focus in background"
        )

    typed: list[str] = []
    op = GhostTypistOperator(
        chunk_size=15,
        read_visual=_visual,
        type_char=lambda ch: (typed.append(ch) or True),
    )
    paragraph = (
        "This is a longer paragraph that the Ghost Typist should pause "
        "midway when the user clicks away from Notepad into another app."
    )
    result = op.run(paragraph, wait_hotkey=False)
    assert result["paused"] is True
    assert result["ok"] is False
    assert typed  # some chars landed before pause
    assert len(typed) < len(paragraph)
    assert "pause_reason" in result


def test_actuator_dispatches_type_stealth_text(
    tmp_path: Path, monkeypatch
) -> None:  # noqa: ANN001
    db = tmp_path / "bb.db"
    monkeypatch.setenv("DONNA_OS_DRY_RUN", "1")
    monkeypatch.setenv("DONNA_GHOST_SKIP_HOTKEY", "1")
    monkeypatch.setenv("DONNA_DISABLE_TOAST", "1")
    init_blackboard(db)

    monkeypatch.setattr(
        "dana.middleware.actuator_executor.resolve_action",
        lambda action_id, status, result="", db_path=None, **kw: __import__(
            "dana.memory.blackboard", fromlist=["resolve_action"]
        ).resolve_action(
            action_id,
            status=status,
            result=result,
            error_context=kw.get("error_context", ""),
            db_path=db,
        ),
    )

    aid = enqueue_action(
        "type_stealth_text",
        {"text": "Stealth hello.", "wait_hotkey": False},
        session_id="ghost-test",
        db_path=db,
    )
    from dana.memory.blackboard import claim_next_pending

    claimed = claim_next_pending(db_path=db)
    assert claimed is not None
    assert claimed["action_id"] == aid

    stats = process_action(claimed, db_path=db)
    assert stats["status"] == "completed"
    row = get_action(aid, db_path=db)
    assert row is not None
    assert "OK: type_stealth_text" in (row.get("result") or "")


def test_type_stealth_text_entry(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("DONNA_OS_DRY_RUN", "1")
    monkeypatch.setenv("DONNA_GHOST_SKIP_HOTKEY", "1")
    out = type_stealth_text("abc", wait_hotkey=False)
    assert out.startswith("OK: type_stealth_text")
