"""Stage 7.3 — Blackboard GC + Ghost Typist is_typing mute flag."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dana.memory.blackboard import (
    enqueue_action,
    get_action,
    init_blackboard,
    is_typing,
    resolve_action,
    set_is_typing,
)
from dana.memory.garbage_collector import prune_action_queue
from dana.operators.ghost_typist import GhostTypistOperator


def test_system_state_is_typing_roundtrip(tmp_path: Path) -> None:
    db = tmp_path / "bb.db"
    init_blackboard(db)
    with sqlite3.connect(str(db)) as conn:
        tables = {
            str(r[0])
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert "system_state" in tables
    set_is_typing(True, db_path=db)
    assert is_typing(db_path=db) is True
    set_is_typing(False, db_path=db)
    assert is_typing(db_path=db) is False


def test_ghost_typist_sets_is_typing_flag(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("DONNA_OS_DRY_RUN", "1")
    monkeypatch.setenv("DONNA_GHOST_SKIP_HOTKEY", "1")
    db = tmp_path / "bb.db"
    init_blackboard(db)
    monkeypatch.setattr(
        "dana.memory.blackboard.BLACKBOARD_DB_PATH",
        db,
    )
    seen: list[bool] = []

    def _type(ch: str) -> bool:
        seen.append(is_typing(db_path=db))
        return True

    op = GhostTypistOperator(
        type_char=_type,
        wait_hotkey_fn=lambda *_a, **_k: True,
        read_visual=lambda: "stable desk",
        chunk_size=20,
    )
    result = op.run("Hi", wait_hotkey=False)
    assert result.get("ok") is True
    assert seen and all(seen), "is_typing should be 1 during keystrokes"
    assert is_typing(db_path=db) is False


def test_prune_action_queue_deletes_old_settled(tmp_path: Path) -> None:
    db = tmp_path / "bb.db"
    init_blackboard(db)
    aid = enqueue_action("web_search", {"query": "x"}, db_path=db)
    resolve_action(aid, status="completed", result="OK", db_path=db)
    # Backdate updated_at beyond the 10-minute window.
    old = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
    with sqlite3.connect(str(db)) as conn:
        conn.execute(
            "UPDATE action_queue SET updated_at = ? WHERE action_id = ?",
            (old, aid),
        )
        conn.commit()
    # Keep a fresh completed row.
    aid2 = enqueue_action("web_search", {"query": "y"}, db_path=db)
    resolve_action(aid2, status="completed", result="OK", db_path=db)

    stats = prune_action_queue(db_path=db, older_than_minutes=10.0, vacuum=True)
    assert stats["deleted"] >= 1
    assert stats["vacuumed"] is True
    assert get_action(aid, db_path=db) is None
    assert get_action(aid2, db_path=db) is not None


def test_jason_finalize_runs_gc(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    db = tmp_path / "bb.db"
    init_blackboard(db)
    called: list[dict] = []

    def _gc(*, db_path=None, older_than_minutes=10.0):  # noqa: ANN001
        out = {"ok": True, "deleted": 0, "vacuumed": True, "db_path": str(db_path)}
        called.append(out)
        return out

    monkeypatch.setattr(
        "dana.memory.garbage_collector.run_blackboard_gc",
        _gc,
    )
    from dana.management.jason_supervisor import build_bulk_evaluate_slides_graph

    graph = build_bulk_evaluate_slides_graph(db_path=db)
    final = graph.invoke(
        {
            "directory": str(tmp_path),
            "session_id": "gc-test",
            "slides": [],
            "index": 0,
            "evaluations": [],
            "enqueued": [],
            "skipped": [],
            "status": "start",
            "history": [],
        }
    )
    assert final.get("status") == "complete"
    assert called, "finalize should invoke blackboard GC"
    hist = final.get("history") or []
    assert any(isinstance(h, dict) and h.get("event") == "done" for h in hist)
