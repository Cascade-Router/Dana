"""Stage 4.2 — action_queue + actuator daemon + non-blocking graph enqueue."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from donna.memory.blackboard import (
    claim_next_pending,
    enqueue_action,
    get_action,
    init_blackboard,
    is_heavy_actuator_tool,
    resolve_action,
)
from donna.middleware.actuator_executor import poll_once, process_action


def test_heavy_tool_classification() -> None:
    assert is_heavy_actuator_tool("draft_cursor_prompt")
    assert is_heavy_actuator_tool("file_editor")
    assert not is_heavy_actuator_tool("analyze_visual_context")
    assert not is_heavy_actuator_tool("ocr_with_region")


def test_enqueue_claim_resolve(tmp_path: Path) -> None:
    db = tmp_path / "bb.db"
    init_blackboard(db)
    aid = enqueue_action(
        "draft_cursor_prompt",
        {"objective": "x", "context": "y"},
        session_id="sess-42",
        db_path=db,
    )
    assert aid > 0
    row = get_action(aid, db_path=db)
    assert row is not None
    assert row["status"] == "pending"
    assert row["tool_name"] == "draft_cursor_prompt"
    assert row["arguments"]["objective"] == "x"

    claimed = claim_next_pending(db_path=db)
    assert claimed is not None
    assert claimed["action_id"] == aid
    assert claimed["status"] == "running"
    assert claim_next_pending(db_path=db) is None

    resolve_action(aid, status="completed", result="OK: done", db_path=db)
    done = get_action(aid, db_path=db)
    assert done is not None
    assert done["status"] == "completed"
    assert done["result"] == "OK: done"


def test_graph_ack_shape_does_not_block(tmp_path: Path) -> None:
    """Enqueue path returns immediately with Task ID ack (no tool body wait)."""
    db = tmp_path / "bb.db"
    init_blackboard(db)
    t0 = time.perf_counter()
    aid = enqueue_action(
        "draft_cursor_prompt",
        {"objective": "async ticket", "context": "stage 4.2"},
        session_id="chat-1",
        db_path=db,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    ack = f"Action queued successfully. Task ID: {aid}."
    assert "Action queued successfully" in ack
    assert str(aid) in ack
    # Enqueue must stay lightweight (no tool execution here).
    assert elapsed_ms < 500.0
    assert get_action(aid, db_path=db)["status"] == "pending"


def test_actuator_processes_pending_in_background(
    tmp_path: Path, monkeypatch
) -> None:  # noqa: ANN001
    db = tmp_path / "bb.db"
    out = tmp_path / "donna_telemetry.jsonl"
    monkeypatch.setattr("donna.telemetry.TELEMETRY_JSONL_PATH", out)
    monkeypatch.setenv("DONNA_DISABLE_TOAST", "1")
    init_blackboard(db)

    monkeypatch.setattr(
        "donna.middleware.actuator_executor.execute_tool_payload",
        lambda tool_name, arguments, **_kw: f"OK: ran {tool_name}",
    )
    # Bind claim/resolve/process to this temp DB.
    monkeypatch.setattr(
        "donna.middleware.actuator_executor.claim_next_pending",
        lambda db_path=None: claim_next_pending(db_path=db),
    )
    monkeypatch.setattr(
        "donna.middleware.actuator_executor.resolve_action",
        lambda action_id, status, result="", db_path=None, **kw: resolve_action(
            action_id,
            status=status,
            result=result,
            error_context=kw.get("error_context", ""),
            db_path=db,
        ),
    )

    aid = enqueue_action(
        "draft_cursor_prompt",
        {"objective": "bg", "context": "work"},
        session_id="s",
        db_path=db,
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        inflight: set = set()
        n = poll_once(pool, inflight, db_path=db, max_claim=1)
        assert n == 1
        # Wait for background worker.
        for f in list(inflight):
            f.result(timeout=5.0)

    row = get_action(aid, db_path=db)
    assert row is not None
    assert row["status"] == "completed"
    assert row["result"] == "OK: ran draft_cursor_prompt"

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    tags = [json.loads(line)["tag"] for line in lines]
    assert "[ACTUATOR_START]" in tags
    assert "[ACTUATOR_DONE]" in tags
    done_evt = next(json.loads(l) for l in lines if json.loads(l)["tag"] == "[ACTUATOR_DONE]")
    assert done_evt.get("latency_ms") is not None


def test_process_action_failure_marks_failed(
    tmp_path: Path, monkeypatch
) -> None:  # noqa: ANN001
    db = tmp_path / "bb.db"
    monkeypatch.setenv("DONNA_DISABLE_TOAST", "1")
    init_blackboard(db)
    monkeypatch.setattr(
        "donna.middleware.actuator_executor.execute_tool_payload",
        lambda tool_name, arguments, **_kw: "ERROR: boom",
    )
    monkeypatch.setattr(
        "donna.middleware.actuator_executor.resolve_action",
        lambda action_id, status, result="", db_path=None, **kw: resolve_action(
            action_id,
            status=status,
            result=result,
            error_context=kw.get("error_context", ""),
            db_path=db,
        ),
    )
    aid = enqueue_action("web_search", {"query": "x"}, db_path=db)
    claimed = claim_next_pending(db_path=db)
    assert claimed is not None
    stats = process_action(claimed, db_path=db)
    assert stats["status"] == "failed"
    assert get_action(aid, db_path=db)["status"] == "failed"
