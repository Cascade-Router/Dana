"""Structured task_queue.json dispatcher (replaces flat input.txt)."""

from __future__ import annotations

import json
from pathlib import Path

from dana.tools.task_queue import (
    load_task_queue,
    migrate_legacy_input_txt,
    save_task_queue,
    update_task_status,
)


def test_migrate_legacy_input_txt(tmp_path: Path) -> None:
    queue = tmp_path / "task_queue.json"
    legacy = tmp_path / "input.txt"
    legacy.write_text(
        '"Dana, please log a ticket for the audio pipeline"',
        encoding="utf-8",
    )
    save_task_queue([], path=queue)

    migrated = migrate_legacy_input_txt(queue_path=queue, input_path=legacy)
    assert migrated is not None
    assert "audio pipeline" in migrated
    assert legacy.read_text(encoding="utf-8") == ""
    tasks = load_task_queue(queue)
    assert len(tasks) == 1
    assert tasks[0]["status"] == "pending"
    assert "audio pipeline" in tasks[0]["command"]
    print("[PASS] legacy input.txt migrates into task_queue.json")


def test_migrate_splits_paragraphs(tmp_path: Path) -> None:
    queue = tmp_path / "task_queue.json"
    legacy = tmp_path / "input.txt"
    legacy.write_text(
        "First ticket about broker safety.\n\n"
        "Second ticket about cascade_router.\n\n"
        "Third ticket about cleanup.\n",
        encoding="utf-8",
    )
    save_task_queue([], path=queue)
    migrate_legacy_input_txt(queue_path=queue, input_path=legacy)
    tasks = load_task_queue(queue)
    assert len(tasks) == 3
    assert all(t["status"] == "pending" for t in tasks)
    assert "broker" in tasks[0]["command"]
    assert "cascade_router" in tasks[1]["command"]
    assert "cleanup" in tasks[2]["command"]
    assert legacy.read_text(encoding="utf-8") == ""
    print("[PASS] migrate splits blank-line paragraphs")


def test_update_task_status_roundtrip(tmp_path: Path) -> None:
    queue = tmp_path / "task_queue.json"
    save_task_queue(
        [{"id": "t1", "status": "pending", "command": "x"}],
        path=queue,
    )
    assert update_task_status("t1", "completed", path=queue) is True
    assert load_task_queue(queue)[0]["status"] == "completed"
    raw = json.loads(queue.read_text(encoding="utf-8"))
    assert isinstance(raw, list)
    print("[PASS] update_task_status roundtrip")
