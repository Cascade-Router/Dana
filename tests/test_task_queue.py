"""Structured task_queue.json dispatcher (replaces flat input.txt)."""

from __future__ import annotations

import json
from pathlib import Path

from dana.tools.broker import IntentBroker
from dana.tools.task_queue import (
    load_task_queue,
    migrate_legacy_input_txt,
    pending_count,
    save_task_queue,
    update_task_status,
)


def test_dispatch_pending_marks_completed_and_failed(tmp_path: Path) -> None:
    queue = tmp_path / "task_queue.json"
    save_task_queue(
        [
            {
                "id": "task_ok",
                "status": "pending",
                "command": "say hello",
            },
            {
                "id": "task_bad",
                "status": "pending",
                "command": "boom",
            },
            {
                "id": "task_done",
                "status": "completed",
                "command": "already done",
            },
            {
                "id": "task_empty",
                "status": "pending",
                "command": "   ",
            },
        ],
        path=queue,
    )

    seen: list[str] = []

    def handler(command: str) -> None:
        seen.append(command)
        if command == "boom":
            raise RuntimeError("simulated failure")

    broker = IntentBroker(registry={})
    results = broker.dispatch_pending_tasks(handler, path=queue)

    assert seen == ["say hello", "boom"]
    assert pending_count(queue) == 0

    by_id = {r["id"]: r["status"] for r in results}
    assert by_id["task_ok"] == "completed"
    assert by_id["task_bad"] == "failed"
    assert by_id["task_empty"] == "failed"

    on_disk = {t["id"]: t for t in load_task_queue(queue)}
    assert on_disk["task_ok"]["status"] == "completed"
    assert on_disk["task_bad"]["status"] == "failed"
    assert "simulated failure" in str(on_disk["task_bad"].get("error") or "")
    assert on_disk["task_done"]["status"] == "completed"
    assert on_disk["task_empty"]["status"] == "failed"
    print("[PASS] dispatch_pending isolates failures")


def test_failed_task_does_not_block_next(tmp_path: Path) -> None:
    queue = tmp_path / "task_queue.json"
    save_task_queue(
        [
            {"id": "a", "status": "pending", "command": "first"},
            {"id": "b", "status": "pending", "command": "second"},
        ],
        path=queue,
    )
    order: list[str] = []

    def handler(command: str) -> None:
        order.append(command)
        if command == "first":
            raise ValueError("nope")

    IntentBroker(registry={}).dispatch_pending_tasks(handler, path=queue)
    assert order == ["first", "second"]
    statuses = {t["id"]: t["status"] for t in load_task_queue(queue)}
    assert statuses == {"a": "failed", "b": "completed"}
    print("[PASS] failure isolation continues queue")


def test_migrate_legacy_input_txt(tmp_path: Path) -> None:
    queue = tmp_path / "task_queue.json"
    legacy = tmp_path / "input.txt"
    legacy.write_text(
        '"Donna, please log a ticket for the audio pipeline"',
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
