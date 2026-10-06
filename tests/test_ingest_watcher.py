"""Tests for hardened input.txt ingest watcher."""

from __future__ import annotations

import json
from pathlib import Path

import ingest


def test_ensure_input_txt_creates_missing(tmp_path, monkeypatch) -> None:
    target = tmp_path / "execution_jail" / "input.txt"
    monkeypatch.setattr(ingest, "INPUT_FILE", target)
    assert not target.exists()
    path = ingest.ensure_input_txt()
    assert path == target
    assert target.is_file()
    assert target.read_text(encoding="utf-8") == ""


def test_ingest_missing_file_does_not_raise(tmp_path, monkeypatch) -> None:
    target = tmp_path / "execution_jail" / "input.txt"
    queue = tmp_path / "execution_jail" / "task_queue.json"
    monkeypatch.setattr(ingest, "INPUT_FILE", target)
    monkeypatch.setattr(ingest, "QUEUE_FILE", queue)
    # ensure creates empty — ingest returns 0, no exception
    n = ingest.ingest_text_to_queue(empty_sleep=0.0)
    assert n == 0
    assert target.is_file()


def test_ingest_queues_and_clears(tmp_path, monkeypatch) -> None:
    target = tmp_path / "execution_jail" / "input.txt"
    queue = tmp_path / "execution_jail" / "task_queue.json"
    monkeypatch.setattr(ingest, "INPUT_FILE", target)
    monkeypatch.setattr(ingest, "QUEUE_FILE", queue)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("say hello from ingest test\n", encoding="utf-8")
    n = ingest.ingest_text_to_queue(empty_sleep=0.0)
    assert n == 1
    assert target.read_text(encoding="utf-8") == ""
    data = queue.read_text(encoding="utf-8")
    assert "say hello from ingest test" in data


def test_ingest_skips_commands_already_pending(tmp_path, monkeypatch) -> None:
    """Duplicates (case/whitespace-insensitive) of a pending command, or of
    another command in the same batch, are not queued twice."""
    target = tmp_path / "execution_jail" / "input.txt"
    queue = tmp_path / "execution_jail" / "task_queue.json"
    monkeypatch.setattr(ingest, "INPUT_FILE", target)
    monkeypatch.setattr(ingest, "QUEUE_FILE", queue)
    queue.parent.mkdir(parents=True, exist_ok=True)
    queue.write_text('[{"id": "a", "status": "pending", "command": "Make a box"}]\n', encoding="utf-8")
    target.write_text("make   a BOX\n\nmake a cylinder\n\nMAKE A CYLINDER\n", encoding="utf-8")
    assert ingest.ingest_text_to_queue(empty_sleep=0.0) == 1
    commands = [t["command"] for t in json.loads(queue.read_text(encoding="utf-8"))]
    assert commands == ["Make a box", "make a cylinder"]
