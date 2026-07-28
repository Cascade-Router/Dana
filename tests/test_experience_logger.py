"""Episodic Watchdog SQLite logger tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from dana.swarm.experience_logger import (
    count_episodes,
    init_db,
    log_watchdog_episode,
)
from dana.swarm.watchdog_graph import log_episode


def test_log_watchdog_episode_writes_row(tmp_path: Path) -> None:
    db = tmp_path / "watchdog_history.db"
    init_db(db)
    row_id = log_watchdog_episode(
        {
            "task": "Alert when Notepad opens",
            "code": "assert True\nprint('ok')\n",
            "feedback": "APPROVED",
            "status": "executed",
            "revisions": 1,
            "history": [
                {
                    "stage": "titan_eval",
                    "revision": 1,
                    "code": "assert True\nprint('ok')\n",
                    "feedback": "APPROVED",
                    "status": "APPROVED",
                }
            ],
        },
        db_path=db,
    )
    assert row_id >= 1
    assert count_episodes(db) == 1

    with sqlite3.connect(str(db)) as conn:
        row = conn.execute(
            "SELECT task_prompt, generated_code, jason_feedback, "
            "revisions_needed, execution_status, trajectory_json FROM watchdog_history"
        ).fetchone()
    assert row is not None
    assert row[0] == "Alert when Notepad opens"
    assert "assert True" in row[1]
    assert row[2] == "APPROVED"
    assert row[3] == 1
    assert row[4] == "executed"
    assert "titan_eval" in row[5]
    print("[PASS] experience_logger row write")


def test_log_episode_node_is_best_effort(monkeypatch, tmp_path: Path) -> None:
    db = tmp_path / "ep.db"
    monkeypatch.setattr(
        "dana.swarm.experience_logger.DEFAULT_DB_PATH",
        db,
    )
    out = log_episode(
        {
            "task": "t",
            "code": "x=1",
            "feedback": "REJECTED: missing tts",
            "status": "REJECTED: missing tts",
            "revisions": 3,
            "history": [],
        }
    )
    assert out == {}
    assert count_episodes(db) == 1
    print("[PASS] log_episode graph node")


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        test_log_watchdog_episode_writes_row(Path(d))
    print("OK")
