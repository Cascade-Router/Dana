"""Episodic memory for Watchdog runs — SQLite log for future fine-tuning."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from dana.paths import PROJECT_ROOT, WATCHDOG_HISTORY_DB

_REPO_ROOT = PROJECT_ROOT
DEFAULT_DB_PATH = WATCHDOG_HISTORY_DB

_SCHEMA = """
CREATE TABLE IF NOT EXISTS watchdog_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    task_prompt TEXT NOT NULL,
    generated_code TEXT NOT NULL,
    jason_feedback TEXT NOT NULL,
    revisions_needed INTEGER NOT NULL,
    execution_status TEXT NOT NULL,
    trajectory_json TEXT NOT NULL DEFAULT '[]'
);
"""

_lock = threading.Lock()


def default_db_path() -> Path:
    return DEFAULT_DB_PATH


def init_db(db_path: Path | str | None = None) -> Path:
    """Ensure parent dirs + ``watchdog_history`` table exist; return resolved path."""
    path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        with sqlite3.connect(str(path)) as conn:
            conn.execute(_SCHEMA)
            conn.commit()
    return path.resolve()


def log_watchdog_episode(
    state: Mapping[str, Any],
    *,
    db_path: Path | str | None = None,
) -> int:
    """Persist a final ``WatchdogState`` into ``watchdog_history``.

    Returns the inserted row id.
    """
    path = init_db(db_path)
    ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    task = str(state.get("task") or "")
    code = str(state.get("code") or "")
    feedback = str(state.get("feedback") or "")
    try:
        revisions = int(state.get("revisions") or 0)
    except (TypeError, ValueError):
        revisions = 0
    status = str(state.get("status") or "unknown")
    trajectory_json = json.dumps(state.get("history") or [])

    with _lock:
        with sqlite3.connect(str(path)) as conn:
            cur = conn.execute(
                """
                INSERT INTO watchdog_history (
                    timestamp,
                    task_prompt,
                    generated_code,
                    jason_feedback,
                    revisions_needed,
                    execution_status,
                    trajectory_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (ts, task, code, feedback, revisions, status, trajectory_json),
            )
            conn.commit()
            return int(cur.lastrowid or 0)


def count_episodes(db_path: Path | str | None = None) -> int:
    path = init_db(db_path)
    with _lock:
        with sqlite3.connect(str(path)) as conn:
            row = conn.execute("SELECT COUNT(*) FROM watchdog_history").fetchone()
            return int(row[0] if row else 0)
