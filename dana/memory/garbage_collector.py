"""Stage 7.3 — Blackboard garbage collector (prune + VACUUM)."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from dana.memory.blackboard import BLACKBOARD_DB_PATH, _LOCK, init_blackboard


def prune_action_queue(
    *,
    db_path: Path | str | None = None,
    older_than_minutes: float = 10.0,
    vacuum: bool = True,
) -> dict[str, Any]:
    """Delete settled action_queue rows older than ``older_than_minutes``, then VACUUM.

    Settled statuses: ``completed``, ``failed``, ``cancelled``.
    Age is measured from ``updated_at`` (ISO timestamps; ticket's ``timestamp``
    maps to this column).
    """
    path = init_blackboard(db_path)
    minutes = max(0.1, float(older_than_minutes))
    cutoff = (
        datetime.now(timezone.utc) - timedelta(minutes=minutes)
    ).isoformat()
    deleted = 0
    with _LOCK:
        with sqlite3.connect(str(path), timeout=60.0) as conn:
            cur = conn.execute(
                "DELETE FROM action_queue "
                "WHERE status IN ('completed', 'failed', 'cancelled') "
                "AND updated_at < ?",
                (cutoff,),
            )
            deleted = int(cur.rowcount or 0)
            conn.commit()

    vacuumed = False
    if vacuum:
        # VACUUM cannot run inside another transaction; open a fresh connection.
        try:
            with sqlite3.connect(str(path), timeout=120.0) as conn:
                conn.execute("VACUUM")
                vacuumed = True
        except sqlite3.Error:
            vacuumed = False

    return {
        "ok": True,
        "deleted": deleted,
        "cutoff": cutoff,
        "vacuumed": vacuumed,
        "db_path": str(path),
    }


def run_blackboard_gc(
    *,
    db_path: Path | str | None = None,
    older_than_minutes: float = 10.0,
) -> dict[str, Any]:
    """Public entry used by Jason finalize and manual maintenance."""
    return prune_action_queue(
        db_path=db_path or BLACKBOARD_DB_PATH,
        older_than_minutes=older_than_minutes,
        vacuum=True,
    )
