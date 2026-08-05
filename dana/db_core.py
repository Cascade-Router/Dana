"""SQLite-backed structured logging for VAD / watchdog runtime events.

Provides a small durable sink so planners can query state transitions without
scraping text logs. File lives under ``logs/dana_events.db``.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from dana.paths import LOGS_DIR

_DB_PATH = Path(LOGS_DIR) / "dana_events.db"
_LOCK = threading.RLock()
_INITIALIZED = False


def db_path() -> Path:
    return _DB_PATH


def _connect() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(_DB_PATH), timeout=30.0)
    con.execute("PRAGMA journal_mode=WAL")
    return con


def ensure_schema() -> None:
    global _INITIALIZED
    with _LOCK:
        if _INITIALIZED and _DB_PATH.is_file():
            return
        with _connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    kind TEXT NOT NULL,
                    source TEXT NOT NULL,
                    message TEXT NOT NULL,
                    payload TEXT
                )
                """
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_kind_ts ON events(kind, ts)"
            )
            con.commit()
        _INITIALIZED = True


def log_event(
    kind: str,
    message: str,
    *,
    source: str = "dana",
    payload: dict[str, Any] | None = None,
) -> None:
    """Append one structured event (best-effort; never raises to callers)."""
    try:
        ensure_schema()
        import json

        blob = json.dumps(payload, ensure_ascii=False) if payload else None
        with _LOCK:
            with _connect() as con:
                con.execute(
                    "INSERT INTO events(ts, kind, source, message, payload) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        time.time(),
                        str(kind or "event"),
                        str(source or "dana"),
                        str(message or ""),
                        blob,
                    ),
                )
                con.commit()
    except Exception:  # noqa: BLE001
        pass


def log_vad_state(state: str, *, detail: str = "", **extra: Any) -> None:
    """Record a VAD state transition (listening / standby / abort / speech)."""
    log_event(
        "vad_state",
        detail or state,
        source="vad",
        payload={"state": state, **extra},
    )


def log_watchdog_event(message: str, *, level: str = "info", **extra: Any) -> None:
    """Record a watchdog graph lifecycle / error breadcrumb."""
    log_event(
        "watchdog",
        message,
        source="watchdog_graph",
        payload={"level": level, **extra},
    )


__all__ = (
    "db_path",
    "ensure_schema",
    "log_event",
    "log_vad_state",
    "log_watchdog_event",
)
