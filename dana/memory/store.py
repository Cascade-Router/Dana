"""Persistent episodic memory store for Dānā (SQLite, coexists with vault/blackboard)."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

CATEGORIES = frozenset({"user_preference", "environment_fact", "task_outcome"})

_DEFAULT_DB = Path(__file__).resolve().parent / "memory.db"
_LOCK = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodic_facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    category TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    confidence_score REAL NOT NULL DEFAULT 1.0,
    ttl_seconds INTEGER,
    UNIQUE(category, key)
);
CREATE INDEX IF NOT EXISTS idx_episodic_category
    ON episodic_facts(category);
CREATE INDEX IF NOT EXISTS idx_episodic_key
    ON episodic_facts(key);
"""

_FACT_COLUMNS = (
    "id, timestamp, category, key, value, confidence_score, ttl_seconds"
)


def _serialize(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9_]+", (text or "").lower()) if len(t) > 1]


def _row_to_fact(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    fact = dict(row)
    # ``timestamp`` is the creation epoch; expose ``created_at`` as an alias.
    fact["created_at"] = fact.get("timestamp")
    return fact


class EpisodicMemoryStore:
    """SQLite-backed episodic facts (preferences, environment, outcomes)."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        self.db_path = Path(db_path) if db_path is not None else _DEFAULT_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._time_fn: Callable[[], float] = time_fn or time.time
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _migrate(self, conn: sqlite3.Connection) -> None:
        cols = {
            str(r[1])
            for r in conn.execute("PRAGMA table_info(episodic_facts)").fetchall()
        }
        if cols and "ttl_seconds" not in cols:
            conn.execute(
                "ALTER TABLE episodic_facts ADD COLUMN ttl_seconds INTEGER"
            )

    def _init_db(self) -> None:
        with _LOCK:
            with self._connect() as conn:
                conn.executescript(_SCHEMA)
                self._migrate(conn)
                conn.commit()

    def _is_expired(
        self,
        fact: dict[str, Any],
        *,
        now: float | None = None,
    ) -> bool:
        ttl = fact.get("ttl_seconds")
        if ttl is None:
            return False
        created = float(fact.get("timestamp") or fact.get("created_at") or 0.0)
        current = float(self._time_fn() if now is None else now)
        return current > created + float(ttl)

    def add_fact(
        self,
        category: str,
        key: str,
        value: Any,
        *,
        confidence_score: float = 1.0,
        ttl_seconds: int | None = None,
    ) -> dict[str, Any]:
        """Upsert a fact by (category, key). Returns the stored row as a dict."""
        cat = str(category or "").strip()
        if cat not in CATEGORIES:
            raise ValueError(
                f"category must be one of {sorted(CATEGORIES)}, got {cat!r}"
            )
        k = str(key or "").strip()
        if not k:
            raise ValueError("key must be non-empty")
        val = _serialize(value)
        conf = float(confidence_score)
        conf = max(0.0, min(1.0, conf))
        ttl = None if ttl_seconds is None else int(ttl_seconds)
        ts = float(self._time_fn())
        with _LOCK:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO episodic_facts
                        (timestamp, category, key, value, confidence_score, ttl_seconds)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(category, key) DO UPDATE SET
                        timestamp = excluded.timestamp,
                        value = excluded.value,
                        confidence_score = excluded.confidence_score,
                        ttl_seconds = excluded.ttl_seconds
                    """,
                    (ts, cat, k, val, conf, ttl),
                )
                row = conn.execute(
                    f"""
                    SELECT {_FACT_COLUMNS}
                    FROM episodic_facts
                    WHERE category = ? AND key = ?
                    """,
                    (cat, k),
                ).fetchone()
                conn.commit()
        fact = _row_to_fact(row)
        return fact if fact is not None else {
            "id": None,
            "timestamp": ts,
            "created_at": ts,
            "category": cat,
            "key": k,
            "value": val,
            "confidence_score": conf,
            "ttl_seconds": ttl,
        }

    def search_facts(self, query_text: str, *, limit: int = 20) -> list[dict[str, Any]]:
        """Keyword / simple TF-style scoring over key + value (no embeddings)."""
        tokens = _tokenize(query_text)
        now = float(self._time_fn())
        with _LOCK:
            with self._connect() as conn:
                rows = conn.execute(
                    f"""
                    SELECT {_FACT_COLUMNS}
                    FROM episodic_facts
                    ORDER BY timestamp DESC
                    """
                ).fetchall()
        facts = [
            f
            for r in rows
            if (f := _row_to_fact(r)) is not None and not self._is_expired(f, now=now)
        ]
        if not tokens:
            return facts[: max(1, int(limit))]

        scored: list[tuple[float, dict[str, Any]]] = []
        for fact in facts:
            blob_tokens = set(_tokenize(fact.get("key"))) | set(
                _tokenize(f"{fact.get('key')} {fact.get('value')}")
            )
            # Prefer exact key token hits; also score value overlap.
            key_toks = set(_tokenize(str(fact.get("key") or "")))
            val_toks = set(_tokenize(str(fact.get("value") or "")))
            hit = 0.0
            for t in tokens:
                if t in key_toks or t == str(fact.get("key") or "").lower():
                    hit += 3.0
                elif t in val_toks:
                    hit += 1.0
                elif any(t in bt for bt in blob_tokens):
                    hit += 0.5
            if hit <= 0:
                continue
            conf = float(fact.get("confidence_score") or 1.0)
            scored.append((hit * conf, fact))

        scored.sort(key=lambda x: (-x[0], -float(x[1].get("timestamp") or 0)))
        return [f for _, f in scored[: max(1, int(limit))]]

    def get_all_preferences(self) -> dict[str, Any]:
        """Active user preferences as ``{key: parsed_value}``."""
        now = float(self._time_fn())
        with _LOCK:
            with self._connect() as conn:
                rows = conn.execute(
                    f"""
                    SELECT {_FACT_COLUMNS}
                    FROM episodic_facts
                    WHERE category = 'user_preference'
                    ORDER BY key ASC
                    """
                ).fetchall()
        out: dict[str, Any] = {}
        for row in rows:
            fact = _row_to_fact(row)
            if fact is None or self._is_expired(fact, now=now):
                continue
            raw = fact["value"]
            try:
                out[fact["key"]] = json.loads(raw)
            except (TypeError, ValueError, json.JSONDecodeError):
                out[fact["key"]] = raw
        return out

    def list_facts(self, *, include_expired: bool = False) -> list[dict[str, Any]]:
        """Return all facts (optionally including TTL-expired rows)."""
        now = float(self._time_fn())
        with _LOCK:
            with self._connect() as conn:
                rows = conn.execute(
                    f"""
                    SELECT {_FACT_COLUMNS}
                    FROM episodic_facts
                    ORDER BY timestamp DESC
                    """
                ).fetchall()
        facts: list[dict[str, Any]] = []
        for row in rows:
            fact = _row_to_fact(row)
            if fact is None:
                continue
            if not include_expired and self._is_expired(fact, now=now):
                continue
            facts.append(fact)
        return facts

    def delete_fact(self, fact_id: int) -> bool:
        """Delete a fact by primary key. Returns True if a row was removed."""
        with _LOCK:
            with self._connect() as conn:
                cur = conn.execute(
                    "DELETE FROM episodic_facts WHERE id = ?",
                    (int(fact_id),),
                )
                conn.commit()
                return int(cur.rowcount or 0) > 0

    def prune_expired_entries(self) -> int:
        """Delete TTL-expired records. Returns number of rows deleted."""
        now = float(self._time_fn())
        with _LOCK:
            with self._connect() as conn:
                cur = conn.execute(
                    """
                    DELETE FROM episodic_facts
                    WHERE ttl_seconds IS NOT NULL
                      AND (? > timestamp + ttl_seconds)
                    """,
                    (now,),
                )
                deleted = int(cur.rowcount or 0)
                conn.commit()
        return deleted


_default_store: EpisodicMemoryStore | None = None


def get_episodic_store(db_path: str | Path | None = None) -> EpisodicMemoryStore:
    """Return a store instance (shared default, or a path-specific instance)."""
    global _default_store
    if db_path is not None:
        return EpisodicMemoryStore(db_path)
    if _default_store is None:
        _default_store = EpisodicMemoryStore()
    return _default_store
