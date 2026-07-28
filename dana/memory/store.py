"""Persistent episodic memory store for Dānā (SQLite, coexists with vault/blackboard)."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
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
    UNIQUE(category, key)
);
CREATE INDEX IF NOT EXISTS idx_episodic_category
    ON episodic_facts(category);
CREATE INDEX IF NOT EXISTS idx_episodic_key
    ON episodic_facts(key);
"""


def _serialize(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9_]+", (text or "").lower()) if len(t) > 1]


class EpisodicMemoryStore:
    """SQLite-backed episodic facts (preferences, environment, outcomes)."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else _DEFAULT_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with _LOCK:
            with self._connect() as conn:
                conn.executescript(_SCHEMA)
                conn.commit()

    def add_fact(
        self,
        category: str,
        key: str,
        value: Any,
        *,
        confidence_score: float = 1.0,
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
        ts = time.time()
        with _LOCK:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO episodic_facts
                        (timestamp, category, key, value, confidence_score)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(category, key) DO UPDATE SET
                        timestamp = excluded.timestamp,
                        value = excluded.value,
                        confidence_score = excluded.confidence_score
                    """,
                    (ts, cat, k, val, conf),
                )
                row = conn.execute(
                    """
                    SELECT id, timestamp, category, key, value, confidence_score
                    FROM episodic_facts
                    WHERE category = ? AND key = ?
                    """,
                    (cat, k),
                ).fetchone()
                conn.commit()
        return dict(row) if row is not None else {
            "id": None,
            "timestamp": ts,
            "category": cat,
            "key": k,
            "value": val,
            "confidence_score": conf,
        }

    def search_facts(self, query_text: str, *, limit: int = 20) -> list[dict[str, Any]]:
        """Keyword / simple TF-style scoring over key + value (no embeddings)."""
        tokens = _tokenize(query_text)
        with _LOCK:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT id, timestamp, category, key, value, confidence_score
                    FROM episodic_facts
                    ORDER BY timestamp DESC
                    """
                ).fetchall()
        facts = [dict(r) for r in rows]
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
        with _LOCK:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT key, value
                    FROM episodic_facts
                    WHERE category = 'user_preference'
                    ORDER BY key ASC
                    """
                ).fetchall()
        out: dict[str, Any] = {}
        for row in rows:
            raw = row["value"]
            try:
                out[row["key"]] = json.loads(raw)
            except (TypeError, ValueError, json.JSONDecodeError):
                out[row["key"]] = raw
        return out


_default_store: EpisodicMemoryStore | None = None


def get_episodic_store(db_path: str | Path | None = None) -> EpisodicMemoryStore:
    """Return a store instance (shared default, or a path-specific instance)."""
    global _default_store
    if db_path is not None:
        return EpisodicMemoryStore(db_path)
    if _default_store is None:
        _default_store = EpisodicMemoryStore()
    return _default_store
