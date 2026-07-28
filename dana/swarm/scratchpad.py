"""Research Scratchpad — SQLite cache for deep-research Search Agent findings.

Planner → Search Agent writes raw ``search_once`` results here; WriterAgent
reads them to compile the final synthesis (never invents from an empty cache).
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dana.paths import RESEARCH_SCRATCHPAD_DB

DEFAULT_DB_PATH = RESEARCH_SCRATCHPAD_DB

_SCHEMA = """
CREATE TABLE IF NOT EXISTS research_sessions (
    session_id TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS research_findings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    objective TEXT NOT NULL,
    query_used TEXT NOT NULL,
    findings_text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(session_id) REFERENCES research_sessions(session_id)
);
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_scratchpad(db_path: Path | str | None = None) -> Path:
    """Ensure parent dirs + tables exist; return resolved DB path."""
    path = Path(db_path or DEFAULT_DB_PATH).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(path)) as conn:
        conn.executescript(_SCHEMA)
        conn.commit()
    return path


def open_session(query: str, *, db_path: Path | str | None = None) -> str:
    """Create a new research session; return ``session_id``."""
    path = init_scratchpad(db_path)
    session_id = uuid.uuid4().hex[:16]
    with sqlite3.connect(str(path)) as conn:
        conn.execute(
            "INSERT INTO research_sessions (session_id, query, created_at) "
            "VALUES (?, ?, ?)",
            (session_id, (query or "").strip(), _utc_now()),
        )
        conn.commit()
    return session_id


def write_finding(
    session_id: str,
    *,
    objective: str,
    query_used: str,
    findings_text: str,
    db_path: Path | str | None = None,
) -> int:
    """Append one Search Agent finding; return row id."""
    path = init_scratchpad(db_path)
    with sqlite3.connect(str(path)) as conn:
        cur = conn.execute(
            "INSERT INTO research_findings "
            "(session_id, objective, query_used, findings_text, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                session_id,
                (objective or "").strip(),
                (query_used or "").strip(),
                (findings_text or "").strip(),
                _utc_now(),
            ),
        )
        conn.commit()
        return int(cur.lastrowid or 0)


def read_findings(
    session_id: str,
    *,
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Return all findings for a session (oldest first)."""
    path = init_scratchpad(db_path)
    with sqlite3.connect(str(path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, objective, query_used, findings_text, created_at "
            "FROM research_findings WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def format_findings_for_writer(findings: list[dict[str, Any]]) -> str:
    """Compact scratchpad dump for WriterAgent prompts."""
    if not findings:
        return "(scratchpad empty — no search findings cached)"
    blocks: list[str] = []
    for i, row in enumerate(findings, 1):
        blocks.append(
            f"### Finding {i}\n"
            f"Objective: {row.get('objective') or ''}\n"
            f"Query used: {row.get('query_used') or ''}\n"
            f"{row.get('findings_text') or ''}\n"
        )
    return "\n".join(blocks)


def findings_as_json(findings: list[dict[str, Any]]) -> str:
    return json.dumps(findings, ensure_ascii=False, indent=2)
