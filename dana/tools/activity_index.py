"""Day-scoped activity index — episodic facts + blackboard + log lines."""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

TOOL_ID = "list_activity_for_day"

_RELATIVE_DAY_RE = re.compile(
    r"(?i)^\s*(yesterday|today|last\s+night|this\s+morning|previous\s+session)\s*$"
)


def resolve_date_str(date_str: str | None = None) -> str:
    """Normalize ``YYYY-MM-DD`` or relative phrases to a local calendar day."""
    raw = str(date_str or "").strip()
    local = datetime.now().astimezone()
    if not raw:
        raw = "yesterday"
    m = _RELATIVE_DAY_RE.match(raw)
    if m:
        key = re.sub(r"\s+", " ", m.group(1).lower())
        if key == "today":
            day = local
        elif key in ("yesterday", "last night", "previous session"):
            day = local - timedelta(days=1)
        elif key == "this morning":
            day = local
        else:
            day = local - timedelta(days=1)
        return day.strftime("%Y-%m-%d")
    # Accept YYYY-MM-DD (optionally with time suffix).
    m2 = re.match(r"^(\d{4}-\d{2}-\d{2})", raw)
    if m2:
        return m2.group(1)
    # Fallback: treat unknown as yesterday.
    return (local - timedelta(days=1)).strftime("%Y-%m-%d")


def _day_bounds(day_label: str) -> tuple[float, float]:
    local = datetime.now().astimezone()
    y, m, d = (int(x) for x in day_label.split("-", 2))
    day = local.replace(year=y, month=m, day=d, hour=0, minute=0, second=0, microsecond=0)
    start = day.timestamp()
    end = (day + timedelta(days=1)).timestamp()
    return start, end


def _query_episodic(day_label: str, *, limit: int = 40) -> list[dict[str, Any]]:
    start, end = _day_bounds(day_label)
    out: list[dict[str, Any]] = []
    try:
        from dana.memory.store import get_episodic_store

        for fact in get_episodic_store().list_facts(include_expired=True):
            ts = float(fact.get("timestamp") or 0)
            if start <= ts < end:
                out.append(
                    {
                        "source": "episodic_facts",
                        "id": fact.get("id"),
                        "key": fact.get("key"),
                        "category": fact.get("category"),
                        "value": str(fact.get("value") or "")[:300],
                        "timestamp": ts,
                    }
                )
            if len(out) >= limit:
                break
    except Exception as exc:  # noqa: BLE001
        out.append({"source": "episodic_facts", "error": str(exc)})
    return out


def _query_blackboard(day_label: str, *, limit: int = 40) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        from dana.memory.blackboard import BLACKBOARD_DB_PATH, load_messages

        db = Path(BLACKBOARD_DB_PATH)
        if not db.is_file():
            return out
        with sqlite3.connect(str(db)) as conn:
            rows = conn.execute(
                "SELECT DISTINCT session_id FROM messages LIMIT 16"
            ).fetchall()
        for (sid,) in rows:
            for msg in load_messages(str(sid), limit=60):
                created = str(msg.get("created_at") or "")
                if day_label not in created:
                    continue
                out.append(
                    {
                        "source": "blackboard_msgs",
                        "session_id": sid,
                        "role": msg.get("role"),
                        "content": str(msg.get("content") or "")[:240],
                        "created_at": created,
                    }
                )
                if len(out) >= limit:
                    return out
    except Exception as exc:  # noqa: BLE001
        out.append({"source": "blackboard_msgs", "error": str(exc)})
    return out


def _query_logs(day_label: str, *, limit: int = 40) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        from dana.paths import LOGS_DIR

        logs_dir = Path(LOGS_DIR)
        if not logs_dir.is_dir():
            return out
        for path in sorted(logs_dir.glob("*.log")):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                continue
            hits = [
                ln.strip()
                for ln in text.splitlines()
                if day_label in ln or day_label.replace("-", "/") in ln
            ]
            if not hits:
                continue
            out.append(
                {
                    "source": "logs",
                    "file": path.name,
                    "count": len(hits),
                    "sample": hits[:8],
                }
            )
            if len(out) >= limit:
                break
    except Exception as exc:  # noqa: BLE001
        out.append({"source": "logs", "error": str(exc)})
    return out


def list_activity_for_day(date_str: str | None = None) -> dict[str, Any]:
    """Union episodic facts, blackboard messages, and dated log lines for a day."""
    day = resolve_date_str(date_str)
    episodic = _query_episodic(day)
    blackboard = _query_blackboard(day)
    logs = _query_logs(day)
    fact_n = len([r for r in episodic if not r.get("error")])
    bb_n = len([r for r in blackboard if not r.get("error")])
    log_n = sum(int(r.get("count") or 0) for r in logs if not r.get("error"))
    has_evidence = bool(fact_n or bb_n or log_n)
    return {
        "ok": True,
        "date": day,
        "has_evidence": has_evidence,
        "counts": {
            "episodic_facts": fact_n,
            "blackboard_msgs": bb_n,
            "log_line_hits": log_n,
        },
        "episodic_facts": episodic[:40],
        "blackboard_msgs": blackboard[:40],
        "log_hits": logs[:20],
        "note": (
            "Summarize concrete activity for the user from this evidence. "
            "If has_evidence is false, admit the gap — do not invent events."
            if has_evidence
            else "No corroborating activity found for this local calendar day."
        ),
    }


def format_activity_observation(payload: dict[str, Any]) -> str:
    """Compact tool observation for the ReAct loop."""
    try:
        body = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    except Exception:  # noqa: BLE001
        body = str(payload)
    if len(body) > 12000:
        body = body[:12000] + "\n…[truncated]"
    day = payload.get("date")
    counts = payload.get("counts") or {}
    return (
        f"OK: list_activity_for_day date={day} "
        f"facts={counts.get('episodic_facts')} "
        f"blackboard={counts.get('blackboard_msgs')} "
        f"log_hits={counts.get('log_line_hits')} "
        f"has_evidence={payload.get('has_evidence')}\n"
        f"{body}"
    )


def handle_list_activity_for_day(date_str: str | None = None) -> str:
    try:
        payload = list_activity_for_day(date_str)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: list_activity_for_day failed: {exc}"
    return format_activity_observation(payload)
