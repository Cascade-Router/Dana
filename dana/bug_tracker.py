"""Autonomous bug tracker — append-only JSON ledger for Titan Repair.

Canonical entry shape (CAMGRASPER/tracker/bug_tracker.json)::

    {"timestamp": "...", "error": "...", "context": "...", "status": "PENDING"}

Optional fields (``id``, ``user_query``, …) may also be present for repair routing.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dana.paths import BUG_TRACKER_PATH, PENDING_PATCHES_DIR, TRACKER_DIR

_LOCK = threading.Lock()
PENDING_STATUS = "PENDING"
_OPEN_STATUSES = frozenset({"PENDING", "pending", "open"})


def _ensure_docs() -> None:
    TRACKER_DIR.mkdir(parents=True, exist_ok=True)
    PENDING_PATCHES_DIR.mkdir(parents=True, exist_ok=True)


def load_bug_tracker(path: Path | None = None) -> list[dict[str, Any]]:
    """Return the bug ledger as a list (empty if missing/corrupt)."""
    target = path or BUG_TRACKER_PATH
    if not target.is_file():
        return []
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(raw, list):
        return [e for e in raw if isinstance(e, dict)]
    if isinstance(raw, dict) and isinstance(raw.get("bugs"), list):
        return [e for e in raw["bugs"] if isinstance(e, dict)]
    return []


def log_bug_to_tracker(
    error: str,
    context: str = "",
    *,
    status: str = PENDING_STATUS,
    path: Path | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Append a terminal-failure / crash entry to ``CAMGRASPER/tracker/bug_tracker.json``.

    Required shape: timestamp, error, context, status=PENDING.
    """
    _ensure_docs()
    target = path or BUG_TRACKER_PATH
    stamp = datetime.now(timezone.utc).isoformat()
    entry: dict[str, Any] = {
        "id": datetime.now(timezone.utc).strftime("bug_%Y%m%dT%H%M%S%f"),
        "timestamp": stamp,
        "error": (error or "").strip()[:4000],
        "context": (context or "").strip()[:8000],
        "status": status or PENDING_STATUS,
    }
    for key, val in extra.items():
        if key in entry or val is None:
            continue
        entry[key] = val
    with _LOCK:
        bugs = load_bug_tracker(target)
        bugs.append(entry)
        tmp = target.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(bugs, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        tmp.replace(target)
    try:
        from dana.logging import log as _log

        _log(
            "BugTracker",
            f"logged {entry['id']} status={entry['status']}: "
            f"{(entry['error'] or '')[:120]}",
        )
    except Exception:
        pass
    return entry


def append_bug_tracker_entry(
    *,
    user_query: str = "",
    error: str,
    traceback_text: str = "",
    tool_trace: list[dict[str, Any]] | None = None,
    status: str = PENDING_STATUS,
    path: Path | None = None,
    context: str = "",
) -> dict[str, Any]:
    """Compatibility wrapper — prefers ``log_bug_to_tracker`` schema."""
    ctx = (context or "").strip()
    if not ctx:
        parts = []
        if user_query:
            parts.append(f"user_query={user_query.strip()[:1500]}")
        if traceback_text:
            parts.append(traceback_text.strip()[:4000])
        if tool_trace:
            parts.append(f"tool_trace={json.dumps(tool_trace[:8], ensure_ascii=False)[:2000]}")
        ctx = "\n".join(parts) if parts else user_query
    return log_bug_to_tracker(
        error,
        ctx,
        status=status or PENDING_STATUS,
        path=path,
        user_query=(user_query or "").strip()[:2000],
        traceback=(traceback_text or "").strip()[:12000],
        tool_trace=list(tool_trace or [])[:20],
    )


def mark_bug_status(
    bug_id: str,
    status: str,
    *,
    path: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> bool:
    """Update ``status`` (and optional fields) for ``bug_id``. Returns True if found."""
    target = path or BUG_TRACKER_PATH
    with _LOCK:
        bugs = load_bug_tracker(target)
        found = False
        for entry in bugs:
            if str(entry.get("id")) == str(bug_id):
                entry["status"] = status
                if extra:
                    entry.update(extra)
                found = True
                break
        if not found:
            return False
        tmp = target.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(bugs, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        tmp.replace(target)
        return True


def open_bugs(path: Path | None = None) -> list[dict[str, Any]]:
    """Return bugs with PENDING (or legacy ``open``) status."""
    return [
        b
        for b in load_bug_tracker(path)
        if str(b.get("status") or PENDING_STATUS) in _OPEN_STATUSES
    ]


def list_todo_basket(*, path: Path | None = None) -> str:
    """Summarize PENDING bugs for the ``list_todo_basket`` tool observation."""
    bugs = open_bugs(path)
    if not bugs:
        return "OK: todo basket empty — no PENDING bugs in CAMGRASPER/tracker/bug_tracker.json."
    lines = [
        f"OK: todo basket has {len(bugs)} PENDING bug(s):",
    ]
    for i, bug in enumerate(bugs[:20], 1):
        bid = bug.get("id") or f"#{i}"
        err = str(bug.get("error") or "").strip().replace("\n", " ")[:160]
        ts = str(bug.get("timestamp") or "")[:32]
        ctx = str(bug.get("context") or bug.get("user_query") or "").strip().replace(
            "\n", " "
        )[:100]
        lines.append(f"{i}. [{bid}] {ts} — {err}" + (f" | ctx={ctx}" if ctx else ""))
    if len(bugs) > 20:
        lines.append(f"...and {len(bugs) - 20} more.")
    lines.append(
        "Say 'run titan repair' to draft fixes into CAMGRASPER/tracker/pending_patches/ "
        "(no automatic source hot-patch)."
    )
    return "\n".join(lines)
