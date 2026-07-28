"""Stage 8.9 — Local human-feedback logging for HITL ticket decisions.

Appends JSON lines to ``memory/feedback_logs.jsonl`` (gitignored).
Never sends data to external APIs.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()


def _log(msg: str) -> None:
    try:
        from dana.logging import log

        log("FeedbackLog", msg)
    except Exception:  # noqa: BLE001
        print(f"[FeedbackLog] {msg}", flush=True)


def feedback_log_path() -> Path:
    """Absolute path to the local JSONL feedback store."""
    try:
        from dana.paths import PROJECT_ROOT

        root = Path(PROJECT_ROOT)
    except Exception:  # noqa: BLE001
        root = Path(__file__).resolve().parents[2]
    return root / "memory" / "feedback_logs.jsonl"


def log_human_feedback(
    task_id: str | None,
    human_decision: str,
    jason_critique: str,
    ticket_content: dict[str, Any] | str | None,
    *,
    session_id: str = "",
    note: str = "",
) -> Path:
    """Append one serialized feedback record to ``memory/feedback_logs.jsonl``.

    Returns the path written (for tests / diagnostics).
    """
    path = feedback_log_path()
    decision = str(human_decision or "").strip().lower() or "unknown"
    if decision in {"approved", "yes", "true", "1", "submit"}:
        decision = "approve"
    elif decision in {"denied", "no", "false", "0", "reject", "cancelled"}:
        decision = "deny"

    if isinstance(ticket_content, dict):
        ticket_obj: Any = {
            "objective": str(ticket_content.get("objective") or ""),
            "context": str(ticket_content.get("context") or ""),
            "tool": str(ticket_content.get("tool") or "draft_cursor_prompt"),
        }
    else:
        ticket_obj = str(ticket_content or "")

    record = {
        "ts": time.time(),
        "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "task_id": str(task_id or uuid.uuid4().hex[:12]),
        "session_id": str(session_id or ""),
        "human_decision": decision,
        "jason_critique": str(jason_critique or "").strip(),
        "ticket_content": ticket_obj,
        "note": str(note or ""),
    }
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)
    _log(
        f"appended decision={decision} task_id={record['task_id']} "
        f"path={path}"
    )
    return path


def clear_feedback_logs() -> dict[str, Any]:
    """Stage 8.9.1 — safely truncate ``memory/feedback_logs.jsonl`` to 0 bytes.

    Opens the file in ``\"w\"`` mode under the same lock as appends so GUI /
    worker writers cannot race. Missing files are a no-op success.
    """
    path = feedback_log_path()
    with _LOCK:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8"):
                pass
            size = int(path.stat().st_size) if path.is_file() else 0
            _log(f"cleared feedback logs path={path} size={size}B")
            return {
                "ok": True,
                "path": str(path),
                "bytes": size,
                "message": f"Logs Cleared ({size} B)",
            }
        except FileNotFoundError:
            _log(f"clear skipped — file not found ({path})")
            return {
                "ok": True,
                "path": str(path),
                "bytes": 0,
                "message": "Logs Cleared (0 B)",
                "missing": True,
            }
        except OSError as exc:
            _log(f"WARNING: clear failed ({exc})")
            return {
                "ok": False,
                "path": str(path),
                "bytes": -1,
                "message": f"Clear failed: {exc}",
                "error": str(exc),
            }
