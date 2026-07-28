"""Structured task queue for batched agentic dispatch (execution_jail).

Replaces the legacy flat ``input.txt`` interceptor. The on-disk schema is a
JSON array of task objects::

    [
      {"id": "task_001", "status": "pending", "command": "Donna, please ..."}
    ]

Valid statuses: ``pending``, ``completed``, ``failed``.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable

from dana.paths import (
    DONNA_WORKSPACE,
    EXECUTION_JAIL_DIR,
    TASK_QUEUE_PATH,
    TEXT_INJECTION_PATH,
)

_LOCK = threading.RLock()
_VALID_STATUSES = frozenset({"pending", "running", "completed", "failed"})

# Suppress duplicate prompt strings within this window (seconds).
_RECENT_CMD_WINDOW_S = 10.0
_recent_commands: dict[str, float] = {}
_RECENT_LOCK = threading.Lock()


def normalize_command_key(command: str) -> str:
    return " ".join((command or "").strip().lower().split())


def is_recent_duplicate_command(command: str) -> bool:
    """True when the same prompt was enqueued/completed within the last 10s."""
    key = normalize_command_key(command)
    if not key:
        return True
    now = time.time()
    with _RECENT_LOCK:
        stale = [k for k, ts in _recent_commands.items() if now - ts > _RECENT_CMD_WINDOW_S]
        for k in stale:
            _recent_commands.pop(k, None)
        ts = _recent_commands.get(key)
        return ts is not None and (now - ts) < _RECENT_CMD_WINDOW_S


def remember_command(command: str) -> None:
    """Record a command for the 10s dedupe window."""
    key = normalize_command_key(command)
    if not key:
        return
    with _RECENT_LOCK:
        _recent_commands[key] = time.time()


def try_record_command(command: str) -> bool:
    """Return False if duplicate within 10s; otherwise record and return True."""
    key = normalize_command_key(command)
    if not key:
        return False
    now = time.time()
    with _RECENT_LOCK:
        stale = [k for k, ts in _recent_commands.items() if now - ts > _RECENT_CMD_WINDOW_S]
        for k in stale:
            _recent_commands.pop(k, None)
        ts = _recent_commands.get(key)
        if ts is not None and (now - ts) < _RECENT_CMD_WINDOW_S:
            return False
        _recent_commands[key] = now
        return True


def shadow_backup_before_write(src: Path | str) -> None:
    """Copy ``src`` to ``DONNA_WORKSPACE/.shadow_state/<name>.bak`` before a write.

    No-op when the source file does not exist. Never raises into callers.
    """
    try:
        path = Path(src)
        if not path.is_file():
            return
        shadow_dir = Path(DONNA_WORKSPACE) / ".shadow_state"
        shadow_dir.mkdir(parents=True, exist_ok=True)
        dest = shadow_dir / f"{path.name}.bak"
        shutil.copy2(path, dest)
    except Exception:  # noqa: BLE001
        return


def default_queue_path() -> Path:
    return Path(TASK_QUEUE_PATH)


def ensure_queue_file(path: Path | None = None) -> Path:
    """Create an empty ``[]`` queue file if missing."""
    target = Path(path) if path is not None else default_queue_path()
    with _LOCK:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file():
            target.write_text("[]\n", encoding="utf-8")
    return target


def _normalize_task(raw: Any, *, index: int) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    command = str(raw.get("command") or "").strip()
    status = str(raw.get("status") or "pending").strip().lower() or "pending"
    if status not in _VALID_STATUSES:
        status = "pending"
    tid = str(raw.get("id") or "").strip() or f"task_{index:03d}"
    task: dict[str, Any] = {
        "id": tid,
        "status": status,
        "command": command,
    }
    for key in ("error", "completed_at", "started_at"):
        if key in raw and raw[key] is not None:
            task[key] = raw[key]
    return task


def load_task_queue(path: Path | None = None) -> list[dict[str, Any]]:
    """Load and normalize the queue. Corrupt/missing → empty list (never raises)."""
    target = Path(path) if path is not None else default_queue_path()
    with _LOCK:
        try:
            if not target.is_file():
                return []
            raw = target.read_text(encoding="utf-8-sig", errors="replace").strip()
            if not raw:
                return []
            data = json.loads(raw)
        except (OSError, json.JSONDecodeError):
            return []
    if not isinstance(data, list):
        return []
    out: list[dict[str, Any]] = []
    for i, item in enumerate(data, start=1):
        task = _normalize_task(item, index=i)
        if task is not None:
            out.append(task)
    return out


def save_task_queue(tasks: list[dict[str, Any]], path: Path | None = None) -> None:
    """Atomically rewrite the queue file."""
    target = Path(path) if path is not None else default_queue_path()
    payload = json.dumps(tasks, ensure_ascii=False, indent=2) + "\n"
    with _LOCK:
        target.parent.mkdir(parents=True, exist_ok=True)
        shadow_backup_before_write(target)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(target)


def pending_tasks(path: Path | None = None) -> list[dict[str, Any]]:
    return [t for t in load_task_queue(path) if t.get("status") == "pending"]


def pending_count(path: Path | None = None) -> int:
    return len(pending_tasks(path))


def claim_pending_tasks(path: Path | None = None) -> list[dict[str, Any]]:
    """Atomically pop pending tasks: mark ``running`` before any handler runs.

    Prevents InputIngest / a second drain from re-reading the same work.
    Returns claimed task copies (id/command/status=running).
    """
    with _LOCK:
        tasks = load_task_queue(path)
        claimed: list[dict[str, Any]] = []
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for task in tasks:
            if task.get("status") != "pending":
                continue
            command = str(task.get("command") or "").strip()
            task["status"] = "running"
            task["started_at"] = stamp
            claimed.append(
                {
                    "id": str(task.get("id") or ""),
                    "status": "running",
                    "command": command,
                    "started_at": stamp,
                }
            )
        if claimed:
            save_task_queue(tasks, path)
        return claimed


def update_task_status(
    task_id: str,
    status: str,
    *,
    path: Path | None = None,
    error: str | None = None,
) -> bool:
    """Set status for ``task_id``. Returns True if a matching task was updated."""
    status = (status or "").strip().lower()
    if status not in _VALID_STATUSES:
        raise ValueError(f"invalid status: {status!r}")
    with _LOCK:
        tasks = load_task_queue(path)
        found = False
        for task in tasks:
            if str(task.get("id")) != str(task_id):
                continue
            task["status"] = status
            if status in ("completed", "failed"):
                task["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                remember_command(str(task.get("command") or ""))
            if error:
                task["error"] = error[:2000]
            elif "error" in task and status == "completed":
                task.pop("error", None)
            found = True
            break
        if found:
            save_task_queue(tasks, path)
        return found


def migrate_legacy_input_txt(
    *,
    queue_path: Path | None = None,
    input_path: Path | None = None,
) -> str | None:
    """If legacy ``input.txt`` has content, enqueue paragraph tasks and clear the file.

    Splits on blank lines (same as ``ingest.ingest_text_to_queue``) so batched
    tickets never collapse into one oversized command.

    Returns a short summary string, or ``None`` when there was nothing to migrate.
    For the default jail paths, prefers the root ``ingest`` converter when available.
    """
    # Default paths: use the shared paragraph-splitting ingest converter.
    if queue_path is None and input_path is None:
        try:
            import ingest

            before = pending_count()
            ingest.ingest_text_to_queue()
            after = pending_count()
            added = max(0, after - before)
            return f"ingested {added} task(s)" if added else None
        except Exception:  # noqa: BLE001
            pass

    target = Path(input_path) if input_path is not None else Path(TEXT_INJECTION_PATH)
    try:
        if not target.is_file():
            return None
        raw = target.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return None

    text = (raw or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    if not text:
        return None

    # Paragraph split — one queue object per blank-line-separated block.
    raw_tasks = [part.strip() for part in text.split("\n\n") if part.strip()]
    if not raw_tasks:
        return None

    qpath = Path(queue_path) if queue_path is not None else default_queue_path()
    with _LOCK:
        tasks = load_task_queue(qpath)
        pending_cmds = {
            normalize_command_key(str(t.get("command") or ""))
            for t in tasks
            if t.get("status") in ("pending", "running")
        }
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        added = 0
        for i, command in enumerate(raw_tasks, start=1):
            key = normalize_command_key(command)
            if not key or key in pending_cmds:
                continue
            if not try_record_command(command):
                continue
            tasks.append(
                {
                    "id": f"legacy_input_{stamp}_{i}",
                    "status": "pending",
                    "command": command,
                }
            )
            pending_cmds.add(key)
            added += 1
        if added:
            save_task_queue(tasks, qpath)
        try:
            target.write_text("", encoding="utf-8")
        except OSError:
            pass
    return raw_tasks[0] if added else None


def ensure_execution_jail_queue() -> Path:
    """Ensure jail dir + empty queue exist (and migrate legacy input.txt)."""
    EXECUTION_JAIL_DIR.mkdir(parents=True, exist_ok=True)
    path = ensure_queue_file()
    migrate_legacy_input_txt()
    return path


def drain(
    handler: Callable[[str], Any],
    *,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Atomic drain entrypoint — claim pending, then dispatch via broker."""
    from dana.tools.broker import dispatch_pending_tasks

    return dispatch_pending_tasks(handler, path=path)
