"""Realtime ChromaDB sync — filesystem watcher → purge / re-embed.

Watches active workspace roots with the ``watchdog`` library (optional polling
fallback). On create/modify, old chunks for that filepath are stripped and the
file is re-embedded. On delete, matching document IDs are purged.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from dana.memory.vault import (
    _INGEST_EXTENSIONS,
    _SKIP_DIR_NAMES,
    CodebaseVault,
    get_codebase_vault,
    normalize_vault_filepath,
)
from dana.paths import DONNA_WORKSPACE, PROJECT_ROOT

logger = logging.getLogger(__name__)

# Extra path segments that must never trigger vault churn.
_EXTRA_SKIP_PARTS = frozenset(
    {
        ".dana",
        ".dana_scratch",
        ".venv",
        "venv",
        ".cursor",
        "tts_models",
        "assets",
        "node_modules",
        ".pytest_cache",
        ".mypy_cache",
        ".chroma",
        "vault",
        # Runtime logs / dashboards churn constantly — never re-embed.
        "logs",
        "__pycache__",
    }
)

# SQLite journals / WAL / locks / scratch temps — never purge/re-embed or log.
_IGNORE_NAME_SUFFIXES = (
    ".db-journal",
    ".db-wal",
    ".db-shm",
    ".sqlite3-journal",
    ".sqlite-journal",
    ".tmp",
    ".lock",
)

DEFAULT_DEBOUNCE_S = 0.75
DEFAULT_WORKERS = 2

SyncCallback = Callable[[str, str], None]  # (event_kind, filepath_key)


def is_ephemeral_db_noise(path: Path | str) -> bool:
    """True for SQLite journal/WAL/SHM/tmp/lock files (silent ignore)."""
    name = Path(str(path or "")).name.lower()
    if not name:
        return False
    return any(name.endswith(suf) for suf in _IGNORE_NAME_SUFFIXES)


def _log(msg: str) -> None:
    try:
        from dana.logging import log

        log("VectorSync", msg)
    except Exception:  # noqa: BLE001
        logger.info(msg)


def default_watch_roots() -> list[Path]:
    """Active workspace directories eligible for vault sync."""
    roots = [Path(DONNA_WORKSPACE).resolve(), Path(PROJECT_ROOT).resolve()]
    # Deduplicate while preserving order.
    out: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = root.as_posix()
        if key in seen:
            continue
        if root.is_dir():
            out.append(root)
            seen.add(key)
    return out


def should_sync_path(path: Path | str) -> bool:
    """True when ``path`` is an ingestible source file outside skip trees."""
    if is_ephemeral_db_noise(path):
        return False
    try:
        p = Path(path).expanduser().resolve()
    except OSError:
        return False
    if is_ephemeral_db_noise(p):
        return False
    parts = set(p.parts)
    if parts & _SKIP_DIR_NAMES or parts & _EXTRA_SKIP_PARTS:
        return False
    # Also skip nested .git / __pycache__ style segments.
    if any(part in _SKIP_DIR_NAMES or part in _EXTRA_SKIP_PARTS for part in p.parts):
        return False
    return p.suffix.lower() in _INGEST_EXTENSIONS


class VectorIndexSync:
    """Filesystem observer that keeps ``dana_codebase_vault`` fresh."""

    def __init__(
        self,
        *,
        vault: CodebaseVault | None = None,
        watch_roots: list[Path | str] | None = None,
        debounce_s: float = DEFAULT_DEBOUNCE_S,
        max_workers: int = DEFAULT_WORKERS,
        on_event: SyncCallback | None = None,
    ) -> None:
        self.vault = vault or get_codebase_vault()
        self.watch_roots = [
            Path(r).expanduser().resolve()
            for r in (watch_roots if watch_roots is not None else default_watch_roots())
        ]
        self.debounce_s = max(0.05, float(debounce_s))
        self._on_event = on_event
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(max_workers)),
            thread_name_prefix="vector-sync",
        )
        self._pending: dict[str, tuple[str, float]] = {}
        self._pending_lock = threading.Lock()
        self._observer: Any = None
        self._poll_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._flush_thread: threading.Thread | None = None
        self._started = False
        self._stats = {
            "reembedded": 0,
            "purged": 0,
            "errors": 0,
            "events": 0,
        }

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    def start(self) -> str:
        """Begin watching workspace roots (idempotent)."""
        if self._started:
            return "OK: vector sync already running"
        self._stop.clear()
        self._flush_thread = threading.Thread(
            target=self._debounce_loop,
            name="vector-sync-debounce",
            daemon=True,
        )
        self._flush_thread.start()

        mode = self._start_watchdog() or self._start_polling_fallback()
        self._started = True
        roots = ", ".join(r.as_posix() for r in self.watch_roots) or "(none)"
        msg = f"OK: vector sync started mode={mode} roots=[{roots}]"
        _log(msg)
        return msg

    def stop(self, *, wait: bool = False) -> str:
        """Stop the observer and drain background work."""
        self._stop.set()
        if self._observer is not None:
            try:
                self._observer.stop()
                if wait:
                    self._observer.join(timeout=5)
            except Exception:  # noqa: BLE001
                pass
            self._observer = None
        if self._flush_thread is not None and wait:
            self._flush_thread.join(timeout=5)
        self._flush_thread = None
        if wait:
            self._executor.shutdown(wait=True, cancel_futures=False)
        self._started = False
        _log("vector sync stopped")
        return "OK: vector sync stopped"

    def _start_watchdog(self) -> str | None:
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
        except Exception as exc:  # noqa: BLE001
            _log(f"watchdog unavailable ({exc}); will try polling fallback")
            return None

        sync = self

        class _Handler(FileSystemEventHandler):
            def on_created(self, event: Any) -> None:  # noqa: ANN401
                if getattr(event, "is_directory", False):
                    return
                sync.enqueue("created", getattr(event, "src_path", ""))

            def on_modified(self, event: Any) -> None:  # noqa: ANN401
                if getattr(event, "is_directory", False):
                    return
                sync.enqueue("modified", getattr(event, "src_path", ""))

            def on_deleted(self, event: Any) -> None:  # noqa: ANN401
                if getattr(event, "is_directory", False):
                    return
                sync.enqueue("deleted", getattr(event, "src_path", ""))

            def on_moved(self, event: Any) -> None:  # noqa: ANN401
                if getattr(event, "is_directory", False):
                    return
                # Treat move as delete(src) + create(dest).
                sync.enqueue("deleted", getattr(event, "src_path", ""))
                sync.enqueue("created", getattr(event, "dest_path", ""))

        observer = Observer()
        handler = _Handler()
        scheduled = 0
        for root in self.watch_roots:
            if not root.is_dir():
                continue
            observer.schedule(handler, str(root), recursive=True)
            scheduled += 1
        if scheduled == 0:
            return None
        observer.daemon = True
        observer.start()
        self._observer = observer
        return "watchdog"

    def _start_polling_fallback(self) -> str:
        """Lightweight mtime poller when ``watchdog`` is not installed."""
        snapshots: dict[str, float] = {}

        def _scan() -> dict[str, float]:
            found: dict[str, float] = {}
            for root in self.watch_roots:
                if not root.is_dir():
                    continue
                for path in root.rglob("*"):
                    if not path.is_file() or not should_sync_path(path):
                        continue
                    try:
                        found[normalize_vault_filepath(path)] = path.stat().st_mtime
                    except OSError:
                        continue
            return found

        snapshots.update(_scan())

        def _loop() -> None:
            while not self._stop.wait(1.0):
                current = _scan()
                for key, mtime in current.items():
                    prev = snapshots.get(key)
                    if prev is None:
                        self.enqueue("created", key)
                    elif mtime > prev:
                        self.enqueue("modified", key)
                for key in list(snapshots):
                    if key not in current:
                        self.enqueue("deleted", key)
                snapshots.clear()
                snapshots.update(current)

        self._poll_thread = threading.Thread(
            target=_loop, name="vector-sync-poll", daemon=True
        )
        self._poll_thread.start()
        return "polling"

    def enqueue(self, kind: str, path: str | Path) -> None:
        """Debounce a filesystem event for background processing."""
        raw = str(path or "").strip()
        if not raw:
            return
        # Silent early-out: SQLite journals / WAL / locks / tmp produce zero logs.
        if is_ephemeral_db_noise(raw):
            return
        try:
            p = Path(raw)
        except Exception:  # noqa: BLE001
            return
        if is_ephemeral_db_noise(p):
            return
        # Deletes may already be gone — still allow purge by path key.
        if kind != "deleted" and p.exists() and not should_sync_path(p):
            return
        if kind == "deleted":
            # Filter by suffix / skip dirs using the path string alone.
            try:
                if is_ephemeral_db_noise(p) or is_ephemeral_db_noise(raw):
                    return
                if p.suffix.lower() not in _INGEST_EXTENSIONS:
                    return
                if any(
                    part in _SKIP_DIR_NAMES or part in _EXTRA_SKIP_PARTS
                    for part in Path(raw).parts
                ):
                    return
            except Exception:  # noqa: BLE001
                return
        try:
            # resolve(strict=False) keeps absolute keys for deleted paths.
            key = Path(raw).expanduser().resolve(strict=False).as_posix()
        except Exception:  # noqa: BLE001
            key = Path(raw).as_posix()

        with self._pending_lock:
            self._pending[key] = (kind, time.monotonic())
        self._stats["events"] += 1

    def _debounce_loop(self) -> None:
        while not self._stop.wait(0.2):
            due: list[tuple[str, str]] = []
            now = time.monotonic()
            with self._pending_lock:
                for key, (kind, ts) in list(self._pending.items()):
                    if now - ts >= self.debounce_s:
                        due.append((kind, key))
                        self._pending.pop(key, None)
            for kind, key in due:
                self._executor.submit(self._process_event, kind, key)

    def _process_event(self, kind: str, filepath_key: str) -> None:
        # Belt-and-suspenders: never log or mutate vault for journal noise.
        if is_ephemeral_db_noise(filepath_key):
            return
        try:
            if self._on_event is not None:
                try:
                    self._on_event(kind, filepath_key)
                except Exception:  # noqa: BLE001
                    pass
            if kind == "deleted":
                msg = self.vault.purge_filepath(filepath_key)
                self._stats["purged"] += 1
                _log(msg)
                return
            # created / modified → strip old vectors then re-embed.
            msg = self.vault.reembed_file(filepath_key)
            if "re-embedded" in msg:
                self._stats["reembedded"] += 1
            elif "purged" in msg and "skip" in msg:
                self._stats["purged"] += 1
            _log(msg)
        except Exception as exc:  # noqa: BLE001
            self._stats["errors"] += 1
            _log(f"ERROR: vector sync failed for {filepath_key}: {exc}")

    def flush(self, *, timeout_s: float = 10.0) -> None:
        """Process pending debounced events synchronously (tests / shutdown)."""
        with self._pending_lock:
            items = list(self._pending.items())
            self._pending.clear()
        futures = [
            self._executor.submit(self._process_event, kind, key)
            for key, (kind, _ts) in items
        ]
        deadline = time.time() + max(0.1, float(timeout_s))
        for fut in futures:
            remaining = max(0.0, deadline - time.time())
            try:
                fut.result(timeout=remaining)
            except Exception:  # noqa: BLE001
                continue


_GLOBAL_SYNC: VectorIndexSync | None = None
_GLOBAL_LOCK = threading.Lock()


def get_vector_sync() -> VectorIndexSync | None:
    return _GLOBAL_SYNC


def start_vector_sync(
    *,
    vault: CodebaseVault | None = None,
    watch_roots: list[Path | str] | None = None,
    debounce_s: float = DEFAULT_DEBOUNCE_S,
) -> VectorIndexSync:
    """Start (or return) the process-wide vector filesystem syncer."""
    global _GLOBAL_SYNC
    with _GLOBAL_LOCK:
        if _GLOBAL_SYNC is not None and _GLOBAL_SYNC._started:
            return _GLOBAL_SYNC
        sync = VectorIndexSync(
            vault=vault,
            watch_roots=watch_roots,
            debounce_s=debounce_s,
        )
        sync.start()
        _GLOBAL_SYNC = sync
        return sync


def stop_vector_sync(*, wait: bool = False) -> str:
    global _GLOBAL_SYNC
    with _GLOBAL_LOCK:
        if _GLOBAL_SYNC is None:
            return "OK: vector sync not running"
        msg = _GLOBAL_SYNC.stop(wait=wait)
        _GLOBAL_SYNC = None
        return msg


__all__ = (
    "VectorIndexSync",
    "default_watch_roots",
    "get_vector_sync",
    "is_ephemeral_db_noise",
    "should_sync_path",
    "start_vector_sync",
    "stop_vector_sync",
)

