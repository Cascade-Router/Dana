"""Dana logging: light runtime log + clean latest conversation log.

Runtime (``CAMGRASPER/logs/dana_runtime.log``):
  - Circular last-100-lines buffer across the process life.
  - ``log()`` / ``log_debug()`` — debug is silenced unless ``DONNA_DEBUG=1``.

Conversation (``CAMGRASPER/logs/dana_conversation.log``):
  - Truncated (cleared) on every new agent run.
  - User ↔ Dana turns only — no Tracker / wake / mic / YOLO noise.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
import traceback
from typing import Any, Optional

from dana.paths import LOGS_DIR, PROJECT_ROOT
from dana.sanitize import sanitize_log_message
from dana.stdio_boot import NullStdio, ensure_stdio, ensure_stdio_for_pythonw

_PROJECT_DIR = str(PROJECT_ROOT)
# Any import of dana.logging (hot path) hardens pythonw stdio immediately.
ensure_stdio()
RUNTIME_LOG_DIR = str(LOGS_DIR)
RUNTIME_LOG_PATH = str(LOGS_DIR / "dana_runtime.log")
CONVERSATION_LOG_PATH = str(LOGS_DIR / "dana_conversation.log")
# Legacy filenames — migrated on first enable when still present.
_LEGACY_RUNTIME_LOG = str(LOGS_DIR / "donna_runtime.log")
_LEGACY_CONVERSATION_LOG = str(LOGS_DIR / "donna_conversation.log")
# Keep enough headroom for multi-line ``log_exception`` stack traces.
RUNTIME_LOG_MAX_LINES = 250

_stdlib_logger = logging.getLogger("dana")

_runtime_log_lock = threading.Lock()
_conversation_log_lock = threading.Lock()
_runtime_log_tee_installed = False


def debug_logging_enabled() -> bool:
    return os.environ.get("DONNA_DEBUG", "").strip().lower() in ("1", "true", "yes")


def _stamp() -> str:
    return time.strftime("%H:%M:%S")


def _migrate_legacy_log(legacy: str, modern: str) -> None:
    """Rename legacy donna_*.log → dana_*.log when the modern file is absent."""
    try:
        if os.path.isfile(modern) or not os.path.isfile(legacy):
            return
        os.makedirs(os.path.dirname(modern) or ".", exist_ok=True)
        os.replace(legacy, modern)
    except OSError:
        pass


def append_runtime_log(text: str) -> None:
    """Append raw text to ``logs/dana_runtime.log`` (thread-safe, last 100 lines)."""
    if not text:
        return
    try:
        os.makedirs(RUNTIME_LOG_DIR, exist_ok=True)
        with _runtime_log_lock:
            _trim_runtime_log_to_last_lines(RUNTIME_LOG_PATH)
            with open(
                RUNTIME_LOG_PATH,
                "a",
                encoding="utf-8",
                errors="replace",
                newline="",
            ) as fh:
                fh.write(text)
            _trim_runtime_log_to_last_lines(RUNTIME_LOG_PATH)
    except Exception:
        pass


def _trim_runtime_log_to_last_lines(
    path: str,
    *,
    max_lines: int = RUNTIME_LOG_MAX_LINES,
) -> None:
    """Keep only the last ``max_lines`` physical lines in the runtime log."""
    try:
        if not os.path.isfile(path):
            return
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
        if len(lines) <= max_lines:
            return
        tail = lines[-max_lines:]
        with open(path, "w", encoding="utf-8", errors="replace", newline="") as fh:
            fh.writelines(tail)
    except OSError:
        pass


def _print_line(line: str) -> None:
    ensure_stdio()
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        safe = line.encode(encoding, errors="replace").decode(encoding, errors="replace")
        print(safe, flush=True)
    except Exception:
        try:
            buf = getattr(sys.stdout, "buffer", None)
            if buf is not None:
                buf.write((line + "\n").encode("utf-8", errors="replace"))
                buf.flush()
            append_runtime_log(line + "\n")
        except Exception:
            try:
                append_runtime_log(line + "\n")
            except Exception:
                pass


def log(thread: str, message: str, *, level: str = "info") -> None:
    """Emit a runtime log line. ``level=\"debug\"`` is no-op unless DONNA_DEBUG=1."""
    level_l = (level or "info").strip().lower()
    if level_l == "debug" and not debug_logging_enabled():
        return
    message = sanitize_log_message(str(message))
    line = f"[{_stamp()}] [{thread}] {message}"
    _print_line(line)


def log_debug(thread: str, message: str) -> None:
    """Verbose diagnostics — skipped in normal runs."""
    log(thread, message, level="debug")


def log_exception(
    thread: str,
    message: str,
    *,
    exc: Optional[BaseException] = None,
) -> None:
    """Force a full Python stack trace into ``dana_runtime.log``.

    Also calls ``logging.exception`` so stdlib handlers (if any) see the failure.
    Prefer calling from an ``except`` block so ``sys.exc_info()`` is populated.
    """
    message = sanitize_log_message(str(message))
    # Stdlib path (user-requested): full traceback via logging.exception.
    if exc is not None:
        _stdlib_logger.exception("%s [%s]", message, thread, exc_info=exc)
    else:
        _stdlib_logger.exception("%s [%s]", message, thread)

    if exc is not None:
        tb_text = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
    else:
        tb_text = traceback.format_exc()
    if not tb_text or tb_text.strip() == "NoneType: None":
        tb_text = "(no active exception traceback)\n"

    stamp = _stamp()
    block = (
        f"[{stamp}] [{thread}] EXCEPTION: {message}\n"
        f"{tb_text.rstrip()}\n"
    )
    _print_line(f"[{stamp}] [{thread}] EXCEPTION: {message}")
    # Write the full traceback as one append so trim keeps the whole block longer.
    append_runtime_log(block)
    ensure_stdio()
    try:
        if sys.stderr is not None:
            sys.stderr.write(tb_text)
            if not tb_text.endswith("\n"):
                sys.stderr.write("\n")
            sys.stderr.flush()
    except Exception:
        pass


def reset_conversation_log() -> str:
    """Clear and recreate the latest Dana conversation log for this run."""
    os.makedirs(RUNTIME_LOG_DIR, exist_ok=True)
    header = (
        f"===== Dana conversation session {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n"
        "# Latest User ↔ Dana turns only (system noise excluded).\n"
    )
    with _conversation_log_lock:
        with open(
            CONVERSATION_LOG_PATH,
            "w",
            encoding="utf-8",
            errors="replace",
            newline="",
        ) as fh:
            fh.write(header)
    return CONVERSATION_LOG_PATH


def log_conversation(role: str, text: str, *, extra: str = "") -> None:
    """Append one conversation turn to the latest-only conversation log (file only).

    Does **not** write to the runtime log — call ``log()`` separately for essential
    console breadcrumbs if needed.
    """
    role_s = (role or "Dana").strip() or "Dana"
    if role_s.lower() == "donna":
        role_s = "Dana"
    body = sanitize_log_message(str(text or "").strip())
    if not body:
        return
    suffix = f" ({extra})" if extra else ""
    conv_line = f"[{_stamp()}] {role_s}: {body}{suffix}\n"
    try:
        os.makedirs(RUNTIME_LOG_DIR, exist_ok=True)
        with _conversation_log_lock:
            with open(
                CONVERSATION_LOG_PATH,
                "a",
                encoding="utf-8",
                errors="replace",
                newline="",
            ) as fh:
                fh.write(conv_line)
    except Exception:
        pass

class _RuntimeLogTee:
    """Mirror writes to the original stream and the persistent runtime log."""

    def __init__(self, stream: Any) -> None:
        self._stream = stream if stream is not None else NullStdio()

    def write(self, data: Any) -> int:
        text = data if isinstance(data, str) else str(data)
        written = len(text)
        try:
            result = self._stream.write(data)
            if isinstance(result, int):
                written = result
        except Exception:
            # pythonw / broken console: never abort Whisper/mic on log writes.
            pass
        try:
            append_runtime_log(text)
        except Exception:
            pass
        return written

    def flush(self) -> None:
        try:
            self._stream.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        try:
            return bool(self._stream.isatty())
        except Exception:
            return False

    def reconfigure(self, **kwargs: Any) -> None:
        try:
            reconf = getattr(self._stream, "reconfigure", None)
            if callable(reconf):
                reconf(**kwargs)
        except Exception:
            pass

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def enable_runtime_file_logging() -> str:
    """Install stdout/stderr tees; start a fresh conversation log for this run."""
    global _runtime_log_tee_installed
    ensure_stdio()
    os.makedirs(RUNTIME_LOG_DIR, exist_ok=True)
    _migrate_legacy_log(_LEGACY_RUNTIME_LOG, RUNTIME_LOG_PATH)
    _migrate_legacy_log(_LEGACY_CONVERSATION_LOG, CONVERSATION_LOG_PATH)
    with _runtime_log_lock:
        _trim_runtime_log_to_last_lines(RUNTIME_LOG_PATH)
    reset_conversation_log()
    if not _runtime_log_tee_installed:
        if not isinstance(sys.stdout, _RuntimeLogTee):
            sys.stdout = _RuntimeLogTee(sys.stdout)  # type: ignore[assignment]
        if not isinstance(sys.stderr, _RuntimeLogTee):
            sys.stderr = _RuntimeLogTee(sys.stderr)  # type: ignore[assignment]
        _runtime_log_tee_installed = True
        append_runtime_log(
            f"\n===== Dana runtime session {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n"
        )
    return RUNTIME_LOG_PATH
