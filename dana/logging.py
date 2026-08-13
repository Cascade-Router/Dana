"""Dana logging: light runtime log + clean latest conversation log.

Runtime (``CAMGRASPER/logs/dana_runtime.log``):
  - Circular last-100-lines buffer across the process life.
  - ``log()`` / ``log_debug()`` — debug is silenced unless ``DANA_DEBUG=1``.

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
from dana.stdio_boot import NullStdio, ensure_stdio

_PROJECT_DIR = str(PROJECT_ROOT)
# Any import of dana.logging (hot path) hardens pythonw stdio immediately.
ensure_stdio()
RUNTIME_LOG_DIR = str(LOGS_DIR)
RUNTIME_LOG_PATH = str(LOGS_DIR / "dana_runtime.log")
CONVERSATION_LOG_PATH = str(LOGS_DIR / "dana_conversation.log")
FATAL_CRASH_LOG_PATH = str(LOGS_DIR / "fatal_crash.log")
# Keep enough headroom for multi-line ``log_exception`` stack traces.
RUNTIME_LOG_MAX_LINES = 250
_fatal_hooks_installed = False
_fatal_log_lock = threading.Lock()

_stdlib_logger = logging.getLogger("dana")

_runtime_log_lock = threading.Lock()
_conversation_log_lock = threading.Lock()
_runtime_log_tee_installed = False


def debug_logging_enabled() -> bool:
    return os.environ.get("DANA_DEBUG", "").strip().lower() in ("1", "true", "yes")


def _stamp() -> str:
    return time.strftime("%H:%M:%S")


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
    """Emit a runtime log line. ``level=\"debug\"`` is no-op unless DANA_DEBUG=1."""
    level_l = (level or "info").strip().lower()
    if level_l == "debug" and not debug_logging_enabled():
        return
    message = sanitize_log_message(str(message))
    line = f"[{_stamp()}] [{thread}] {message}"
    _print_line(line)


def log_debug(thread: str, message: str) -> None:
    """Verbose diagnostics — skipped in normal runs."""
    log(thread, message, level="debug")


def write_fatal_crash_log(
    kind: str,
    exc_type: Any,
    exc_value: Any,
    exc_tb: Any,
    *,
    thread_name: str = "",
) -> str | None:
    """Append a full traceback to ``logs/fatal_crash.log`` (never trimmed).

    Returns the log path on success, else ``None``. Safe for hooks / workers.
    """
    try:
        os.makedirs(RUNTIME_LOG_DIR, exist_ok=True)
    except Exception:
        return None
    try:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        thr = (thread_name or threading.current_thread().name or "").strip()
        header = (
            f"\n===== FATAL {kind} {stamp} "
            f"pid={os.getpid()} thread={thr or '-'} =====\n"
        )
        if exc_type is None:
            body = "(no exception info)\n"
        else:
            body = "".join(
                traceback.format_exception(exc_type, exc_value, exc_tb)
            )
        with _fatal_log_lock:
            with open(
                FATAL_CRASH_LOG_PATH,
                "a",
                encoding="utf-8",
                errors="replace",
                newline="",
            ) as fh:
                fh.write(header)
                fh.write(body)
                if not body.endswith("\n"):
                    fh.write("\n")
        # Mirror a short breadcrumb into the circular runtime log.
        try:
            name = getattr(exc_type, "__name__", str(exc_type))
            append_runtime_log(
                f"[{_stamp()}] [Fatal] {kind}: {name}: {exc_value} "
                f"→ {FATAL_CRASH_LOG_PATH}\n"
            )
        except Exception:
            pass
        return FATAL_CRASH_LOG_PATH
    except Exception:
        return None


def install_fatal_crash_hooks() -> None:
    """Bind ``sys.excepthook`` + ``threading.excepthook`` → ``fatal_crash.log``.

    Idempotent. Does not replace ``SystemExit`` / ``KeyboardInterrupt`` handling
    beyond writing a note for unexpected ``SystemExit`` codes.
    """
    global _fatal_hooks_installed
    if _fatal_hooks_installed:
        return

    prev_sys_hook = sys.excepthook

    def _sys_hook(exc_type, exc_value, exc_tb):  # noqa: ANN001
        # Quiet exits — do not spam fatal_crash.log.
        if exc_type is KeyboardInterrupt:
            return prev_sys_hook(exc_type, exc_value, exc_tb)
        if exc_type is SystemExit:
            try:
                code = getattr(exc_value, "code", exc_value)
            except Exception:
                code = None
            if code in (0, None):
                return prev_sys_hook(exc_type, exc_value, exc_tb)
        write_fatal_crash_log(
            "sys.excepthook",
            exc_type,
            exc_value,
            exc_tb,
            thread_name="MainThread",
        )
        return prev_sys_hook(exc_type, exc_value, exc_tb)

    sys.excepthook = _sys_hook

    if hasattr(threading, "excepthook"):
        prev_thread_hook = threading.excepthook

        def _thread_hook(args):  # noqa: ANN001
            try:
                tname = ""
                thr = getattr(args, "thread", None)
                if thr is not None:
                    tname = str(getattr(thr, "name", "") or "")
                write_fatal_crash_log(
                    "threading.excepthook",
                    getattr(args, "exc_type", None),
                    getattr(args, "exc_value", None),
                    getattr(args, "exc_traceback", None),
                    thread_name=tname,
                )
            except Exception:
                pass
            try:
                return prev_thread_hook(args)
            except Exception:
                return None

        threading.excepthook = _thread_hook

    _fatal_hooks_installed = True


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
    if role_s.lower() == "dana":
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
