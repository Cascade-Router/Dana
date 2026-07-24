"""Root entry point for Donna.

Always resolves the repo root onto ``sys.path`` and as the process cwd so
``python run.py`` works from any working directory.
"""

from __future__ import annotations

import logging
import os
import sys

_ROOT = os.path.abspath(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
try:
    os.chdir(_ROOT)
except OSError:
    pass

# pythonw.exe leaves stdout/stderr as None — patch before any library prints/tqdm.
try:
    from donna.stdio_boot import ensure_stdio

    ensure_stdio()
except Exception:
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8", errors="replace")  # type: ignore[assignment]
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8", errors="replace")  # type: ignore[assignment]

# Ensure workspace dirs exist + migrate legacy artifacts before agent boot.
try:
    from donna.workspace import ensure_donna_workspace

    ensure_donna_workspace(migrate=True)
except Exception as exc:  # noqa: BLE001
    print(f"[Workspace] WARNING: ensure_donna_workspace failed: {exc}", file=sys.stderr)

# Held for process lifetime so the OS releases the bind only on exit.
_DONNA_INSTANCE_LOCK_SOCK = None
# Dedicated loopback port — not the telemetry dashboard (47474).
_DONNA_INSTANCE_LOCK_PORT = 47473
_HEADLESS_LOG_NAME = "donna_headless.log"


def _wants_no_gui(argv: list[str] | None = None) -> bool:
    args = list(sys.argv[1:] if argv is None else argv)
    return "--no-gui" in args


def _configure_headless_logging() -> str | None:
    """File logging fallback when OS-level stdout redirect is missing/broken."""
    log_path = os.path.join(_ROOT, _HEADLESS_LOG_NAME)
    try:
        root = logging.getLogger()
        # Avoid duplicate handlers on reload.
        for h in list(root.handlers):
            if getattr(h, "_donna_headless", False):
                return log_path
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        handler._donna_headless = True  # type: ignore[attr-defined]
        root.addHandler(handler)
        root.setLevel(logging.DEBUG)
        logging.captureWarnings(True)
        logging.getLogger("donna").info(
            "Headless file logging active → %s (cwd=%s)",
            log_path,
            os.path.abspath(os.getcwd()),
        )
        return log_path
    except Exception as exc:  # noqa: BLE001
        print(f"[Main] WARNING: headless file logging failed: {exc}", file=sys.stderr)
        return None


def _acquire_single_instance_lock() -> bool:
    """Bind a loopback TCP socket; False if another Donna already holds it."""
    global _DONNA_INSTANCE_LOCK_SOCK
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # Exclusive bind (no SO_REUSEADDR) — second instance must fail.
        sock.bind(("127.0.0.1", _DONNA_INSTANCE_LOCK_PORT))
        sock.listen(1)
    except OSError:
        try:
            sock.close()
        except OSError:
            pass
        return False
    _DONNA_INSTANCE_LOCK_SOCK = sock
    return True


if __name__ == "__main__":
    if _wants_no_gui():
        _configure_headless_logging()

    if not _acquire_single_instance_lock():
        msg = (
            "[Main] ERROR: Another instance of Donna is already running. "
            "Aborting to protect execution jail."
        )
        print(msg, flush=True)
        if _wants_no_gui():
            logging.getLogger("donna").error(msg)
        sys.exit(1)

    # Defer core_agent import until launch so torch/transformers/YOLO stay off
    # the interpreter's critical path during ``run.py`` module load.
    try:
        from donna.core_agent import main  # noqa: E402
    except Exception:
        if _wants_no_gui():
            logging.getLogger("donna").exception("Failed importing donna.core_agent")
        raise

    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        # Closing the GUI / Ctrl+C should exit quietly (workers log "Stopped.").
        try:
            from donna.core_agent import _shutdown_agent_threads

            _shutdown_agent_threads(join_timeout=5.0)
        except Exception:
            pass
        raise SystemExit(130)
    except Exception:
        if _wants_no_gui():
            logging.getLogger("donna").exception("Unhandled exception in headless boot")
        raise
