"""Root entry point for Dana.

Always resolves the repo root onto ``sys.path`` and as the process cwd so
``python run.py`` works from any working directory.
"""

from __future__ import annotations

import logging
import os
import sys

# Windows taskbar: claim an explicit AppUserModelID BEFORE any Tk/CTk window.
if sys.platform == "win32":
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "dana.assistant.desktop.v1"
        )
    except Exception:
        pass

_ROOT = os.path.abspath(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
for _sub in ("scripts", os.path.join("scripts", "diagnostics")):
    _p = os.path.join(_ROOT, _sub)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
try:
    os.chdir(_ROOT)
except OSError:
    pass

# pythonw.exe leaves stdout/stderr as None — patch before any library prints/tqdm.
try:
    from dana.stdio_boot import ensure_stdio

    ensure_stdio()
except Exception:
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8", errors="replace")  # type: ignore[assignment]
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8", errors="replace")  # type: ignore[assignment]

# Capture fatal main-thread + background-thread crashes before anything else.
try:
    from dana.logging import install_fatal_crash_hooks

    install_fatal_crash_hooks()
except Exception as _fatal_hook_exc:  # noqa: BLE001
    print(
        f"[Main] WARNING: fatal crash hooks unavailable ({_fatal_hook_exc})",
        file=sys.stderr,
        flush=True,
    )

# Ensure workspace dirs exist + migrate legacy artifacts before agent boot.
try:
    from dana.workspace import ensure_dana_workspace

    ensure_dana_workspace(migrate=True)
except Exception as exc:  # noqa: BLE001
    print(f"[Workspace] WARNING: ensure_dana_workspace failed: {exc}", file=sys.stderr)

# Held for process lifetime so the OS releases the bind only on exit.
_DANA_INSTANCE_LOCK_SOCK = None
# Dedicated loopback port — not the telemetry dashboard (47474).
_DANA_INSTANCE_LOCK_PORT = 47473
_HEADLESS_LOG_NAME = "dana_headless.log"


def _wants_no_gui(argv: list[str] | None = None) -> bool:
    args = list(sys.argv[1:] if argv is None else argv)
    return "--no-gui" in args


def _configure_headless_logging() -> str | None:
    """File logging fallback when OS-level stdout redirect is missing/broken."""
    # Mark headless early so Meta-Broker IPC starts its queue drainer.
    os.environ.setdefault("DANA_NO_GUI", "1")
    os.environ.setdefault("DANA_HEADLESS", "1")
    if not (os.environ.get("DANA_META_BROKER_LOG") or "").strip():
        suite = os.path.join(_ROOT, "logs", "lru_cache_suite.log")
        default = os.path.join(_ROOT, "logs", "meta_broker_headless.log")
        os.environ["DANA_META_BROKER_LOG"] = (
            suite if os.path.isfile(suite) else default
        )
    try:
        from dana.graph.meta_broker_process import start_headless_telemetry_drainer

        start_headless_telemetry_drainer(
            log_path=os.environ.get("DANA_META_BROKER_LOG")
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"[Main] WARNING: Meta-Broker headless drainer unavailable ({exc})",
            file=sys.stderr,
            flush=True,
        )
    log_path = os.path.join(_ROOT, _HEADLESS_LOG_NAME)
    try:
        root = logging.getLogger()
        # Avoid duplicate handlers on reload.
        for h in list(root.handlers):
            if getattr(h, "_dana_headless", False):
                return log_path
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        handler._dana_headless = True  # type: ignore[attr-defined]
        root.addHandler(handler)
        root.setLevel(logging.DEBUG)
        logging.captureWarnings(True)
        logging.getLogger("dana").info(
            "Headless file logging active → %s (cwd=%s)",
            log_path,
            os.path.abspath(os.getcwd()),
        )
        return log_path
    except Exception as exc:  # noqa: BLE001
        print(f"[Main] WARNING: headless file logging failed: {exc}", file=sys.stderr)
        return None


def _acquire_single_instance_lock() -> bool:
    """Bind a loopback TCP socket; False if another Dana already holds it."""
    global _DANA_INSTANCE_LOCK_SOCK
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # Exclusive bind (no SO_REUSEADDR) — second instance must fail.
        sock.bind(("127.0.0.1", _DANA_INSTANCE_LOCK_PORT))
        sock.listen(1)
    except OSError:
        try:
            sock.close()
        except OSError:
            pass
        return False
    _DANA_INSTANCE_LOCK_SOCK = sock
    return True


# Must match the major of torch pinned in requirements-cuda.txt (2.13.0+cu126).
_EXPECTED_TORCH_MAJOR = 2


def verify_environment() -> None:
    """Lightweight startup guard: CUDA visibility + torch major pin check.

    Runs before ``dana.core_agent`` so transformers / Whisper / Florence stay
    off the critical path until the environment looks sane.
    """
    try:
        import torch
    except ImportError as exc:
        print(
            "[Env] ERROR: PyTorch is not installed. "
            "Install from requirements-cuda.txt (CUDA cu126 index) after "
            "requirements.txt.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1) from exc

    version = getattr(torch, "__version__", "0")
    try:
        major = int(str(version).split(".", 1)[0].split("+", 1)[0])
    except ValueError:
        major = -1
    if major != _EXPECTED_TORCH_MAJOR:
        print(
            f"[Env] ERROR: PyTorch major version drift: got {version!r}, "
            f"expected major {_EXPECTED_TORCH_MAJOR} "
            f"(see requirements-cuda.txt torch==2.13.0+cu126).",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1)

    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "CUDA device 0"
        print(f"[Env] CUDA available - using GPU ({name}).", flush=True)
    else:
        print(
            "[Env] WARNING: CUDA not available - falling back to CPU. "
            "Vision / Whisper will be slower.",
            flush=True,
        )


def _startup_crash_log_path() -> str:
    return os.path.join(_ROOT, "logs", "startup_crash.log")


def _write_startup_crash_log(exc: BaseException) -> str | None:
    """Persist a full traceback for packaged / pythonw startups."""
    import traceback

    path = _startup_crash_log_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", errors="replace") as fh:
            fh.write(traceback.format_exc())
            if not str(exc):
                fh.write(f"\n{type(exc).__name__}\n")
        return path
    except Exception as write_exc:  # noqa: BLE001
        print(
            f"[Main] WARNING: could not write startup_crash.log: {write_exc}",
            file=sys.stderr,
            flush=True,
        )
        return None


def _show_startup_crash_dialog(exc: BaseException, log_path: str | None) -> None:
    """Surface the exception via MessageBoxW (Windows) or tkinter.messagebox."""
    detail = f"{type(exc).__name__}: {exc}"
    if log_path:
        detail = f"{detail}\n\nFull traceback written to:\n{log_path}"
    title = "Dānā Startup Error"

    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(0, detail, title, 0x10)
            return
        except Exception:  # noqa: BLE001
            pass
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        try:
            messagebox.showerror(title, detail)
        finally:
            try:
                root.destroy()
            except Exception:  # noqa: BLE001
                pass
        return
    except Exception:  # noqa: BLE001
        pass
    print(f"[Main] {title}: {detail}", file=sys.stderr, flush=True)


def _run_desktop_main() -> int:
    if _wants_no_gui():
        _configure_headless_logging()

    if not _acquire_single_instance_lock():
        msg = (
            "[Main] ERROR: Another instance of Dana is already running. "
            "Aborting to protect execution jail."
        )
        print(msg, flush=True)
        if _wants_no_gui():
            logging.getLogger("dana").error(msg)
        return 1

    verify_environment()

    # Defer core_agent import until launch so torch/transformers/YOLO stay off
    # the interpreter's critical path during ``run.py`` module load.
    from dana.core_agent import main  # noqa: E402

    return int(main() or 0)


if __name__ == "__main__":
    try:
        raise SystemExit(_run_desktop_main())
    except SystemExit:
        raise
    except KeyboardInterrupt:
        # Closing the GUI / Ctrl+C should exit quietly (workers log "Stopped.").
        try:
            from dana.core.app_runtime import _shutdown_agent_threads

            _shutdown_agent_threads(join_timeout=5.0)
        except Exception:
            pass
        raise SystemExit(130)
    except Exception as exc:
        if _wants_no_gui():
            logging.getLogger("dana").exception("Unhandled exception in headless boot")
        try:
            from dana.logging import write_fatal_crash_log

            write_fatal_crash_log(
                "run.py.__main__",
                type(exc),
                exc,
                exc.__traceback__,
                thread_name="MainThread",
            )
        except Exception:
            pass
        log_path = _write_startup_crash_log(exc)
        if not _wants_no_gui():
            _show_startup_crash_dialog(exc, log_path)
        else:
            print(
                f"[Main] Startup crash — see {log_path or 'stderr'}",
                file=sys.stderr,
                flush=True,
            )
        raise SystemExit(1) from exc
