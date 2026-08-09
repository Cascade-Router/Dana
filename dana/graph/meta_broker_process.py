"""Isolated Meta-Broker runner — ``multiprocessing.Process`` + Queue IPC.

Headless ``--no-gui`` runs drain telemetry on a dedicated daemon thread so the
child never blocks on a saturated pipe. Child puts are always non-blocking.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue as queue_mod
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable

# Child-process telemetry sink (set only inside the worker).
_CHILD_QUEUE: Any = None

# Parent-process headless drain (thread-safe).
_HEADLESS_Q: queue_mod.Queue | None = None
_HEADLESS_DRAINER: threading.Thread | None = None
_HEADLESS_STOP = threading.Event()
_HEADLESS_LOCK = threading.Lock()
_HEADLESS_LOG_PATH: Path | None = None

DEFAULT_ISOLATED_TIMEOUT_S = 300.0
_TELEMETRY_QUEUE_MAX = 256


def _json_safe(value: Any, *, depth: int = 0) -> Any:
    if depth > 8:
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            out[str(k)] = _json_safe(v, depth=depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        return [_json_safe(v, depth=depth + 1) for v in value]
    return str(value)


def _is_full_error(exc: BaseException) -> bool:
    name = type(exc).__name__
    return name in {"Full", "QueueFull"} or "Full" in name or "full" in str(exc).lower()


def _mp_put_nowait(q: Any, payload: dict[str, Any], *, critical: bool = False) -> bool:
    """put_nowait with optional drain-and-retry for critical result payloads."""
    try:
        q.put_nowait(payload)
        return True
    except Exception as exc:
        if not _is_full_error(exc):
            try:
                q.put_nowait(payload)
                return True
            except Exception:
                return False
    if not critical:
        return False
    # Make room for the result by discarding oldest telemetry.
    for _ in range(_TELEMETRY_QUEUE_MAX):
        try:
            _ = q.get_nowait()
        except Exception:
            break
        try:
            q.put_nowait(payload)
            return True
        except Exception:
            continue
    return False


def resolve_headless_log_path() -> Path:
    """Prefer ``DANA_META_BROKER_LOG``, else ``logs/meta_broker_headless.log``."""
    override = (os.environ.get("DANA_META_BROKER_LOG") or "").strip()
    if override:
        return Path(override)
    try:
        from dana.paths import PROJECT_ROOT

        root = Path(PROJECT_ROOT)
    except Exception:  # noqa: BLE001
        root = Path.cwd()
    # Suite harnesses often set DANA_META_BROKER_LOG=logs/lru_cache_suite.log.
    # Fall back to a dedicated headless IPC log under logs/.
    preferred = root / "logs" / "lru_cache_suite.log"
    if preferred.is_file() or (os.environ.get("DANA_SUITE_LOG") or "").strip():
        return preferred
    return root / "logs" / "meta_broker_headless.log"


def _format_event_line(event: dict[str, Any]) -> str:
    kind = str(event.get("type") or "telemetry")
    msg = str(event.get("message") or event.get("error") or "").strip()
    phase = str(event.get("phase") or "")
    status = str(event.get("status") or "")
    bits = [f"[MetaBrokerIPC] kind={kind}"]
    if phase:
        bits.append(f"phase={phase}")
    if status:
        bits.append(f"status={status}")
    if msg:
        bits.append(msg)
    return " ".join(bits)


def _write_headless_line(line: str) -> None:
    text = line if line.endswith("\n") else line + "\n"
    try:
        print(text, end="", flush=True)
    except Exception:
        pass
    path = _HEADLESS_LOG_PATH
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", errors="replace") as fh:
            fh.write(text)
    except Exception:
        pass


def _headless_drainer_loop() -> None:
    """Continuously drain the parent telemetry queue → log + stdout."""
    while not _HEADLESS_STOP.is_set():
        q = _HEADLESS_Q
        if q is None:
            time.sleep(0.1)
            continue
        try:
            item = q.get(timeout=0.1)
        except queue_mod.Empty:
            continue
        except Exception:
            time.sleep(0.1)
            continue
        if not isinstance(item, dict):
            continue
        try:
            _write_headless_line(_format_event_line(item))
        except Exception:
            pass


def start_headless_telemetry_drainer(
    *,
    log_path: str | Path | None = None,
) -> None:
    """Launch the headless Queue consumer (idempotent). Safe for ``--no-gui``."""
    global _HEADLESS_Q, _HEADLESS_DRAINER, _HEADLESS_LOG_PATH
    with _HEADLESS_LOCK:
        if log_path is not None:
            _HEADLESS_LOG_PATH = Path(log_path)
        elif _HEADLESS_LOG_PATH is None:
            _HEADLESS_LOG_PATH = resolve_headless_log_path()
        if _HEADLESS_Q is None:
            _HEADLESS_Q = queue_mod.Queue(maxsize=_TELEMETRY_QUEUE_MAX)
        if _HEADLESS_DRAINER is not None and _HEADLESS_DRAINER.is_alive():
            return
        _HEADLESS_STOP.clear()
        _HEADLESS_DRAINER = threading.Thread(
            target=_headless_drainer_loop,
            name="MetaBrokerHeadlessDrainer",
            daemon=True,
        )
        _HEADLESS_DRAINER.start()
        _write_headless_line(
            f"[MetaBrokerIPC] headless drainer started log={_HEADLESS_LOG_PATH}"
        )


def stop_headless_telemetry_drainer() -> None:
    """Signal the headless drainer to stop (tests / shutdown)."""
    _HEADLESS_STOP.set()


def enqueue_headless_event(event: dict[str, Any]) -> None:
    """Parent-side fan-in for telemetry (non-blocking)."""
    q = _HEADLESS_Q
    if q is None:
        return
    try:
        q.put_nowait(dict(event))
    except queue_mod.Full:
        try:
            _ = q.get_nowait()
        except Exception:
            return
        try:
            q.put_nowait(dict(event))
        except Exception:
            pass
    except Exception:
        pass


def _broker_worker(prompt: str, queue: Any, env_extra: dict[str, str] | None) -> None:
    """Process entry: run LangGraph Meta-Broker; never kill the parent GUI."""
    global _CHILD_QUEUE
    _CHILD_QUEUE = queue
    if env_extra:
        os.environ.update({str(k): str(v) for k, v in env_extra.items()})
    log_path = (os.environ.get("DANA_META_BROKER_LOG") or "").strip()
    _log_fh = None
    if log_path:
        try:
            p = Path(log_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            _log_fh = p.open("a", encoding="utf-8", errors="replace")
            _log_fh.write(f"\n# meta_broker child pid={os.getpid()}\n")
            _log_fh.flush()

            class _Tee:
                def __init__(self, primary: Any, secondary: Any) -> None:
                    self.primary = primary
                    self.secondary = secondary

                def write(self, data: str) -> int:
                    try:
                        self.primary.write(data)
                    except Exception:
                        pass
                    try:
                        self.secondary.write(data)
                        self.secondary.flush()
                    except Exception:
                        pass
                    return len(data or "")

                def flush(self) -> None:
                    for s in (self.primary, self.secondary):
                        try:
                            s.flush()
                        except Exception:
                            pass

            import sys

            sys.stdout = _Tee(sys.stdout, _log_fh)  # type: ignore[assignment]
            sys.stderr = _Tee(sys.stderr, _log_fh)  # type: ignore[assignment]
        except Exception:
            _log_fh = None
    try:
        _mp_put_nowait(
            queue,
            {
                "type": "telemetry",
                "message": f"Meta-Broker process started pid={os.getpid()}",
                "phase": "start",
                "status": "planning",
            },
            critical=False,
        )
        from dana.graph.builder import run_meta_broker

        final = run_meta_broker(str(prompt or ""))
        _mp_put_nowait(
            queue,
            {
                "type": "result",
                "ok": True,
                "state": _json_safe(dict(final or {})),
            },
            critical=True,
        )
    except Exception as exc:  # noqa: BLE001
        _mp_put_nowait(
            queue,
            {
                "type": "result",
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "state": {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "final_response": f"Meta-Broker process crashed: {exc}",
                    "epics": [],
                    "epic_log": [f"process_crash: {type(exc).__name__}: {exc}"],
                },
            },
            critical=True,
        )
    finally:
        _CHILD_QUEUE = None
        if _log_fh is not None:
            try:
                _log_fh.close()
            except Exception:
                pass


def child_queue_put(payload: dict[str, Any]) -> None:
    """Non-blocking telemetry push from inside the Meta-Broker child.

    Never falls back to blocking ``put`` — a full queue drops the event so
    artifact generation continues.
    """
    q = _CHILD_QUEUE
    if q is None:
        return
    try:
        q.put_nowait(dict(payload))
    except Exception:
        return


def run_meta_broker_isolated(
    prompt: str,
    *,
    timeout_s: float | None = DEFAULT_ISOLATED_TIMEOUT_S,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    env_extra: dict[str, str] | None = None,
    stop_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Spawn Meta-Broker in a child process; forward Queue events to the parent.

    ``timeout_s`` defaults to 300s. On expiry the child is terminated and a
    clean failure dict is returned (no parent deadlock).

    When ``stop_event`` is set, the parent terminates the child and returns a
    cancelled failure dict (Gradio / HF Space Stop button).
    """
    if timeout_s is None:
        timeout_s = DEFAULT_ISOLATED_TIMEOUT_S
    try:
        timeout_s = float(timeout_s)
    except (TypeError, ValueError):
        timeout_s = DEFAULT_ISOLATED_TIMEOUT_S
    if timeout_s <= 0:
        timeout_s = DEFAULT_ISOLATED_TIMEOUT_S

    # Ensure headless drainer is alive when no GUI is draining via Tk after().
    if (os.environ.get("DANA_HEADLESS") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    } or (os.environ.get("DANA_NO_GUI") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        start_headless_telemetry_drainer()

    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue(maxsize=_TELEMETRY_QUEUE_MAX)
    child_env = dict(env_extra or {})
    if "DANA_META_BROKER_LOG" not in child_env:
        try:
            child_env["DANA_META_BROKER_LOG"] = str(resolve_headless_log_path())
        except Exception:
            pass

    proc = ctx.Process(
        target=_broker_worker,
        args=(str(prompt or ""), queue, child_env),
        name="DanaMetaBroker",
        daemon=True,
    )
    proc.start()
    enqueue_headless_event(
        {
            "type": "telemetry",
            "message": f"spawned Meta-Broker child pid={proc.pid}",
            "phase": "start",
            "status": "planning",
        }
    )

    result: dict[str, Any] | None = None
    deadline = time.time() + float(timeout_s)
    timed_out = False

    def _dispatch(item: dict[str, Any]) -> None:
        enqueue_headless_event(item)
        if on_event is not None:
            try:
                on_event(item)
            except Exception:  # noqa: BLE001
                pass

    while True:
        if stop_event is not None and stop_event.is_set():
            timed_out = True
            try:
                proc.terminate()
            except Exception:  # noqa: BLE001
                pass
            try:
                proc.join(timeout=5.0)
            except Exception:  # noqa: BLE001
                pass
            if proc.is_alive():
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass
            result = {
                "status": "failed",
                "error": "meta_broker cancelled by user",
                "final_response": "Meta-Broker stopped by user.",
                "epics": [],
                "epic_log": ["process_cancelled"],
            }
            enqueue_headless_event(
                {
                    "type": "telemetry",
                    "message": "STOPPED: Meta-Broker child terminated by user",
                    "phase": "cancelled",
                    "status": "failed",
                    "terminal": True,
                }
            )
            break

        if time.time() >= deadline:
            timed_out = True
            try:
                proc.terminate()
            except Exception:  # noqa: BLE001
                pass
            try:
                proc.join(timeout=5.0)
            except Exception:  # noqa: BLE001
                pass
            if proc.is_alive():
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass
            result = {
                "status": "failed",
                "error": f"meta_broker process timed out after {timeout_s}s",
                "final_response": f"Meta-Broker timed out after {timeout_s}s",
                "epics": [],
                "epic_log": ["process_timeout"],
            }
            enqueue_headless_event(
                {
                    "type": "telemetry",
                    "message": (
                        f"TIMEOUT: terminated Meta-Broker child after {timeout_s}s"
                    ),
                    "phase": "timeout",
                    "status": "failed",
                }
            )
            break

        try:
            item = queue.get(timeout=0.1)
        except Exception:
            item = None

        if item is not None and isinstance(item, dict):
            kind = str(item.get("type") or "")
            if kind == "telemetry":
                _dispatch(item)
            elif kind == "result":
                _dispatch(item)
                if item.get("ok"):
                    state = item.get("state")
                    result = (
                        dict(state)
                        if isinstance(state, dict)
                        else {
                            "status": "completed",
                            "final_response": str(state or ""),
                            "epics": [],
                            "epic_log": [],
                        }
                    )
                else:
                    state = item.get("state")
                    if isinstance(state, dict) and state:
                        result = dict(state)
                    else:
                        result = {
                            "status": "failed",
                            "error": str(
                                item.get("error") or "meta_broker process failed"
                            ),
                            "final_response": str(
                                item.get("error") or "Meta-Broker process failed"
                            ),
                            "epics": [],
                            "epic_log": [
                                str(item.get("error") or "process_failed")
                            ],
                        }
                break

        if not proc.is_alive() and result is None:
            try:
                while True:
                    item = queue.get_nowait()
                    if isinstance(item, dict) and str(item.get("type") or "") == "result":
                        _dispatch(item)
                        state = item.get("state")
                        result = (
                            dict(state)
                            if isinstance(state, dict)
                            else {
                                "status": "failed",
                                "error": str(item.get("error") or "exit"),
                                "epics": [],
                                "epic_log": [],
                            }
                        )
                        break
                    if isinstance(item, dict):
                        _dispatch(item)
            except Exception:  # noqa: BLE001
                pass
            if result is None:
                result = {
                    "status": "failed",
                    "error": f"meta_broker process exited (code={proc.exitcode})",
                    "final_response": (
                        f"Meta-Broker process exited with code={proc.exitcode}"
                    ),
                    "epics": [],
                    "epic_log": [f"process_exit_code={proc.exitcode}"],
                }
            break

    if not timed_out:
        try:
            proc.join(timeout=5.0)
        except Exception:  # noqa: BLE001
            pass
    elif proc.is_alive():
        try:
            proc.join(timeout=2.0)
        except Exception:  # noqa: BLE001
            pass

    return result or {
        "status": "failed",
        "error": "meta_broker returned no result",
        "final_response": "Meta-Broker returned no result",
        "epics": [],
        "epic_log": [],
    }


__all__ = (
    "DEFAULT_ISOLATED_TIMEOUT_S",
    "child_queue_put",
    "enqueue_headless_event",
    "resolve_headless_log_path",
    "run_meta_broker_isolated",
    "start_headless_telemetry_drainer",
    "stop_headless_telemetry_drainer",
)
