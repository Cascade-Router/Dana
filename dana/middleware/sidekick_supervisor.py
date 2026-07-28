"""Sidekick process supervisor — owns vision poller + actuator lifecycle.

Starts eyes/hands as subprocesses, publishes/reads Blackboard heartbeats, and
exposes a degraded-mode summary for Chat. Does not merge workers into one
thread; keeps process isolation with first-class health.

Run standalone::

    python -m dana.middleware.sidekick_supervisor

Or let ``core_agent.agent_loop`` start it as a daemon thread.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from dana.memory.blackboard import (
    HEARTBEAT_ACTUATOR_KEY,
    HEARTBEAT_VISION_KEY,
    read_heartbeat,
    sidekick_health,
)

_LOCK = threading.Lock()
_STARTED = False
_STOP = threading.Event()
_CHILDREN: dict[str, subprocess.Popen[Any]] = {}
_SUPERVISOR_THREAD: threading.Thread | None = None

_CHECK_INTERVAL_S = 5.0
_HEARTBEAT_STALE_S = 45.0


def _python() -> str:
    return sys.executable or "python"


def _spawn(name: str, module: str) -> subprocess.Popen[Any]:
    creationflags = 0
    startupinfo = None
    if sys.platform == "win32":
        # Hide child consoles on spawn/restart (no flash during stop/cleanup).
        creationflags |= int(getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000))
        creationflags |= int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = int(getattr(subprocess, "SW_HIDE", 0))
    proc = subprocess.Popen(  # noqa: S603
        [_python(), "-m", module],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
        startupinfo=startupinfo,
    )
    print(
        f"[sidekick_supervisor] started {name} pid={proc.pid} module={module}",
        flush=True,
    )
    return proc


def _alive(proc: subprocess.Popen[Any] | None) -> bool:
    return proc is not None and proc.poll() is None


def ensure_children(*, restart: bool = True) -> dict[str, Any]:
    """Start or restart vision_poller + actuator_executor subprocesses."""
    with _LOCK:
        status: dict[str, Any] = {}
        specs = {
            "vision": "dana.middleware.vision_poller",
            "actuator": "dana.middleware.actuator_executor",
        }
        for name, module in specs.items():
            proc = _CHILDREN.get(name)
            if _alive(proc):
                status[name] = {"pid": proc.pid, "running": True, "restarted": False}
                continue
            if proc is not None and not restart:
                status[name] = {"pid": proc.pid, "running": False, "restarted": False}
                continue
            _CHILDREN[name] = _spawn(name, module)
            status[name] = {
                "pid": _CHILDREN[name].pid,
                "running": True,
                "restarted": proc is not None,
            }
        return status


def stop_children() -> None:
    """Terminate supervised children (best-effort)."""
    with _LOCK:
        for name, proc in list(_CHILDREN.items()):
            if proc is None:
                continue
            try:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=3.0)
                    except Exception:  # noqa: BLE001
                        proc.kill()
            except Exception:  # noqa: BLE001
                pass
            print(f"[sidekick_supervisor] stopped {name}", flush=True)
        _CHILDREN.clear()


def health_summary(*, stale_s: float = _HEARTBEAT_STALE_S) -> dict[str, Any]:
    """Return eyes/hands health + child process status."""
    health = sidekick_health()
    with _LOCK:
        children = {
            name: {
                "pid": getattr(proc, "pid", None),
                "running": _alive(proc),
            }
            for name, proc in _CHILDREN.items()
        }
    vision_hb = read_heartbeat(HEARTBEAT_VISION_KEY, stale_s=stale_s)
    actuator_hb = read_heartbeat(HEARTBEAT_ACTUATOR_KEY, stale_s=stale_s)
    return {
        **health,
        "children": children,
        "vision_heartbeat": vision_hb,
        "actuator_heartbeat": actuator_hb,
    }


def format_degraded_chat_hint(health: dict[str, Any] | None = None) -> str:
    """One-line Chat hint when eyes and/or hands are offline."""
    h = health or health_summary()
    missing: list[str] = []
    if not h.get("vision_alive"):
        missing.append("vision")
    if not h.get("actuator_alive"):
        missing.append("actuator")
    if not missing:
        return ""
    return (
        "[SIDEKICK DEGRADED: I can talk, but "
        + " and ".join(missing)
        + " appear offline.]"
    )


def _supervisor_loop() -> None:
    ensure_children(restart=True)
    while not _STOP.is_set():
        try:
            ensure_children(restart=True)
            h = health_summary()
            if h.get("degraded"):
                print(
                    f"[sidekick_supervisor] degraded vision_alive={h.get('vision_alive')} "
                    f"actuator_alive={h.get('actuator_alive')} children={h.get('children')}",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001
            print(f"[sidekick_supervisor] ERROR: {exc}", flush=True)
        _STOP.wait(timeout=_CHECK_INTERVAL_S)


def start_sidekick_supervisor(*, as_thread: bool = True) -> threading.Thread | None:
    """Idempotent start of the supervisor loop (daemon thread by default)."""
    global _STARTED, _SUPERVISOR_THREAD
    with _LOCK:
        if _STARTED and _SUPERVISOR_THREAD is not None and _SUPERVISOR_THREAD.is_alive():
            return _SUPERVISOR_THREAD
        _STOP.clear()
        if as_thread:
            t = threading.Thread(
                target=_supervisor_loop,
                name="SidekickSupervisor",
                daemon=True,
            )
            t.start()
            _SUPERVISOR_THREAD = t
            _STARTED = True
            return t
    _STARTED = True
    _supervisor_loop()
    return None


def stop_sidekick_supervisor() -> None:
    """Signal supervisor stop and terminate children."""
    global _STARTED
    _STOP.set()
    stop_children()
    with _LOCK:
        _STARTED = False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Donna sidekick eyes/hands supervisor")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Start children once, print health, exit (no monitor loop)",
    )
    args = parser.parse_args(argv)
    if args.once:
        ensure_children(restart=True)
        time.sleep(1.0)
        print(health_summary(), flush=True)
        return 0
    try:
        start_sidekick_supervisor(as_thread=False)
    except KeyboardInterrupt:
        stop_sidekick_supervisor()
    return 0


if __name__ == "__main__":
    # Ensure repo root on path when launched as __main__.
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    raise SystemExit(main())
