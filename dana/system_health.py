"""OS-level safety rails: RAM circuit breaker, LLM concurrency, process trees."""

from __future__ import annotations

import os
import threading
from typing import Any

import psutil

# Serialize ALL LLM generations (Ollama + Cloud) — never double VRAM load.
llm_lock = threading.RLock()

# Raised to tolerate heavy browser / Chrome RAM without false-positive aborts.
RAM_THRESHOLD_PERCENT = 92.0
_RAM_ABORT_PCT = RAM_THRESHOLD_PERCENT


def check_system_health(*, ram_limit_pct: float = _RAM_ABORT_PCT) -> dict[str, Any]:
    """Return a small health snapshot; abort if RAM usage exceeds the safe limit.

    Raises
    ------
    SystemError
        When virtual memory ``percent`` is above ``ram_limit_pct``
        (default ``RAM_THRESHOLD_PERCENT`` = 92%).
    """
    mem = psutil.virtual_memory()
    cpu = float(psutil.cpu_percent(interval=None))
    snap = {
        "ram_percent": float(mem.percent),
        "ram_available_mb": float(mem.available) / (1024.0 * 1024.0),
        "cpu_percent": cpu,
        "ram_threshold_percent": float(ram_limit_pct),
    }
    skip = (os.environ.get("DANA_SKIP_RAM_BREAKER") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if (not skip) and float(mem.percent) > float(ram_limit_pct):
        try:
            from dana.audio.tts_manager import get_tts_manager

            get_tts_manager().notify(
                "Warning: RAM usage exceeds safe limits."
            )
        except Exception:  # noqa: BLE001
            pass
        raise SystemError(
            "CRITICAL: RAM usage exceeds safe limits. Aborting to prevent OS crash."
        )
    if skip and float(mem.percent) > float(ram_limit_pct):
        print(
            f"[SystemHealth] RAM breaker skipped "
            f"(DANA_SKIP_RAM_BREAKER=1, ram={mem.percent:.1f}%)",
            flush=True,
        )
    return snap


def kill_process_tree(pid: int) -> None:
    """Force-kill ``pid`` and all descendants (rogue pytest / compile children)."""
    try:
        pid_i = int(pid)
    except (TypeError, ValueError):
        return
    if pid_i <= 0:
        return
    try:
        parent = psutil.Process(pid_i)
    except (psutil.NoSuchProcess, psutil.Error):
        return
    try:
        children = parent.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.Error):
        children = []
    for child in children:
        try:
            child.kill()
        except (psutil.NoSuchProcess, psutil.Error):
            pass
    try:
        parent.kill()
    except (psutil.NoSuchProcess, psutil.Error):
        pass
    try:
        gone, alive = psutil.wait_procs([*children, parent], timeout=3)
    except (psutil.Error, OSError):
        return
    for p in alive:
        try:
            p.kill()
        except (psutil.NoSuchProcess, psutil.Error):
            pass
    _ = gone


__all__ = (
    "RAM_THRESHOLD_PERCENT",
    "check_system_health",
    "kill_process_tree",
    "llm_lock",
)
