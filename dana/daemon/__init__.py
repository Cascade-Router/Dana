"""Process-isolated Agent Engine sidecar (Phase 2A).

The GUI shell talks to this package over loopback TCP (JSON lines).
Prefer importing concrete symbols from submodules in production code.
"""

from __future__ import annotations

from dana.daemon.engine import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    EngineDaemon,
    run_engine_daemon,
)
from dana.daemon.watchdog import ProcessWatchdog, default_session_path

__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "EngineDaemon",
    "ProcessWatchdog",
    "default_session_path",
    "run_engine_daemon",
]
