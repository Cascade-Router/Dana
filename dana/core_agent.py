"""Dana core agent — backwards-compatible facade.

Decomposed across Phases 5-8 into dana.core.shared_state/constants/
agent_loop/app_runtime, dana.audio.*, dana.ui.*, dana.vision.*, and
dana.ingestion.* -- see docs/architecture/phase7_core_agent_decomposition.md.
Kept here: the handful of names still reached via ``dana.core_agent`` by
historical convention (``run.py``, ``dana.ui.main``, tests using ``DanaGUI``).

  python -m dana.core_agent [--download]
"""

from __future__ import annotations

import os
import sys

# Bootstrap BEFORE package imports: running ``python dana/core_agent.py`` puts
# ``dana/`` on sys.path[0], which breaks ``import dana`` and root modules.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dana.paths import apply_windows_process_hardening, ensure_project_root_on_syspath
ensure_project_root_on_syspath()
apply_windows_process_hardening()

# Order is load-bearing: dana.audio's module-level init calls back into
# dana.core.shared_state (lazily); importing audio first avoids a
# partial-module AttributeError if this is the process's first touch of
# either side. See docs/architecture/phase7_core_agent_decomposition.md.
import dana.audio  # noqa: F401
import dana.core.shared_state as state
from dana.core.app_runtime import agent_loop, main
from dana.ui.app_gui import DanaGUI

__all__ = ["DanaGUI", "agent_loop", "main", "state"]

if __name__ == "__main__":
    try:
        from dana.stdio_boot import ensure_stdio

        ensure_stdio()
    except Exception:
        pass
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        try:
            from dana.core.app_runtime import _shutdown_agent_threads

            _shutdown_agent_threads(join_timeout=5.0)
        except Exception:
            pass
        sys.exit(130)
