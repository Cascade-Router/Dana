"""Dana — local voice agent package.

Submodules are importable as::

    from dana import agentic, tools, prompts, core_agent
    from dana.tools import broker
    from dana.paths import PROJECT_ROOT
"""

from __future__ import annotations

# Harden pythonw stdio before any submodule prints / tqdm / sounddevice.
try:
    from dana.stdio_boot import ensure_stdio

    ensure_stdio()
except Exception:
    pass

from dana.paths import DANA_WORKSPACE, PROJECT_ROOT

__version__ = "0.1.0"

__all__ = ["PROJECT_ROOT", "DANA_WORKSPACE", "__version__"]
