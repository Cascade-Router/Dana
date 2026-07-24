"""Donna — local voice agent package.

Submodules are importable as::

    from donna import agentic, tools, prompts, core_agent
    from donna.tools import broker
    from donna.paths import PROJECT_ROOT
"""

from __future__ import annotations

# Harden pythonw stdio before any submodule prints / tqdm / sounddevice.
try:
    from donna.stdio_boot import ensure_stdio

    ensure_stdio()
except Exception:
    pass

from donna.paths import DONNA_WORKSPACE, PROJECT_ROOT

__all__ = ["PROJECT_ROOT", "DONNA_WORKSPACE"]
