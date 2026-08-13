"""OS/CAD platform abstraction layer.

Everything above this package (the unified Gradio UI, tool broker, etc.)
should depend only on :mod:`dana.platform.base`'s interfaces and
:mod:`dana.platform.factory`'s driver selection — never import
``dana.platform.win32``/``dana.platform.mock``/``dana.platform.darwin``
directly. That's what lets the same call site run against real Win32/FreeCAD
on desktop and simulated telemetry on a Hugging Face Space without an
``if IS_HF_SPACE`` branch at every call site.
"""

from __future__ import annotations

from dana.platform.base import BaseCADEngine, BaseControlPlane
from dana.platform.factory import get_cad_engine, get_control_plane

__all__ = (
    "BaseCADEngine",
    "BaseControlPlane",
    "get_cad_engine",
    "get_control_plane",
)
