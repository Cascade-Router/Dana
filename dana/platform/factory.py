"""Dynamic platform driver selection — the single place that decides which
concrete ``BaseControlPlane``/``BaseCADEngine`` implementation runs.

Call sites should use ``get_control_plane()``/``get_cad_engine()`` and never
import ``dana.platform.win32``/``dana.platform.mock``/``dana.platform.darwin``
directly — that keeps the "which driver am I on" decision in exactly one
place instead of scattered ``if IS_HF_SPACE`` checks throughout the app.
"""

from __future__ import annotations

import os
import sys

from dana.platform.base import BaseCADEngine, BaseControlPlane

IS_HF_SPACE = os.getenv("SPACE_ID") is not None
IS_WINDOWS = sys.platform == "win32"
IS_MAC = sys.platform == "darwin"


def get_control_plane() -> BaseControlPlane:
    if IS_HF_SPACE:
        from dana.platform.mock import MockControlPlane

        return MockControlPlane()
    if IS_WINDOWS:
        from dana.platform.win32 import Win32ControlPlane

        return Win32ControlPlane()
    if IS_MAC:
        from dana.platform.darwin import MacOSControlPlane

        return MacOSControlPlane()
    from dana.platform.mock import MockControlPlane

    return MockControlPlane()


def get_cad_engine() -> BaseCADEngine:
    if IS_HF_SPACE:
        from dana.platform.mock import MockFreeCADEngine

        return MockFreeCADEngine()
    if IS_WINDOWS:
        from dana.platform.win32 import RealFreeCADEngine

        return RealFreeCADEngine()
    # macOS / other: FreeCADCmd IPC is cross-platform in principle, but
    # untested off Windows here — fall back to the mock rather than assume
    # dana.plugins.freecad.engine's binary-discovery globs work unmodified.
    from dana.platform.mock import MockFreeCADEngine

    return MockFreeCADEngine()


__all__ = ("IS_HF_SPACE", "IS_MAC", "IS_WINDOWS", "get_cad_engine", "get_control_plane")
