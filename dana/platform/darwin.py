"""macOS control-plane stub (``sys.platform == "darwin"``).

Not implemented yet — real window actuation on macOS needs AppleScript
(``osascript``, System Events) or PyObjC/Quartz window-server calls, neither
of which exist elsewhere in this codebase yet. This class exists so
:mod:`dana.platform.factory` has a real branch to dispatch to once that
lands, instead of silently falling through to ``MockControlPlane`` on a Mac.
There is deliberately no ``MacOSCADEngine`` — FreeCAD's headless IPC
(:mod:`dana.plugins.freecad.engine`) shells out to ``FreeCADCmd``, which is
cross-platform as-is, so a macOS build should reuse
``dana.platform.win32.RealFreeCADEngine`` (or a renamed platform-neutral
sibling) rather than needing its own CAD driver.
"""

from __future__ import annotations

from typing import Any

from dana.platform.base import BaseControlPlane

_NOT_IMPLEMENTED = (
    "MacOSControlPlane.{method}() is not implemented yet — macOS window "
    "actuation needs AppleScript/Quartz support that doesn't exist in this "
    "codebase yet. Use dana.platform.factory.get_control_plane() so this "
    "gap is visible instead of silently falling back to the mock driver."
)


class MacOSControlPlane(BaseControlPlane):
    def resync_workspace(self) -> dict[str, Any]:
        raise NotImplementedError(_NOT_IMPLEMENTED.format(method="resync_workspace"))

    def prevent_focus_steal(self) -> dict[str, Any]:
        raise NotImplementedError(_NOT_IMPLEMENTED.format(method="prevent_focus_steal"))

    def get_active_display(self) -> dict[str, Any]:
        raise NotImplementedError(_NOT_IMPLEMENTED.format(method="get_active_display"))


__all__ = ("MacOSControlPlane",)
