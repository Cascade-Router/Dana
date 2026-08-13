"""Abstract driver interfaces for OS window actuation and parametric CAD.

Every concrete driver (:mod:`dana.platform.win32`, :mod:`dana.platform.mock`,
:mod:`dana.platform.darwin`) implements these two interfaces so the rest of
the app — the unified Gradio UI, the tool broker — can call
``control_plane.resync_workspace()`` without knowing or caring whether it's
talking to real Win32 APIs or a mocked telemetry stream. Every method
returns a plain ``dict`` (not a JSON string) with at least an ``"ok"`` key,
so callers never need to guess the shape before deciding what to render.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseControlPlane(ABC):
    """OS-level window/display management, abstracted over the host platform."""

    @abstractmethod
    def resync_workspace(self) -> dict[str, Any]:
        """Reconcile managed background-app windows (e.g. FreeCAD) onto their
        target monitor without activating them.

        Returns a report dict, at minimum ``{"ok": bool, "moved": [...]}``.
        """

    @abstractmethod
    def prevent_focus_steal(self) -> dict[str, Any]:
        """Assert the zero-focus contract: report the current foreground
        window without changing it, so a caller can diff before/after an
        actuation and confirm nothing stole focus.

        Returns ``{"ok": bool, "foreground": {...} | None}``.
        """

    @abstractmethod
    def get_active_display(self) -> dict[str, Any]:
        """Return the current display topology: primary size plus any
        secondary monitor geometry actuators can target.

        Returns ``{"ok": bool, "primary": {...}, "secondary": {...} | None}``.
        """


class BaseCADEngine(ABC):
    """Parametric CAD geometry generation, abstracted over the host platform."""

    @abstractmethod
    def create_box(
        self, length: float, width: float, height: float, name: str = "Box"
    ) -> dict[str, Any]:
        """Create a parametric box primitive. Returns a result dict including
        at least ``{"ok": bool, "path": str, "dimensions": {...}}``."""

    @abstractmethod
    def create_cylinder(
        self, radius: float, height: float, name: str = "Cylinder"
    ) -> dict[str, Any]:
        """Create a parametric cylinder primitive. Same result shape as
        ``create_box``."""

    @abstractmethod
    def apply_boolean_cut(
        self, base_path: str, tool_path: str, name: str = "Cut"
    ) -> dict[str, Any]:
        """Subtract the ``tool_path`` solid from the ``base_path`` solid —
        both previously returned by ``create_box``/``create_cylinder`` (or
        another cut) on the SAME engine instance. Returns a result dict
        including ``{"ok": bool, "path": str}``."""

    @abstractmethod
    def export_mesh_stl(self, source_path: str, name: str | None = None) -> dict[str, Any]:
        """Tessellate/export the solid at ``source_path`` to a standalone
        ``.stl`` mesh file. Returns ``{"ok": bool, "path": str}``."""
