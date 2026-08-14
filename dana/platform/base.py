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
        self,
        length: float,
        width: float,
        height: float,
        name: str = "Box",
        placement: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> dict[str, Any]:
        """Create a parametric box primitive, translated by ``placement``
        (global X/Y/Z offset in mm) on top of its normal local origin.
        Returns a result dict including at least
        ``{"ok": bool, "path": str, "dimensions": {...}}``."""

    @abstractmethod
    def create_cylinder(
        self,
        radius: float,
        height: float,
        name: str = "Cylinder",
        placement: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> dict[str, Any]:
        """Create a parametric cylinder primitive, translated by
        ``placement`` (global X/Y/Z offset in mm). Same result shape as
        ``create_box``."""

    @abstractmethod
    def apply_boolean(
        self, operation: str, base_path: str, tool_path: str, name: str | None = None
    ) -> dict[str, Any]:
        """Combine two previously-created solids with a Boolean operation:
        ``"cut"`` subtracts the tool from the base, ``"union"`` fuses them,
        ``"intersect"`` keeps only their overlapping volume. Both paths must
        have been previously returned by ``create_box``/``create_cylinder``
        (or another ``apply_boolean``) on the SAME engine instance. Returns
        a result dict including ``{"ok": bool, "path": str, "name": str}``."""

    @abstractmethod
    def apply_edge_operation(
        self,
        operation: str,
        target_path: str,
        value: float,
        face_centroid: tuple[float, float, float] | None = None,
        name: str | None = None,
    ) -> dict[str, Any]:
        """Round (``"fillet"``) or bevel (``"chamfer"``) the edges of a
        previously-created solid by ``value`` mm. Without ``face_centroid``,
        every edge of the object is targeted (a global fillet/chamfer);
        with it, only the edges bounding the object's face nearest that
        point are targeted. Returns a result dict including
        ``{"ok": bool, "path": str, "name": str}``."""

    @abstractmethod
    def create_extrusion(
        self, profile_points: list[list[float]], height: float, name: str = "Extrusion"
    ) -> dict[str, Any]:
        """Extrude a closed 2D (XY) polyline ``profile_points`` ``height``
        units along Z into a solid — no arbitrary extrusion axis; a caller
        anchoring this to a clicked face's normal must itself confirm that
        normal is close enough to Z for a straight-up extrusion to be
        geometrically meaningful. Same result shape as ``create_box``."""

    @abstractmethod
    def create_pyramid(
        self,
        length: float,
        width: float,
        height: float,
        name: str = "Pyramid",
        placement: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> dict[str, Any]:
        """Create a sharp-edged rectangular pyramid: a ``length`` x ``width``
        base centered at the origin, apex at ``(0, 0, height)``, translated
        by ``placement`` (global X/Y/Z offset in mm). Same result shape as
        ``create_box``."""

    @abstractmethod
    def create_star_prism(
        self,
        points: int,
        outer_radius: float,
        inner_radius: float,
        height: float,
        name: str = "StarPrism",
        placement: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> dict[str, Any]:
        """Create a sharp-edged, ``points``-pointed star prism: an N-point
        star polygon (vertices alternating ``outer_radius``/``inner_radius``)
        extruded ``height`` units along Z, translated by ``placement``
        (global X/Y/Z offset in mm). Same result shape as ``create_box``."""

    @abstractmethod
    def export_mesh_stl(self, source_path: str, name: str | None = None) -> dict[str, Any]:
        """Tessellate/export the solid at ``source_path`` to a standalone
        ``.stl`` mesh file. Returns ``{"ok": bool, "path": str}``."""

    @abstractmethod
    def modify_parameter(self, target_path: str, parameter_name: str, new_value: float) -> dict[str, Any]:
        """Change a single dimensional property (e.g. ``"Height"``,
        ``"Radius"``) on a previously-created object, in place — reopens
        and overwrites the SAME document/path rather than creating a new
        one. Returns a result dict including ``{"ok": bool, "path": str,
        "name": str}``."""

    @abstractmethod
    def get_bounding_box(self, target_path: str) -> dict[str, Any]:
        """Read-only: the physical bounding box of a previously-created
        object, in mm. Never mutates anything. Returns
        ``{"ok": bool, "x_min": float, "y_min": float, "z_min": float,
        "x_max": float, "y_max": float, "z_max": float}``."""

    @abstractmethod
    def create_pipe(
        self,
        pipe_radius: float,
        path_type: str,
        length_or_angle: float,
        name: str = "Pipe",
        placement: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> dict[str, Any]:
        """Sweep a circular profile (``pipe_radius`` mm) into a tubular
        solid, translated by ``placement`` (global X/Y/Z offset in mm).
        ``path_type="straight"`` sweeps ``length_or_angle`` mm along a
        straight line; ``path_type="arc"`` sweeps ``length_or_angle``
        degrees along a circular arc (a curved elbow). Same result shape
        as ``create_box``."""
