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
    def inspect_spatial_properties(self, target_path: str) -> dict[str, Any]:
        """Read-only: richer topology introspection than ``get_bounding_box``
        — volume, surface area, center of mass, solid validity, and face/
        edge/vertex counts for a previously-created object. Never mutates
        anything. Lets a caller check topology complexity/validity before a
        risky fillet/chamfer/boolean rather than discovering infeasibility
        only after it fails. Returns a result dict including
        ``{"ok": bool, "volume": float, "area": float, "center_of_mass":
        [x, y, z], "is_valid": bool, "face_count": int, "edge_count": int,
        "vertex_count": int}``."""

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

    @abstractmethod
    def align_objects(self, source_path: str, target_path: str, alignment_type: str) -> dict[str, Any]:
        """Snap the ``source_path`` object directly to the ``target_path``
        object's bounding box (``alignment_type`` one of ``top_center``/
        ``bottom_center``/``flush_left``/``flush_right``), translating the
        source object's placement in place — reopens and overwrites the
        SAME source document/path, like ``modify_parameter``. Returns a
        result dict including ``{"ok": bool, "path": str, "placement":
        [x, y, z]}``."""

    @abstractmethod
    def export_model(self, target_paths: list[str], format: str, filename: str) -> dict[str, Any]:
        """Export one or more previously-created objects together into a
        single named ``.stl`` (3D printing) or ``.step`` (external CAD
        interchange) file. Returns a result dict including
        ``{"ok": bool, "path": str}``."""

    @abstractmethod
    def create_assembly_mate(
        self,
        fixed_path: str,
        moving_path: str,
        mate_type: str,
        mate_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Position the MOVING object relative to the FIXED object as a
        named kinematic mate (``mate_type`` one of ``concentric``/
        ``coincident_planar``/``offset_axial``), translating the moving
        object's placement in place — reopens and overwrites the SAME
        moving-object document/path, like ``align_objects``. Returns a
        result dict including ``{"ok": bool, "path": str, "mate_type": str,
        "placement": [x, y, z]}``."""

    @abstractmethod
    def create_sketch_extrude(
        self,
        segments: list[dict[str, Any]],
        height: float,
        start: tuple[float, float] = (0.0, 0.0),
        plane: str = "XY",
        name: str = "Sketch",
        placement: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> dict[str, Any]:
        """Draw a closed 2D profile from an ordered list of line/arc
        segments (each ``{"type": "line", "to": [x, y]}`` or ``{"type":
        "arc", "to": [x, y], "via": [x, y]}``) on ``plane`` ("XY"/"XZ"/"YZ"),
        then extrude it ``height`` units along the plane's normal into a
        solid — a higher-leverage primitive than ``create_extrusion`` for
        profiles with rounded/arc edges a straight-edged polyline can't
        express. Same result shape as ``create_box``."""

    @abstractmethod
    def batch_pattern_array(
        self,
        source_path: str,
        pattern_type: str,
        *,
        count_x: int = 1,
        count_y: int = 1,
        spacing_x: float | None = None,
        spacing_y: float | None = None,
        count: int = 1,
        radius: float = 0.0,
        name: str = "Pattern",
    ) -> dict[str, Any]:
        """Copy a previously-created object into a linear, grid, or
        circular arrangement (``pattern_type``), combined into a single
        compound — ONE call instead of one create_* call per copy, so a
        repetitive layout (e.g. an 8x8 grid of 64 tiles) doesn't burn
        through the ReAct loop's per-turn iteration cap. Returns a result
        dict including ``{"ok": bool, "path": str, "name": str}``."""
