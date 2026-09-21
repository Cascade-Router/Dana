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
from collections.abc import Sequence
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
        self,
        operation: str,
        base_object: str | None = None,
        tool_object: str | None = None,
        name: str | None = None,
        objects: list[str] | None = None,
    ) -> dict[str, Any]:
        """Combine previously-created solids with a Boolean operation:
        ``"cut"`` subtracts the tool from the base (always exactly
        ``base_object``/``tool_object`` — a cut has no N-ary equivalent),
        ``"union"`` fuses every given shape into one, ``"intersect"`` keeps
        only their overlapping volume. ``base_object``/``tool_object`` are
        the exact object NAMES returned by ``create_box``/
        ``create_cylinder``/``insert_standard_part`` (or a prior
        ``apply_boolean``) — resolved by name against this driver's own
        shared session state, not a file path (every session-scoped creation
        call shares one underlying document/session, so a path alone can no
        longer tell two objects apart). ``objects`` (2+ names), when given
        INSTEAD of ``base_object``/``tool_object``, fuses/intersects every
        one of them in a single call — valid only for ``"union"``/
        ``"intersect"``. Returns a result dict including ``{"ok": bool,
        "path": str, "name": str}``."""

    @abstractmethod
    def apply_edge_operation(
        self,
        operation: str,
        target_object: str,
        value: float,
        face_centroid: tuple[float, float, float] | None = None,
        name: str | None = None,
    ) -> dict[str, Any]:
        """Round (``"fillet"``) or bevel (``"chamfer"``) the edges of a
        previously-created solid by ``value`` mm, resolved by NAME —
        exactly ``apply_boolean``'s by-name/no-path story — against this
        driver's own shared session state. Without ``face_centroid``, every
        edge of the object is targeted (a global fillet/chamfer); with it,
        only the edges bounding the object's face nearest that point are
        targeted. Returns a result dict including
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
    def create_polygon(
        self,
        sides: int,
        radius: float,
        height: float,
        name: str = "Polygon",
        placement: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> dict[str, Any]:
        """Create an extruded regular N-gon (``sides`` >= 3, evenly spaced,
        inscribed in ``radius``), translated by ``placement`` (global X/Y/Z
        offset in mm). Same result shape as ``create_box``. The dedicated
        primitive for a hexagon/pentagon/octagon/... — no ``create_star_prism``
        degenerate-radius trick needed."""

    @abstractmethod
    def export_mesh_stl(
        self, source_path: str, name: str | None = None, target_object: str | None = None
    ) -> dict[str, Any]:
        """Tessellate/export the solid at ``source_path`` to a standalone
        ``.stl`` mesh file. ``target_object``, when given, exports ONLY
        that resolved object rather than every object in the document —
        required once ``source_path`` can be a shared multi-object session
        document. Returns ``{"ok": bool, "path": str}``."""

    @abstractmethod
    def modify_parameter(
        self,
        target_object: str,
        parameter_name: str,
        new_value: float | Sequence[float],
        yaw: float | None = None,
        pitch: float | None = None,
        roll: float | None = None,
    ) -> dict[str, Any]:
        """Change a single dimensional property (e.g. ``"Height"``,
        ``"Radius"``) on a previously-created object, by NAME, in place —
        reopens and overwrites the SAME shared session document/path rather
        than creating a new one. ``parameter_name`` of ``"Placement"``/
        ``"Placement.Base"`` is special: ``new_value`` must then be a
        3-number ``[x, y, z]`` vector (mm), applied to the object's
        ``Placement.Base``. Rotation is NEVER packed into ``new_value`` —
        it's set via the separate ``yaw``/``pitch``/``roll`` DEGREES
        parameters instead: all three omitted preserves the object's
        current ``Placement.Rotation``; any one given replaces the whole
        rotation with a fresh ``FreeCAD.Rotation(yaw, pitch, roll)``
        (omitted axes default to 0.0). Returns a result dict including
        ``{"ok": bool, "path": str, "name": str}``."""

    @abstractmethod
    def get_bounding_box(self, target_path: str, target_object: str | None = None) -> dict[str, Any]:
        """Read-only: the physical bounding box of a previously-created
        object, in mm. Never mutates anything. ``target_object``, when
        given, is resolved by NAME (exact Name, then Label, then
        case-insensitive Name) rather than by blindly picking the one
        object nothing else references — required once ``target_path`` can
        point at a shared multi-object document. Returns
        ``{"ok": bool, "x_min": float, "y_min": float, "z_min": float,
        "x_max": float, "y_max": float, "z_max": float}``."""

    @abstractmethod
    def inspect_spatial_properties(self, target_path: str, target_object: str | None = None) -> dict[str, Any]:
        """Read-only: richer topology introspection than ``get_bounding_box``
        — volume, surface area, center of mass, solid validity, and face/
        edge/vertex counts for a previously-created object. Never mutates
        anything. Lets a caller check topology complexity/validity before a
        risky fillet/chamfer/boolean rather than discovering infeasibility
        only after it fails. ``target_object`` resolves the same way as
        ``get_bounding_box``'s. Returns a result dict including
        ``{"ok": bool, "volume": float, "area": float, "center_of_mass":
        [x, y, z], "is_valid": bool, "face_count": int, "edge_count": int,
        "vertex_count": int}``."""

    @abstractmethod
    def query_topology(self, part_name: str) -> dict[str, Any]:
        """Read-only: per-face topology of a previously-created object,
        resolved by NAME against this driver's own shared session state —
        for every face on ``part_name``'s ``Shape``: its 1-based face index
        (e.g. ``"Face3"``, matching ``apply_assembly_constraint``'s own
        ``part1_element``/``part2_element`` numbering), area, whether it's
        planar, surface type, centroid, and normal (``None``, with an
        explanatory ``normal_warning``, for any curved face — a face's
        normal is only a single well-defined vector when it's planar).
        Never mutates anything. Intended as a "look before you leap" query
        ahead of ``apply_assembly_constraint`` — see the real engine's own
        docstring for the exact contract. Returns a result dict including
        ``{"ok": bool, "part_name": str, "path": str, "face_count": int,
        "faces": [...]}``."""

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
    def create_helix(
        self,
        coil_radius: float,
        pitch: float,
        height: float,
        pipe_radius: float,
        name: str = "Helix",
        placement: tuple[float, float, float] = (0.0, 0.0, 0.0),
        angle_offset: float = 0.0,
    ) -> dict[str, Any]:
        """Sweep a circular profile (``pipe_radius`` mm cross-section) along
        a cylindrical helical spine (``coil_radius`` mm coil radius,
        ``pitch`` mm rise per turn, ``height`` mm total vertical extent)
        into a solid coil tube, translated by ``placement`` (global X/Y/Z
        offset in mm) and rotated ``angle_offset`` degrees about the global
        Z axis — lets several coils share one central hub without
        interpenetrating (e.g. three coils at 0/120/240 degrees). Number of
        turns is implied (``height / pitch``), not a separate parameter.
        Same result shape as ``create_pipe``."""

    @abstractmethod
    def align_objects(
        self,
        source_path: str,
        target_path: str,
        alignment_type: str,
        source_object: str | None = None,
        target_object: str | None = None,
    ) -> dict[str, Any]:
        """Snap the ``source_path`` object directly to the ``target_path``
        object's bounding box (``alignment_type`` one of ``top_center``/
        ``bottom_center``/``flush_left``/``flush_right``), translating the
        source object's placement in place — reopens and overwrites the
        SAME source document/path, like ``modify_parameter``.
        ``source_object``/``target_object``, when given, are each resolved
        by NAME (exact Name, then Label, then case-insensitive Name) —
        required once ``source_path``/``target_path`` can be the SAME
        shared session document, where two distinct objects can no longer
        be told apart by path alone. Returns a result dict including
        ``{"ok": bool, "path": str, "placement": [x, y, z]}``."""

    @abstractmethod
    def export_model(
        self,
        target_paths: list[str],
        format: str,
        filename: str,
        target_objects: list[str] | None = None,
    ) -> dict[str, Any]:
        """Export one or more previously-created objects together into a
        single named ``.stl`` (3D printing) or ``.step`` (external CAD
        interchange) file. ``target_objects``, when given, must line up
        index-for-index with ``target_paths`` — the object at each index is
        resolved by NAME within that index's document, the same resolution
        ``get_bounding_box`` uses, rather than blindly picking whichever
        object nothing else references. Returns a result dict including
        ``{"ok": bool, "path": str}``."""

    @abstractmethod
    def create_assembly_mate(
        self,
        fixed_path: str,
        moving_path: str,
        mate_type: str,
        mate_params: dict[str, Any] | None = None,
        fixed_object: str | None = None,
        moving_object: str | None = None,
    ) -> dict[str, Any]:
        """Position the MOVING object relative to the FIXED object as a
        named kinematic mate (``mate_type`` one of ``concentric``/
        ``coincident_planar``/``offset_axial``), translating the moving
        object's placement in place — reopens and overwrites the SAME
        moving-object document/path, like ``align_objects``.
        ``fixed_object``/``moving_object`` resolve the same way as
        ``align_objects``'s ``source_object``/``target_object``. Returns a
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
    def create_sketch(
        self,
        name: str,
        plane: str,
        geometry: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Create a real, constrainable ``Sketcher::SketchObject`` mapped onto
        one of the three standard base planes (``"XY"``/``"XZ"``/``"YZ"``),
        from an ordered list of raw 2D geometry elements — each ``{"type":
        "line", "start": [x, y], "end": [x, y]}`` / ``{"type": "circle",
        "center": [x, y], "radius": r}`` / ``{"type": "arc", "center":
        [x, y], "radius": r, "start_angle": deg, "end_angle": deg}``. List
        position ``i`` becomes that sketch's ``Geometry[i]`` — the exact
        0-based index ``apply_sketch_constraint``'s ``geometry_indices``
        reference. Unlike ``create_sketch_extrude``, this never extrudes —
        the sketch stays a live parametric object a later
        ``apply_sketch_constraint`` call can lock down. Same result shape as
        ``create_box``."""

    @abstractmethod
    def apply_sketch_constraint(
        self,
        sketch_name: str,
        constraint_type: str,
        geometry_indices: list[int],
        value: float | None = None,
    ) -> dict[str, Any]:
        """Lock down a previously-created sketch's raw geometry with a real
        ``Sketcher::Constraint``, resolved by NAME — same by-name story as
        ``apply_boolean``/``modify_parameter``. ``geometry_indices`` are
        0-based positions into the target sketch's own ``Geometry`` list:
        ``"Coincident"`` needs exactly 4 ints ``[geoId1, posId1, geoId2,
        posId2]`` (``posId`` 1=start, 2=end, 3=center); ``"Horizontal"``/
        ``"Vertical"`` need exactly 1 int ``[geoId]``; ``"Distance"``/
        ``"Radius"`` need exactly 1 int ``[geoId]`` plus a required
        ``value`` (edge length / circle-arc radius, in mm). Returns a
        result dict including ``{"ok": bool, "path": str, "name": str}``."""

    @abstractmethod
    def create_pad(
        self,
        sketch_name: str,
        length: float,
        symmetric_to_plane: bool = False,
        reversed_direction: bool = False,
    ) -> dict[str, Any]:
        """Extrude a previously-created ``create_sketch`` profile into a real
        ``PartDesign::Pad`` solid, resolved by NAME — same by-name story as
        ``apply_boolean``/``modify_parameter``. Ensures PartDesign
        containment itself: if the sketch isn't already inside a
        ``PartDesign::Body``, one is created (or an existing one reused) and
        the sketch is moved into it before the Pad is applied.
        ``symmetric_to_plane`` extrudes evenly on both sides of the sketch
        plane (``Midplane``) instead of only forward from it;
        ``reversed_direction`` flips which side a non-symmetric extrusion
        grows into (``Reversed``). Same result shape as ``create_box``."""

    @abstractmethod
    def create_pocket(
        self,
        sketch_name: str,
        depth: float,
        through_all: bool = False,
        symmetric_to_plane: bool = False,
        reversed_direction: bool = False,
    ) -> dict[str, Any]:
        """Subtract material from the active ``PartDesign::Body``'s existing
        solid using a previously-created ``create_sketch`` profile, via a
        real ``PartDesign::Pocket`` — same by-name/containment story as
        ``create_pad``. Requires the body to already have some solid to cut
        into. ``through_all`` cuts all the way through regardless of
        ``depth``; otherwise the cut goes exactly ``depth`` mm deep.
        ``symmetric_to_plane``/``reversed_direction`` mirror ``create_pad``'s
        own ``Midplane``/``Reversed`` semantics. Same result shape as
        ``create_box``."""

    @abstractmethod
    def create_polar_pattern(
        self,
        feature_name: str,
        occurrences: int,
        angle: float = 360.0,
        axis: str = "Z",
        reversed_direction: bool = False,
    ) -> dict[str, Any]:
        """Repeat a previously-created ``create_pad``/``create_pocket`` 3D
        feature evenly around one of its own ``PartDesign::Body``'s
        principal axes (``"X"``/``"Y"``/``"Z"``), via a real
        ``PartDesign::PolarPattern``, resolved by NAME — same by-name story
        as ``apply_boolean``/``modify_parameter``. ``feature_name`` must
        already be a 3D feature inside a body (never a bare ``create_sketch``
        profile). ``occurrences`` (>= 2) is the TOTAL copy count including
        the original; ``angle`` is the total angular span those copies are
        spread across, in degrees; ``reversed_direction`` spreads them in
        the opposite rotational direction. Same result shape as
        ``create_box``."""

    @abstractmethod
    def create_linear_pattern(
        self,
        feature_name: str,
        occurrences: int,
        length: float,
        direction: str = "X",
        reversed_direction: bool = False,
    ) -> dict[str, Any]:
        """Repeat a previously-created ``create_pad``/``create_pocket`` 3D
        feature evenly along one of its own ``PartDesign::Body``'s principal
        axes (``"X"``/``"Y"``/``"Z"``), via a real
        ``PartDesign::LinearPattern`` — same by-name/containment story as
        ``create_polar_pattern``. ``occurrences`` (>= 2) is the TOTAL copy
        count including the original; ``length`` is the total span those
        copies are spread across, in mm, from the first copy to the last;
        ``reversed_direction`` spreads them in the opposite direction along
        the axis. Same result shape as ``create_box``."""

    @abstractmethod
    def create_sweep(self, profile_sketch: str, path_sketch: str, frenet: bool = True) -> dict[str, Any]:
        """Sweep a previously-created ``create_sketch`` profile along
        another ``create_sketch`` path into a real solid, via a real
        ``PartDesign::AdditivePipe``, resolved by NAME — same by-name story
        as ``apply_boolean``/``modify_parameter``. Both sketches must
        already exist; neither needs to already be inside a
        ``PartDesign::Body`` — one is found/created and BOTH are moved into
        it before the sweep is applied. ``frenet``, when true, makes the
        profile's orientation follow the path's own Frenet frame instead of
        keeping a fixed orientation throughout. Same result shape as
        ``create_box``."""

    @abstractmethod
    def create_loft(
        self,
        cross_section_sketches: list[str],
        ruled: bool = False,
        closed: bool = False,
    ) -> dict[str, Any]:
        """Blend a smooth solid through an ORDERED list of 2+ previously-
        created ``create_sketch`` cross-sections, via a real
        ``PartDesign::AdditiveLoft`` — same by-name/containment story as
        ``create_sweep``. The first entry becomes the feature's own
        ``Profile``; every other entry becomes an additional ``Sections``
        cross-section, blended through in list order. ``ruled``, when true,
        connects consecutive cross-sections with straight ruled surfaces
        instead of a smooth blend; ``closed``, when true, loops the loft
        back from the last cross-section to the first. Same result shape as
        ``create_box``."""

    @abstractmethod
    def create_assembly(self, name: str) -> dict[str, Any]:
        """Create a real ``App::Part`` assembly container — a plain
        organizational grouping object with no geometry of its own, used to
        gather independent ``PartDesign::Body`` instances into one
        positioned sub-assembly via ``add_parts_to_assembly``/
        ``position_assembly_part``. Returns a result dict including at
        least ``{"ok": bool, "path": str, "name": str}`` — no
        ``bounding_box``, since an ``App::Part`` has no ``Shape`` of its
        own."""

    @abstractmethod
    def add_parts_to_assembly(self, assembly_name: str, part_names: list[str]) -> dict[str, Any]:
        """Move the named parts (typically ``PartDesign::Body`` instances)
        into a previously-created ``create_assembly`` container, resolved
        by NAME — same by-name story as ``apply_boolean``/
        ``modify_parameter``. A part already in the assembly is silently
        left alone rather than re-added. Same result shape as
        ``create_assembly``."""

    @abstractmethod
    def position_assembly_part(
        self,
        part_name: str,
        placement_x: float = 0.0,
        placement_y: float = 0.0,
        placement_z: float = 0.0,
        yaw: float = 0.0,
        pitch: float = 0.0,
        roll: float = 0.0,
    ) -> dict[str, Any]:
        """Move and/or orient a previously-created part (typically a
        ``PartDesign::Body`` inside a ``create_assembly`` container, but
        works on any named object) by REPLACING its whole ``Placement``,
        resolved by NAME — same by-name story as ``apply_boolean``/
        ``modify_parameter``. ``(placement_x, placement_y, placement_z)``
        is the new position in mm; ``(yaw, pitch, roll)`` is a fresh Euler
        rotation in DEGREES, replacing any prior rotation rather than
        composing with it. Same result shape as ``create_assembly``."""

    @abstractmethod
    def apply_assembly_constraint(
        self,
        assembly_name: str,
        part1_name: str,
        part1_element: str,
        part2_name: str,
        part2_element: str,
        constraint_type: str,
        offset: float = 0.0,
        uv_tensor: Sequence[float] | None = None,
        world_fractions: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Move ``part2_name`` so ``part2_element`` (a ``Face``/``Edge``
        reference, e.g. ``"Face1"``) satisfies ``constraint_type``
        (``"Coincident"``/``"Concentric"``/``"Parallel"``/``"Perpendicular"``/
        ``"Distance"``) against ``part1_name``'s ``part1_element`` — a
        ONE-SHOT geometric Placement computed from each element's real
        BRep geometry, not a live re-solvable constraint (see
        ``dana.plugins.freecad.engine.apply_assembly_constraint``'s own
        docstring for exactly why the native Assembly workbench's real
        joints aren't reachable from this headless execution model). Both
        parts must already be members of ``assembly_name``.

        ``uv_tensor`` (Continuous Parametric UV Placement, Coincident/
        Distance + a Face ``part1_element`` only): an optional ``[u, v]``
        pair, each in ``[0.0, 1.0]`` — shifts the target from that face's
        center (the default, ``[0.5, 0.5]``) to any continuous point on it,
        deterministically resolved against that face's own real (u, v)
        range — see the real engine's own docstring for the exact
        contract.

        ``world_fractions`` (same Coincident/Distance + Face restriction,
        mutually exclusive with ``uv_tensor``): a ``{"X"/"Y"/"Z": fraction}``
        dict resolving which of the face's own ``u``/``v`` axes actually
        points along each given world-space direction, instead of the
        caller guessing — see the real engine's own docstring for the
        exact contract."""

    @abstractmethod
    def anchor_assembly_root(self, assembly_name: str, part_name: str) -> dict[str, Any]:
        """Deterministically pins ``part_name`` (already a member of
        ``assembly_name``) to the assembly's origin: resets its
        ``Placement`` to IDENTITY (position and rotation both zero) and
        marks it ``DanaAnchored``, which every placement-mutating tool
        here (``position_assembly_part``, ``modify_freecad_parameter``'s
        ``Placement``/``Placement.Base`` branch, ``apply_assembly_constraint``
        when this part is passed as ``part2``) then refuses to move or
        rotate — see the real engine's own docstring for exactly why this
        is the deterministic substitute for a real FreeCAD "Fixed" joint
        constraint."""

    @abstractmethod
    def define_kinematic_joint(
        self,
        assembly_name: str,
        child_link: str,
        parent_link: str = "base_link",
        joint_type: str = "fixed",
        axis: Sequence[float] = (0.0, 0.0, 1.0),
        joint_name: str | None = None,
        limit_lower: float | None = None,
        limit_upper: float | None = None,
        limit_effort: float | None = None,
        limit_velocity: float | None = None,
    ) -> dict[str, Any]:
        """Declares a real parent/child kinematic joint between two
        members of ``assembly_name`` — persisted state a later
        ``export_assembly_to_urdf`` call reads back to build an actual
        kinematic TREE instead of a flat "every part fixed to base_link"
        star (``parent_link`` may also be the literal ``"base_link"``, the
        synthetic root, meaning "attach to the world" — see
        ``dana.plugins.freecad.engine.define_kinematic_joint``'s own
        docstring for exactly how/where this is stored and why redefining
        the same ``child_link`` replaces its prior joint). ``joint_type``:
        ``"fixed"``, ``"revolute"``/``"prismatic"`` (need ``axis``, and
        accept ``limit_lower``/``limit_upper``/``limit_effort``/
        ``limit_velocity``), or ``"continuous"`` (unlimited rotation about
        ``axis``, no limit)."""

    @abstractmethod
    def validate_assembly_collisions(self, assembly_name: str) -> dict[str, Any]:
        """Volumetric Validation Gate: TRUE solid-intersection audit (via
        real boolean intersection, not a bounding-box overlap) across every
        pair of ``assembly_name``'s members — see the real engine's own
        docstring for exactly why this is a distinct, whole-assembly,
        called-once-before-export check rather than real-time coordinate
        feedback. Returns ``collisions`` (a list of ``{"part_a", "part_b",
        "overlap_volume"}`` dicts) and ``has_collisions``/
        ``checked_members`` alongside it."""

    @abstractmethod
    def export_assembly_to_urdf(
        self,
        assembly_name: str,
        export_directory: str | None = None,
        density_kg_m3: float | None = None,
    ) -> dict[str, Any]:
        """Export ``assembly_name`` (a real ``create_assembly`` container)
        into a ``.urdf`` robot description — each member becomes a link
        (its own ``.stl`` under ``meshes/``, reused as both ``<visual>``
        and ``<collision>``) jointed onto whatever parent a prior
        ``define_kinematic_joint`` call declared for it (or the synthetic
        ``base_link`` root by default, for a flat "star" topology, same as
        before ``define_kinematic_joint`` existed), at that member's real
        Placement RELATIVE TO THAT PARENT, and carries a real
        ``<inertial>`` block (mass/center-of-mass/inertia tensor from that
        member's own Shape volume times ``density_kg_m3``) — see
        ``dana.plugins.freecad.engine.export_assembly_to_urdf``'s own
        docstring. ``export_directory`` defaults to a
        ``<assembly_name>_urdf`` folder under this session's own output
        directory. ``density_kg_m3`` defaults to aluminum (see that same
        docstring)."""

    @abstractmethod
    def create_feature_on_face(
        self,
        object_name: str,
        face: str,
        shape: str,
        u: float,
        v: float,
        extent: float,
        operation: str,
        radius: float | None = None,
        width: float | None = None,
        length: float | None = None,
        name: str | None = None,
    ) -> dict[str, Any]:
        """Add or cut a circle/rectangle feature directly on a named face
        (``"top"``/``"bottom"``/``"front"``/``"back"``/``"left"``/
        ``"right"``, FreeCAD's own standard-view convention) of an existing
        object, at local 2D (``u``, ``v``) coordinates on that face —
        offloads the 3D placement/rotation math a caller would otherwise
        need for a feature on a non-Z face onto the driver instead.
        ``operation="cut"`` subtracts the shape (a hole/pocket/slot);
        ``operation="add"`` unions it onto the surface (a boss/tab). Only
        supports a genuinely flat target face (a box-like/prismatic
        object) — fails with a clear error on a curved one. Returns the
        same result shape as ``apply_boolean``, since ``object_name`` is
        CONSUMED by the underlying boolean step exactly like any other
        boolean call — only the returned name is valid afterward."""

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
