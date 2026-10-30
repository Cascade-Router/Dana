"""The mock CAD engine answers query_topology, check_printability and
validate_assembly_collisions from its trimesh meshes (they used to return
"not supported"), builds real swept geometry for create_pipe/create_helix,
and moves meshes in position_assembly_part — so the hosted demo and Linux
CI can run the inspect -> assemble -> validate -> print-check path.

Each result is checked against the real engine's schema and against
geometry worked out by hand, not just ``ok``.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("manifold3d")

import dana.core.react_dispatch as rd
import dana.platform.mock as mock
from dana.platform.mock import MockControlPlane, MockFreeCADEngine
from dana.plugins.freecad import engine as fc_engine
from dana.plugins.freecad.ir import pipe_arc_radius
from dana.plugins.freecad.printability import build_report
from dana.tools.urdf_builder import ROOT_LINK_NAME
from dana.tools.schema import ToolCall

# Every key the real engine's query_topology reports per face.
_REAL_FACE_KEYS = {
    "face_index", "area", "is_planar", "surface_type", "normal", "normal_warning", "centroid",
    "u_span_physical_mm", "v_span_physical_mm", "u_direction_vector", "v_direction_vector",
}


@pytest.fixture
def cad() -> MockFreeCADEngine:
    return MockFreeCADEngine()


# --- query_topology -------------------------------------------------------------------


def test_box_topology_has_six_planar_faces_in_part_box_order(cad: MockFreeCADEngine) -> None:
    cad.create_box(60, 40, 5, name="Plate")
    topo = cad.query_topology("Plate")
    assert topo["ok"] is True and topo["part_name"] == "Plate" and "name" not in topo
    faces = topo["faces"]
    assert [f["face_index"] for f in faces] == [f"Face{i}" for i in range(1, 7)]
    assert [f["direction"] for f in faces] == ["-X", "+X", "-Y", "+Y", "-Z", "+Z"]
    assert [round(f["area"], 6) for f in faces] == [200, 200, 300, 300, 2400, 2400]
    assert all(_REAL_FACE_KEYS <= f.keys() and f["is_planar"] for f in faces)
    top = faces[5]
    assert top["surface_type"] == "Part::GeomPlane" and top["normal"] == pytest.approx([0, 0, 1])
    assert top["centroid"] == pytest.approx([0, 0, 2.5])
    assert sorted([top["u_span_physical_mm"], top["v_span_physical_mm"]]) == pytest.approx([40, 60])

    assert topo["edge_count"] == 12 and all(e["kind"] == "line" for e in topo["edges"])
    assert sorted(round(e["length"], 6) for e in topo["edges"]) == [5] * 4 + [40] * 4 + [60] * 4
    bbox = topo["bounding_box"]
    assert (bbox["xlen"], bbox["ylen"], bbox["zlen"]) == pytest.approx((60, 40, 5))
    assert topo["volume"] == pytest.approx(12_000) and topo["is_watertight"] is True
    assert topo["area"] == pytest.approx(2 * (2400 + 300 + 200))


def test_cylinder_topology_is_two_caps_and_one_curved_side(cad: MockFreeCADEngine) -> None:
    cad.create_cylinder(5, 20, name="Rod")
    faces = cad.query_topology("Rod")["faces"]
    assert [(f["direction"], f["is_planar"]) for f in faces] == [("-Z", True), ("+Z", True), (None, False)]
    side = faces[2]
    assert side["surface_type"] == "Part::GeomCylinder"
    assert side["normal"] is None and "not planar" in side["normal_warning"]
    assert side["u_span_physical_mm"] is None
    edges = cad.query_topology("Rod")["edges"]
    assert [e["kind"] for e in edges] == ["curve", "curve"]
    assert edges[0]["length"] == pytest.approx(2 * math.pi * 5, rel=0.01)


def test_a_cut_hole_shrinks_the_top_face_and_adds_a_bore(cad: MockFreeCADEngine) -> None:
    cad.create_box(60, 40, 5, name="Plate")
    cad.create_feature_on_face("Plate", "top", "circle", 0, 0, 5, "cut", radius=4, name="Drilled")
    faces = cad.query_topology("Drilled")["faces"]
    top = next(f for f in faces if f["direction"] == "+Z")
    assert top["area"] == pytest.approx(2400 - math.pi * 16, rel=0.01)
    bore = [f for f in faces if not f["is_planar"]]
    assert len(bore) == 1 and bore[0]["area"] == pytest.approx(2 * math.pi * 4 * 5, rel=0.01)


def test_topology_resolves_names_like_the_real_engine(cad: MockFreeCADEngine) -> None:
    cad.create_box(10, 10, 10, name="Bracket")
    assert cad.query_topology("bracket")["part_name"] == "Bracket"
    missing = cad.query_topology("Nope")
    assert missing["ok"] is False and "Nope" in missing["error"]


def test_query_topology_runs_through_dispatch() -> None:
    rd._set_has_plan(True, "test plan")
    cad, cp = MockFreeCADEngine(), MockControlPlane()
    built = rd.dispatch_tool_call(
        ToolCall(tool_id="create_freecad_box", arguments={"name": "Base", "length": 30, "width": 20, "height": 4}),
        cad, cp,
    )
    assert built.ok is True
    topo = rd.dispatch_tool_call(ToolCall(tool_id="query_topology", arguments={"part_name": "Base"}), cad, cp)
    assert topo.ok is True and topo.payload["face_count"] == 6


# --- swept geometry ----------------------------------------------------------------------


def test_arc_pipe_is_a_real_swept_elbow(cad: MockFreeCADEngine) -> None:
    result = cad.create_pipe(3, "arc", 90, name="Elbow")
    bend = pipe_arc_radius(3)
    assert result["dimensions"]["arc_radius"] == bend
    assert result["bounding_box"] == pytest.approx([0, 0, -3, bend + 3, bend + 3, 3], abs=1e-6)
    mesh = trimesh.load(result["path"], force="mesh")
    assert mesh.is_watertight
    # A quarter torus; 32-gon cross-section, so ~0.6% under the true circle.
    assert mesh.volume == pytest.approx(math.pi * 3**2 * (bend * math.pi / 2), rel=0.01)
    assert "placeholder" not in result["note"]


def test_straight_pipe_stands_on_the_origin(cad: MockFreeCADEngine) -> None:
    result = cad.create_pipe(2, "straight", 30, name="Tube", placement=(5, 0, 0))
    assert result["bounding_box"] == pytest.approx([3, -2, 0, 7, 2, 30], abs=1e-3)


def test_helix_is_a_coil_with_the_pipe_volume(cad: MockFreeCADEngine) -> None:
    result = cad.create_helix(10, 5, 20, 1, name="Spring")
    x_min, y_min, z_min, x_max, y_max, z_max = result["bounding_box"]
    assert (x_min, y_min, x_max, y_max) == pytest.approx((-11, -11, 11, 11), abs=0.05)
    assert z_min == pytest.approx(-1, abs=0.05) and z_max == pytest.approx(21, abs=0.05)
    mesh = trimesh.load(result["path"], force="mesh")
    assert mesh.is_watertight
    length = (20 / 5) * math.hypot(2 * math.pi * 10, 5)
    assert mesh.volume == pytest.approx(math.pi * length, rel=0.02)
    # It's a coil, not the old solid placeholder cylinder (~8,000 mm³).
    assert mesh.volume < 1_000
    assert "placeholder" not in result["note"]


def test_helix_angle_offset_rotates_the_coil_start(cad: MockFreeCADEngine) -> None:
    plain = trimesh.load(cad.create_helix(10, 5, 5, 1, name="H0")["path"], force="mesh")
    turned = trimesh.load(cad.create_helix(10, 5, 5, 1, name="H90", angle_offset=90)["path"], force="mesh")
    # The start cap's center vertex is the path's first point.
    def has_vertex_at(mesh: trimesh.Trimesh, point: list[float]) -> bool:
        return float(((mesh.vertices - point) ** 2).sum(axis=1).min()) < 1e-9

    assert has_vertex_at(plain, [10, 0, 0]) and not has_vertex_at(plain, [0, 10, 0])
    assert has_vertex_at(turned, [0, 10, 0])


def test_swept_and_extruded_objects_can_be_used_by_name(cad: MockFreeCADEngine) -> None:
    for build in (
        lambda: cad.create_pipe(2, "straight", 30, name="Tube"),
        lambda: cad.create_helix(10, 5, 10, 1, name="Coil"),
        lambda: cad.create_pyramid(10, 10, 10, name="Pyr"),
        lambda: cad.create_polygon(6, 5, 5, name="Hex"),
        lambda: cad.create_star_prism(5, 10, 4, 3, name="Star"),
        lambda: cad.create_extrusion([[0, 0], [5, 0], [0, 5]], 2, name="Wedge"),
    ):
        build()
    cad.create_box(100, 100, 2, name="Slab", placement=(0, 0, -1))
    assert cad.apply_boolean("cut", "Slab", "Tube", name="Holed")["ok"] is True
    for name in ("Coil", "Pyr", "Hex", "Star", "Wedge"):
        assert cad.query_topology(name)["ok"] is True, name


# --- check_printability --------------------------------------------------------------------


def test_printability_report_has_the_real_schema(cad: MockFreeCADEngine) -> None:
    cad.create_box(60, 40, 5, name="Plate")
    report = cad.check_printability("Plate")
    assert report["ok"] is True
    assert report["printable"] is True and report["requires_supports"] is False and report["warnings"] == []
    assert report["build_volume"]["name"] == "Prusa MK4" and report["build_volume"]["fit"] == "fits"
    assert report["dimensions_mm"] == pytest.approx([60, 40, 5])
    assert report["checks"]["open_edges"] == 0 and report["checks"]["components"] == 1
    assert report["checks"]["self_intersections"] is None  # not checked on the mock
    assert report["current_orientation_printable"] is True
    real_keys = set(
        build_report(target="x", checks={}, bbox=[0, 0, 0, 1, 1, 1], facets=[], solid_facets=None)
    ) | {"current_orientation_printable", "mesh_sound", "recommended_orientation", "remediation_hint", "orientations"}
    assert real_keys <= report.keys()
    stl = Path(report["stl_path"])
    assert stl.is_file() and stl.name == "Plate.stl" and stl.parent.parent.parent.name == "prints"


def test_printability_flags_a_part_too_big_for_the_configured_bed(
    cad: MockFreeCADEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    cad.create_box(200, 40, 5, name="Beam")
    assert cad.check_printability("Beam")["printable"] is True  # fits the MK4's 250 mm
    monkeypatch.setenv("DANA_PRINTER_BED_MM", "180,180,180")
    report = cad.check_printability("Beam")
    assert report["printable"] is False and report["build_volume"]["fit"] == "too_large"
    assert "180 x 180 x 180" in report["warnings"][0]
    monkeypatch.setenv("DANA_PRINTER_BED_MM", "nonsense")
    assert cad.check_printability("Beam")["ok"] is False


def test_printability_finds_the_overhang_of_a_mushroom(cad: MockFreeCADEngine) -> None:
    cad.create_cylinder(3, 20, name="Stem", placement=(0, 0, 10))
    cad.create_box(40, 40, 4, name="Cap", placement=(0, 0, 22))
    cad.apply_boolean("union", "Stem", "Cap", name="Mushroom")
    report = cad.check_printability()  # default target: the newest top-level solid
    assert report["target"] == "Mushroom"
    assert report["requires_supports"] is True
    assert report["overhangs"]["area_mm2"] == pytest.approx(40 * 40 - math.pi * 9, rel=0.01)
    # Upside down it prints support-free on the cap.
    assert report["recommended_orientation"]["up_axis"] == "-Z"


def test_printability_refuses_an_unknown_object(cad: MockFreeCADEngine) -> None:
    assert cad.check_printability("Ghost")["ok"] is False


# --- validate_assembly_collisions ---------------------------------------------------------


def _rig(cad: MockFreeCADEngine, offset_x: float) -> None:
    cad.create_box(20, 20, 20, name="A")
    cad.create_box(20, 20, 20, name="B", placement=(offset_x, 0, 0))
    cad.create_assembly("Rig")
    cad.add_parts_to_assembly("Rig", ["A", "B"])


def test_overlapping_parts_collide_with_the_exact_overlap_volume(cad: MockFreeCADEngine) -> None:
    _rig(cad, 10)
    report = cad.validate_assembly_collisions("Rig")
    assert report["ok"] is True and report["has_collisions"] is True
    assert report["collisions"] == [{"part_a": "A", "part_b": "B", "overlap_volume": pytest.approx(4000)}]
    assert report["clearances"] == [{"part_a": "A", "part_b": "B", "distance_mm": 0.0}]
    assert report["checked_members"] == 2 and report["name"] == "Rig"


def test_spaced_parts_report_their_true_clearance(cad: MockFreeCADEngine) -> None:
    _rig(cad, 35)
    report = cad.validate_assembly_collisions("Rig")
    assert report["ok"] is True and report["has_collisions"] is False and report["collisions"] == []
    assert report["clearances"][0]["distance_mm"] == pytest.approx(15)
    assert "collision_check_errors" not in report


def test_touching_faces_are_not_a_collision(cad: MockFreeCADEngine) -> None:
    _rig(cad, 20)
    report = cad.validate_assembly_collisions("Rig")
    assert report["has_collisions"] is False
    assert report["clearances"][0]["distance_mm"] == pytest.approx(0, abs=1e-6)


def test_curved_parts_whose_boxes_overlap_but_solids_dont(cad: MockFreeCADEngine) -> None:
    # Two r=10 cylinders 25 mm apart diagonally: boxes overlap, solids are 5.36 mm apart.
    cad.create_cylinder(10, 10, name="P")
    cad.create_cylinder(10, 10, name="Q", placement=(17.68, 17.68, 0))
    cad.create_assembly("Pair")
    cad.add_parts_to_assembly("Pair", ["P", "Q"])
    report = cad.validate_assembly_collisions("Pair")
    assert report["has_collisions"] is False
    assert report["clearances"][0]["distance_mm"] == pytest.approx(math.hypot(17.68, 17.68) - 20, abs=0.1)


def test_boolean_operands_are_not_checked_against_their_result(cad: MockFreeCADEngine) -> None:
    cad.create_box(20, 20, 20, name="Block")
    cad.create_cylinder(4, 30, name="Drill")
    cad.apply_boolean("cut", "Block", "Drill", name="Bored")
    cad.create_assembly("Kit")
    cad.add_parts_to_assembly("Kit", ["Block", "Drill", "Bored"])
    report = cad.validate_assembly_collisions("Kit")
    assert report["checked_members"] == 1 and report["has_collisions"] is False


def test_an_open_mesh_member_fails_the_check_instead_of_passing(cad: MockFreeCADEngine) -> None:
    cad.create_box(10, 10, 10, name="Solid")
    sheet = trimesh.Trimesh(vertices=[[0, 0, 0], [10, 0, 0], [0, 10, 0]], faces=[[0, 1, 2]])
    sheet_path = fc_engine._session_dir() / "sheet.glb"
    sheet.export(sheet_path)
    mock._mock_object_registry()["Sheet"] = str(sheet_path)
    cad.create_assembly("Bad")
    cad.add_parts_to_assembly("Bad", ["Solid", "Sheet"])
    report = cad.validate_assembly_collisions("Bad")
    assert report["ok"] is False and "could not check 1 part pair" in report["error"]
    assert report["collision_check_errors"][0]["part_b"] == "Sheet"
    assert report["has_collisions"] is False and report["collisions"] == []


def test_unknown_assembly_is_refused(cad: MockFreeCADEngine) -> None:
    assert cad.validate_assembly_collisions("Nothing")["ok"] is False


# --- position_assembly_part moves the mesh ------------------------------------------------


def test_positioning_a_part_moves_it_and_clears_the_collision(cad: MockFreeCADEngine) -> None:
    _rig(cad, 0)
    assert cad.validate_assembly_collisions("Rig")["has_collisions"] is True
    moved = cad.position_assembly_part("B", placement_x=30, yaw=90)
    assert moved["bounding_box"] == pytest.approx([20, -10, -10, 40, 10, 10], abs=1e-6)
    report = cad.validate_assembly_collisions("Rig")
    assert report["has_collisions"] is False and report["clearances"][0]["distance_mm"] == pytest.approx(10)
    # Absolute, not cumulative: the same call again changes nothing.
    again = cad.position_assembly_part("B", placement_x=30, yaw=90)
    assert again["bounding_box"] == pytest.approx(moved["bounding_box"], abs=1e-6)
    # The assembly preview follows the move.
    rig = trimesh.load(mock._mock_object_registry()["Rig"], force="mesh")
    assert rig.bounds[1][0] == pytest.approx(40)


def test_placements_survive_a_session_snapshot(cad: MockFreeCADEngine) -> None:
    cad.create_box(10, 10, 10, name="Lid", placement=(1, 2, 3))
    cad.position_assembly_part("Lid", placement_z=50)
    from dana.session_context import get_session_id

    # The ambient session, not "default": an earlier test may have set another.
    snapshot = mock.snapshot_mock_session(get_session_id())
    assert snapshot["placements"]["Lid"] == [0.0, 0.0, 50.0, 0.0, 0.0, 0.0]


# --- outputs stay in the session's workspace ------------------------------------------------


def test_mock_outputs_live_under_the_session_directory(cad: MockFreeCADEngine) -> None:
    mesh_path = Path(cad.create_box(1, 1, 1, name="Cube")["path"])
    assert mesh_path.parent == fc_engine._session_dir() / "mock"
    exported = Path(cad.export_model([str(mesh_path)], "stl", "cube")["path"])
    assert exported == fc_engine._export_dir() / "cube.stl" and exported.is_file()


def test_printability_rejects_an_inside_out_mesh(cad: MockFreeCADEngine) -> None:
    flipped = trimesh.creation.box(extents=[10, 10, 10])
    flipped.invert()
    path = fc_engine._session_dir() / "flipped.glb"
    flipped.export(path)
    mock._mock_object_registry()["Flipped"] = str(path)
    report = cad.check_printability("Flipped")
    assert report["checks"]["positive_volume"] is False
    assert report["printable"] is False and any("no positive volume" in w for w in report["warnings"])


def test_printability_reports_the_new_mesh_checks_on_a_clean_part(cad: MockFreeCADEngine) -> None:
    cad.create_box(10, 10, 10, name="Cube")
    checks = cad.check_printability("Cube")["checks"]
    assert checks["winding_consistent"] is True and checks["positive_volume"] is True
    assert checks["degenerate_faces"] == 0


def test_build_report_blocks_inconsistent_winding_and_warns_on_degenerate_faces() -> None:
    clean = {"shape_valid": True, "closed_solid": True, "mesh_solid": True, "non_manifold": False,
             "self_intersections": False, "open_edges": 0, "components": 1}
    bbox = [0, 0, 0, 10, 10, 10]
    assert build_report(target="x", checks=clean, bbox=bbox, facets=[])["printable"] is True  # keys absent: unchecked
    bad = build_report(target="x", checks={**clean, "winding_consistent": False}, bbox=bbox, facets=[])
    assert bad["printable"] is False and "winding" in bad["warnings"][0]
    sliver = build_report(target="x", checks={**clean, "degenerate_faces": 3}, bbox=bbox, facets=[])
    assert sliver["printable"] is True and "3 degenerate" in sliver["warnings"][0]


# --- export_assembly_to_urdf ---------------------------------------------------------------


def test_mock_urdf_export_writes_links_meshes_and_inertia(cad: MockFreeCADEngine) -> None:
    cad.create_box(40, 40, 10, name="Base")
    cad.create_cylinder(5, 30, name="Post")
    cad.create_assembly("Stand")
    cad.add_parts_to_assembly("Stand", ["Base", "Post"])
    cad.position_assembly_part("Post", placement_z=21)
    cad.define_kinematic_joint("Stand", "Post", "Base", "revolute", axis=[0, 0, 1])
    result = cad.export_assembly_to_urdf("Stand")
    assert result["ok"] is True, result
    urdf = Path(result["path"])
    assert urdf.is_file() and urdf.parent == fc_engine._export_dir() / "Stand_urdf"
    root = ET.parse(urdf).getroot()
    links = {link.get("name"): link for link in root.findall("link")}
    # Base is the only root, so no synthetic base_link; Post hangs off Base.
    assert set(links) == {"Base", "Post"} and ROOT_LINK_NAME not in links
    joint = root.find("joint")
    assert (joint.get("type"), joint.find("parent").get("link"), joint.find("child").get("link")) == (
        "revolute", "Base", "Post")
    assert [float(v) for v in joint.find("origin").get("xyz").split()] == pytest.approx([0, 0, 0.021])

    for name in ("Base", "Post"):
        mesh_file = links[name].find("visual/geometry/mesh").get("filename")
        local = trimesh.load(urdf.parent / mesh_file, force="mesh")
        assert local.centroid == pytest.approx([0, 0, 0], abs=1e-6)  # written in the part's own frame
    # Base: 40 x 40 x 10 mm of the default density (2,700 kg/m³ aluminium).
    mass = float(links["Base"].find("inertial/mass").get("value"))
    assert mass == pytest.approx(2700 * 16_000e-9, rel=1e-3)
    izz = float(links["Base"].find("inertial/inertia").get("izz"))
    assert izz == pytest.approx(mass * (0.04**2 + 0.04**2) / 12, rel=1e-3)


def test_mock_urdf_export_refuses_an_unknown_assembly(cad: MockFreeCADEngine) -> None:
    assert cad.export_assembly_to_urdf("Nope")["ok"] is False
