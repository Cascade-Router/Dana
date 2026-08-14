"""Geometry-correctness tests for MockFreeCADEngine's headless trimesh
primitives — watertightness and volume, not just "did it return ok=True".

These exist because the mock engine's fan-triangulated extrusion helper
had a real off-by-one indexing bug (top-cap/side-wall vertices pointed one
index short) that produced non-watertight meshes while still reporting
ok=True with plausible-looking bounding boxes — caught only by actually
loading the exported mesh and checking manifoldness/volume, not by
asserting on the JSON result shape alone.
"""

from __future__ import annotations

import math

import pytest

trimesh = pytest.importorskip("trimesh")

from dana.platform.mock import MockFreeCADEngine, _star_polygon_vertices


@pytest.fixture
def engine() -> MockFreeCADEngine:
    return MockFreeCADEngine()


def test_box_is_watertight_with_correct_volume(engine: MockFreeCADEngine) -> None:
    result = engine.create_box(40, 25, 15)
    mesh = trimesh.load(result["path"])
    assert mesh.is_watertight
    assert math.isclose(mesh.volume, 40 * 25 * 15, rel_tol=1e-6)


def test_pyramid_is_watertight_with_correct_volume(engine: MockFreeCADEngine) -> None:
    result = engine.create_pyramid(50, 50, 75)
    mesh = trimesh.load(result["path"])
    assert mesh.is_watertight
    assert math.isclose(mesh.volume, 50 * 50 * 75 / 3, rel_tol=1e-6)


def test_square_extrusion_is_watertight_with_correct_volume(engine: MockFreeCADEngine) -> None:
    result = engine.create_extrusion([[-10, -10], [10, -10], [10, 10], [-10, 10]], 25)
    mesh = trimesh.load(result["path"])
    assert mesh.is_watertight
    assert math.isclose(mesh.volume, 20 * 20 * 25, rel_tol=1e-6)


def test_star_prism_is_watertight(engine: MockFreeCADEngine) -> None:
    result = engine.create_star_prism(8, 60, 20, 5)
    mesh = trimesh.load(result["path"])
    assert mesh.is_watertight
    assert mesh.volume > 0


def test_star_prism_volume_matches_analytic_polygon_area_times_height() -> None:
    """The fan-from-centroid triangulation is exact for a star-shaped-from-
    center polygon, so the prism volume should equal (shoelace area of the
    2D star polygon) * height, not just "some positive number"."""
    points, outer, inner, height = 8, 60.0, 20.0, 5.0
    vertices = _star_polygon_vertices(points, outer, inner)

    n = len(vertices)
    shoelace = 0.0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        shoelace += x1 * y2 - x2 * y1
    polygon_area = abs(shoelace) / 2.0

    result = MockFreeCADEngine().create_star_prism(points, outer, inner, height)
    mesh = trimesh.load(result["path"])
    assert math.isclose(mesh.volume, polygon_area * height, rel_tol=1e-6)


def test_star_polygon_vertices_alternate_outer_and_inner_radius() -> None:
    vertices = _star_polygon_vertices(5, 50.0, 20.0)
    assert len(vertices) == 10
    for i, (x, y) in enumerate(vertices):
        expected_radius = 50.0 if i % 2 == 0 else 20.0
        assert math.isclose(math.hypot(x, y), expected_radius, rel_tol=1e-9)


def test_star_prism_rejects_too_few_points(engine: MockFreeCADEngine) -> None:
    result = engine.create_star_prism(2, 50, 20, 5)
    assert result["ok"] is False
