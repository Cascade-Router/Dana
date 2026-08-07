"""Pytest coverage for vision_math root-finding + bounding-box geometry."""
import math
from pathlib import Path

import pytest

from vision_math import (
    calculate_iou,
    calculate_roots,
    f,
    find_nearest_in_direction,
    generate_equation_image,
    get_centroid,
    inset_bbox,
    normalize_coordinates,
)


def test_calculate_roots_are_real_and_sorted() -> None:
    roots = calculate_roots()
    assert len(roots) == 3
    assert roots == sorted(roots)
    for r in roots:
        assert math.isfinite(r)
        assert abs(f(r)) < 1e-08


def test_known_root_signs() -> None:
    """x^3 - 4x + 1 has one negative and two positive real roots."""
    roots = calculate_roots()
    assert roots[0] < 0.0
    assert roots[1] > 0.0
    assert roots[2] > roots[1]


def test_generate_equation_image_tmp(tmp_path: Path) -> None:
    dest = tmp_path / 'eq.ppm'
    out = generate_equation_image(str(dest))
    path = Path(out)
    assert path.is_file()
    text = path.read_text(encoding='ascii', errors='replace')
    assert text.startswith('P3')
    assert 'equation=' in text


def test_custom_brackets_still_converge() -> None:
    roots = calculate_roots(brackets=[(-2.5, -1.5), (0.1, 0.9), (1.5, 2.5)])
    assert len(roots) == 3
    for r in roots:
        assert abs(f(r)) < 1e-07


# --------------------------------------------------------------------------
# get_centroid
# --------------------------------------------------------------------------


def test_get_centroid_basic() -> None:
    assert get_centroid([0, 0, 10, 20]) == (5.0, 10.0)


def test_get_centroid_negative_coords() -> None:
    assert get_centroid([-10, -10, 10, 10]) == (0.0, 0.0)


def test_get_centroid_accepts_floats_and_tuples() -> None:
    assert get_centroid((1.5, 2.5, 3.5, 4.5)) == (2.5, 3.5)


# --------------------------------------------------------------------------
# inset_bbox
# --------------------------------------------------------------------------


def test_inset_bbox_shrinks_symmetrically() -> None:
    assert inset_bbox([0, 0, 100, 100], 10) == (10.0, 10.0, 90.0, 90.0)


def test_inset_bbox_zero_padding_is_noop() -> None:
    assert inset_bbox([0, 0, 100, 50], 0) == (0.0, 0.0, 100.0, 50.0)


def test_inset_bbox_clamps_high_padding_without_inverting() -> None:
    xmin, ymin, xmax, ymax = inset_bbox([0, 0, 100, 100], 90)
    assert xmin < xmax
    assert ymin < ymax
    # Near-collapse to the centroid, but never past it.
    assert xmin == pytest.approx(49.999, abs=1e-2)
    assert xmax == pytest.approx(50.001, abs=1e-2)


def test_inset_bbox_clamps_negative_padding_to_zero() -> None:
    assert inset_bbox([0, 0, 100, 100], -25) == (0.0, 0.0, 100.0, 100.0)


def test_inset_bbox_centroid_is_unchanged() -> None:
    bbox = [10, 20, 210, 120]
    inset = inset_bbox(bbox, 15)
    assert get_centroid(inset) == pytest.approx(get_centroid(bbox))


# --------------------------------------------------------------------------
# calculate_iou
# --------------------------------------------------------------------------


def test_calculate_iou_identical_boxes_is_one() -> None:
    box = [0, 0, 10, 10]
    assert calculate_iou(box, box) == pytest.approx(1.0)


def test_calculate_iou_no_overlap_is_zero() -> None:
    assert calculate_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_calculate_iou_partial_overlap_known_value() -> None:
    # A = [0,0,10,10] area 100; B = [5,5,15,15] area 100.
    # Intersection = [5,5,10,10] area 25. Union = 175. IoU = 25/175.
    iou = calculate_iou([0, 0, 10, 10], [5, 5, 15, 15])
    assert iou == pytest.approx(25.0 / 175.0)


def test_calculate_iou_is_symmetric() -> None:
    a, b = [0, 0, 10, 10], [5, 5, 15, 15]
    assert calculate_iou(a, b) == pytest.approx(calculate_iou(b, a))


def test_calculate_iou_degenerate_box_is_zero() -> None:
    assert calculate_iou([5, 5, 5, 5], [0, 0, 10, 10]) == 0.0


# --------------------------------------------------------------------------
# find_nearest_in_direction
# --------------------------------------------------------------------------


def test_find_nearest_in_direction_right_picks_closest() -> None:
    reference = [0, 0, 10, 10]
    near = [20, 0, 30, 10]
    far = [100, 0, 110, 10]
    result = find_nearest_in_direction(reference, [far, near], "right")
    assert result == (20.0, 0.0, 30.0, 10.0)


def test_find_nearest_in_direction_down() -> None:
    reference = [0, 0, 10, 10]
    below = [0, 50, 10, 60]
    above = [0, -50, 10, -40]
    result = find_nearest_in_direction(reference, [above, below], "down")
    assert result == (0.0, 50.0, 10.0, 60.0)


def test_find_nearest_in_direction_up() -> None:
    reference = [0, 100, 10, 110]
    above = [0, 0, 10, 10]
    result = find_nearest_in_direction(reference, [above], "up")
    assert result == (0.0, 0.0, 10.0, 10.0)


def test_find_nearest_in_direction_left() -> None:
    reference = [100, 0, 110, 10]
    leftward = [0, 0, 10, 10]
    result = find_nearest_in_direction(reference, [leftward], "left")
    assert result == (0.0, 0.0, 10.0, 10.0)


def test_find_nearest_in_direction_no_qualifying_candidate_returns_none() -> None:
    reference = [0, 0, 10, 10]
    # Both candidates are to the left, so "right" should find nothing.
    candidates = [[-20, 0, -10, 10], [-40, 0, -30, 10]]
    assert find_nearest_in_direction(reference, candidates, "right") is None


def test_find_nearest_in_direction_empty_candidates_returns_none() -> None:
    assert find_nearest_in_direction([0, 0, 10, 10], [], "right") is None


def test_find_nearest_in_direction_invalid_direction_raises() -> None:
    with pytest.raises(ValueError):
        find_nearest_in_direction([0, 0, 10, 10], [[20, 0, 30, 10]], "diagonal")


# --------------------------------------------------------------------------
# normalize_coordinates
# --------------------------------------------------------------------------


def test_normalize_coordinates_scales_up() -> None:
    x, y = normalize_coordinates(50, 50, (100, 100), (200, 200))
    assert (x, y) == pytest.approx((100.0, 100.0))


def test_normalize_coordinates_scales_down() -> None:
    x, y = normalize_coordinates(100, 100, (200, 200), (100, 100))
    assert (x, y) == pytest.approx((50.0, 50.0))


def test_normalize_coordinates_identity_when_resolutions_match() -> None:
    x, y = normalize_coordinates(42, 84, (1920, 1080), (1920, 1080))
    assert (x, y) == pytest.approx((42.0, 84.0))


def test_normalize_coordinates_handles_independent_axis_scaling() -> None:
    # Source 100x50 -> target 400x50: only x scales, by 4x.
    x, y = normalize_coordinates(10, 10, (100, 50), (400, 50))
    assert (x, y) == pytest.approx((40.0, 10.0))


def test_normalize_coordinates_rejects_non_positive_source() -> None:
    with pytest.raises(ValueError):
        normalize_coordinates(1, 1, (0, 100), (100, 100))


def test_normalize_coordinates_rejects_non_positive_target() -> None:
    with pytest.raises(ValueError):
        normalize_coordinates(1, 1, (100, 100), (100, -1))
