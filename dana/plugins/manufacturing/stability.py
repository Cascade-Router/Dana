"""Static tip-over check: is an assembly's center of mass, dropped straight
down onto the ground plane, inside the polygon it stands on?

Pure geometry on XY points (millimetres). The support polygon is the convex
hull of the ground-contact points (what the assembly touches the ground
with): a body resting on that set can't tip while its center of mass is over
the hull, even if the contact itself is several separate feet.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

# Below this margin the assembly stands, but only just: a small push or an
# uneven surface can tip it.
MIN_STABILITY_MARGIN_MM = 2.0

Point = tuple[float, float]


def convex_hull(points: Iterable[Sequence[float]]) -> list[Point]:
    """Counter-clockwise hull (Andrew's monotone chain), collinear points dropped."""
    pts = sorted({(float(p[0]), float(p[1])) for p in points})
    if len(pts) <= 2:
        return pts

    def cross(o: Point, a: Point, b: Point) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[Point] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[Point] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _segment_distance(p: Point, a: Point, b: Point) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length_sq = dx * dx + dy * dy
    t = 0.0 if length_sq == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length_sq))
    return math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy))


def signed_margin(point: Point, hull: Sequence[Point]) -> float:
    """Distance from ``point`` to the hull's boundary: positive inside,
    negative outside. A hull with no area (fewer than three corners: the
    assembly balances on a point or an edge) has no inside, so the result is
    never positive."""
    if not hull:
        raise ValueError("empty support polygon")
    if len(hull) == 1:
        return -math.hypot(point[0] - hull[0][0], point[1] - hull[0][1])
    edges = list(zip(hull, hull[1:] + hull[:1])) if len(hull) > 2 else [(hull[0], hull[1])]
    distance = min(_segment_distance(point, a, b) for a, b in edges)
    inside = len(hull) > 2 and all(
        (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0]) > 0 for a, b in edges
    )
    return distance if inside else -distance


def stability_report(center_of_mass: Sequence[float], footprint_xy: Iterable[Sequence[float]]) -> dict[str, Any]:
    """``statically_stable`` (the CoM's ground projection is strictly inside
    the support polygon), ``stability_margin_mm`` (signed, see
    ``signed_margin``) and ``stability_warning`` (None when stable with at
    least ``MIN_STABILITY_MARGIN_MM`` to spare)."""
    hull = convex_hull(footprint_xy)
    if not hull:
        return {
            "statically_stable": None,
            "stability_margin_mm": None,
            "stability_warning": "No ground contact found, so stability could not be checked.",
            "support_polygon_xy": [],
        }
    margin = signed_margin((float(center_of_mass[0]), float(center_of_mass[1])), hull)
    if margin <= 0:
        warning = (
            f"Assembly center of mass projects {abs(margin):.1f} mm outside the base footprint, so it will tip "
            "over. Widen or extend the base under the heavy side, move mass over the base, or give the base a "
            "denser material (part_materials)."
        )
    elif margin < MIN_STABILITY_MARGIN_MM:
        warning = (
            f"Assembly center of mass is only {margin:.1f} mm inside the base footprint edge; a small push or "
            "uneven surface can tip it. Widen the base or give it a denser material (part_materials)."
        )
    else:
        warning = None
    return {
        "statically_stable": margin > 0,
        "stability_margin_mm": round(margin, 3),
        "stability_warning": warning,
        "support_polygon_xy": [[round(x, 3), round(y, 3)] for x, y in hull],
    }
