"""Deterministic bounding-box geometry for vision-grounded OS navigation.

Stdlib-only (no NumPy). Bounding boxes are ``[xmin, ymin, xmax, ymax]``
sequences in pixel space.
"""
import math
from typing import Callable, Literal, Sequence

BBox = Sequence[float]
Direction = Literal["up", "down", "left", "right"]

def get_centroid(bbox: BBox) -> tuple[float, float]:
    """Return the exact ``(x, y)`` center of a ``[xmin, ymin, xmax, ymax]`` box."""
    xmin, ymin, xmax, ymax = (float(v) for v in bbox)
    return (xmin + xmax) / 2.0, (ymin + ymax) / 2.0


def inset_bbox(bbox: BBox, padding_percent: float) -> tuple[float, float, float, float]:
    """Shrink ``bbox`` inward by ``padding_percent`` of its own width/height.

    Keeps clicks off dead-space borders (rounded corners, focus rings) by
    biasing the actionable area toward the center. ``padding_percent`` is
    clamped to ``[0, 50)`` per side so the box can shrink but never invert;
    values at or above 50 are treated as 49.999 (near-collapse to centroid).
    """
    xmin, ymin, xmax, ymax = (float(v) for v in bbox)
    pct = max(0.0, min(49.999, float(padding_percent))) / 100.0
    dx = (xmax - xmin) * pct
    dy = (ymax - ymin) * pct
    return (xmin + dx, ymin + dy, xmax - dx, ymax - dy)


def calculate_iou(bbox1: BBox, bbox2: BBox) -> float:
    """Intersection-over-Union of two ``[xmin, ymin, xmax, ymax]`` boxes.

    Returns ``0.0`` for degenerate (zero/negative-area) or non-overlapping
    boxes rather than raising, so it is safe to use in a dedup sweep over
    noisy vision-model output.
    """
    ax1, ay1, ax2, ay2 = (float(v) for v in bbox1)
    bx1, by1, bx2, by2 = (float(v) for v in bbox2)
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def find_nearest_in_direction(
    reference_bbox: BBox,
    candidate_bboxes: Sequence[BBox],
    direction: Direction,
) -> tuple[float, float, float, float] | None:
    """Return the candidate box closest to ``reference_bbox`` in ``direction``.

    A candidate only qualifies if its centroid lies strictly on the requested
    side of the reference centroid (e.g. ``'right'`` requires
    ``candidate_cx > reference_cx``); among qualifying candidates the one
    with the smallest Euclidean centroid distance wins. Returns ``None`` when
    no candidate qualifies.
    """
    ref_cx, ref_cy = get_centroid(reference_bbox)
    checks: dict[str, Callable[[float, float], bool]] = {
        "up": lambda cx, cy: cy < ref_cy,
        "down": lambda cx, cy: cy > ref_cy,
        "left": lambda cx, cy: cx < ref_cx,
        "right": lambda cx, cy: cx > ref_cx,
    }
    check = checks.get(direction)
    if check is None:
        raise ValueError(
            f"unknown direction {direction!r}; expected one of {sorted(checks)}"
        )

    best: tuple[float, float, float, float] | None = None
    best_dist = math.inf
    for candidate in candidate_bboxes:
        cx, cy = get_centroid(candidate)
        if not check(cx, cy):
            continue
        dist = math.hypot(cx - ref_cx, cy - ref_cy)
        if dist < best_dist:
            best_dist = dist
            best = tuple(float(v) for v in candidate)
    return best


def normalize_coordinates(
    x: float,
    y: float,
    source_resolution: tuple[float, float],
    target_resolution: tuple[float, float],
) -> tuple[float, float]:
    """Rescale a point from ``source_resolution`` to ``target_resolution``.

    Use when a vision model's coordinates were computed against a resized
    screenshot (``source_resolution``) but must land at the OS's actual
    display scale (``target_resolution``), e.g. a 1280x720 model input vs a
    3840x2160 physical desktop.
    """
    sw, sh = float(source_resolution[0]), float(source_resolution[1])
    tw, th = float(target_resolution[0]), float(target_resolution[1])
    if sw <= 0.0 or sh <= 0.0:
        raise ValueError(f"source_resolution must be positive, got {source_resolution!r}")
    if tw <= 0.0 or th <= 0.0:
        raise ValueError(f"target_resolution must be positive, got {target_resolution!r}")
    return x * (tw / sw), y * (th / sh)


__all__ = (
    "BBox",
    "Direction",
    "calculate_iou",
    "find_nearest_in_direction",
    "get_centroid",
    "inset_bbox",
    "normalize_coordinates",
)
