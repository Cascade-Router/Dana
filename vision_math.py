"""Root-finding for the vision-grounded cubic ``f(x) = x^3 - 4x + 1``.

Stdlib-only (no NumPy). Uses Brent-style bisection on known isolating brackets
plus optional Newton polish.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence


def f(x: float) -> float:
    """Evaluate ``x^3 - 4x + 1``."""
    return (x * x * x) - (4.0 * x) + 1.0


def df(x: float) -> float:
    """Derivative ``3x^2 - 4``."""
    return (3.0 * x * x) - 4.0


def _bisect(
    fn: Callable[[float], float],
    a: float,
    b: float,
    *,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> float:
    fa, fb = fn(a), fn(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0.0:
        raise ValueError(f"no sign change on [{a}, {b}]")
    lo, hi = a, b
    flo = fa
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = fn(mid)
        if abs(fmid) < tol or abs(hi - lo) < tol:
            return mid
        if flo * fmid <= 0.0:
            hi = mid
        else:
            lo, flo = mid, fmid
    return 0.5 * (lo + hi)


def _newton(x0: float, *, tol: float = 1e-12, max_iter: int = 40) -> float:
    x = float(x0)
    for _ in range(max_iter):
        y = f(x)
        if abs(y) < tol:
            return x
        d = df(x)
        if abs(d) < 1e-14:
            break
        x = x - y / d
    return x


def calculate_roots(
    brackets: Sequence[tuple[float, float]] | None = None,
) -> list[float]:
    """Return the three real roots of ``x^3 - 4x + 1``, sorted ascending.

    Default isolating intervals cover the three real roots of this cubic.
    """
    intervals = list(brackets) if brackets is not None else [
        (-3.0, -1.0),
        (0.0, 1.0),
        (1.0, 3.0),
    ]
    roots: list[float] = []
    for a, b in intervals:
        rough = _bisect(f, float(a), float(b))
        roots.append(_newton(rough))
    roots.sort()
    return roots


def generate_equation_image(
    path: str = "vision_math_equation.png",
    equation: str = "f(x) = x^3 - 4x + 1",
) -> str:
    """Write a tiny PPM/PNG-less placeholder text image via raw PPM (stdlib).

    Returns the output path. Used when a camera is unavailable for vision grounding.
    """
    # Minimal ASCII art rendered into a P3 PPM (portable, no third-party deps).
    width, height = 320, 80
    lines = [
        "P3",
        f"{width} {height}",
        "255",
    ]
    # Dark background with a bright horizontal band (stand-in for OCR target).
    for y in range(height):
        row: list[str] = []
        for x in range(width):
            if 28 <= y <= 52:
                row.extend(["34", "211", "238"])  # cyan band
            else:
                row.extend(["15", "23", "42"])
        lines.append(" ".join(row))
    out = path if path.lower().endswith(".ppm") else path.rsplit(".", 1)[0] + ".ppm"
    with open(out, "w", encoding="ascii") as fh:
        fh.write("\n".join(lines))
        fh.write(f"\n# equation={equation}\n")
    return out


__all__ = (
    "calculate_roots",
    "df",
    "f",
    "generate_equation_image",
)
