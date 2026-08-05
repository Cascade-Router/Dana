"""Pytest coverage for vision_math root-finding."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from vision_math import calculate_roots, f, generate_equation_image


def test_calculate_roots_are_real_and_sorted() -> None:
    roots = calculate_roots()
    assert len(roots) == 3
    assert roots == sorted(roots)
    for r in roots:
        assert math.isfinite(r)
        assert abs(f(r)) < 1e-8


def test_known_root_signs() -> None:
    """x^3 - 4x + 1 has one negative and two positive real roots."""
    roots = calculate_roots()
    assert roots[0] < 0.0
    assert roots[1] > 0.0
    assert roots[2] > roots[1]


def test_generate_equation_image_tmp(tmp_path: Path) -> None:
    dest = tmp_path / "eq.ppm"
    out = generate_equation_image(str(dest))
    path = Path(out)
    assert path.is_file()
    text = path.read_text(encoding="ascii", errors="replace")
    assert text.startswith("P3")
    assert "equation=" in text


def test_custom_brackets_still_converge() -> None:
    roots = calculate_roots(brackets=[(-2.5, -1.5), (0.1, 0.9), (1.5, 2.5)])
    assert len(roots) == 3
    for r in roots:
        assert abs(f(r)) < 1e-7
