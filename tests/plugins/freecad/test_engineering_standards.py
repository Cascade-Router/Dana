"""Minimal tests for dana.plugins.freecad.engineering_standards — verifies
the local lookup table returns the correct standard dimensions and handles
no-match/ambiguous queries sanely. Not a full test suite by design.
"""

from __future__ import annotations

from dana.plugins.freecad.engineering_standards import query_engineering_standard


def test_nema17_query_returns_correct_mounting_dimensions():
    result = query_engineering_standard("NEMA 17 dimensions")
    assert result["ok"] is True
    assert result["standard"] == "nema17"
    dims = result["dimensions"]
    assert dims["mounting_hole_spacing_mm"] == 31.0
    assert dims["mounting_hole_count"] == 4
    assert dims["shaft_diameter_mm"] == 5.0


def test_m3_clearance_query_returns_correct_hole_diameter():
    result = query_engineering_standard("M3 clearance hole")
    assert result["ok"] is True
    assert result["standard"] == "m3_clearance"
    assert result["dimensions"]["close_fit_mm"] == 3.2


def test_m4_and_m5_queries_are_distinguished_from_m3():
    m4 = query_engineering_standard("M4 bolt")
    m5 = query_engineering_standard("M5 screw")
    assert m4["standard"] == "m4_clearance"
    assert m4["dimensions"]["nominal_diameter_mm"] == 4.0
    assert m5["standard"] == "m5_clearance"
    assert m5["dimensions"]["nominal_diameter_mm"] == 5.0


def test_unmatched_query_reports_failure_with_available_standards():
    result = query_engineering_standard("flux capacitor bracket")
    assert result["ok"] is False
    assert "NEMA 17 Stepper Motor" in result["available_standards"]


def test_empty_query_is_rejected():
    assert query_engineering_standard("").get("ok") is False
