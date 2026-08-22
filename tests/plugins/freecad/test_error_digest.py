"""Minimal tests for dana.plugins.freecad.error_digest — verifies recognized
FreeCAD/kernel failures get enriched with a reason+suggestion, while
already-specific messages (dispatch validation errors, driver limitation
notes) are preserved verbatim rather than replaced with a generic
placeholder. Not a full test suite by design.
"""

from __future__ import annotations

from dana.plugins.freecad.error_digest import digest_error


def test_recognized_fillet_failure_gets_enriched_reason_and_suggestion():
    raw = "Part::TopoShape::makeFillet failed: BRepFilletAPI: fillet not possible on selected edges"
    result = digest_error("perform_freecad_edge_operation", raw)
    assert result["status"] == "error"
    assert "fillet" in result["reason"].lower()
    assert "smaller" in result["suggestion"].lower()
    assert result["raw_error"] == raw


def test_validation_message_mentioning_fillet_is_preserved_not_misclassified():
    """A plain enum-validation error that happens to mention 'fillet' as a
    valid option must NOT be misclassified as a fillet kernel failure — it
    already IS a clear, specific message and should survive verbatim."""
    raw = "perform_freecad_edge_operation requires operation to be one of fillet, chamfer"
    result = digest_error("perform_freecad_edge_operation", raw)
    assert result["reason"] == raw


def test_unrecognized_message_is_preserved_verbatim_not_replaced():
    raw = "unknown target_object 'Widget' — create it first with a create_freecad_* tool"
    result = digest_error("modify_freecad_parameter", raw)
    assert result["reason"] == raw
    assert result["raw_error"] == raw


def test_real_freecad_timeout_still_classified_correctly():
    """dana.plugins.freecad.engine's own real timeout message, verbatim —
    must still match the FreeCAD-specific timeout signature."""
    raw = "FreeCADCmd timed out after 90s"
    result = digest_error("create_freecad_box", raw)
    assert "FreeCAD subprocess" in result["reason"]


def test_non_freecad_timeout_is_not_misclassified_as_freecad():
    """Regression: dispatch_tool_call/digest_error runs for EVERY tool, not
    just FreeCAD's — a bare 'timed out' substring from an unrelated plugin
    (e.g. coder_plugin's run_verification_command) must never get
    overwritten with FreeCAD-specific text just because it shares that one
    word. Preserved verbatim like any other unrecognized message."""
    raw = "pytest timed out after 120 seconds"
    result = digest_error("run_verification_command", raw)
    assert "FreeCAD" not in result["reason"]
    assert result["reason"] == raw
