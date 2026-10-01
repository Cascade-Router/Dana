"""Multi-feature patterns: create_freecad_polar_pattern / _linear_pattern take
one feature name or a list of them, end to end (schema, dispatch handler,
topology redirect, mock engine, macro replay). Runs without FreeCAD; the
real-engine geometry is covered in test_engine_partdesign_live.py."""

from __future__ import annotations

import uuid

import pytest

from dana.core import react_dispatch as rd
from dana.platform.mock import MockFreeCADEngine, _mock_object_registry
from dana.plugins.freecad.call_log import CadCallLog
from dana.plugins.freecad.engine import pattern_feature_names
from dana.plugins.freecad.py_export import render_macro_script
from dana.session_context import DEFAULT_SESSION_ID, set_session_id
from dana.tools.schema import load_tool_registry, to_openai_function_schema, validate_tool_arguments


@pytest.fixture(autouse=True)
def _own_session():
    set_session_id(f"multi-pattern-{uuid.uuid4().hex[:8]}")
    yield
    set_session_id(DEFAULT_SESSION_ID)


@pytest.mark.parametrize("tool_id", ["create_freecad_polar_pattern", "create_freecad_linear_pattern"])
def test_schema_advertises_a_list_and_still_accepts_one_name(tool_id: str) -> None:
    spec = load_tool_registry()[tool_id]
    prop = to_openai_function_schema(spec)["function"]["parameters"]["properties"]["feature_name"]
    assert prop["type"] == "array" and prop["items"] == {"type": "string"}
    base = {"occurrences": 4} if "polar" in tool_id else {"occurrences": 3, "length": 20}
    assert validate_tool_arguments(spec, {**base, "feature_name": ["Pad001", "Pad002"]}) is None
    assert validate_tool_arguments(spec, {**base, "feature_name": "Pad001"}) is None


def test_feature_name_normalization() -> None:
    assert pattern_feature_names("Pad") == ["Pad"]
    assert pattern_feature_names(["Pad001", "Pocket004"]) == ["Pad001", "Pocket004"]
    assert pattern_feature_names('["Pad001", "Pocket004"]') == ["Pad001", "Pocket004"]
    assert "at least one" in pattern_feature_names([])
    assert "at least one" in pattern_feature_names(["Pad", " "])
    assert "more than once" in pattern_feature_names(["Pad", "Pad"])
    assert "must be a feature name" in pattern_feature_names(7)


def _mock_pads(engine: MockFreeCADEngine, *names: str) -> None:
    """Real mock Pad meshes registered under each name, in both the mock
    engine's registry and the dispatcher's object registry (what a real
    create_freecad_pad dispatch would have recorded)."""
    for i, name in enumerate(names):
        geometry = [{"type": "circle", "center": [10.0 * (i + 1), 0.0], "radius": 2.0}]
        assert engine.create_sketch(f"S{i}", "XY", geometry)["ok"]
        assert engine.create_pad(f"S{i}", 4)["ok"]
        path = _mock_object_registry()["Pad"]
        _mock_object_registry()[name] = path
        rd._object_registry()[name] = path


def test_handler_patterns_every_listed_feature_on_the_mock_engine() -> None:
    engine = MockFreeCADEngine()
    _mock_pads(engine, "Pad001", "Pad002")
    result = rd._tool_create_freecad_polar_pattern(
        {"feature_name": ["Pad001", "Pad002"], "occurrences": 4}, engine, None
    )
    assert result["ok"] is True, result
    assert result["dimensions"]["features"] == ["Pad001", "Pad002"]

    single = rd._tool_create_freecad_linear_pattern(
        {"feature_name": "Pad001", "occurrences": 3, "length": 20}, engine, None
    )
    assert single["ok"] is True, single
    assert single["dimensions"]["features"] == ["Pad001"]


def test_handler_names_every_unknown_feature() -> None:
    engine = MockFreeCADEngine()
    _mock_pads(engine, "Pad001")
    result = rd._tool_create_freecad_polar_pattern(
        {"feature_name": ["Pad001", "Ghost", "Phantom"], "occurrences": 4}, engine, None
    )
    assert result["ok"] is False
    assert "'Ghost'" in result["error"] and "'Phantom'" in result["error"]
    assert "'Pad001'" not in result["error"]


def test_topology_redirect_records_every_listed_feature_as_consumed() -> None:
    resolved, input_names, _warning = rd._apply_topology_redirects(
        "create_freecad_polar_pattern", {"feature_name": ["Pad001", "Pocket004"], "occurrences": 4}
    )
    assert resolved["feature_name"] == ["Pad001", "Pocket004"]
    assert input_names == ["Pad001", "Pocket004"]


def test_macro_replays_a_multi_feature_pattern() -> None:
    log = CadCallLog()
    log.record(
        "create_freecad_polar_pattern",
        {"feature_name": ["Pad001", "Pad002", "Pocket004"], "occurrences": 4},
        ok=True,
        result={"name": "PolarPattern", "dimensions": {
            "occurrences": 4, "angle": 360.0, "axis": "Z", "reversed_direction": False,
            "features": ["Pad001", "Pad002", "Pocket004"],
        }},
    )
    # An older single-name record (no "features" in dimensions) still replays.
    log.record(
        "create_freecad_linear_pattern",
        {"feature_name": "Pad001", "occurrences": 3, "length": 20, "direction": "Y"},
        ok=True,
        result={"name": "LinearPattern", "dimensions": {
            "occurrences": 3, "length": 20.0, "direction": "Y", "reversed_direction": False,
        }},
    )
    script = render_macro_script(log)
    compile(script, "<generated-macro>", "exec")
    assert "for _src in ['Pad001', 'Pad002', 'Pocket004']:" in script
    assert "for _src in ['Pad001']:" in script
    assert "obj.Originals = _feats" in script
    assert "could not be replayed" not in script
