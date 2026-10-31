"""Array arguments sent as JSON-encoded strings are parsed back into lists
before schema validation (dana.tools.schema.coerce_stringified_arrays)."""

from __future__ import annotations

from typing import Any

import pytest

import dana.core.react_dispatch as rd
from dana.session_context import set_session_id
from dana.tools.schema import ToolCall, coerce_stringified_arrays, load_tool_registry, validate_tool_arguments

_CIRCLE = [{"type": "circle", "center": [20, 15], "radius": 4}]


def _spec(tool_id: str) -> Any:
    return load_tool_registry()[tool_id]


def test_a_stringified_array_is_parsed_and_then_validates() -> None:
    spec = _spec("perform_freecad_boolean")
    args = {"operation": "cut", "objects": '["Base", "Hole"]'}
    coerced = coerce_stringified_arrays(spec, args)
    assert coerced == {"operation": "cut", "objects": ["Base", "Hole"]}
    assert args["objects"] == '["Base", "Hole"]'  # the caller's dict is not mutated
    assert validate_tool_arguments(spec, coerced) is None


@pytest.mark.parametrize("value", ["Base, Hole", '{"a": 1}', "[not json", ""])
def test_anything_that_is_not_a_json_list_is_left_for_validation(value: str) -> None:
    spec = _spec("perform_freecad_boolean")
    args = {"operation": "cut", "objects": value}
    assert coerce_stringified_arrays(spec, args) is args


def test_a_string_parameter_is_never_parsed() -> None:
    spec = _spec("create_freecad_sketch")
    args = {"name": '["looks", "like", "a", "list"]', "plane": "XY", "geometry": _CIRCLE}
    assert coerce_stringified_arrays(spec, args) is args


def test_dispatch_hands_the_handler_the_parsed_list(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Any] = []

    def _spy(args: dict[str, Any], _engine: Any, _cp: Any) -> dict[str, Any]:
        seen.append(args["geometry"])
        return {"ok": True, "name": args["name"]}

    monkeypatch.setitem(rd.TOOL_HANDLERS, "create_freecad_sketch", _spy)
    sid = "coerce-dispatch"
    set_session_id(sid)
    rd._PLAN_STATE_REGISTRY[sid] = {"has_plan": True, "plan_text": "plan", "tasks": [], "fsm_state": "done"}
    call = ToolCall(
        tool_id="create_freecad_sketch",
        arguments={"name": "hole", "plane": "XY", "geometry": '[{"type": "circle", "center": [20, 15], "radius": 4}]'},
    )
    result = rd.dispatch_tool_call(call, None, None)
    assert result.ok, result
    assert seen == [_CIRCLE]
