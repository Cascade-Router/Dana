"""create_freecad_sketch's strict geometry pre-validation: bad payloads
bounce with a recoverable error before the engine is ever called."""

from __future__ import annotations

from typing import Any

import pytest

import dana.core.react_dispatch as rd

_ERR = "Invalid geometry primitive. Only 'line', 'circle', and 'arc' with format [x, y, ...] are supported."

_VALID = [
    {"type": "line", "start": [0, 0], "end": [10, 0]},
    {"type": "circle", "center": [0, 0], "radius": 5},
    {"type": "arc", "center": [0, 0], "radius": 2.5, "start_angle": 0, "end_angle": 90},
]


class _SpyEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def create_sketch(self, name: str, plane: str, geometry: list[dict]) -> dict[str, Any]:
        self.calls.append((name, plane, geometry))
        return {"ok": True, "name": name}


def _call(geometry: Any) -> tuple[dict[str, Any], _SpyEngine]:
    engine = _SpyEngine()
    result = rd._tool_create_freecad_sketch({"name": "S", "plane": "XY", "geometry": geometry}, engine, None)
    return result, engine


def test_valid_geometry_reaches_engine() -> None:
    result, engine = _call(_VALID)
    assert result["ok"] is True
    assert len(engine.calls) == 1


@pytest.mark.parametrize(
    "geometry",
    [
        [{"type": "rectangle", "start": [0, 0], "end": [10, 20]}],
        [{"type": "line", "x1": 0, "y1": 0, "x2": 10, "y2": 0}],
        [{"type": "line", "start": [0, 0], "end": [10, 0], "x1": 0}],
        [{"type": "circle", "cx": 0, "cy": 0, "radius": 5}],
        [{"type": "circle", "center": [0, 0], "radius": 5, "width": 3}],
        [{"type": "arc", "center": [0, 0], "radius": 1, "start_angle": 0}],
        [{"type": "line", "start": [0, 0, 0], "end": [10, 0]}],
        [{"type": "line", "start": ["0", 0], "end": [10, 0]}],
        [{"type": "Line", "start": [0, 0], "end": [10, 0]}],
        ["line"],
        [],
        None,
    ],
)
def test_invalid_geometry_bounces_before_engine(geometry: Any) -> None:
    result, engine = _call(geometry)
    assert result["ok"] is False
    assert result["error"].startswith(_ERR)
    assert engine.calls == []


def test_validator_raises_schema_validation_error() -> None:
    with pytest.raises(rd.SchemaValidationError, match=r"unsupported keys \['x1'\]"):
        rd.validate_sketch_geometry([{"type": "line", "start": [0, 0], "end": [1, 0], "x1": 0}])
    assert issubclass(rd.SchemaValidationError, ValueError)


def test_whitelist_is_exactly_line_circle_arc() -> None:
    assert rd.SKETCH_GEOMETRY_PRIMITIVES == ("line", "circle", "arc")


def test_prompt_forbids_marking_auto_advanced_tasks() -> None:
    prompt = rd._CAD_AGENT_PROTOCOL if hasattr(rd, "_CAD_AGENT_PROTOCOL") else None
    text = prompt or "\n".join(v for v in vars(rd).values() if isinstance(v, str))
    assert "FORBIDDEN from calling" in text and "mark_task_completed" in text
