"""Turn 2+ continuity: the system prompt's "Current CAD model" block
(server._cad_state_block) and update_cad_parameters' id resolution."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

import dana.api.server as server
import dana.core.react_dispatch as rd
from dana.plugins.freecad.call_log import CadCallLog
from dana.session_context import set_session_id


def _box(log: CadCallLog, name: str, length: float, width: float = 30, height: float = 10) -> None:
    log.record(
        "create_freecad_box",
        {"name": name, "length": length, "width": width, "height": height},
        ok=True,
        result={
            "ok": True, "name": name, "type": "Part::Box", "placement": [0, 0, 0],
            "dimensions": {"length": length, "width": width, "height": height},
            "bounding_box": [0, 0, 0, length, width, height],
        },
    )


def _session(log: CadCallLog) -> dict[str, Any]:
    return {"session_id": "cad-state-test", "call_log": log}


def test_no_model_no_block() -> None:
    assert server._cad_state_block(_session(CadCallLog())) == ""


def test_block_lists_objects_and_current_parameter_values() -> None:
    log = CadCallLog()
    _box(log, "Plate", 40)
    block = server._cad_state_block(_session(log))
    assert "Plate 40×30×10 at (0, 0, 0)" in block
    assert "Plate.length=40 mm [0.length]" in block
    assert "update_cad_parameters" in block


def test_a_large_model_stays_within_200_tokens() -> None:
    log = CadCallLog()
    for i in range(40):
        _box(log, f"Block{i:02d}", 10 + i)
    block = server._cad_state_block(_session(log))
    assert rd._count_tokens(block) <= server._CAD_STATE_MAX_TOKENS
    assert "Block39" in block  # the newest object survives the trim
    assert "older" in block and "more" in block  # and the trim says so


def test_labels_resolve_to_ids_and_unknown_names_list_the_options() -> None:
    log = CadCallLog()
    _box(log, "Plate", 40)
    overrides, error = server._resolve_parameter_overrides(_session(log), {"plate.LENGTH": 120, "0.width": 35})
    assert error is None and overrides == {"0.length": 120, "0.width": 35}
    _, error = server._resolve_parameter_overrides(_session(log), {"Hole.radius": 6})
    assert "unknown parameter 'Hole.radius'" in error and "Plate.length [0.length]" in error
    _, error = server._resolve_parameter_overrides(_session(log), "length=120")
    assert "non-empty object" in error


@pytest.mark.parametrize("has_geometry", [False, True])
def test_the_edit_tool_is_offered_only_once_there_is_geometry(
    has_geometry: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = f"edit-tool-offer-{has_geometry}"
    set_session_id(sid)
    if has_geometry:
        rd._OBJECT_PATH_REGISTRY[sid] = {"Plate": "plate.glb"}
    captured: list[list[dict]] = []

    class _Fake:
        def complete_with_tool_calls(self, messages: Any, *, tools: Any, **_k: Any) -> dict:
            captured.append(tools)
            return {"content": "ok", "tool_calls": [], "provider": "test"}

    monkeypatch.setattr(rd, "ModelProvider", lambda **_k: _Fake())
    asyncio.run(
        rd.next_react_turn(
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "make it longer"}],
            None,
            raw_text="make it longer",
            active_plugins=frozenset({"freecad"}),
            session_id=sid,
        )
    )
    offered = {t["function"]["name"] for t in captured[-1]}
    assert ("update_cad_parameters" in offered) is has_geometry
    assert rd._count_tokens(json.dumps(captured[-1])) <= rd._TOOL_TOKEN_BUDGET


def test_dispatching_it_outside_the_server_explains_itself() -> None:
    result = rd.TOOL_HANDLERS["update_cad_parameters"]({"overrides": {"0.length": 1}}, None, None)
    assert result["ok"] is False and "call log" in result["error"]
