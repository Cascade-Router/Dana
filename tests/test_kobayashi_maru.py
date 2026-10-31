"""scripts/run_kobayashi_maru.py on the mock engine, so the gauntlet fails
here the moment a phase gate, a WebSocket event or a tool contract changes
under it. It sat at 1/5 unnoticed once already: its scripted calls predated
create_plan and it still waited for the retired tool_result events."""

from __future__ import annotations

import sys

import pytest

from dana.paths import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import run_kobayashi_maru  # noqa: E402


def test_the_gauntlet_survives_5_of_5_on_the_mock_engine() -> None:
    rows = run_kobayashi_maru.run_gauntlet("mock")
    failures = [f"{name}: {detail}" for name, passed, detail in rows if not passed]
    assert len(rows) == 5
    assert not failures, failures


def test_an_explicit_flag_picks_the_engine() -> None:
    assert run_kobayashi_maru.resolve_engine_mode(mock=True, freecad=False) == "mock"
    assert run_kobayashi_maru.resolve_engine_mode(mock=False, freecad=True) == "freecad"


@pytest.mark.parametrize(
    ("operation", "value", "ok"),
    [("fillet", 4.9, True), ("fillet", 5.0, False), ("fillet", 20, False), ("chamfer", 6, False), ("fillet", 0, False)],
)
def test_the_mock_refuses_an_edge_operation_that_cannot_fit(operation: str, value: float, ok: bool) -> None:
    """OCC can't round or bevel an edge by half or more of its shortest edge;
    the mock refuses the same calls instead of silently "succeeding"."""
    from dana.platform.mock import MockFreeCADEngine

    engine = MockFreeCADEngine()
    assert engine.create_box(10, 10, 10, name="Cube")["ok"]
    result = engine.apply_edge_operation(operation, "Cube", value)
    assert result["ok"] is ok, result
    if not ok and value:
        assert "BRep_API: command not done" in result["error"]
