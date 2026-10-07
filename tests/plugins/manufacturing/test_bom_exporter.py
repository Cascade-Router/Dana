"""generate_assembly_bom: per-part volume -> mass -> cost CSV.

The mock-driver tests run headless everywhere (CI included): the mock's box
primitive is a closed trimesh mesh, so a 10 mm cube's volume is exactly
1 cm^3 there too. The last test drives the real FreeCAD engine and is skipped
when FreeCADCmd isn't installed.
"""

from __future__ import annotations

import csv
import math
import uuid
from pathlib import Path
from typing import Any

import pytest

from dana.core import react_dispatch as rd
from dana.plugins.freecad import engine as fc_engine
from dana.plugins.manufacturing import bom_exporter
from dana.session_context import DEFAULT_SESSION_ID, set_session_id
from dana.tools.schema import ToolCall

PLA = bom_exporter.load_materials()["materials"]["PLA"]
AL = bom_exporter.load_materials()["materials"]["Aluminum 6061"]


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(bom_exporter, "_EXPORT_DIR", tmp_path / "exports")
    set_session_id(f"bom-{uuid.uuid4().hex[:8]}")
    rd._set_has_plan(True, "BOM test")
    yield
    set_session_id(DEFAULT_SESSION_ID)


def _dispatcher(cad: Any):
    from dana.platform.mock import MockControlPlane

    control_plane = MockControlPlane()

    def _call(tool_id: str, **arguments: Any):
        return rd.dispatch_tool_call(ToolCall(tool_id=tool_id, arguments=arguments), cad, control_plane)

    return _call


@pytest.fixture
def mock_call():
    from dana.platform.mock import MockFreeCADEngine

    return _dispatcher(MockFreeCADEngine())


def _read_csv(path: str) -> list[list[str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def test_one_cm3_pla_cube_mass_and_cost(mock_call, tmp_path: Path) -> None:
    assert mock_call("create_freecad_box", name="Cube", length=10, width=10, height=10).ok
    assert mock_call("create_freecad_assembly", name="Kit").ok
    assert mock_call("add_parts_to_assembly", assembly_name="Kit", part_names=["Cube"]).ok

    result = mock_call("generate_assembly_bom", assembly_name="Kit")
    assert result.ok, result.message
    payload = result.payload
    expected_cost = 1.24 / 1000.0 * PLA["cost_per_kg"]

    path = Path(payload["path"])
    assert path == tmp_path / "exports" / "Kit_bom.csv"
    rows = _read_csv(str(path))
    assert rows[0] == list(bom_exporter.CSV_HEADER)
    assert len(rows) == 2
    name, volume_cm3, mass_g, material, density, cost = rows[1]
    assert (name, material, float(density)) == ("Cube", "PLA", 1.24)
    assert float(volume_cm3) == pytest.approx(1.0)
    assert float(mass_g) == pytest.approx(1.24)
    assert float(cost) == pytest.approx(expected_cost, abs=1e-4)
    assert payload["total_mass_g"] == pytest.approx(1.24)
    assert payload["total_cost"] == pytest.approx(expected_cost, abs=1e-4)
    assert "1.24 g" in rd.summarize_result(ToolCall(tool_id="generate_assembly_bom", arguments={}), result)


def test_material_is_matched_case_insensitively_and_by_alias(mock_call) -> None:
    assert mock_call("create_freecad_box", name="Plate", length=100, width=50, height=2).ok
    assert mock_call("create_freecad_assembly", name="Frame").ok
    assert mock_call("add_parts_to_assembly", assembly_name="Frame", part_names=["Plate"]).ok

    result = mock_call("generate_assembly_bom", assembly_name="Frame", material="al6061")
    assert result.ok, result.message
    assert result.payload["material"] == "Aluminum 6061"
    assert result.payload["total_mass_g"] == pytest.approx(10.0 * 2.70)  # 10 cm^3 of aluminum


def test_unknown_material_lists_choices_and_writes_nothing(mock_call, tmp_path: Path) -> None:
    assert mock_call("create_freecad_box", name="Cube", length=10, width=10, height=10).ok
    assert mock_call("create_freecad_assembly", name="Kit").ok
    assert mock_call("add_parts_to_assembly", assembly_name="Kit", part_names=["Cube"]).ok

    result = mock_call("generate_assembly_bom", assembly_name="Kit", material="unobtainium")
    assert not result.ok
    assert "PETG" in result.message
    assert not (tmp_path / "exports").exists()


def test_unknown_assembly_is_refused(mock_call) -> None:
    result = mock_call("generate_assembly_bom", assembly_name="Nope")
    assert not result.ok
    assert "create_freecad_assembly" in result.message


def test_multi_part_rows_and_totals() -> None:
    parts = [{"name": "A", "volume_mm3": 2000.0}, {"name": "B", "volume_mm3": 500.0}]
    result = bom_exporter.build_bom("Pair", parts, "Stainless Steel 304")
    assert result["ok"]
    assert [r["mass_g"] for r in result["parts"]] == [pytest.approx(16.0), pytest.approx(4.0)]
    assert result["total_mass_g"] == pytest.approx(20.0)
    assert result["total_cost"] == pytest.approx(0.02 * result["parts"][0]["cost_per_kg"])
    assert len(_read_csv(result["path"])) == 3


def test_mixed_materials_cost_each_part_in_its_own_material() -> None:
    # A 10 cm^3 aluminium plate and a 2 cm^3 PLA bracket.
    parts = [{"name": "BasePlate", "volume_mm3": 10_000.0}, {"name": "Bracket", "volume_mm3": 2_000.0}]
    result = bom_exporter.build_bom("Mount", parts, "PLA", {"BasePlate": "aluminium 6061"})

    assert result["ok"], result
    plate, bracket = result["parts"]
    assert (plate["material"], plate["density_g_cm3"], plate["mass_g"]) == ("Aluminum 6061", 2.70, pytest.approx(27.0))
    assert (bracket["material"], bracket["mass_g"]) == ("PLA", pytest.approx(2.48))
    plate_cost = 27.0 / 1000 * AL["cost_per_kg"]
    bracket_cost = 2.48 / 1000 * PLA["cost_per_kg"]
    assert plate["cost"] == pytest.approx(plate_cost, abs=1e-4)
    assert result["total_mass_g"] == pytest.approx(29.48)
    assert result["total_cost"] == pytest.approx(plate_cost + bracket_cost, abs=1e-4)
    assert result["materials"] == ["Aluminum 6061", "PLA"]
    rows = _read_csv(result["path"])[1:]
    assert [(r[0], r[3], float(r[4])) for r in rows] == [("BasePlate", "Aluminum 6061", 2.70), ("Bracket", "PLA", 1.24)]


def test_part_materials_match_a_part_label_too() -> None:
    parts = [{"name": "Box001", "label": "Lid", "volume_mm3": 1000.0}]
    result = bom_exporter.build_bom("Case", parts, "PLA", {"Lid": "PETG"})
    assert result["parts"][0]["material"] == "PETG"


@pytest.mark.parametrize(
    ("part_materials", "expected"),
    [
        pytest.param({"BasePlate": "titanium"}, "titanium", id="unknown-material"),
        pytest.param({"Bolt": "PLA"}, "Bolt", id="unknown-part"),
    ],
)
def test_bad_part_materials_fail_before_writing(part_materials: dict[str, str], expected: str, tmp_path: Path) -> None:
    parts = [{"name": "BasePlate", "volume_mm3": 1000.0}]
    result = bom_exporter.build_bom("Mount", parts, "PLA", part_materials)
    assert not result["ok"]
    assert expected in result["error"]
    assert not (tmp_path / "exports").exists()


def test_mixed_materials_through_the_tool(mock_call) -> None:
    assert mock_call("create_freecad_box", name="BasePlate", length=50, width=20, height=10).ok
    assert mock_call("create_freecad_box", name="Bracket", length=10, width=10, height=20).ok
    assert mock_call("create_freecad_assembly", name="Mount").ok
    assert mock_call("add_parts_to_assembly", assembly_name="Mount", part_names=["BasePlate", "Bracket"]).ok

    result = mock_call("generate_assembly_bom", assembly_name="Mount", part_materials={"BasePlate": "6061"})
    assert result.ok, result.message
    by_name = {p["name"]: p for p in result.payload["parts"]}
    assert by_name["BasePlate"]["mass_g"] == pytest.approx(10.0 * 2.70)
    assert by_name["Bracket"]["mass_g"] == pytest.approx(2.0 * 1.24)
    summary = rd.summarize_result(ToolCall(tool_id="generate_assembly_bom", arguments={}), result)
    assert "Aluminum 6061, PLA" in summary


def test_part_materials_must_be_a_name_to_material_mapping(mock_call) -> None:
    assert mock_call("create_freecad_assembly", name="Mount").ok
    result = mock_call("generate_assembly_bom", assembly_name="Mount", part_materials=["PLA"])
    assert not result.ok
    assert "part_materials" in result.message


@pytest.mark.parametrize("volume", [0.0, -1000.0])
def test_non_positive_volume_is_refused(volume: float) -> None:
    result = bom_exporter.build_bom("Bad", [{"name": "Shell", "volume_mm3": volume}])
    assert not result["ok"]
    assert "Shell" in result["error"]


def test_empty_assembly_is_refused() -> None:
    result = bom_exporter.build_bom("Empty", [])
    assert not result["ok"]
    assert "add_parts_to_assembly" in result["error"]


@pytest.mark.e2e
@pytest.mark.skipif(
    fc_engine.detect_freecadcmd() is None,
    reason="real FreeCADCmd not found (PATH or DANA_FREECADCMD_PATH) — this test drives the real engine",
)
def test_real_freecad_volumes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from dana.platform.win32 import RealFreeCADEngine

    monkeypatch.setattr(fc_engine, "_OUTPUT_DIR", tmp_path / "freecad_output")
    monkeypatch.setattr(fc_engine, "_EXPORT_DIR", tmp_path / "freecad_exports")
    monkeypatch.setenv("DANA_HEADLESS", "true")
    monkeypatch.delenv("DANA_OS_DRY_RUN", raising=False)
    call = _dispatcher(RealFreeCADEngine())

    assert call("create_freecad_box", name="Cube", length=10, width=10, height=10).ok
    assert call("create_freecad_cylinder", name="Peg", radius=5, height=10, placement_x=40).ok
    assert call("create_freecad_assembly", name="Kit").ok
    assert call("add_parts_to_assembly", assembly_name="Kit", part_names=["Cube", "Peg"]).ok

    result = call("generate_assembly_bom", assembly_name="Kit", material="PLA")
    assert result.ok, result.message
    rows = {r["name"]: r for r in result.payload["parts"]}
    assert set(rows) == {"Cube", "Peg"}
    assert rows["Cube"]["mass_g"] == pytest.approx(1.24, abs=1e-4)
    assert rows["Peg"]["volume_cm3"] == pytest.approx(math.pi * 0.25, abs=1e-4)  # r=0.5 cm, h=1 cm


@pytest.mark.e2e
@pytest.mark.skipif(
    fc_engine.detect_freecadcmd() is None,
    reason="real FreeCADCmd not found (PATH or DANA_FREECADCMD_PATH) — this test drives the real engine",
)
def test_real_freecad_mixed_materials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from dana.platform.win32 import RealFreeCADEngine

    monkeypatch.setattr(fc_engine, "_OUTPUT_DIR", tmp_path / "freecad_output")
    monkeypatch.setattr(fc_engine, "_EXPORT_DIR", tmp_path / "freecad_exports")
    monkeypatch.setenv("DANA_HEADLESS", "true")
    monkeypatch.delenv("DANA_OS_DRY_RUN", raising=False)
    call = _dispatcher(RealFreeCADEngine())

    assert call("create_freecad_box", name="BasePlate", length=100, width=50, height=4).ok  # 20 cm^3
    assert call("create_freecad_box", name="Bracket", length=20, width=10, height=30, placement_z=4).ok  # 6 cm^3
    assert call("create_freecad_assembly", name="Mount").ok
    assert call("add_parts_to_assembly", assembly_name="Mount", part_names=["BasePlate", "Bracket"]).ok

    result = call(
        "generate_assembly_bom", assembly_name="Mount", material="PLA", part_materials={"BasePlate": "Aluminum 6061"}
    )
    assert result.ok, result.message
    by_name = {p["name"]: p for p in result.payload["parts"]}
    assert (by_name["BasePlate"]["material"], by_name["BasePlate"]["mass_g"]) == ("Aluminum 6061", pytest.approx(54.0))
    assert (by_name["Bracket"]["material"], by_name["Bracket"]["mass_g"]) == ("PLA", pytest.approx(7.44))
    expected_cost = 54.0 / 1000 * AL["cost_per_kg"] + 7.44 / 1000 * PLA["cost_per_kg"]
    assert result.payload["total_cost"] == pytest.approx(expected_cost, abs=1e-4)
