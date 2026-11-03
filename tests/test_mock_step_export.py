"""The mock engine's STEP export (dana.platform.mock.write_faceted_step):
ISO 10303-21 AP214, one faceted B-rep per watertight body. The live test
reads the file back with real FreeCAD; it runs wherever FreeCADCmd is
installed (not in CI)."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from dana.platform.mock import MockFreeCADEngine, write_faceted_step
from dana.plugins.freecad import engine as freecad_engine


def _bracket_step() -> tuple[Path, object]:
    import trimesh

    engine = MockFreeCADEngine()
    engine.create_box(40, 30, 10, name="Plate")
    engine.create_cylinder(4, 10, name="Hole", placement=(20.0, 15.0, 0.0))
    cut = engine.apply_boolean("cut", "Plate", "Hole", name="Bracket")
    result = engine.export_model([cut["path"]], "step", "bracket")
    assert result["ok"] is True, result
    return Path(result["path"]), trimesh.load(cut["path"], force="mesh")


def _entities(text: str) -> dict[int, str]:
    data = text.split("DATA;", 1)[1].split("ENDSEC;", 1)[0]
    return {int(m[1]): m[2] for m in re.finditer(r"^#(\d+)=(.*);$", data, re.M)}


def test_the_file_is_a_well_formed_ap214_exchange_file() -> None:
    path, mesh = _bracket_step()
    text = path.read_text(encoding="ascii")
    assert path.parent.name == freecad_engine._export_dir().name  # the session's export folder
    assert text.startswith("ISO-10303-21;\nHEADER;\n")
    assert "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));" in text
    assert text.rstrip().endswith("ENDSEC;\nEND-ISO-10303-21;")
    entities = _entities(text)
    # Every #reference points at an entity that exists.
    referenced = {int(r) for body in entities.values() for r in re.findall(r"#(\d+)", body)}
    assert referenced <= set(entities)
    kinds = [re.match(r"\(?([A-Z_0-9]+)", body)[1] for body in entities.values()]
    assert kinds.count("FACETED_BREP") == 1 and kinds.count("CLOSED_SHELL") == 1
    assert kinds.count("FACE_SURFACE") == kinds.count("POLY_LOOP") == kinds.count("PLANE") == len(mesh.faces)
    # The mesh's vertices plus the representation's origin.
    assert kinds.count("CARTESIAN_POINT") == len(mesh.vertices) + 1
    coordinates = [c for body in entities.values() if body.startswith("CARTESIAN_POINT")
                   for c in body.split("(", 2)[2].rstrip(")").split(",")]
    assert len(coordinates) == 3 * (len(mesh.vertices) + 1)
    assert all(re.fullmatch(r"-?\d+\.\d*(E[-+]?\d+)?", c) for c in coordinates), coordinates[:6]


def test_step_reals_always_carry_a_decimal_point() -> None:
    from dana.platform.mock import _step_real

    assert [_step_real(v) for v in (10, 0.5, -3.25, 1e-5, 0)] == ["10.", "0.5", "-3.25", "1.E-05", "0."]


def test_an_open_mesh_is_a_surface_model_not_a_solid(tmp_path: Path) -> None:
    import trimesh

    sheet = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]])
    stats = write_faceted_step([("Sheet", sheet)], tmp_path / "sheet.step", title="sheet")
    text = (tmp_path / "sheet.step").read_text(encoding="ascii")
    assert stats == {"step_bodies": 1, "step_vertices": 3, "step_faces": 1}
    assert "SHELL_BASED_SURFACE_MODEL('Sheet'" in text and "OPEN_SHELL" in text and "FACETED_BREP(" not in text


@pytest.mark.e2e
@pytest.mark.skipif(freecad_engine.detect_freecadcmd() is None, reason="real FreeCADCmd not found")
def test_freecad_reads_it_back_as_the_same_solid(tmp_path: Path) -> None:
    path, mesh = _bracket_step()
    script = tmp_path / "read_step.py"
    script.write_text(
        "import Part\n"
        f"shape = Part.read({str(path)!r})\n"
        "print('RESULT', len(shape.Solids), shape.isValid(), len(shape.Faces), round(shape.Volume, 2))\n",
        encoding="utf-8",
    )
    out = subprocess.run([freecad_engine.detect_freecadcmd(), str(script)], capture_output=True, text=True, timeout=240)
    line = next(l for l in (out.stdout + out.stderr).splitlines() if l.startswith("RESULT"))
    solids, valid, faces, volume = line.split()[1:]
    assert (solids, valid, int(faces)) == ("1", "True", len(mesh.faces))
    assert float(volume) == pytest.approx(float(mesh.volume), rel=1e-6)
