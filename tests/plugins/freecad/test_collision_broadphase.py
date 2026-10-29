"""validate_assembly_collisions' bounding-box broadphase: pairs whose
axis-aligned boxes don't meet skip the B-Rep boolean (``Shape.common``);
overlapping pairs still go through it and are still flagged.

The unit tests run the real rendered FreeCADCmd script against a stub
``FreeCAD`` module whose shapes record every ``common()`` call, so they run
anywhere. The live test drives real FreeCAD.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import sys
import types
import uuid
from pathlib import Path
from typing import Any

import pytest

from dana.core import react_dispatch as rd
from dana.plugins.freecad import engine as fc_engine
from dana.session_context import DEFAULT_SESSION_ID, set_session_id
from dana.tools.schema import ToolCall

_LIVE = pytest.mark.skipif(
    fc_engine.detect_freecadcmd() is None,
    reason="real FreeCADCmd not found (PATH or DANA_FREECADCMD_PATH) — this test drives the real engine",
)
_MARKER = fc_engine._OK_MARKER


class _Box:
    def __init__(self, lo: tuple[float, float, float], hi: tuple[float, float, float], *, broken: bool = False):
        self.lo, self.hi, self.broken = lo, hi, broken

    def intersect(self, other: _Box) -> bool:
        if self.broken or other.broken:
            raise RuntimeError("bad bounding box")
        return all(a_lo <= b_hi and b_lo <= a_hi for a_lo, a_hi, b_lo, b_hi in zip(self.lo, self.hi, other.lo, other.hi))


class _Shape:
    def __init__(self, name: str, box: _Box, calls: list[tuple[str, str]], *, overlap: float = 0.0):
        self.name, self.BoundBox, self._calls, self._overlap = name, box, calls, overlap

    def isNull(self) -> bool:
        return False

    def common(self, other: _Shape) -> Any:
        self._calls.append((self.name, other.name))
        return types.SimpleNamespace(Volume=max(self._overlap, other._overlap) if self.BoundBox.intersect(other.BoundBox) else 0.0)

    def distToShape(self, other: _Shape) -> tuple[float, list, list]:
        gap = max(max(b_lo - a_hi, a_lo - b_hi, 0.0)
                  for a_lo, a_hi, b_lo, b_hi in zip(self.BoundBox.lo, self.BoundBox.hi, other.BoundBox.lo, other.BoundBox.hi))
        return (gap, [], [])


class _Obj:
    def __init__(self, name: str, shape: _Shape | None = None, group: list[_Obj] | None = None):
        self.Name = self.Label = name
        if shape is not None:
            self.Shape = shape
        if group is not None:
            self.Group = group


def _run_script(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, members: list[_Obj]) -> dict[str, Any]:
    """Renders and executes the real collision script against stub members;
    returns its parsed marker lines."""
    session_file = tmp_path / "Session_Active.FCStd"
    session_file.write_bytes(b"stub")
    assembly = _Obj("Rig", group=members)
    doc = types.SimpleNamespace(
        Objects=[assembly, *members],
        getObject=lambda name: next((o for o in [assembly, *members] if o.Name == name), None),
    )
    freecad = types.ModuleType("FreeCAD")
    freecad.openDocument = lambda _path: doc
    freecad.newDocument = lambda _name: doc
    monkeypatch.setitem(sys.modules, "FreeCAD", freecad)

    script = fc_engine._VALIDATE_ASSEMBLY_COLLISIONS_SCRIPT.format(
        assembly_name="Rig",
        epsilon=fc_engine._COLLISION_VOLUME_EPSILON,
        session_path=str(session_file),
        session_doc_name=fc_engine._SESSION_DOCUMENT_NAME,
        marker=_MARKER,
    )
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(script, "<collision-script>", "exec"), {"__name__": "__collision__"})
    stdout = out.getvalue()
    return {
        "collisions": fc_engine._extract_collisions(stdout) or [],
        "clearances": json.loads(fc_engine._CLEARANCES_RE.search(stdout).group(1)),
        "errors": json.loads(fc_engine._CHECK_ERRORS_RE.search(stdout).group(1)),
        "skipped": int(re.search(re.escape(_MARKER) + r"_BBOX_SKIPPED (\d+)", stdout).group(1)),
        "checked": fc_engine._extract_checked_count(stdout),
    }


def test_disjoint_boxes_skip_common_and_overlapping_pairs_are_still_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str]] = []
    a = _Obj("A", _Shape("A", _Box((0, 0, 0), (10, 10, 10)), calls, overlap=125.0))
    b = _Obj("B", _Shape("B", _Box((5, 5, 5), (15, 15, 15)), calls, overlap=125.0))
    far = _Obj("Far", _Shape("Far", _Box((100, 0, 0), (110, 10, 10)), calls))

    report = _run_script(tmp_path, monkeypatch, [a, b, far])

    assert calls == [("A", "B")]  # A/Far and B/Far never reached the boolean
    assert report["skipped"] == 2
    assert report["collisions"] == [{"part_a": "A", "part_b": "B", "overlap_volume": 125.0}]
    distances = {(c["part_a"], c["part_b"]): c["distance_mm"] for c in report["clearances"]}
    assert distances == {("A", "B"): 0.0, ("A", "Far"): 90.0, ("B", "Far"): 85.0}
    assert report["errors"] == []
    assert report["checked"] == 3


def test_boxes_that_only_touch_still_run_the_boolean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []
    a = _Obj("A", _Shape("A", _Box((0, 0, 0), (10, 10, 10)), calls))
    b = _Obj("B", _Shape("B", _Box((10, 0, 0), (20, 10, 10)), calls))

    report = _run_script(tmp_path, monkeypatch, [a, b])

    assert calls == [("A", "B")]
    assert (report["skipped"], report["collisions"]) == (0, [])


def test_an_unreadable_bounding_box_falls_back_to_the_boolean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The broadphase may only ever skip work, never hide a pair: a box that
    can't be read sends the pair through common() as before."""
    calls: list[tuple[str, str]] = []
    a = _Obj("A", _Shape("A", _Box((0, 0, 0), (10, 10, 10), broken=True), calls))
    far = _Obj("Far", _Shape("Far", _Box((100, 0, 0), (110, 10, 10)), calls))

    def failing_common(other: _Shape) -> Any:
        calls.append(("A", other.name))
        raise RuntimeError("degenerate solid")

    a.Shape.common = failing_common

    report = _run_script(tmp_path, monkeypatch, [a, far])

    assert calls == [("A", "Far")]
    assert report["skipped"] == 0
    assert report["errors"] == [{"part_a": "A", "part_b": "Far", "stage": "intersection", "error": "degenerate solid"}]


# -- live: real FreeCAD -------------------------------------------------------------------


@pytest.mark.e2e
@_LIVE
def test_live_broadphase_skips_distant_parts_and_still_flags_the_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dana.platform.mock import MockControlPlane
    from dana.platform.win32 import RealFreeCADEngine

    set_session_id(f"broadphase-{uuid.uuid4().hex[:8]}")
    rd._set_has_plan(True, "broadphase test")
    monkeypatch.setattr(fc_engine, "_OUTPUT_DIR", tmp_path / "freecad_output")
    monkeypatch.setattr(fc_engine, "_EXPORT_DIR", tmp_path / "exports")
    monkeypatch.setenv("DANA_HEADLESS", "true")
    monkeypatch.delenv("DANA_OS_DRY_RUN", raising=False)
    stdouts: list[str] = []
    real_run = fc_engine._run_freecad_script

    def recording_run(script: str, **kwargs: Any) -> dict[str, Any]:
        result = real_run(script, **kwargs)
        stdouts.append(result.get("stdout") or "")
        return result

    monkeypatch.setattr(fc_engine, "_run_freecad_script", recording_run)
    cad, control_plane = RealFreeCADEngine(), MockControlPlane()

    def call(tool_id: str, **arguments: Any) -> dict[str, Any]:
        result = rd.dispatch_tool_call(ToolCall(tool_id=tool_id, arguments=arguments), cad, control_plane)
        assert result.ok or tool_id == "validate_assembly_collisions", (tool_id, result.message)
        return result.payload

    try:
        call("create_freecad_box", name="Left", length=10, width=10, height=10)
        call("create_freecad_box", name="Clash", length=10, width=10, height=10, placement_x=5)
        call("create_freecad_box", name="Far", length=10, width=10, height=10, placement_x=200)
        call("create_freecad_assembly", name="Rig")
        call("add_parts_to_assembly", assembly_name="Rig", part_names=["Left", "Clash", "Far"])
        stdouts.clear()

        report = call("validate_assembly_collisions", assembly_name="Rig")
    finally:
        set_session_id(DEFAULT_SESSION_ID)

    skipped = [int(m.group(1)) for s in stdouts for m in [re.search(re.escape(_MARKER) + r"_BBOX_SKIPPED (\d+)", s)] if m]
    assert skipped == [2]  # Left/Far and Clash/Far
    assert [{c["part_a"], c["part_b"]} for c in report["collisions"]] == [{"Left", "Clash"}]
    assert report["collisions"][0]["overlap_volume"] == pytest.approx(500.0, rel=1e-6)
    distances = {frozenset((c["part_a"], c["part_b"])): c["distance_mm"] for c in report["clearances"]}
    assert distances[frozenset(("Left", "Far"))] == pytest.approx(190.0, abs=1e-6)
    assert distances[frozenset(("Clash", "Far"))] == pytest.approx(185.0, abs=1e-6)
    assert report["checked_members"] == 3
