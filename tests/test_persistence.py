"""Atomic persistence of the Sketcher/PartDesign tools in
dana.plugins.freecad.engine.

Each op runs as one FreeCADCmd script: open Session_Active.FCStd -> op code
-> _check_feature (Tip + doc.recompute() + validity check, raises on
failure) -> save. An exception anywhere before the save ends the script, so
the file on disk keeps its last good state, which is the "restore" half of an
atomic operation, with no in-process document to roll back.

The structural tests below pin that ordering for every op and run anywhere.
The live tests force a failure AFTER an op's code and recompute have run
(the latest point it can fail), reload the saved document in FreeCADCmd, and
require the earlier Pad to still be there as a real solid.
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

import pytest

from dana.plugins.freecad import engine
from dana.session_context import DEFAULT_SESSION_ID, set_session_id

_OP_SENTINEL = "# <op code>\n"
_INJECTED = "injected failure after recompute"


# -- structural: every op saves last, only after its own checks -------------------


@pytest.mark.parametrize("op", sorted(engine._PARTDESIGN_OPS))
def test_save_is_the_last_document_write_in_every_op_script(op: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine, "_partdesign_code", lambda _op, **_fmt: _OP_SENTINEL)
    script, _path = engine._partdesign_script(op)
    save = engine._SESSION_SAVE_SNIPPET
    assert script.count(save) == 1
    assert script.index(_OP_SENTINEL) < script.index(save)
    # Nothing after the save touches the document again.
    tail = script[script.index(save) + len(save) :]
    assert "recompute" not in tail and "addObject" not in tail and ".save" not in tail


@pytest.mark.parametrize(
    "op", ["create_pad", "create_pocket", "pattern", "create_sweep", "create_loft"]
)
def test_every_feature_op_recomputes_and_validates_before_saving(op: str) -> None:
    body = engine._PARTDESIGN_OPS[op][0]
    assert "_check_feature(doc," in body
    assert "doc.recompute()" in engine._PARTDESIGN_HELPERS.split("def _check_feature", 1)[1].split("\ndef ", 1)[0]


# -- live: a failure after recompute leaves the last good part on disk ------------

_requires_freecad = pytest.mark.skipif(
    engine.detect_freecadcmd() is None,
    reason="real FreeCADCmd not found (PATH or DANA_FREECADCMD_PATH) — this test drives the real engine",
)


@pytest.fixture
def live_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(engine, "_OUTPUT_DIR", tmp_path / "freecad_output")
    monkeypatch.setattr(engine, "_EXPORT_DIR", tmp_path / "exports")
    monkeypatch.setenv("DANA_HEADLESS", "true")
    monkeypatch.delenv("DANA_OS_DRY_RUN", raising=False)
    set_session_id(f"persistence-{uuid.uuid4().hex[:8]}")
    yield
    set_session_id(DEFAULT_SESSION_ID)


def _inject_failure(monkeypatch: pytest.MonkeyPatch, target_op: str) -> None:
    """Make ``target_op`` raise right after its own code (including
    _check_feature's recompute) has run, just before the save."""
    original = engine._partdesign_code

    def failing(op: str, **fmt: Any) -> str:
        code = original(op, **fmt)
        if op == target_op:
            code += f"raise RuntimeError({_INJECTED!r})\n"
        return code

    monkeypatch.setattr(engine, "_partdesign_code", failing)


def _ok(raw: str) -> dict[str, Any]:
    result = json.loads(raw)
    assert result.get("ok") is True, result
    return result


def _rect(x0: float, y0: float, x1: float, y1: float) -> list[dict[str, Any]]:
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return [{"type": "line", "start": list(corners[i]), "end": list(corners[(i + 1) % 4])} for i in range(4)]


def _reload() -> dict[str, dict[str, Any]]:
    """Reopen the saved session document in a fresh FreeCADCmd process and
    describe every object in it."""
    script = (
        "import FreeCAD as App, json\n"
        f"doc = App.openDocument({str(engine._session_document_path())!r})\n"
        "doc.recompute()\n"
        "out = {}\n"
        "for o in doc.Objects:\n"
        "    s = getattr(o, 'Shape', None)\n"
        "    parent = o.getParentGeoFeatureGroup()\n"
        "    out[o.Name] = {'type': o.TypeId, 'valid': o.isValid(),\n"
        "                   'solids': len(s.Solids) if s is not None else 0,\n"
        "                   'volume': s.Volume if s is not None and s.Solids else 0.0,\n"
        "                   'body': parent.Name if parent is not None else None,\n"
        "                   'tip': o.Tip.Name if getattr(o, 'Tip', None) is not None else None}\n"
        f"print({engine._OK_MARKER!r} + '_RELOAD ' + json.dumps(out))\n"
    )
    result = engine._run_freecad_script(script)
    assert result["ok"], result
    match = re.search(re.escape(engine._OK_MARKER + "_RELOAD ") + r"(\{.*\})", result["stdout"])
    assert match, result["stdout"][-2000:]
    return json.loads(match.group(1))


def _assert_pad_survived(objects: dict[str, dict[str, Any]], pad: str, tip: str) -> None:
    assert objects[pad]["type"] == "PartDesign::Pad"
    assert objects[pad]["body"] == "Body"
    assert objects[pad]["valid"] is True
    assert objects[pad]["solids"] >= 1 and objects[pad]["volume"] > 0  # a solid, not a 2D wireframe
    assert objects["Body"]["tip"] == tip
    assert objects["Body"]["solids"] >= 1 and objects["Body"]["volume"] > 0


def _plate() -> str:
    _ok(engine.create_sketch("Plate", "XY", _rect(-30, -30, 30, 30)))
    return _ok(engine.create_pad("Plate", 4))["name"]


def _hole() -> None:
    _ok(engine.create_sketch("Hole", "XY", [{"type": "circle", "center": [15, 0], "radius": 3}]))


def _failing_call(op: str, pocket: str | None) -> Any:
    """The call that should fail, plus any extra sketches it needs (built
    before the failure is armed)."""
    if op == "create_pad":
        _ok(engine.create_sketch("Boss", "XY", [{"type": "circle", "center": [0, 0], "radius": 4}]))
        return lambda: engine.create_pad("Boss", 10)
    if op == "create_pocket":
        _hole()
        return lambda: engine.create_pocket("Hole", 1, through_all=True)
    if op == "polar":
        return lambda: engine.create_polar_pattern(pocket, 4)
    if op == "linear":
        return lambda: engine.create_linear_pattern(pocket, 3, 20, direction="Y")
    if op == "create_sweep":
        _ok(engine.create_sketch("Section", "XY", [{"type": "circle", "center": [0, 0], "radius": 2}]))
        _ok(engine.create_sketch("Spine", "XZ", [{"type": "line", "start": [0, 0], "end": [0, 20]}]))
        return lambda: engine.create_sweep("Section", "Spine")
    assert op == "create_loft"
    _ok(engine.create_sketch("Base", "XY", _rect(-10, -10, 10, 10)))
    _ok(engine.create_sketch("Top", "XY", [{"type": "circle", "center": [0, 0], "radius": 5}]))
    _ok(engine.modify_parameter("Top", "Placement", [0, 0, 15]))
    return lambda: engine.create_loft(["Base", "Top"])


@pytest.mark.parametrize(
    ("op", "engine_op"),
    [
        ("create_pad", "create_pad"),
        ("create_pocket", "create_pocket"),
        ("polar", "pattern"),
        ("linear", "pattern"),
        ("create_sweep", "create_sweep"),
        ("create_loft", "create_loft"),
    ],
)
@pytest.mark.e2e
@_requires_freecad
@pytest.mark.usefixtures("live_session")
def test_failure_after_pad_leaves_a_loadable_document_with_the_pad(
    op: str, engine_op: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    pad = _plate()
    pocket = None
    if op in ("polar", "linear"):
        _hole()
        pocket = _ok(engine.create_pocket("Hole", 1, through_all=True))["name"]
    tip = pocket or pad
    call = _failing_call(op, pocket)

    session = engine._session_document_path()
    before_bytes = session.read_bytes()
    before = _reload()
    _assert_pad_survived(before, pad, tip)

    _inject_failure(monkeypatch, engine_op)
    result = json.loads(call())
    assert result["ok"] is False
    assert _INJECTED in result["error"]

    # Restored = never overwritten: the file is byte-for-byte the last good save.
    assert session.read_bytes() == before_bytes
    after = _reload()
    assert set(after) == set(before)
    _assert_pad_survived(after, pad, tip)


@pytest.mark.e2e
@_requires_freecad
def test_a_failing_script_runs_once_and_reports_its_own_error() -> None:
    """FreeCADCmd re-runs a script that raised as __main__ in the same
    process; _run_freecad_script must stop that so the first error is the
    one reported and nothing from a second pass can reach disk."""
    result = engine._run_freecad_script('print("PASS", __name__)\nraise RuntimeError("first failure")\n')
    passes = [line for line in result["stdout"].splitlines() if line.startswith("PASS")]
    assert passes == ["PASS __main__"]
    assert result["ok"] is False
    assert "first failure" in result["error"]
    assert result["error"].startswith(engine._FREECAD_INTERNAL_EXCEPTION_BANNER)


@pytest.mark.e2e
@_requires_freecad
def test_a_succeeding_script_still_runs_normally() -> None:
    result = engine._run_freecad_script(
        f'import sys\nprint({engine._OK_MARKER!r})\nprint("argv ok", len(sys.argv) >= 1)\n'
    )
    assert result["ok"] is True, result
    assert "argv ok True" in result["stdout"]
