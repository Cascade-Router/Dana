#!/usr/bin/env python3
"""Headless regression check for Continuous Parametric UV Placement —
bypasses the LLM entirely by calling ``dana.plugins.freecad.engine``
directly.

``apply_assembly_constraint``'s ``uv_tensor`` parameter replaces the old
``semantic_alignment`` 5-token enum (``"center"``/``"top_left"``/...),
which itself replaced an even older ``corner_offset`` escape hatch (a raw
float for a face's u/v parameter that let the LLM hallucinate a value
outside the face's real bounds, e.g. ``v: 20`` on a face only 10mm tall).
This script proves the continuous replacement is still deterministic: it
spawns a 40x20x10 box and a cylinder, mates the cylinder to the box's Face1
once per representative ``uv_tensor`` value (the four corners, dead center,
and one continuous in-between blend), and asserts the resulting
``Placement.Base`` always lands inside that face's real ``BoundBox`` — never
floating off in space.

Usage (from repo root)::

    python scripts/test_uv_tensor.py
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dana.api.sessions import new_session_id  # noqa: E402
from dana.plugins.freecad import engine  # noqa: E402
from dana.session_context import set_session_id  # noqa: E402

# The five old semantic_alignment tokens, expressed as their equivalent
# uv_tensor value (see _face_alignment_delta's old _uv_by_alignment dict,
# now retired: "top_left" -> (u_lo, v_hi) i.e. u=0/v=1, etc.), plus one
# genuinely continuous in-between blend the old discrete enum could never
# express at all.
_UV_TENSORS = (
    (0.5, 0.5),  # former "center"
    (0.0, 1.0),  # former "top_left"
    (1.0, 1.0),  # former "top_right"
    (0.0, 0.0),  # former "bottom_left"
    (1.0, 0.0),  # former "bottom_right"
    (0.25, 0.75),  # continuous blend -- no discrete-token equivalent
)

# Box's Face1 (-X, per create_box's verified fixed face order) mated
# against the cylinder's flat bottom end-cap (Face2, confirmed live for
# Part::Cylinder: Face1=lateral, Face2=bottom disc, Face3=top disc).
_BOX_FACE = "Face1"
_WHEEL_FACE = "Face2"

_PROBE_SCRIPT = """\
import FreeCAD as App

doc = App.openDocument({session_path!r})
box = doc.getObject({box_name!r})
face = box.Shape.Faces[{face_index!r} - 1]
bb = face.BoundBox
wheel = doc.getObject({wheel_name!r})
p = wheel.Placement.Base
print("{marker}_BOUNDS " + str([bb.XMin, bb.XMax, bb.YMin, bb.YMax, bb.ZMin, bb.ZMax]))
print("{marker}_PLACEMENT " + str([p.x, p.y, p.z]))
print("{marker} ok")
"""


def _probe(session_path: Path, box_name: str, wheel_name: str) -> tuple[list[float], list[float]]:
    script = _PROBE_SCRIPT.format(
        session_path=str(session_path),
        box_name=box_name,
        wheel_name=wheel_name,
        face_index=int(_BOX_FACE.removeprefix("Face")),
        marker="PROBE",
    )
    result = engine._run_freecad_script(script, require_marker=False)
    if not result["ok"]:
        raise RuntimeError(f"probe script failed: {result['error']}")
    stdout = result["stdout"]
    bounds = ast.literal_eval(re.search(r"PROBE_BOUNDS (\[.*\])", stdout).group(1))
    placement = ast.literal_eval(re.search(r"PROBE_PLACEMENT (\[.*\])", stdout).group(1))
    return bounds, placement


def main() -> int:
    set_session_id(f"test-uv-tensor-{new_session_id()}")

    box = json.loads(engine.create_box(40.0, 20.0, 10.0, name="Chassis"))
    assert box["ok"], box
    box_name = box["name"]

    cyl = json.loads(engine.create_cylinder(2.0, 3.0, name="Wheel"))
    assert cyl["ok"], cyl
    wheel_name = cyl["name"]

    asm = json.loads(engine.create_assembly("Asm"))
    assert asm["ok"], asm
    asm_name = asm["name"]

    added = json.loads(engine.add_parts_to_assembly(asm_name, [box_name, wheel_name]))
    assert added["ok"], added

    session_path = Path(box["path"])
    failures = []
    for uv_tensor in _UV_TENSORS:
        result = json.loads(
            engine.apply_assembly_constraint(
                asm_name,
                box_name,
                _BOX_FACE,
                wheel_name,
                _WHEEL_FACE,
                "Coincident",
                uv_tensor=list(uv_tensor),
            )
        )
        if not result["ok"]:
            failures.append(f"{uv_tensor}: apply_assembly_constraint failed: {result['error']}")
            continue

        (x_min, x_max, y_min, y_max, z_min, z_max), (px, py, pz) = _probe(
            session_path, box_name, wheel_name
        )
        in_bounds = (
            x_min - 1e-6 <= px <= x_max + 1e-6
            and y_min - 1e-6 <= py <= y_max + 1e-6
            and z_min - 1e-6 <= pz <= z_max + 1e-6
        )
        status = "OK" if in_bounds else "FAIL"
        print(
            f"[{status}] uv_tensor={list(uv_tensor)!r} -> Placement.Base="
            f"({px:.3f}, {py:.3f}, {pz:.3f}) vs Face1 bounds "
            f"x=[{x_min:.3f}, {x_max:.3f}] y=[{y_min:.3f}, {y_max:.3f}] z=[{z_min:.3f}, {z_max:.3f}]"
        )
        if not in_bounds:
            failures.append(f"{uv_tensor}: Placement.Base ({px}, {py}, {pz}) outside Face1 BoundBox")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f" - {f}")
        return 1

    print(f"\nAll {len(_UV_TENSORS)} uv_tensor values landed inside Face1's bounding box.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
