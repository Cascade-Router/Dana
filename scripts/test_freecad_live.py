#!/usr/bin/env python3
"""Live diagnostic: verify FreeCAD is installed and can generate real geometry.

Detects ``FreeCADCmd`` (via ``dana.plugins.freecad.engine``) and, as a
secondary signal, whether FreeCAD's own Python module is importable
directly in this interpreter. Then drives a real box + cylinder boolean
cut through FreeCADCmd and saves the result as ``dana_test.FCStd`` (and
optionally exports an ``.STL`` mesh) — a smoke test for the whole FreeCAD
Co-Pilot path (detection, subprocess execution, real geometry) in one run,
independent of pytest/mocks.

Usage (from repo root)::

    python scripts/test_freecad_live.py
    python scripts/test_freecad_live.py --export-stl
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_OK_MARKER = "DANA_FREECAD_LIVE_OK"

_BOOLEAN_CUT_SCRIPT = """\
import FreeCAD as App

doc = App.newDocument("DanaLiveTest")
box = doc.addObject("Part::Box", "Box")
box.Length = 10
box.Width = 10
box.Height = 10

cyl = doc.addObject("Part::Cylinder", "Cylinder")
cyl.Radius = 3
cyl.Height = 12
cyl.Placement = App.Placement(App.Vector(5, 5, -1), App.Rotation())

doc.recompute()
cut = doc.addObject("Part::Cut", "BoxCutCylinder")
cut.Base = box
cut.Tool = cyl
doc.recompute()

doc.saveAs({out_fcstd!r})
{export_stl_line}
print("{marker} fcstd=" + {out_fcstd!r})
"""

_EXPORT_STL_LINE = "cut.Shape.exportStl({out_stl!r})"


def _detect() -> tuple[str | None, str | None]:
    """Return ``(freecadcmd_path, freecad_pyd_hint)`` — either may be ``None``."""
    from dana.plugins.freecad.engine import detect_freecadcmd

    cmd_path = detect_freecadcmd()
    pyd_hint = None
    try:
        import importlib.util

        spec = importlib.util.find_spec("FreeCAD")
        if spec is not None and spec.origin:
            pyd_hint = spec.origin
    except (ImportError, ValueError):
        pyd_hint = None
    return cmd_path, pyd_hint


def run_live_test(*, export_stl: bool) -> int:
    print("=" * 72)
    print("Dana FreeCAD Live Diagnostic")
    print("=" * 72)

    cmd_path, pyd_hint = _detect()
    if cmd_path:
        print(f"[OK] FreeCADCmd detected: {cmd_path}")
    else:
        print(
            "[FAIL] FreeCADCmd not found on PATH, DANA_FREECADCMD_PATH, "
            "or common install dirs."
        )
    if pyd_hint:
        print(f"[INFO] FreeCAD Python module importable directly at: {pyd_hint}")
    else:
        print(
            "[INFO] FreeCAD Python module not importable in this interpreter "
            "(expected - FreeCADCmd runs its own bundled interpreter)."
        )

    if not cmd_path:
        print("\nRESULT: FAIL - cannot proceed without FreeCADCmd.")
        return 1

    tmp_dir = Path(tempfile.mkdtemp(prefix="dana_freecad_live_"))
    out_fcstd = tmp_dir / "dana_test.FCStd"
    out_stl = tmp_dir / "dana_test.stl"
    export_line = _EXPORT_STL_LINE.format(out_stl=str(out_stl)) if export_stl else ""
    script = _BOOLEAN_CUT_SCRIPT.format(
        out_fcstd=str(out_fcstd), export_stl_line=export_line, marker=_OK_MARKER
    )

    print("\n[RUN] Generating box + cylinder boolean cut via FreeCADCmd...")
    print(f"      Output document: {out_fcstd}")
    if export_stl:
        print(f"      STL export:      {out_stl}")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(script)
        script_path = tmp.name

    try:
        proc = subprocess.run(
            [cmd_path, script_path], capture_output=True, text=True, timeout=90, check=False
        )
    except subprocess.TimeoutExpired:
        print("\n[FAIL] FreeCADCmd timed out after 90s.")
        return 1
    finally:
        try:
            Path(script_path).unlink()
        except OSError:
            pass

    print("\n--- FreeCADCmd stdout ---")
    print(proc.stdout.strip() or "(empty)")
    if proc.stderr.strip():
        print("--- FreeCADCmd stderr ---")
        print(proc.stderr.strip())

    marker_seen = _OK_MARKER in (proc.stdout or "")
    ok = proc.returncode == 0 and marker_seen
    fcstd_exists = out_fcstd.is_file()
    stl_exists = out_stl.is_file() if export_stl else True

    print("\n" + "=" * 72)
    if ok and fcstd_exists and stl_exists:
        print(f"RESULT: PASS - {out_fcstd.name} generated ({out_fcstd.stat().st_size} bytes)")
        if export_stl:
            print(f"        {out_stl.name} exported ({out_stl.stat().st_size} bytes)")
        print(f"        Full output kept at: {tmp_dir}")
        return 0

    print("RESULT: FAIL - see stdout/stderr above.")
    print(
        f"        returncode={proc.returncode} marker_seen={marker_seen} "
        f"fcstd_exists={fcstd_exists} stl_exists={stl_exists}"
    )
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export-stl",
        action="store_true",
        help="Also export the cut solid as an .STL mesh.",
    )
    args = parser.parse_args()
    return run_live_test(export_stl=args.export_stl)


if __name__ == "__main__":
    sys.exit(main())
