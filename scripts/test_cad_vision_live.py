#!/usr/bin/env python3
"""Live diagnostic: verify Dana's VLM vision pipeline can read the FreeCAD viewport.

Generates a real cylinder via FreeCAD, sends the GUI to the secondary
monitor WITHOUT stealing OS focus, screenshots that window directly (not a
full-monitor grab — works whether or not it's the active window), and asks
a Vision-Language Model (local Ollama first, cloud OpenAI on fallback) to
confirm it can actually see the geometry — the same
``analyze_cad_blueprint``/``verify_cad_rendering`` path the CAD_MICRO_LOOP_RULE
tells the agent to use after every CAD operation.

Usage (from repo root)::

    python scripts/test_cad_vision_live.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_CYLINDER_DIMS = {"radius": 6.0, "height": 15.0}
_SETTLE_S = 3.0
_EXPECTED_SPEC = {"geometry": "cylinder", "visible": True}


def _banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def _freecad_pids() -> set[int]:
    import psutil

    pids = set()
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            if (proc.info.get("name") or "").lower() == "freecad.exe":
                pids.add(int(proc.info["pid"]))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return pids


def main() -> int:
    from dana.security.dry_run import is_dry_run_enabled

    if is_dry_run_enabled():
        print("[SKIP] DANA_OS_DRY_RUN is set — this test needs real FreeCAD + VLM calls.")
        return 1

    from dana.plugins.freecad import engine as fe
    from dana.tools.cad_vision import analyze_cad_blueprint, capture_cad_viewport, verify_cad_rendering

    pids_before = _freecad_pids()

    try:
        # --- Step 1: generate a cylinder ---------------------------------
        _banner("STEP 1: create_freecad_cylinder(radius=6, height=15)")
        cyl = json.loads(
            fe.create_cylinder(_CYLINDER_DIMS["radius"], _CYLINDER_DIMS["height"], name="VisionTestCylinder")
        )
        if not cyl.get("ok"):
            print(f"[FAIL] create_cylinder failed: {cyl.get('error')}")
            return 1
        print(f"[OK] cylinder created -> {cyl['path']} (gui_shown={cyl.get('gui_shown')})")

        # --- Step 2: send it to the secondary monitor, zero focus stolen --
        _banner("STEP 2: show_in_freecad_gui — send to Display 2, no OS focus stolen")
        gui = json.loads(fe.show_in_freecad_gui(cyl["path"]))
        print(f"[INFO] show_in_freecad_gui -> {json.dumps(gui, indent=2)}")
        if not gui.get("title_matched") or not gui.get("moved_to_secondary"):
            print(
                "[WARN] title_matched="
                f"{gui.get('title_matched')} moved_to_secondary={gui.get('moved_to_secondary')} "
                "— a toast was sent instead. The window-targeted capture below "
                "still runs regardless, so this is informational, not fatal."
            )

        # --- Step 3: let the UI settle ------------------------------------
        _banner(f"STEP 3: waiting {_SETTLE_S:.0f}s for the UI to settle")
        time.sleep(_SETTLE_S)

        # --- Step 4: screenshot the viewport -------------------------------
        _banner("STEP 4: capture_cad_viewport() — screenshot the live FreeCAD window")
        capture = capture_cad_viewport()
        if not capture.get("ok"):
            print(f"[FAIL] viewport capture failed: {capture.get('error')}")
            return 1
        print(f"[OK] screenshot saved: {capture.get('path')} (window_found={capture.get('window_found')})")

        # --- Step 5: ask the VLM directly what it sees ---------------------
        _banner("STEP 5: analyze_cad_blueprint() — raw VLM read of the screenshot")
        import base64

        with open(capture["path"], "rb") as fh:
            image_b64 = base64.b64encode(fh.read()).decode("ascii")
        analysis_raw = analyze_cad_blueprint(image_b64)
        print("[VLM RAW RESPONSE]")
        print(analysis_raw)
        try:
            analysis = json.loads(analysis_raw)
        except json.JSONDecodeError:
            analysis = {}
        if analysis.get("ok"):
            print(f"\n[INFO] VLM provider used: {analysis.get('provider')}")
            print(f"[INFO] VLM summary: {analysis.get('summary')}")
        else:
            print(f"\n[WARN] analyze_cad_blueprint reported failure: {analysis.get('error')}")

        # --- Step 6: verify against the expected spec -----------------------
        _banner(f"STEP 6: verify_cad_rendering(expected_spec_json={_EXPECTED_SPEC})")
        verify_raw = verify_cad_rendering(json.dumps(_EXPECTED_SPEC))
        print("[VLM VERIFICATION RESPONSE]")
        print(verify_raw)
        try:
            verify = json.loads(verify_raw)
        except json.JSONDecodeError:
            verify = {}

        _banner("REPORT")
        if not verify.get("ok"):
            print(f"RESULT: INCONCLUSIVE — verify_cad_rendering failed: {verify.get('error') or verify.get('detail')}")
            return 1

        # The blueprint prompt's structured schema uses 2D AutoCAD-drawing
        # vocabulary (line/circle/polyline/solid) with no "cylinder" type at
        # all — a 3D FreeCAD primitive can only ever show up in the model's
        # free-text summary, not that fixed entity-type list. Check both.
        analysis = verify.get("analysis") or {}
        found_types = [str(e.get("type") or "").lower() for e in analysis.get("entities") or []]
        summary = str(analysis.get("summary") or "")
        cylinder_seen = any("cylind" in t for t in found_types) or "cylind" in summary.lower()
        print(f"Entities the VLM reported: {found_types or '(none)'}")
        print(f"VLM summary: {summary or '(none)'}")
        print(f"Did it identify a cylinder? {'YES' if cylinder_seen else 'NO'}")
        print(f"match_ratio vs expected spec: {verify.get('match_ratio')}")
        return 0 if cylinder_seen else 1
    finally:
        pids_now = _freecad_pids()
        spawned_by_us = pids_now - pids_before
        if spawned_by_us:
            import psutil

            print(f"\n[CLEANUP] closing FreeCAD.exe process(es) this script spawned: {sorted(spawned_by_us)}")
            for pid in spawned_by_us:
                try:
                    psutil.Process(pid).kill()
                except psutil.NoSuchProcess:
                    pass


if __name__ == "__main__":
    sys.exit(main())
