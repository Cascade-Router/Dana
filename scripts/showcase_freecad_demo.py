#!/usr/bin/env python3
"""Live showcase: non-intrusive GUI handoff + "Modify Existing" revision strategy.

Demonstrates the updated ``CAD_MICRO_LOOP_RULE`` end-to-end:

  1. create_freecad_box(...)                    -> new project file, auto-shown
  2. show_in_freecad_gui(box_path)               -> no duplicate FreeCAD.exe spawn
  3. wait 3s, then modify_existing_document(...) -> boolean cut saved to the
                                                     SAME path (no new .FCStd)
  4. show_in_freecad_gui(box_path)               -> reload/show the updated file
  5. simulated "focus blocked" state             -> verify the toast fallback fires
  6. capture_cad_viewport()                      -> screenshot for visual QA

Any FreeCAD.exe process this script itself spawns is closed on exit; any
FreeCAD.exe that was already running before the script started is left
completely untouched (never closes a pre-existing user session).

Usage (from repo root)::

    python scripts/showcase_freecad_demo.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_BOX_DIMS = {"length": 100.0, "width": 50.0, "height": 10.0}
_CYLINDER_DIMS = {"radius": 8.0, "height": 20.0}
_GUI_SETTLE_S = 3.0

_CUT_MODIFICATION_SCRIPT = """\
cyl = doc.addObject("Part::Cylinder", "Cylinder")
cyl.Radius = {radius}
cyl.Height = {cyl_height}
cyl.Placement = App.Placement(
    App.Vector({length} / 2.0, {width} / 2.0, -1.0), App.Rotation()
)
base = doc.getObject({box_name!r})
cut = doc.addObject("Part::Cut", "BoxCutCylinder")
cut.Base = base
cut.Tool = cyl
"""


def _banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def _parse_ok(observation: str, step_name: str) -> dict:
    try:
        data = json.loads(observation)
    except json.JSONDecodeError:
        print(f"[FAIL] {step_name}: non-JSON observation: {observation!r}")
        raise SystemExit(1)
    if not data.get("ok"):
        print(f"[FAIL] {step_name}: {data.get('error')}")
        raise SystemExit(1)
    print(f"[OK] {step_name} ->\n{json.dumps(data, indent=2)}")
    return data


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


def _test_notification_fallback(existing_path: str) -> bool:
    """Force a focus-steal failure and confirm the toast fallback actually fires."""
    import dana.middleware.toast_notify as toast_mod
    import dana.tools.os_control as os_control_mod
    from dana.plugins.freecad import engine as fe

    calls: list[tuple[str, str]] = []
    orig_toast = toast_mod.show_silent_toast_async
    orig_focus = os_control_mod.set_foreground_window

    def _fake_toast(title: str, message: str, **_kwargs) -> None:
        calls.append((title, message))

    def _fake_focus(_hwnd: int) -> bool:
        return False  # simulate Windows denying the focus-steal

    toast_mod.show_silent_toast_async = _fake_toast
    os_control_mod.set_foreground_window = _fake_focus
    try:
        result = json.loads(fe.show_in_freecad_gui(existing_path))
    finally:
        toast_mod.show_silent_toast_async = orig_toast
        os_control_mod.set_foreground_window = orig_focus

    if result.get("focused") is not False:
        print(f"[FAIL] expected focused=False under simulated block, got {result.get('focused')!r}")
        return False
    if result.get("spawned"):
        print("[FAIL] simulated-block call should not have spawned a new FreeCAD.exe (was already running)")
        return False
    if not calls:
        print("[FAIL] expected the toast notification fallback to fire, but it did not")
        return False
    print(f"[VERIFIED] simulated focus-blocked state correctly triggered toast: {calls[0]}")
    return True


def main() -> int:
    from dana.security.dry_run import is_dry_run_enabled

    if is_dry_run_enabled():
        print(
            "[SKIP] DANA_OS_DRY_RUN is set — this showcase needs a real "
            "FreeCADCmd/FreeCAD GUI run to verify actual geometry and the "
            "live viewport handoff. Unset DANA_OS_DRY_RUN and re-run."
        )
        return 1

    from dana.plugins.freecad import engine as fe
    from dana.tools.cad_vision import capture_cad_viewport

    pids_before = _freecad_pids()
    print(f"[INFO] FreeCAD.exe processes present before this run: {sorted(pids_before)}")

    try:
        # --- Step 1: generate the box (new project file) --------------------
        _banner("STEP 1: create_freecad_box(length=100, width=50, height=10)")
        box = _parse_ok(
            fe.create_box(
                _BOX_DIMS["length"], _BOX_DIMS["width"], _BOX_DIMS["height"], name="ShowcaseBox"
            ),
            "create_box",
        )
        expected_box_bbox = [0.0, 0.0, 0.0, _BOX_DIMS["length"], _BOX_DIMS["width"], _BOX_DIMS["height"]]
        if box.get("bounding_box") != expected_box_bbox:
            print(f"[FAIL] box bounding_box mismatch: got {box.get('bounding_box')}")
            return 1
        print(f"[VERIFIED] box bounding box correct. gui_shown={box.get('gui_shown')}")

        # --- Step 2: explicit GUI pop-up, no duplicate spawn -----------------
        _banner("STEP 2: show_in_freecad_gui(box_path) — must not spawn a duplicate process")
        pids_after_create = _freecad_pids()
        gui1 = _parse_ok(fe.show_in_freecad_gui(box["path"]), "show_in_freecad_gui (box)")
        pids_after_show = _freecad_pids()
        new_spawns = pids_after_show - pids_after_create
        if gui1.get("spawned") and not new_spawns:
            print("[FAIL] reported spawned=True but no new FreeCAD.exe process was observed")
            return 1
        if not gui1.get("spawned") and new_spawns:
            print(f"[FAIL] reported spawned=False but new FreeCAD.exe process(es) appeared: {new_spawns}")
            return 1
        print(f"[VERIFIED] no duplicate spawn (was_running={gui1.get('was_running')}, spawned={gui1.get('spawned')}, new_pids={new_spawns}).")

        # --- Step 3: wait, then modify the SAME file in place ----------------
        _banner(f"STEP 3: waiting {_GUI_SETTLE_S:.0f}s, then modify_existing_document (boolean cut)")
        time.sleep(_GUI_SETTLE_S)
        mod_script = _CUT_MODIFICATION_SCRIPT.format(
            radius=_CYLINDER_DIMS["radius"],
            cyl_height=_CYLINDER_DIMS["height"],
            length=_BOX_DIMS["length"],
            width=_BOX_DIMS["width"],
            box_name=box["name"],
        )
        mod = _parse_ok(
            fe.modify_existing_document(box["path"], mod_script),
            "modify_existing_document (boolean cut)",
        )
        if mod.get("path") != box.get("path"):
            print(f"[FAIL] modify_existing_document changed the file path: {mod.get('path')} != {box.get('path')}")
            return 1
        print("[VERIFIED] boolean cut saved back to the SAME file (no new .FCStd created).")

        # --- Step 4: reload/show the updated file in the GUI -----------------
        _banner("STEP 4: show_in_freecad_gui(box_path) — reload/show the updated (same) file")
        gui2 = _parse_ok(fe.show_in_freecad_gui(box["path"]), "show_in_freecad_gui (updated box)")
        print(f"[VERIFIED] GUI open-file request sent (was_running={gui2.get('was_running')}, spawned={gui2.get('spawned')}, focused={gui2.get('focused')}).")
        time.sleep(2.0)

        # --- Step 5: simulated focus-blocked state -> toast fallback --------
        _banner("STEP 5: simulate a blocked focus-steal — verify the toast fallback fires")
        if not _test_notification_fallback(box["path"]):
            return 1

        # --- Step 6: screenshot the live viewport for visual QA -------------
        _banner("STEP 6: capture_cad_viewport() — screenshot the live FreeCAD window")
        capture = capture_cad_viewport()
        if not capture.get("ok"):
            print(f"[FAIL] viewport capture failed: {capture.get('error')}")
            return 1
        print(f"[VERIFIED] screenshot saved: {capture.get('path')} (window_found={capture.get('window_found')})")

        pids_final = _freecad_pids()
        total_new = pids_final - pids_before
        print(f"\n[INFO] New FreeCAD.exe processes spawned across the whole run: {sorted(total_new)}")
        if len(total_new) > 1:
            print(f"[FAIL] expected at most 1 new FreeCAD.exe process across this multi-step run, got {len(total_new)}")
            return 1
        print("[VERIFIED] no process clutter — at most one new FreeCAD.exe across the entire multi-step operation.")

        _banner("SHOWCASE SUMMARY")
        print("All 6 steps passed:")
        print(f"  1. Box (new file)         -> {box['path']}")
        print(f"  2. GUI pop-up, no dup spawn -> new_pids={new_spawns}")
        print(f"  3. Modify existing (cut)  -> same path, {box['path']}")
        print(f"  4. GUI reload             -> was_running={gui2.get('was_running')}")
        print("  5. Simulated focus-block  -> toast fallback fired")
        print(f"  6. Viewport screenshot    -> {capture.get('path')}")
        return 0
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
