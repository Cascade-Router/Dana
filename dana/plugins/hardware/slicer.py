"""``slice_stl_to_gcode`` — slices a model into G-code with the PrusaSlicer
command line, the middle step of check_printability -> slice_stl_to_gcode ->
dispatch_to_printer.

Runs::

    <prusa-slicer> --export-gcode --load <profile.ini> --output <part>_sliced.gcode <part>.stl

* **Profiles are the user's own exported configs.** ``printer_profile`` names
  ``<workspace>/slicer_profiles/<name>.ini`` (or ``DANA_SLICER_PROFILE_DIR``),
  exported from PrusaSlicer's File > Export > Export Config with the right
  printer, filament and print settings selected. Nothing is guessed: without
  ``--load`` PrusaSlicer would slice for its generic default printer (wrong bed,
  temperatures and start G-code), so a missing profile is an error, not a
  fallback.
* **The slicer** is ``DANA_SLICER_PATH``, else the first of
  ``prusa-slicer-console`` / ``prusa-slicer`` / ``PrusaSlicer`` on PATH, else
  PrusaSlicer's default Windows install location.
* **Files stay in the sandbox:** the model must resolve inside the agent
  workspace (or a mounted directory), and the G-code is written next to it, so
  dispatch_to_printer can send it.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from dana.plugins.os import file_system
from dana.plugins.os.file_system import PathEscapeError, resolve_sandboxed_path

DEFAULT_PROFILE = "mk4_default"
MODEL_SUFFIXES = frozenset({".stl", ".3mf", ".obj"})
SLICE_TIMEOUT_S = 600.0
_PROFILE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_SLICER_NAMES = ("prusa-slicer-console", "prusa-slicer", "PrusaSlicer")
_WINDOWS_DEFAULT = Path(r"C:\Program Files\Prusa3D\PrusaSlicer\prusa-slicer-console.exe")


def find_slicer() -> str:
    """The slicer executable to run. Falls back to the bare name
    ``prusa-slicer`` so a missing install surfaces as FileNotFoundError."""
    override = (os.environ.get("DANA_SLICER_PATH") or "").strip()
    if override:
        return override
    for name in _SLICER_NAMES:
        found = shutil.which(name)
        if found:
            return found
    if os.name == "nt" and _WINDOWS_DEFAULT.is_file():
        return str(_WINDOWS_DEFAULT)
    return "prusa-slicer"


def profile_dir() -> Path:
    override = (os.environ.get("DANA_SLICER_PROFILE_DIR") or "").strip()
    return Path(override) if override else file_system._SANDBOX_ROOT / "slicer_profiles"


def output_path_for(model: Path) -> Path:
    return model.with_name(f"{model.stem}_sliced.gcode")


def _tail(text: str | bytes | None, lines: int = 8) -> str:
    if isinstance(text, bytes):
        text = text.decode("utf-8", "replace")
    return "\n".join((text or "").strip().splitlines()[-lines:])


def slice_stl_to_gcode(
    stl_filepath: str,
    printer_profile: str = DEFAULT_PROFILE,
    *,
    allowed_mounts: list[str] | None = None,
) -> dict[str, Any]:
    """Slice ``stl_filepath`` with ``printer_profile``. Returns ``{"ok": True,
    "gcode_path": <absolute path>, ...}`` or ``{"ok": False, "error": ...}``."""
    try:
        model = resolve_sandboxed_path(stl_filepath, allowed_mounts)
    except PathEscapeError as exc:
        return {"ok": False, "error": f"slice_stl_to_gcode: {exc}"}
    if model.suffix.lower() not in MODEL_SUFFIXES:
        return {
            "ok": False,
            "error": f"slice_stl_to_gcode: {model.name} is not a model file ({', '.join(sorted(MODEL_SUFFIXES))})",
        }
    if not model.is_file():
        return {"ok": False, "error": f"slice_stl_to_gcode: no such file: {model}"}

    profile_name = (printer_profile or DEFAULT_PROFILE).strip()
    if not _PROFILE_NAME_RE.fullmatch(profile_name):
        return {"ok": False, "error": f"slice_stl_to_gcode: invalid printer_profile name {profile_name!r}"}
    profile = profile_dir() / f"{profile_name}.ini"
    if not profile.is_file():
        return {
            "ok": False,
            "error": (
                f"slice_stl_to_gcode: no slicer profile '{profile_name}' at {profile}. In PrusaSlicer, select "
                "the printer, filament and print settings, then File > Export > Export Config and save it there."
            ),
        }

    output = output_path_for(model)
    slicer = find_slicer()
    command = [slicer, "--export-gcode", "--load", str(profile), "--output", str(output), str(model)]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=SLICE_TIMEOUT_S)
    except FileNotFoundError:
        return {
            "ok": False,
            "error": (
                f"slice_stl_to_gcode: slicer not found ({slicer}). Install PrusaSlicer or set DANA_SLICER_PATH "
                "to its command-line executable (prusa-slicer-console.exe on Windows)."
            ),
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"slice_stl_to_gcode: slicer timed out after {SLICE_TIMEOUT_S:g}s"}
    except subprocess.CalledProcessError as exc:
        detail = _tail(exc.stderr) or _tail(exc.stdout) or "no output"
        return {"ok": False, "error": f"slice_stl_to_gcode: slicer failed (exit {exc.returncode}): {detail}"}

    if not output.is_file():
        return {"ok": False, "error": f"slice_stl_to_gcode: slicer exited cleanly but wrote no G-code at {output}"}
    return {
        "ok": True,
        "gcode_path": str(output.resolve()),
        "stl_path": str(model),
        "printer_profile": profile_name,
        "slicer": slicer,
    }


__all__ = ("DEFAULT_PROFILE", "find_slicer", "output_path_for", "profile_dir", "slice_stl_to_gcode")
