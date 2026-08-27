"""Auto-provisioning for FreeCAD's community "Fasteners" workbench
(https://github.com/shaise/FreeCAD_FastenersWB) — ``insert_standard_part``'s
``part_type="fastener"`` needs ``import Fasteners`` to succeed inside the
disposable FreeCADCmd subprocess it shells out to (see ``standard_parts.py``'s
own ``_FASTENER_SCRIPT``), which only works if the workbench is installed
into FreeCAD's user Mod directory — normally a manual Tools -> Addon Manager
step. This module resolves that directory and, if the workbench isn't there
yet, installs it (git clone, falling back to a zip download) so a fresh
machine doesn't need that manual step before "fastener" parts work.

Deliberately does all of this in THIS (Dana's own venv) process rather than
inside the templated FreeCADCmd script: ``_FASTENER_SCRIPT`` is rendered via
``str.format()``, and the git-clone/zip-download/extract logic below is
ordinary Python full of ``{``/``}`` (f-strings, dict/set literals) — embedding
it there would mean escaping every brace as ``{{``/``}}``, fragile and hard
to review. Instead, ``ensure_fasteners_workbench()`` runs here, and the
resolved Mod directory is passed to the FreeCADCmd subprocess as the
``DANA_FREECAD_MOD_PATH`` env var (see ``standard_parts.py``'s
``insert_standard_part`` and ``engine.py``'s ``_run_freecad_script``
``extra_env`` param) — NOT ``PYTHONPATH``, which FreeCADCmd.exe's embedded
Python interpreter ignores on Windows (confirmed live). ``_FASTENER_SCRIPT``'s
own preamble reads that var and does the ``sys.path.append`` itself, in
plain statements with no ``{``/``}`` at all, so it's safe inside the
``.format()``-rendered template. The script's own ``import Fasteners``
(already guarded with its own ``FASTENERS_WORKBENCH_MISSING`` message for
the case this can't resolve anything) is untouched.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

_REPO_URL = "https://github.com/shaise/FreeCAD_FastenersWB.git"
_ZIP_URL = "https://github.com/shaise/FreeCAD_FastenersWB/archive/refs/heads/master.zip"
_WORKBENCH_DIR_NAME = "Fasteners"

_CLONE_TIMEOUT_S = 120.0
_DOWNLOAD_TIMEOUT_S = 60.0


def _versioned_and_flat(freecad_root: Path) -> list[Path]:
    """FreeCAD >= 1.0 nests user data under a version-specific profile
    directory directly inside the FreeCAD root (observed on a real 1.1
    install: ``.../FreeCAD/v1-1/Mod``, containing an Addon-Manager-installed
    ``fasteners/`` — NOT ``.../FreeCAD/Mod`` directly, which doesn't exist
    at all on that machine) so a version upgrade doesn't clobber an older
    version's Mod/config. Versioned dirs are tried first (newest-looking
    name first — lexicographic on ``v<major>-<minor>`` sorts newest-first
    for any realistic version range); the flat, unversioned ``Mod`` is
    still tried after for pre-1.0 installs that never had a profile dir."""
    versioned = sorted(freecad_root.glob("v*/Mod"), reverse=True)
    return [*versioned, freecad_root / "Mod"]


def _candidate_mod_dirs() -> list[Path]:
    """FreeCAD's per-OS user Mod directory/directories, most-likely-correct
    first."""
    system = platform.system()
    if system == "Windows":
        appdata = (os.environ.get("APPDATA") or "").strip()
        return _versioned_and_flat(Path(appdata) / "FreeCAD") if appdata else []
    # Linux/Mac: both base-directory conventions exist across FreeCAD
    # versions/distros — try the modern XDG path first, then the older
    # dotfile layout, each with the same versioned-profile-dir handling.
    home = Path.home()
    return [
        *_versioned_and_flat(home / ".local" / "share" / "FreeCAD"),
        *_versioned_and_flat(home / ".FreeCAD"),
    ]


def _is_installed(mod_dir: Path) -> bool:
    return (mod_dir / _WORKBENCH_DIR_NAME).is_dir()


def _clone_via_git(dest: Path) -> bool:
    if shutil.which("git") is None:
        return False
    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", _REPO_URL, str(dest)],
            capture_output=True,
            text=True,
            timeout=_CLONE_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and dest.is_dir()


def _download_via_zip(dest: Path) -> bool:
    """``git clone`` fallback: download the ``master`` branch archive and
    extract it in place of a real clone. GitHub's branch-archive zip always
    extracts to a single top-level ``<repo>-<branch>/`` directory, which
    gets renamed to ``dest`` (i.e. ``Fasteners``) once extracted."""
    try:
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            zip_path = tmp_dir / "fasteners.zip"
            with urllib.request.urlopen(_ZIP_URL, timeout=_DOWNLOAD_TIMEOUT_S) as resp:
                zip_path.write_bytes(resp.read())
            extract_dir = tmp_dir / "extracted"
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extract_dir)
            extracted_root = next((p for p in extract_dir.iterdir() if p.is_dir()), None)
            if extracted_root is None:
                return False
            shutil.move(str(extracted_root), str(dest))
    except (OSError, urllib.error.URLError, zipfile.BadZipFile):
        return False
    return dest.is_dir()


def ensure_fasteners_workbench() -> Path | None:
    """Return the Mod directory containing a usable "Fasteners" workbench,
    installing it (git clone, falling back to a zip download) if it isn't
    present in any candidate directory yet.

    Returns ``None`` if no Mod directory could be resolved/created or every
    install attempt failed — callers should treat that exactly like
    "workbench not installed" (the FreeCADCmd script's own
    ``FASTENERS_WORKBENCH_MISSING`` message already covers that case).
    """
    candidates = _candidate_mod_dirs()
    for mod_dir in candidates:
        if _is_installed(mod_dir):
            return mod_dir

    if not candidates:
        return None
    target_mod_dir = candidates[0]
    dest = target_mod_dir / _WORKBENCH_DIR_NAME

    if dest.exists():
        # A previous partial/failed attempt left something behind, or it's
        # a file rather than a directory — either way not a clean target
        # git clone would refuse anyway, so don't attempt to install here.
        return None

    try:
        target_mod_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None

    print(f"[fasteners_bootstrap] Fasteners workbench not found — installing into {dest} ...", flush=True)
    if _clone_via_git(dest):
        print("[fasteners_bootstrap] installed via git clone.", flush=True)
        return target_mod_dir

    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)  # partial git-clone leftovers

    if _download_via_zip(dest):
        print("[fasteners_bootstrap] installed via zip download.", flush=True)
        return target_mod_dir

    print("[fasteners_bootstrap] auto-install failed (no git, and zip download/extract failed).", flush=True)
    return None


__all__ = ("ensure_fasteners_workbench",)
