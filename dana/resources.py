"""Bundled resource path resolution for dev and PyInstaller freezes.

Kept separate from ``dana.paths`` so ToolForge / offline routing constants stay
untouched. Prefer this helper (or ``dana.ui.logo`` asset roots) for icons,
logos, and other packaged data files.
"""

from __future__ import annotations

import sys
from pathlib import Path


def frozen_meipass() -> Path | None:
    """Return ``sys._MEIPASS`` when running under PyInstaller, else None."""
    meipass = getattr(sys, "_MEIPASS", None)
    if not meipass:
        return None
    try:
        return Path(meipass).resolve()
    except Exception:  # noqa: BLE001
        return Path(str(meipass))


def get_resource_path(relative_path: str | Path) -> Path:
    """Resolve a packaged resource path (MEIPASS-aware).

    Search order:
      1. ``sys._MEIPASS / relative_path`` when frozen
      2. Project root (``dana.paths.PROJECT_ROOT`` or repo root via this file)
      3. Onedir directory next to ``sys.executable`` when ``sys.frozen``

    Returns the first existing file/dir match, otherwise the preferred
    candidate (MEIPASS when frozen, else project-root join) so callers can
    still inspect ``.is_file()`` / ``.is_dir()``.
    """
    rel = Path(relative_path)
    if rel.is_absolute():
        return rel

    # Normalize "dana\\assets\\x" and "./dana/assets/x"
    parts = [p for p in rel.parts if p not in ("", ".", "..")]
    rel = Path(*parts) if parts else Path()

    candidates: list[Path] = []
    meipass = frozen_meipass()
    if meipass is not None:
        candidates.append(meipass / rel)

    try:
        from dana.paths import PROJECT_ROOT

        candidates.append(Path(PROJECT_ROOT) / rel)
    except Exception:  # noqa: BLE001
        pass

    # dana/resources.py → parents[1] is repo root in source trees; under a
    # freeze it is typically the same as ``_MEIPASS``.
    pkg_root = Path(__file__).resolve().parents[1]
    candidates.append(pkg_root / rel)

    if bool(getattr(sys, "frozen", False)):
        try:
            candidates.append(Path(sys.executable).resolve().parent / rel)
        except Exception:  # noqa: BLE001
            pass

    seen: set[str] = set()
    ordered: list[Path] = []
    for cand in candidates:
        try:
            key = str(cand.resolve()) if cand.exists() else str(cand)
        except Exception:  # noqa: BLE001
            key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(cand)

    for cand in ordered:
        try:
            if cand.exists():
                return cand.resolve()
        except Exception:  # noqa: BLE001
            if cand.is_file() or cand.is_dir():
                return cand

    if ordered:
        preferred = ordered[0]
        try:
            return preferred.resolve()
        except Exception:  # noqa: BLE001
            return preferred
    return Path(rel)
