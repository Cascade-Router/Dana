"""Keeps tests (and the Playwright mock backend) out of the real workspace.

Dana's runtime directories — ``freecad_output/``, ``exports/`` and
``agent_workspace/`` under the repo root — are copied into module-level
``Path`` constants all over the package (``engine._OUTPUT_DIR``,
``py_export._EXPORT_DIR``, ``sessions.SESSIONS_DIR``, ...). Patching them one
name at a time misses the next module that adds one, so this sweeps every
loaded ``dana`` module for ``Path`` globals under those directories instead.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

from dana.paths import PROJECT_ROOT

ISOLATED_DIRS: tuple[str, ...] = ("freecad_output", "exports", "agent_workspace")
_REAL_ROOTS: tuple[Path, ...] = tuple(PROJECT_ROOT / name for name in ISOLATED_DIRS)

# module name -> (module object, names of its Path globals seen under a real root)
_SCANNED: dict[str, tuple[ModuleType, tuple[str, ...]]] = {}


def _under_real_root(value: Path) -> bool:
    return any(value == root or value.is_relative_to(root) for root in _REAL_ROOTS)


def _path_globals(module: ModuleType) -> tuple[str, ...]:
    cached = _SCANNED.get(module.__name__)
    if cached is not None and cached[0] is module:
        return cached[1]
    names = tuple(
        attr for attr, value in list(vars(module).items()) if isinstance(value, Path) and _under_real_root(value)
    )
    _SCANNED[module.__name__] = (module, names)
    return names


def workspace_redirects(tmp_root: Path) -> Iterator[tuple[ModuleType, str, Path]]:
    """``(module, attribute, replacement)`` for every loaded ``dana`` module
    global that still points into a real workspace directory; the
    replacement keeps the same layout under ``tmp_root``."""
    for name, module in list(sys.modules.items()):
        if module is None or not (name == "dana" or name.startswith("dana.")):
            continue
        for attr in _path_globals(module):
            value = getattr(module, attr, None)
            if isinstance(value, Path) and _under_real_root(value):
                yield module, attr, tmp_root / value.relative_to(PROJECT_ROOT)


def snapshot_real_workspace() -> dict[str, tuple[int, int]]:
    """``relative path -> (size, mtime_ns)`` for every file in the real
    workspace directories."""
    out: dict[str, tuple[int, int]] = {}
    for root in _REAL_ROOTS:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file():
                st = path.stat()
                out[path.relative_to(PROJECT_ROOT).as_posix()] = (st.st_size, st.st_mtime_ns)
    return out


def workspace_changes(before: dict[str, tuple[int, int]], after: dict[str, tuple[int, int]]) -> list[str]:
    """Human-readable added/removed/modified entries between two snapshots."""
    changes = [f"+ {p}" for p in sorted(after.keys() - before.keys())]
    changes += [f"- {p}" for p in sorted(before.keys() - after.keys())]
    changes += [f"~ {p}" for p in sorted(before.keys() & after.keys()) if before[p] != after[p]]
    return changes
