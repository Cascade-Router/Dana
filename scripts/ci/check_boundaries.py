"""Architectural boundary check: ``dana/core/`` must not import domain code.

Parses every ``.py`` file under ``dana/core/`` with ``ast`` and fails on any
``import``/``from ... import`` that reaches into ``dana.plugins`` or
``dana.manufacturing`` (absolute or relative), unless that exact
(file, module) edge is listed in the baseline file.

The baseline holds the coupling that existed when this check was introduced
(react_dispatch.py and skill_loader.py import plugin modules directly). It
only lets existing edges through: a core file importing a plugin module it
didn't import before fails. Importing more names from an already-listed
module does not, since the edge is per module, not per name. When a refactor
removes an edge, the check says so; drop the line (or run
``--update-baseline``) so the baseline only ever shrinks.

Usage:
    python scripts/ci/check_boundaries.py [--update-baseline]
Exit status 1 on any new violation.
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = Path(__file__).with_name("boundary_baseline.txt")
FORBIDDEN_PREFIXES = ("dana.plugins", "dana.manufacturing")


@dataclass(frozen=True)
class Violation:
    file: str  # repo-relative, forward slashes
    line: int
    module: str

    @property
    def edge(self) -> str:
        return f"{self.file} {self.module}"


def _is_forbidden(module: str) -> bool:
    return any(module == p or module.startswith(p + ".") for p in FORBIDDEN_PREFIXES)


def _module_name(path: Path, root: Path) -> str:
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolve_relative(path: Path, root: Path, module: str | None, level: int) -> str:
    """``from ..plugins import x`` inside dana/core/a.py -> ``dana.plugins``."""
    package = _module_name(path, root).split(".")
    if path.name != "__init__.py":
        package = package[:-1]
    base = package[: len(package) - (level - 1)] if level > 1 else package
    return ".".join(base + ([module] if module else []))


def imported_modules(path: Path, root: Path) -> list[tuple[int, str]]:
    """Every module a file imports, as (line, dotted name). ``from a import b``
    yields ``a``; only when ``a`` itself is allowed does it also yield
    ``a.b`` (b may be a submodule), so ``from dana import plugins`` is caught
    without ``from dana.plugins.x import func`` counting ``func`` as a module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_relative(path, root, node.module, node.level) if node.level else (node.module or "")
            if not base:
                continue
            found.append((node.lineno, base))
            if not _is_forbidden(base):
                found.extend((node.lineno, f"{base}.{alias.name}") for alias in node.names if alias.name != "*")
    return found


def find_violations(root: Path | None = None) -> list[Violation]:
    root = root or REPO_ROOT
    violations: list[Violation] = []
    for path in sorted((root / "dana" / "core").rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        violations.extend(
            Violation(rel, line, module) for line, module in imported_modules(path, root) if _is_forbidden(module)
        )
    return sorted(violations, key=lambda v: (v.file, v.line, v.module))


def load_baseline(path: Path | None = None) -> set[str]:
    path = path or BASELINE_PATH
    if not path.exists():
        return set()
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return {line for line in lines if line and not line.startswith("#")}


def write_baseline(violations: list[Violation], path: Path | None = None) -> None:
    path = path or BASELINE_PATH
    edges = sorted({v.edge for v in violations})
    header = (
        "# Existing dana/core -> dana.plugins/dana.manufacturing imports, one\n"
        "# '<file> <module>' edge per line. scripts/ci/check_boundaries.py fails on\n"
        "# any edge not listed here. Remove lines as refactors remove the imports;\n"
        "# never add to it to get a new import through.\n"
    )
    path.write_text(header + "".join(f"{edge}\n" for edge in edges), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--update-baseline", action="store_true", help="rewrite the baseline from the current tree")
    args = parser.parse_args(argv)

    violations = find_violations()
    if args.update_baseline:
        write_baseline(violations)
        print(f"Wrote {len({v.edge for v in violations})} edges to {BASELINE_PATH.relative_to(REPO_ROOT).as_posix()}")
        return 0

    baseline = load_baseline()
    new = [v for v in violations if v.edge not in baseline]
    stale = sorted(baseline - {v.edge for v in violations})

    for v in new:
        print(f"{v.file}:{v.line}: dana/core imports {v.module} - core must not depend on domain plugins")
    if stale:
        print(f"{len(stale)} baseline edge(s) no longer exist - remove them from {BASELINE_PATH.name}:")
        for edge in stale:
            print(f"  {edge}")
    if new:
        print(f"FAIL: {len(new)} new core -> plugin import(s).")
        return 1
    print(f"OK: no new core -> plugin imports ({len(baseline)} baselined edges).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
