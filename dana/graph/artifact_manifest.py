"""Inter-epic artifact manifest — contract schema under ``.dana_scratch/manifest.json``."""

from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MANIFEST_REL = Path(".dana_scratch") / "manifest.json"


def manifest_path(*, workspace: str | Path | None = None) -> Path:
    from dana.paths import PROJECT_ROOT

    root = Path(workspace or PROJECT_ROOT).resolve()
    return root / _MANIFEST_REL


def _module_name_from_path(rel: str) -> str:
    p = Path(str(rel or "").replace("\\", "/"))
    stem = p.stem
    if not stem:
        return "unknown"
    # tests/test_foo.py → test_foo (import path still documented via file)
    parts = list(p.with_suffix("").parts)
    if parts and parts[0] in {"", "."}:
        parts = parts[1:]
    return ".".join(parts) if parts else stem


def extract_exports_from_source(source: str) -> dict[str, list[str]]:
    """Parse top-level function / class names from Python source.

    Explicitly walks ``ast.ClassDef`` **and** ``ast.FunctionDef`` /
    ``ast.AsyncFunctionDef`` so classes like ``VectorDocument`` (including
    ``@dataclass``-decorated forms) are always written into the manifest.
    Nested methods / inner classes are ignored (not import-contract exports).
    """
    functions: list[str] = []
    classes: list[str] = []
    try:
        tree = ast.parse(source or "")
    except SyntaxError:
        return {"functions": [], "classes": []}

    class _ExportVisitor(ast.NodeVisitor):
        """Collect module-level ClassDef / FunctionDef names only."""

        def __init__(self) -> None:
            self._scope_depth = 0

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            # Always record ClassDef at module scope (decorators are ignored).
            if self._scope_depth == 0 and node.name not in classes:
                classes.append(node.name)
            self._scope_depth += 1
            self.generic_visit(node)
            self._scope_depth -= 1

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if self._scope_depth == 0 and node.name not in functions:
                functions.append(node.name)
            # Do not descend — nested defs are not module exports.

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            if self._scope_depth == 0 and node.name not in functions:
                functions.append(node.name)

    _ExportVisitor().visit(tree)
    return {"functions": functions, "classes": classes}


def extract_exports_from_file(path: Path) -> dict[str, list[str]]:
    try:
        src = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {"functions": [], "classes": []}
    return extract_exports_from_source(src)


def load_manifest(*, workspace: str | Path | None = None) -> dict[str, Any]:
    path = manifest_path(workspace=workspace)
    if not path.is_file():
        return {"version": 1, "artifacts": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("manifest load failed: %s", exc)
        return {"version": 1, "artifacts": []}
    if not isinstance(data, dict):
        return {"version": 1, "artifacts": []}
    arts = data.get("artifacts")
    if not isinstance(arts, list):
        data["artifacts"] = []
    data.setdefault("version", 1)
    return data


def write_manifest(
    artifacts: list[dict[str, Any]],
    *,
    workspace: str | Path | None = None,
) -> Path:
    path = manifest_path(workspace=workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "artifacts": list(artifacts)}
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def upsert_artifact_record(
    *,
    file_path: str,
    module_name: str = "",
    functions: list[str] | None = None,
    classes: list[str] | None = None,
    epic_id: Any = None,
    workspace: str | Path | None = None,
) -> dict[str, Any]:
    """Merge one artifact into ``manifest.json`` (keyed by file path)."""
    rel = str(file_path or "").replace("\\", "/").lstrip("./")
    data = load_manifest(workspace=workspace)
    arts: list[dict[str, Any]] = [
        dict(a) for a in (data.get("artifacts") or []) if isinstance(a, dict)
    ]
    record = {
        "file_path": rel,
        "module_name": module_name or _module_name_from_path(rel),
        "functions": list(functions or []),
        "classes": list(classes or []),
        "epic_id": epic_id,
    }
    replaced = False
    for i, row in enumerate(arts):
        if str(row.get("file_path") or "").replace("\\", "/") == rel:
            arts[i] = record
            replaced = True
            break
    if not replaced:
        arts.append(record)
    write_manifest(arts, workspace=workspace)
    return record


def update_manifest_from_epic_artifacts(
    artifacts: list[dict[str, Any]],
    *,
    epic_id: Any = None,
    workspace: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Parse each epic artifact file and upsert exports into the manifest."""
    from dana.paths import PROJECT_ROOT

    root = Path(workspace or PROJECT_ROOT).resolve()
    written: list[dict[str, Any]] = []
    for row in artifacts:
        rel = str(row.get("path") or "").replace("\\", "/").strip()
        if not rel:
            continue
        path = root / rel
        exports = extract_exports_from_file(path) if path.is_file() else {
            "functions": [],
            "classes": [],
        }
        # Prefer content blob when file missing from disk.
        if not path.is_file() and row.get("content"):
            exports = extract_exports_from_source(str(row.get("content") or ""))
        rec = upsert_artifact_record(
            file_path=rel,
            functions=exports.get("functions") or [],
            classes=exports.get("classes") or [],
            epic_id=epic_id if epic_id is not None else row.get("epic_id"),
            workspace=workspace,
        )
        written.append(rec)
    return written


# Injected into Meta-Broker / worker epic prompts (stdlib-only codegen).
META_BROKER_STDLIB_RULE = (
    "When generating Python code for Epics, rely strictly on Python Standard "
    "Library modules (e.g., json, math, os, sys, deque) unless third-party "
    "packages are explicitly requested in the prompt."
)


def format_manifest_contract_block(
    *,
    workspace: str | Path | None = None,
) -> str:
    """Human-readable contract block prepended to the next epic prompt."""
    data = load_manifest(workspace=workspace)
    arts = [a for a in (data.get("artifacts") or []) if isinstance(a, dict)]
    lines = [
        "### Artifact Manifest Contract (STRICT)",
        META_BROKER_STDLIB_RULE,
        "",
    ]
    if not arts:
        return "\n".join(lines)
    lines.extend(
        [
            "Prior epics exported the following modules. Import ONLY these names;",
            "do not invent alternate module paths.",
            "",
        ]
    )
    for a in arts:
        path = str(a.get("file_path") or "")
        mod = str(a.get("module_name") or _module_name_from_path(path))
        fns = ", ".join(a.get("functions") or []) or "(none)"
        cls = ", ".join(a.get("classes") or []) or "(none)"
        lines.append(f"- file=`{path}` module=`{mod}`")
        lines.append(f"  classes: {cls}")
        lines.append(f"  functions: {fns}")
        # Import hint for tests: maze_solver.py → from maze_solver import ...
        bare = Path(path).stem
        if bare.startswith("test_"):
            continue
        export_names = list(a.get("classes") or []) + list(a.get("functions") or [])
        if export_names:
            lines.append(
                f"  import example: `from {bare} import {', '.join(export_names[:4])}`"
            )
    lines.append("")
    return "\n".join(lines)


__all__ = (
    "META_BROKER_STDLIB_RULE",
    "extract_exports_from_file",
    "extract_exports_from_source",
    "format_manifest_contract_block",
    "load_manifest",
    "manifest_path",
    "update_manifest_from_epic_artifacts",
    "upsert_artifact_record",
    "write_manifest",
)
