"""Deterministic AST Sanitizer — fix trivial import flaws before Pytest."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Any

AST_SANITIZER_NODE = "ast_sanitizer"

# Bare names → preferred stdlib module (from-import or module import).
_NAME_TO_MODULE: dict[str, str] = {
    # collections
    "deque": "collections",
    "defaultdict": "collections",
    "Counter": "collections",
    "OrderedDict": "collections",
    "namedtuple": "collections",
    "ChainMap": "collections",
    # asyncio
    "asyncio": "asyncio",
    # pathlib
    "Path": "pathlib",
    # dataclasses
    "dataclass": "dataclasses",
    "field": "dataclasses",
    # typing (common)
    "Any": "typing",
    "Optional": "typing",
    "Union": "typing",
    "List": "typing",
    "Dict": "typing",
    "Tuple": "typing",
    "Set": "typing",
    "Callable": "typing",
    "Iterable": "typing",
    "Iterator": "typing",
    "Mapping": "typing",
    "Sequence": "typing",
    "Literal": "typing",
    "TypedDict": "typing",
    "Protocol": "typing",
    "TypeVar": "typing",
    "Generic": "typing",
    "NotRequired": "typing",
    # heapq / functools / itertools / copy / json / re / math / time / os / sys
    "heappush": "heapq",
    "heappop": "heapq",
    "heapify": "heapq",
    "lru_cache": "functools",
    "partial": "functools",
    "reduce": "functools",
    "wraps": "functools",
    "chain": "itertools",
    "islice": "itertools",
    "groupby": "itertools",
    "product": "itertools",
    "combinations": "itertools",
    "deepcopy": "copy",
    "copy": "copy",
    "json": "json",
    "re": "re",
    "math": "math",
    "time": "time",
    "os": "os",
    "sys": "sys",
    "threading": "threading",
    "subprocess": "subprocess",
    "logging": "logging",
    "argparse": "argparse",
    "dataclasses": "dataclasses",
    "collections": "collections",
    "pathlib": "pathlib",
    "functools": "functools",
    "itertools": "itertools",
    "heapq": "heapq",
    "enum": "enum",
    "Enum": "enum",
    "IntEnum": "enum",
    "contextlib": "contextlib",
    "contextmanager": "contextlib",
    "abstractmethod": "abc",
    "ABC": "abc",
    "Queue": "queue",
    "PriorityQueue": "queue",
    "sleep": "time",  # bare sleep() → time.sleep (asyncio.sleep is attr)
}

# Attribute roots that imply ``import <module>``.
_ATTR_ROOT_MODULES = frozenset(
    {
        "math",
        "collections",
        "asyncio",
        "json",
        "re",
        "os",
        "sys",
        "time",
        "pathlib",
        "functools",
        "itertools",
        "heapq",
        "dataclasses",
        "typing",
        "threading",
        "subprocess",
        "logging",
        "argparse",
        "copy",
        "enum",
        "queue",
        "abc",
        "contextlib",
        "concurrent",
        "urllib",
        "http",
        "socket",
        "struct",
        "hashlib",
        "base64",
        "csv",
        "io",
        "tempfile",
        "shutil",
        "glob",
        "fnmatch",
        "uuid",
        "random",
        "statistics",
        "decimal",
        "fractions",
        "array",
        "bisect",
        "weakref",
        "pprint",
        "textwrap",
        "string",
        "datetime",
        "calendar",
        "zoneinfo",
    }
)

_FILE_TOKEN_RE = re.compile(
    r"([\w./\\-]+\.py)\b",
    re.I,
)


def _collect_imports(tree: ast.AST) -> tuple[set[str], set[str], list[ast.AST]]:
    """Return (imported_modules, imported_names, import_nodes)."""
    modules: set[str] = set()
    names: set[str] = set()
    nodes: list[ast.AST] = []
    for node in tree.body if isinstance(tree, ast.Module) else []:  # type: ignore[attr-defined]
        if isinstance(node, ast.Import):
            nodes.append(node)
            for alias in node.names:
                mod = alias.name.split(".")[0]
                modules.add(mod)
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            nodes.append(node)
            root = (node.module or "").split(".")[0] if node.module else ""
            if root:
                modules.add(root)
            for alias in node.names:
                if alias.name == "*":
                    continue
                names.add(alias.asname or alias.name)
    return modules, names, nodes


def _used_names_and_attr_roots(tree: ast.AST) -> tuple[set[str], set[str]]:
    used: set[str] = set()
    attr_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used.add(node.id)
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            attr_roots.add(node.value.id)
            used.add(node.value.id)
    return used, attr_roots


def sanitize_python_source(source: str) -> tuple[str, dict[str, Any]]:
    """Return (possibly fixed source, report dict).

    Fixes:
      * Missing stdlib imports for known symbols / attribute roots
      * Unused dangling imports (Import / ImportFrom) that are never referenced
    """
    report: dict[str, Any] = {
        "changed": False,
        "added_imports": [],
        "removed_imports": [],
        "error": None,
    }
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        report["error"] = f"SyntaxError: {exc}"
        return source, report

    modules, names, _import_nodes = _collect_imports(tree)
    used, attr_roots = _used_names_and_attr_roots(tree)

    # --- Add missing imports -------------------------------------------------
    to_import_module: set[str] = set()
    to_from_import: dict[str, set[str]] = {}

    for root in attr_roots:
        if root in _ATTR_ROOT_MODULES and root not in modules and root not in names:
            to_import_module.add(root)

    for name in used:
        if name in names or name in modules:
            continue
        mod = _NAME_TO_MODULE.get(name)
        if not mod:
            continue
        if name == mod:
            # ``json.loads`` style already covered by attr_roots; bare ``json`` → import json
            if name not in modules:
                to_import_module.add(name)
            continue
        if name in {"sleep"} and "asyncio" in modules:
            # Prefer asyncio.sleep when asyncio already imported.
            continue
        to_from_import.setdefault(mod, set()).add(name)

    new_import_nodes: list[ast.stmt] = []
    for mod in sorted(to_import_module):
        new_import_nodes.append(ast.Import(names=[ast.alias(name=mod, asname=None)]))
        report["added_imports"].append(f"import {mod}")
        modules.add(mod)
        names.add(mod)

    for mod, syms in sorted(to_from_import.items()):
        # Skip symbols already available via ``import mod``.
        if mod in modules:
            continue
        aliases = [ast.alias(name=s, asname=None) for s in sorted(syms)]
        new_import_nodes.append(ast.ImportFrom(module=mod, names=aliases, level=0))
        report["added_imports"].append(f"from {mod} import {', '.join(sorted(syms))}")
        for s in syms:
            names.add(s)
        modules.add(mod)

    # --- Drop unused imports -------------------------------------------------
    # Rebuild body: keep non-import stmts; filter import nodes.
    body = list(tree.body)
    kept: list[ast.stmt] = []
    # Future / docstring handling preserved by position.
    for node in body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            keep_aliases: list[ast.alias] = []
            if isinstance(node, ast.Import):
                for alias in node.names:
                    bound = alias.asname or alias.name.split(".")[0]
                    # Keep if bound name or any attr-root use of the module.
                    if bound in used or bound in attr_roots:
                        keep_aliases.append(alias)
                    else:
                        report["removed_imports"].append(
                            f"import {alias.name}"
                            + (f" as {alias.asname}" if alias.asname else "")
                        )
                if keep_aliases:
                    node.names = keep_aliases
                    kept.append(node)
            else:
                if node.names and any(a.name == "*" for a in node.names):
                    kept.append(node)
                    continue
                for alias in node.names:
                    bound = alias.asname or alias.name
                    if bound in used:
                        keep_aliases.append(alias)
                    else:
                        report["removed_imports"].append(
                            f"from {node.module} import {alias.name}"
                        )
                if keep_aliases:
                    node.names = keep_aliases
                    kept.append(node)
        else:
            kept.append(node)

    # Insert new imports after module docstring and __future__ imports.
    insert_at = 0
    if (
        kept
        and isinstance(kept[0], ast.Expr)
        and isinstance(getattr(kept[0], "value", None), ast.Constant)
        and isinstance(kept[0].value.value, str)
    ):
        insert_at = 1
    while insert_at < len(kept):
        n = kept[insert_at]
        if isinstance(n, ast.ImportFrom) and n.module == "__future__":
            insert_at += 1
            continue
        break

    if new_import_nodes:
        kept = kept[:insert_at] + new_import_nodes + kept[insert_at:]

    if not report["added_imports"] and not report["removed_imports"]:
        return source, report

    tree.body = kept
    ast.fix_missing_locations(tree)
    try:
        fixed = ast.unparse(tree)
    except Exception as exc:  # noqa: BLE001
        report["error"] = f"unparse failed: {exc}"
        return source, report
    # Prefer a trailing newline for POSIX friendliness.
    if not fixed.endswith("\n"):
        fixed += "\n"
    report["changed"] = True
    return fixed, report


def sanitize_python_file(path: str | Path) -> dict[str, Any]:
    """Sanitize one ``.py`` file in place. Returns the report (+ path)."""
    p = Path(path)
    report: dict[str, Any] = {"path": str(p), "changed": False}
    if not p.is_file() or p.suffix.lower() != ".py":
        report["error"] = "not a .py file"
        return report
    try:
        original = p.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        report["error"] = str(exc)
        return report
    fixed, detail = sanitize_python_source(original)
    report.update(detail)
    if detail.get("changed"):
        try:
            p.write_text(fixed, encoding="utf-8")
        except OSError as exc:
            report["error"] = str(exc)
            report["changed"] = False
    return report


def _epic_py_paths(state: dict[str, Any]) -> list[Path]:
    from dana.paths import PROJECT_ROOT

    workspace = Path(
        str(state.get("workspace_path") or PROJECT_ROOT)
    ).expanduser()
    try:
        workspace = workspace.resolve()
    except OSError:
        pass
    epics = list(state.get("epics") or [])
    idx = int(state.get("active_epic_index") or 0)
    epic = epics[idx] if 0 <= idx < len(epics) else {}
    blobs = [
        str((epic or {}).get("goal") or ""),
        str((epic or {}).get("validation_command") or ""),
        str(state.get("validation_command") or ""),
        str(state.get("user_prompt") or ""),
    ]
    # Also include recently completed artifact paths from this epic.
    for row in state.get("completed_epic_artifacts") or []:
        if isinstance(row, dict) and row.get("path"):
            blobs.append(str(row["path"]))
    found: list[Path] = []
    seen: set[str] = set()
    for blob in blobs:
        for m in _FILE_TOKEN_RE.finditer(blob):
            rel = m.group(1).replace("\\", "/")
            key = rel.lower()
            if key in seen:
                continue
            seen.add(key)
            # Skip protected package trees.
            if rel.startswith(("dana/", "dana_security/", "website/", "legacy/")):
                continue
            cand = workspace / rel
            if cand.is_file():
                found.append(cand)
    return found


def make_ast_sanitizer_node():
    """LangGraph node: sanitize epic ``.py`` files before the runtime harness."""

    def _node(state: dict[str, Any]) -> dict[str, Any]:
        reports: list[dict[str, Any]] = []
        changed = 0
        for path in _epic_py_paths(state):
            rep = sanitize_python_file(path)
            reports.append(rep)
            if rep.get("changed"):
                changed += 1
                print(
                    f"[AstSanitizer] fixed {path.name}: "
                    f"+{rep.get('added_imports')} -{rep.get('removed_imports')}",
                    flush=True,
                )
        log = list(state.get("epic_log") or [])
        if reports:
            log.append(
                f"ast_sanitizer: scanned={len(reports)} changed={changed}"
            )
        return {
            "epic_log": log,
            "ast_sanitizer_report": {
                "scanned": len(reports),
                "changed": changed,
                "files": reports,
            },
        }

    return _node


__all__ = (
    "AST_SANITIZER_NODE",
    "make_ast_sanitizer_node",
    "sanitize_python_file",
    "sanitize_python_source",
)

# Python 3.9+ required for ast.unparse; guard for clarity.
if sys.version_info < (3, 9):  # pragma: no cover
    raise RuntimeError("ast_sanitizer requires Python 3.9+")
