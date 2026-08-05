"""AST-aware symbol navigation for Python and C++ (worker / ReAct tools).

Python uses the stdlib ``ast`` module. C++ uses a lightweight brace-balanced
scanner (tree-sitter is optional — used when importable, otherwise the
scanner). Paths are jailed to ``PROJECT_ROOT`` via ``system_repl``.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

from dana.paths import PROJECT_ROOT

_ROOT = Path(PROJECT_ROOT).resolve()

_PY_SUFFIXES = {".py", ".pyi"}
_CPP_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".inl"}

# C++ class / struct / union / enum / namespace openers.
_CPP_TYPE_RE = re.compile(
    r"(?m)^[ \t]*(?:template\s*<[^;{}]*>\s*)?"
    r"(?:class|struct|union|enum(?:\s+class)?|namespace)\s+"
    r"([A-Za-z_]\w*)\b[^{;]*\{",
)
# C++ function / method definitions (exclude control keywords).
_CPP_FUNC_RE = re.compile(
    r"(?m)^[ \t]*(?:inline\s+|static\s+|virtual\s+|constexpr\s+|explicit\s+)*"
    r"(?:[\w:<>,\s\*&]+?\s+)?"
    r"(?:([A-Za-z_]\w*)\s*::\s*)?"
    r"([A-Za-z_]\w*)\s*\([^;{}]*\)\s*"
    r"(?:const\s*)?(?:override\s*)?(?:final\s*)?(?:noexcept(?:\([^)]*\))?\s*)?"
    r"(?:->\s*[^;{]+)?\s*\{",
)
_CPP_CONTROL = frozenset(
    {
        "if",
        "for",
        "while",
        "switch",
        "catch",
        "else",
        "do",
        "try",
        "return",
        "sizeof",
    }
)
_CPP_INCLUDE_RE = re.compile(r"(?m)^\s*#\s*include\s*[<\"][^>\"]+[>\"]")
_CPP_USING_RE = re.compile(r"(?m)^\s*using\s+(?:namespace\s+)?[\w:]+\s*;")


def _resolve_jailed(file_path: str) -> Path:
    from dana.tools.system_repl import _resolve_jailed as _jail

    return _jail(file_path)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _lang_for(path: Path) -> str:
    suf = path.suffix.lower()
    if suf in _PY_SUFFIXES:
        return "python"
    if suf in _CPP_SUFFIXES:
        return "cpp"
    return "unknown"


def _brace_end(src: str, open_brace_idx: int) -> int:
    """Return index after the matching ``}`` for ``src[open_brace_idx] == '{'``."""
    if open_brace_idx < 0 or open_brace_idx >= len(src) or src[open_brace_idx] != "{":
        return -1
    depth = 0
    i = open_brace_idx
    in_str: str | None = None
    in_line_comment = False
    in_block_comment = False
    while i < len(src):
        ch = src[i]
        nxt = src[i + 1] if i + 1 < len(src) else ""
        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue
        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue
        if in_str:
            if ch == "\\" and in_str != "'":
                i += 2
                continue
            if ch == in_str:
                in_str = None
            i += 1
            continue
        if ch == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue
        if ch == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue
        if ch in {'"', "'"}:
            in_str = ch
            i += 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return -1


def _line_of(src: str, idx: int) -> int:
    return src.count("\n", 0, max(0, idx)) + 1


def _slice_lines(src: str, start_line: int, end_line: int) -> str:
    lines = src.splitlines()
    lo = max(1, start_line) - 1
    hi = min(len(lines), end_line)
    return "\n".join(lines[lo:hi])


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------


def _py_outline(src: str) -> list[str]:
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return [f"ERROR: Python parse failed: {exc}"]

    lines: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            seg = ast.get_source_segment(src, node) or ""
            lines.append(seg.strip() or f"import@L{node.lineno}")
        elif isinstance(node, ast.ClassDef):
            lines.append(f"class {node.name}  # L{node.lineno}-L{node.end_lineno}")
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    async_p = "async " if isinstance(item, ast.AsyncFunctionDef) else ""
                    args = ast.unparse(item.args) if hasattr(ast, "unparse") else "..."
                    lines.append(
                        f"  {async_p}def {item.name}({args})  "
                        f"# L{item.lineno}-L{item.end_lineno}"
                    )
                elif isinstance(item, ast.ClassDef):
                    lines.append(
                        f"  class {item.name}  # L{item.lineno}-L{item.end_lineno}"
                    )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            async_p = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
            args = ast.unparse(node.args) if hasattr(ast, "unparse") else "..."
            lines.append(
                f"{async_p}def {node.name}({args})  # L{node.lineno}-L{node.end_lineno}"
            )
        elif isinstance(node, ast.Assign):
            targets = ", ".join(
                ast.unparse(t) if hasattr(ast, "unparse") else getattr(t, "id", "?")
                for t in node.targets
            )
            if targets.isupper() or targets.endswith("_PATH") or "URL" in targets:
                lines.append(f"{targets} = ...  # L{node.lineno}")
    return lines


def _py_find_symbol(src: str, symbol_name: str) -> tuple[int, int, str] | None:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    want = (symbol_name or "").strip()
    if not want:
        return None

    hits: list[ast.AST] = []

    class _Visitor(ast.NodeVisitor):
        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            if node.name == want:
                hits.append(node)
            self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if node.name == want:
                hits.append(node)
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            if node.name == want:
                hits.append(node)
            self.generic_visit(node)

    _Visitor().visit(tree)
    if not hits:
        # Qualified Class.method
        if "." in want:
            cls_name, meth = want.split(".", 1)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == cls_name:
                    for item in node.body:
                        if (
                            isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                            and item.name == meth
                        ):
                            hits.append(item)
                            break
    if not hits:
        return None
    node = hits[0]
    start = int(getattr(node, "lineno", 1) or 1)
    end = int(getattr(node, "end_lineno", start) or start)
    body = ast.get_source_segment(src, node) or _slice_lines(src, start, end)
    return start, end, body


# ---------------------------------------------------------------------------
# C++ (tree-sitter optional, else brace scanner)
# ---------------------------------------------------------------------------


def _try_tree_sitter_cpp(src: str) -> list[dict[str, Any]] | None:
    try:
        import tree_sitter_cpp  # type: ignore
        from tree_sitter import Language, Parser  # type: ignore
    except Exception:  # noqa: BLE001
        return None
    try:
        language = Language(tree_sitter_cpp.language())
        parser = Parser(language)
        tree = parser.parse(src.encode("utf-8"))
    except Exception:  # noqa: BLE001
        return None

    root = tree.root_node
    symbols: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        kind = getattr(node, "type", "") or ""
        if kind in {
            "function_definition",
            "class_specifier",
            "struct_specifier",
            "namespace_definition",
            "enum_specifier",
        }:
            name = ""
            for child in node.children:
                if child.type in {"type_identifier", "identifier", "namespace_identifier"}:
                    name = src[child.start_byte : child.end_byte]
                    break
                if child.type == "function_declarator":
                    for gc in child.children:
                        if gc.type == "identifier":
                            name = src[gc.start_byte : gc.end_byte]
                            break
            symbols.append(
                {
                    "kind": kind,
                    "name": name or "?",
                    "start_line": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1,
                    "start_byte": node.start_byte,
                    "end_byte": node.end_byte,
                }
            )
        for child in node.children:
            walk(child)

    walk(root)
    return symbols


def _cpp_scan_symbols(src: str) -> list[dict[str, Any]]:
    ts = _try_tree_sitter_cpp(src)
    if ts is not None:
        return ts

    symbols: list[dict[str, Any]] = []
    for m in _CPP_TYPE_RE.finditer(src):
        brace = src.find("{", m.start())
        end = _brace_end(src, brace)
        if end < 0:
            continue
        kind_tok = re.search(
            r"\b(class|struct|union|enum(?:\s+class)?|namespace)\b", m.group(0)
        )
        symbols.append(
            {
                "kind": (kind_tok.group(1) if kind_tok else "type").replace(" ", "_"),
                "name": m.group(1),
                "start_line": _line_of(src, m.start()),
                "end_line": _line_of(src, end - 1),
                "start_byte": m.start(),
                "end_byte": end,
            }
        )
    for m in _CPP_FUNC_RE.finditer(src):
        qual = m.group(1) or ""
        name = m.group(2) or ""
        if name in _CPP_CONTROL:
            continue
        # Skip if this match sits inside an already-recorded type opener line
        # that used the same `{` (class bodies contain methods — keep methods).
        brace = src.find("{", m.start())
        end = _brace_end(src, brace)
        if end < 0:
            continue
        full = f"{qual}::{name}" if qual else name
        symbols.append(
            {
                "kind": "function",
                "name": full,
                "start_line": _line_of(src, m.start()),
                "end_line": _line_of(src, end - 1),
                "start_byte": m.start(),
                "end_byte": end,
            }
        )
    symbols.sort(key=lambda s: (int(s["start_line"]), int(s["end_line"])))
    return symbols


def _cpp_outline(src: str) -> list[str]:
    lines: list[str] = []
    for m in _CPP_INCLUDE_RE.finditer(src):
        lines.append(m.group(0).strip())
    for m in _CPP_USING_RE.finditer(src):
        lines.append(m.group(0).strip())
    for sym in _cpp_scan_symbols(src):
        kind = str(sym.get("kind") or "symbol")
        name = str(sym.get("name") or "?")
        lo = sym.get("start_line")
        hi = sym.get("end_line")
        if kind == "function":
            lines.append(f"fn {name}()  # L{lo}-L{hi}")
        else:
            lines.append(f"{kind} {name}  # L{lo}-L{hi}")
    return lines


def _cpp_find_symbol(src: str, symbol_name: str) -> tuple[int, int, str] | None:
    want = (symbol_name or "").strip()
    if not want:
        return None
    want_base = want.split("::")[-1]
    for sym in _cpp_scan_symbols(src):
        name = str(sym.get("name") or "")
        if name == want or name.endswith(f"::{want_base}") or name == want_base:
            start = int(sym["start_line"])
            end = int(sym["end_line"])
            body = src[int(sym["start_byte"]) : int(sym["end_byte"])]
            return start, end, body
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_file_outline(file_path: str) -> str:
    """Return a structural skeleton (imports/classes/methods) without bodies."""
    try:
        path = _resolve_jailed(file_path)
    except ValueError as exc:
        return f"ERROR: {exc}"
    if not path.is_file():
        return f"ERROR: file not found: {path}"
    try:
        src = _read_text(path)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: read failed: {exc}"

    rel = path.relative_to(_ROOT).as_posix()
    lang = _lang_for(path)
    if lang == "python":
        outline = _py_outline(src)
    elif lang == "cpp":
        outline = _cpp_outline(src)
    else:
        return (
            f"ERROR: unsupported language for outline ({path.suffix!r}). "
            "Supported: Python (.py/.pyi) and C/C++ (.c/.cc/.cpp/.h/.hpp/…)."
        )
    if outline and outline[0].startswith("ERROR:"):
        return outline[0]
    body = "\n".join(outline) if outline else "(empty outline)"
    return f"OK: outline {rel} lang={lang} symbols={len(outline)}\n{body}"


def get_symbol_definition(file_path: str, symbol_name: str) -> str:
    """Extract exact class/function source boundaries for ``symbol_name``."""
    try:
        path = _resolve_jailed(file_path)
    except ValueError as exc:
        return f"ERROR: {exc}"
    if not path.is_file():
        return f"ERROR: file not found: {path}"
    try:
        src = _read_text(path)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: read failed: {exc}"

    rel = path.relative_to(_ROOT).as_posix()
    lang = _lang_for(path)
    hit: tuple[int, int, str] | None
    if lang == "python":
        hit = _py_find_symbol(src, symbol_name)
    elif lang == "cpp":
        hit = _cpp_find_symbol(src, symbol_name)
    else:
        return (
            f"ERROR: unsupported language for symbol lookup ({path.suffix!r}). "
            "Supported: Python and C/C++."
        )
    if hit is None:
        return f"ERROR: symbol {symbol_name!r} not found in {rel}"
    start, end, body = hit
    return (
        f"OK: symbol {symbol_name!r} in {rel} lang={lang} "
        f"L{start}-L{end} ({end - start + 1} lines)\n{body}"
    )


__all__ = (
    "get_file_outline",
    "get_symbol_definition",
)
