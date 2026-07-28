"""Tool Forge Template Method — fixed BaseTool topology for forged tools.

The LLM never authors the import / @tool scaffolding. It only supplies
``docstring`` + ``python_code`` (function body). ``assemble_forged_tool`` injects
them into a pre-validated LangChain ``@tool`` wrapper.
"""

from __future__ import annotations

import json
import re
import textwrap
from typing import Any

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.I)
_TOOL_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")

# Allowed optional import roots the assembler may inject when the body hints need them.
_BODY_IMPORT_HINTS: tuple[tuple[str, str], ...] = (
    ("Image.open", "from PIL import Image"),
    ("resolve_safe_path", "from dana.tools.sandbox_io import resolve_safe_path"),
    ("sandbox_read", "from dana.tools.sandbox_io import sandbox_read"),
    ("psutil.", "import psutil"),
    ("psutil(", "import psutil"),
    ("Path(", "from pathlib import Path"),
    ("Path.", "from pathlib import Path"),
    ("base64.", "import base64"),
    ("io.", "import io"),
    ("math.", "import math"),
    ("json.", "import json"),
    ("re.", "import re"),
)


def safe_tool_name(name: str, *, fallback: str = "forged_tool") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "", (name or "").strip())
    if not cleaned or not cleaned[0].isalpha():
        cleaned = fallback
    if not _TOOL_NAME_RE.match(cleaned):
        cleaned = fallback
    return cleaned[:64]


def _indent_body(body: str, spaces: int = 4) -> str:
    """Normalize LLM function-body text into a uniformly indented block.

    llama3.2 often emits mixed / over-indented bodies (or leftover ``import`` /
    ``def`` lines). Flat bodies are left-stripped then re-indented so
    ``ast.parse`` never hits ``unexpected indent`` after assembly.
    """
    blob = textwrap.dedent(body or "").strip("\n")
    if not blob.strip():
        return " " * spaces + "return ''"

    lines = blob.splitlines()
    # Strip a leading ``def ...:`` if the model ignored instructions.
    if lines and re.match(r"^\s*def\s+\w+\s*\(", lines[0]):
        rest = "\n".join(lines[1:])
        blob = textwrap.dedent(rest).strip("\n") or "return ''"
        lines = blob.splitlines()

    # Assembler owns imports / @tool — drop any the model smuggled into the body.
    kept: list[str] = []
    for ln in lines:
        if re.match(r"^\s*(from\s+\S+\s+import\b|import\s+\w|@tool\b)", ln):
            continue
        kept.append(ln)
    blob = textwrap.dedent("\n".join(kept)).strip("\n") or "return ''"

    # Nested control flow keeps relative indent via dedent; flat bodies are
    # forcibly left-aligned to kill "unexpected indent" from LLM noise.
    needs_structure = bool(
        re.search(
            r"^\s*(if|for|while|try|with|elif|else|except|finally)\b",
            blob,
            re.M,
        )
    )
    if not needs_structure:
        flat_lines = [ln.strip() for ln in blob.splitlines() if ln.strip()]
        blob = "\n".join(flat_lines) if flat_lines else "return ''"
    else:
        blob = textwrap.dedent(blob).strip("\n") or "return ''"

    indented = textwrap.indent(blob, " " * spaces)
    # Self-check: assembled fragment must parse when wrapped in a dummy def.
    probe = f"def _probe():\n{indented}\n"
    try:
        compile(probe, "<forge_body>", "exec")
    except SyntaxError:
        # Last-resort flatten.
        flat_lines = [ln.strip() for ln in blob.splitlines() if ln.strip()]
        indented = textwrap.indent(
            "\n".join(flat_lines) if flat_lines else "return ''",
            " " * spaces,
        )
    return indented


def _infer_imports(body: str) -> list[str]:
    imports = ["from langchain_core.tools import tool"]
    seen = {imports[0]}
    blob = body or ""
    for needle, stmt in _BODY_IMPORT_HINTS:
        if needle in blob and stmt not in seen:
            # Prefer bare "psutil" token match without false positives.
            if needle.startswith("psutil") and "psutil" not in blob:
                continue
            imports.append(stmt)
            seen.add(stmt)
    if re.search(r"\bpsutil\b", blob) and "import psutil" not in seen:
        imports.append("import psutil")
        seen.add("import psutil")
    if re.search(r"\bPath\b", blob) and "from pathlib import Path" not in seen:
        imports.append("from pathlib import Path")
        seen.add("from pathlib import Path")
    return imports


def assemble_forged_tool(
    *,
    tool_name: str,
    docstring: str,
    python_code: str,
    description: str = "",
) -> str:
    """Inject docstring + function body into a pre-validated @tool module."""
    name = safe_tool_name(tool_name)
    doc = (docstring or description or f"Forged tool `{name}`").strip().replace('"""', "'")
    body = _indent_body(python_code)
    imports = "\n".join(_infer_imports(python_code))
    # Fixed signature: text + filepath cover forge + image tools without free-form defs.
    return (
        f"{imports}\n"
        f"\n\n"
        f"@tool\n"
        f"def {name}(text: str = '', filepath: str = '') -> str:\n"
        f'    """{doc}"""\n'
        f"{body}\n"
    )


def extract_coder_json(raw: str) -> dict[str, Any] | None:
    """Parse coder output; reject anything that is not a JSON object payload."""
    text = (raw or "").strip()
    if not text:
        return None
    fence = _JSON_FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()
    # Prefer the outermost object.
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        blob = text[start : end + 1]
        try:
            data = json.loads(blob)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            repaired = _soft_extract_fields(blob)
            if repaired:
                return repaired
            # Brace soup / truncated JSON — still try field scrape on full text.
            repaired = _soft_extract_fields(text)
            if repaired:
                return repaired
    else:
        repaired = _soft_extract_fields(text)
        if repaired:
            return repaired

    # Last resort: model dumped a Python function body / module instead of JSON.
    py_fallback = _python_body_fallback(text)
    if py_fallback:
        return py_fallback
    return None


def _soft_extract_fields(blob: str) -> dict[str, Any] | None:
    """Best-effort field scrape when json.loads fails on multiline python_code."""
    out: dict[str, Any] = {}
    for key in ("tool_name", "description", "docstring", "python_code", "function_body", "code"):
        # 1) Standard JSON string (possibly with escapes).
        m = re.search(
            rf'"{key}"\s*:\s*"((?:\\.|[^"\\])*)"',
            blob,
            re.DOTALL,
        )
        if m:
            try:
                out[key] = json.loads(f'"{m.group(1)}"')
            except json.JSONDecodeError:
                out[key] = (
                    m.group(1)
                    .replace("\\n", "\n")
                    .replace("\\t", "\t")
                    .replace('\\"', '"')
                    .replace("\\\\", "\\")
                )
            continue
        # 2) Triple-quoted / markdown-ish value (common llama3.2 failure).
        m = re.search(
            rf'"{key}"\s*:\s*"""([\s\S]*?)"""',
            blob,
        )
        if m:
            out[key] = m.group(1).strip()
            continue
        m = re.search(
            rf"'{key}'\s*:\s*'((?:\\.|[^'\\])*)'",
            blob,
            re.DOTALL,
        )
        if m:
            out[key] = (
                m.group(1)
                .replace("\\n", "\n")
                .replace("\\t", "\t")
                .replace("\\'", "'")
            )
            continue
        # 3) Unescaped multiline string until next JSON key or closing brace.
        m = re.search(
            rf'"{key}"\s*:\s*"(.*?)(?="\s*,\s*"[A-Za-z_]+"\s*:|"\s*\}})',
            blob,
            re.DOTALL,
        )
        if m:
            out[key] = m.group(1).replace("\\n", "\n").strip()
            continue
    if not out:
        return None
    # Require a body-like field for a usable forge payload.
    if not any(out.get(k) for k in ("python_code", "function_body", "code")):
        return None
    return out


def _python_body_fallback(text: str) -> dict[str, Any] | None:
    """If the coder returned raw Python, treat it as python_code (body or module)."""
    blob = textwrap.dedent(text or "").strip()
    if not blob:
        return None
    # Strip markdown python fences.
    fence = re.search(r"```(?:python)?\s*([\s\S]*?)```", blob, re.I)
    if fence:
        blob = fence.group(1).strip()
    looks_py = bool(
        re.search(r"^\s*(def |return |from |import |@tool)", blob, re.M)
    )
    if not looks_py:
        return None
    name = "forged_tool"
    m = re.search(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", blob, re.M)
    if m:
        name = m.group(1)
    # Prefer function body when a def is present (assembler wraps @tool).
    body = blob
    if m:
        lines = blob.splitlines()
        # Drop import / decorator lines before def; keep body under def.
        def_idx = next(
            (i for i, ln in enumerate(lines) if re.match(r"^\s*def\s+", ln)),
            None,
        )
        if def_idx is not None:
            rest = "\n".join(lines[def_idx + 1 :])
            body = textwrap.dedent(rest).strip() or "return ''"
    return {
        "tool_name": name,
        "description": f"Forged tool `{name}` (recovered from non-JSON coder output)",
        "docstring": f"Forged tool `{name}`",
        "python_code": body,
    }


def normalize_coder_payload(data: dict[str, Any] | None, *, fallback_name: str) -> dict[str, str]:
    """Map coder JSON → tool_name / description / docstring / python_code (body)."""
    data = dict(data or {})
    name = safe_tool_name(str(data.get("tool_name") or fallback_name), fallback=fallback_name)
    description = str(data.get("description") or "").strip()
    docstring = str(data.get("docstring") or description or f"Forged tool `{name}`").strip()
    # Preferred field is python_code (function body). Accept aliases.
    body = str(
        data.get("python_code")
        or data.get("function_body")
        or data.get("code")
        or ""
    ).strip()
    return {
        "tool_name": name,
        "description": description,
        "docstring": docstring,
        "python_code": body,
    }


JSON_SCHEMA_FAILURE = (
    "FATAL: Your output is not valid JSON. Ensure your code is contained "
    "within the `python_code` JSON field "
    '(example: {"tool_name":"...", "description":"...", "docstring":"...", '
    '"python_code":"return text[::-1]"}). '
    "Any text outside this JSON object triggers immediate failure. "
    "If you must emit Python, emit ONLY the function body (no markdown)."
)
