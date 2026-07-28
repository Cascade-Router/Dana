"""Plugin: file_jail_enforcer — jailed local file read under CAMGRASPER/docs.

Safety:
  - All paths resolve under DOCS_JAIL (no ../ escape, no absolute outside jail).
  - No shell / subprocess / OS binaries.
  - Text/Markdown via stdlib; PDF via optional pypdf only.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

from dana.paths import DOCS_DIR, PROJECT_ROOT

TOOL_ID = "file_jail_enforcer"

# Strict directory jail — never read outside this folder.
_ROOT = PROJECT_ROOT
DOCS_JAIL = DOCS_DIR.resolve()

_MAX_PATH_CHARS = 260
_MAX_READ_CHARS = 8_000
_ALLOWED_SUFFIXES = frozenset({".txt", ".md", ".markdown", ".pdf", ".log", ".csv"})


class JailError(ValueError):
    """Raised when a path escapes the docs jail or is otherwise unsafe."""


def _sanitize_rel_path(raw: str) -> str:
    text = unicodedata.normalize("NFC", str(raw or "")).strip().replace("\\", "/")
    if len(text) > _MAX_PATH_CHARS:
        raise JailError("path too long")
    # Strip control chars.
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
    text = text.lstrip("/")
    if not text:
        raise JailError("empty path")
    if text.startswith("~") or re.match(r"^[A-Za-z]:", text):
        raise JailError("absolute / home paths are forbidden — use jail-relative paths")
    parts = [p for p in text.split("/") if p and p != "."]
    if any(p == ".." for p in parts):
        raise JailError("path traversal (..) is forbidden")
    return "/".join(parts)


def resolve_jailed_path(user_path: str) -> Path:
    """Map a user-supplied relative path onto DOCS_JAIL (raises JailError)."""
    rel = _sanitize_rel_path(user_path)
    # Ensure jail exists so resolve() semantics are stable.
    DOCS_JAIL.mkdir(parents=True, exist_ok=True)
    candidate = (DOCS_JAIL / rel).resolve()
    try:
        candidate.relative_to(DOCS_JAIL)
    except ValueError as exc:
        raise JailError("path escapes docs jail") from exc
    if candidate.suffix.lower() not in _ALLOWED_SUFFIXES:
        raise JailError(
            f"suffix '{candidate.suffix}' not allowed; "
            f"use one of {sorted(_ALLOWED_SUFFIXES)}"
        )
    return candidate


def _read_text_file(path: Path, max_chars: int) -> str:
    data = path.read_text(encoding="utf-8", errors="replace")
    if len(data) > max_chars:
        return data[:max_chars] + "\n…[truncated]"
    return data


def _read_pdf_file(path: Path, max_chars: int) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise JailError(
            "PDF support requires the 'pypdf' package (offline, pinned). "
            "Install with: pip install pypdf"
        ) from exc
    reader = PdfReader(str(path))
    chunks: list[str] = []
    total = 0
    for page in reader.pages:
        piece = (page.extract_text() or "").strip()
        if not piece:
            continue
        chunks.append(piece)
        total += len(piece)
        if total >= max_chars:
            break
    text = "\n\n".join(chunks)
    if len(text) > max_chars:
        return text[:max_chars] + "\n…[truncated]"
    return text


def file_jail_enforcer(
    path: str,
    *,
    max_chars: int | None = None,
) -> dict[str, Any]:
    """Read a file strictly inside DOCS_JAIL and return sanitized text."""
    try:
        limit = int(max_chars) if max_chars is not None else _MAX_READ_CHARS
    except (TypeError, ValueError):
        limit = _MAX_READ_CHARS
    limit = max(200, min(limit, _MAX_READ_CHARS))

    try:
        target = resolve_jailed_path(path)
    except JailError as exc:
        return {"ok": False, "error": str(exc), "text": "", "path": None}

    if not target.is_file():
        return {
            "ok": False,
            "error": f"file not found in jail: {target.name}",
            "text": "",
            "path": str(target.relative_to(DOCS_JAIL)).replace("\\", "/"),
        }

    try:
        if target.suffix.lower() == ".pdf":
            text = _read_pdf_file(target, limit)
        else:
            text = _read_text_file(target, limit)
    except JailError as exc:
        return {"ok": False, "error": str(exc), "text": "", "path": None}
    except OSError as exc:
        return {"ok": False, "error": f"read failed: {exc}", "text": "", "path": None}

    rel = str(target.relative_to(DOCS_JAIL)).replace("\\", "/")
    return {
        "ok": True,
        "error": None,
        "text": text,
        "path": rel,
        "chars": len(text),
        "jail": str(DOCS_JAIL),
    }


def handle_tool_call(call: Any) -> str:
    """Broker-compatible handler: ToolCall → Observation string."""
    args = getattr(call, "arguments", None) or {}
    if not isinstance(args, dict):
        args = {}
    path = args.get("path")
    if path is None or not str(path).strip():
        return "ERROR: missing path (jail-relative, e.g. sample_notes.txt)"
    max_chars = args.get("max_chars")
    result = file_jail_enforcer(str(path), max_chars=max_chars)
    if not result.get("ok"):
        return f"ERROR: file_jail_enforcer blocked/failed: {result.get('error')}"
    preview = result["text"]
    if len(preview) > 1200:
        preview = preview[:1200] + "\n…[truncated for observation]"
    return (
        f"OK: file_jail_enforcer path={result['path']!r} chars={result['chars']} "
        f"jail=docs text={preview!r}"
    )
