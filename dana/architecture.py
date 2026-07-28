"""Read-only architecture / capability self-awareness for Donna."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dana.paths import ARCHITECTURE_MD, PROJECT_ROOT, TOOLS_JSON

_ROOT = PROJECT_ROOT
_ALLOWED_DOCS = frozenset(
    {
        ARCHITECTURE_MD.resolve(),
        TOOLS_JSON.resolve(),
    }
)
_MAX_ARCH_CHARS = 24_000
_MAX_SCHEMA_CHARS = 8_000


class ArchitectureAccessError(PermissionError):
    pass


def _assert_allowed(path: Path) -> Path:
    resolved = path.resolve()
    if resolved not in _ALLOWED_DOCS:
        raise ArchitectureAccessError(
            f"Path rejected — outside documentation scope: {resolved}"
        )
    # Extra belt: never allow .py / .env / settings via this API.
    suffix = resolved.suffix.lower()
    if suffix in {".py", ".env", ".enc", ".key", ".pem"}:
        raise ArchitectureAccessError(f"Blocked file type: {suffix}")
    name = resolved.name.lower()
    if name in {"settings.json", ".env", "donna_memory.enc"}:
        raise ArchitectureAccessError(f"Blocked config path: {name}")
    return resolved


def read_architecture_markdown() -> str:
    path = _assert_allowed(ARCHITECTURE_MD)
    text = path.read_text(encoding="utf-8")
    if len(text) > _MAX_ARCH_CHARS:
        return text[:_MAX_ARCH_CHARS] + "\n\n[truncated]"
    return text


def summarize_tools_schema() -> dict[str, Any]:
    path = _assert_allowed(TOOLS_JSON)
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    tools = []
    for item in payload.get("tools") or []:
        params = item.get("parameters") or []
        tools.append(
            {
                "id": item.get("id"),
                "description_en": item.get("description_en") or "",
                "parameters": [
                    {
                        "name": p.get("name"),
                        "type": p.get("type", "string"),
                        "required": bool(p.get("required", True)),
                        "enum": list(p.get("enum") or []),
                    }
                    for p in params
                ],
            }
        )
    return {
        "version": payload.get("version"),
        "tool_count": len(tools),
        "tools": tools,
    }


def read_system_architecture() -> dict[str, Any]:
    """Safe payload for the read_system_architecture tool."""
    schema = summarize_tools_schema()
    schema_text = json.dumps(schema, ensure_ascii=False, indent=2)
    if len(schema_text) > _MAX_SCHEMA_CHARS:
        schema_text = schema_text[:_MAX_SCHEMA_CHARS] + "\n[truncated]"
    return {
        "ok": True,
        "architecture_md": read_architecture_markdown(),
        "tools_schema_summary": schema,
        "tools_schema_summary_text": schema_text,
        "note": (
            "Summarize for the user; do not dump raw markdown verbatim. "
            "Keep technical explanations clear and concise in English."
        ),
    }
