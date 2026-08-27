"""Read-only architecture / capability self-awareness for Dana."""

from __future__ import annotations

import json
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
    if name in {"settings.json", ".env", "dana_memory.enc"}:
        raise ArchitectureAccessError(f"Blocked config path: {name}")
    return resolved


def read_architecture_markdown() -> str:
    path = _assert_allowed(ARCHITECTURE_MD)
    text = path.read_text(encoding="utf-8")
    if len(text) > _MAX_ARCH_CHARS:
        return text[:_MAX_ARCH_CHARS] + "\n\n[truncated]"
    return text


def summarize_tools_schema() -> dict[str, Any]:
    """``id`` + ``description_en`` only, per tool — no ``parameters``.

    With 110+ tools, a full per-parameter schema dump (name/type/required/
    enum for every parameter of every tool) previously blew well past
    Groq's TPM limit on its own, and wasn't even bounded by
    ``read_system_architecture``'s own ``_MAX_SCHEMA_CHARS`` truncation:
    that only capped the JSON-text serialization
    (``tools_schema_summary_text``), while the untruncated dict this
    function returns is ALSO handed back verbatim as
    ``tools_schema_summary`` — so the heavy, unbounded field was still
    reaching the model regardless of the text cap. A caller that actually
    needs a tool's exact parameter schema should reach for
    ``search_tool_catalog``/``load_specific_tool`` instead (see
    ``read_system_architecture``'s ``note``), not this overview.
    """
    path = _assert_allowed(TOOLS_JSON)
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    tools = [
        {
            "id": item.get("id"),
            "description_en": item.get("description_en") or "",
        }
        for item in payload.get("tools") or []
    ]
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
    # No "tools_schema_summary" (raw dict) field — it had no other consumer
    # in the codebase and, unlike tools_schema_summary_text above, was never
    # bounded by _MAX_SCHEMA_CHARS at all, so it stayed a real TPM-crash
    # risk even after summarize_tools_schema() stopped including parameters.
    return {
        "ok": True,
        "architecture_md": read_architecture_markdown(),
        "tools_schema_summary_text": schema_text,
        "note": (
            "Summarize for the user; do not dump raw markdown verbatim. "
            "Keep technical explanations clear and concise in English. "
            "Note: Parameters have been omitted for brevity. If you need "
            "the exact parameter schema for a tool, use search_tool_catalog "
            "or load_specific_tool."
        ),
    }
