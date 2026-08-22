"""Language-agnostic tool Intermediate Representation (IR) for Dana."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolParameterSpec:
    name: str
    type: str
    required: bool = True
    enum: tuple[str, ...] = ()
    # JSON-schema element type for ``type == "array"`` params (e.g. "number"
    # for a [x, y, z] vector) — kept as a plain string rather than a nested
    # dict so this dataclass stays trivially hashable.
    items_type: str = ""
    description_en: str = ""
    description_fa: str = ""


@dataclass(frozen=True)
class ToolSpec:
    id: str
    description_en: str
    description_fa: str
    parameters: tuple[ToolParameterSpec, ...] = ()
    aliases_en: dict[str, tuple[str, ...]] = field(default_factory=dict)
    aliases_fa: dict[str, tuple[str, ...]] = field(default_factory=dict)
    dynamic: bool = False
    # The SOLE source of truth for HITL gating, for every tool regardless of
    # origin (tools.json OR a manifest.json plugin) — see
    # dana.core.react_dispatch.is_mutating_tool. Defaults to False (HITL-
    # gated) so ANY tool — a brand-new native handler someone forgot to
    # annotate, or a third-party plugin — fails SAFE by default: its author
    # must explicitly opt OUT of approval gating (``"read_only": true`` in
    # its declaration) rather than opt in to being dangerous. This
    # deliberately replaced an older design (a hardcoded
    # ``MUTATING_TOOLS`` allow-list in react_dispatch.py that tools.json
    # tools were checked against, with this field only read for plugins) —
    # that design fails OPEN: a new native tool nobody remembered to add to
    # the list would silently dispatch with no human approval.
    read_only: bool = False


@dataclass
class ToolCall:
    """Normalized, language-agnostic tool invocation."""

    tool_id: str
    arguments: dict[str, Any]
    source_lang: str = "en"  # en | fa | mixed
    raw_text: str = ""
    confidence: float = 1.0


def _as_tuple_map(raw: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    out: dict[str, tuple[str, ...]] = {}
    for key, values in (raw or {}).items():
        if isinstance(values, list):
            out[str(key)] = tuple(str(v) for v in values)
        elif isinstance(values, str):
            out[str(key)] = (values,)
    return out


def load_tool_registry(path: str | None = None) -> dict[str, ToolSpec]:
    registry_path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools.json")
    with open(registry_path, encoding="utf-8") as fh:
        payload = json.load(fh)
    tools: dict[str, ToolSpec] = {}
    for item in payload.get("tools", []):
        params = tuple(
            ToolParameterSpec(
                name=str(p["name"]),
                type=str(p.get("type", "string")),
                required=bool(p.get("required", True)),
                enum=tuple(str(x) for x in (p.get("enum") or [])),
                items_type=str(p.get("items_type") or ""),
                description_en=str(p.get("description_en") or ""),
                description_fa=str(p.get("description_fa") or ""),
            )
            for p in (item.get("parameters") or [])
        )
        spec = ToolSpec(
            id=str(item["id"]),
            description_en=str(item.get("description_en") or ""),
            description_fa=str(item.get("description_fa") or ""),
            parameters=params,
            aliases_en=_as_tuple_map(item.get("aliases_en") or {}),
            aliases_fa=_as_tuple_map(item.get("aliases_fa") or {}),
            dynamic=bool(item.get("dynamic", False)),
            # Fail-closed: absent/false means HITL-gated — see ToolSpec.
            # read_only's own docstring. A tools.json entry must explicitly
            # declare "read_only": true to dispatch without approval.
            read_only=bool(item.get("read_only", False)),
        )
        tools[spec.id] = spec
    return tools


def tool_schema_public(registry: dict[str, ToolSpec]) -> list[dict[str, Any]]:
    """Compact IR for prompts / debugging (language-agnostic ids + enums)."""
    out: list[dict[str, Any]] = []
    for spec in registry.values():
        out.append(
            {
                "id": spec.id,
                "parameters": [
                    {
                        "name": p.name,
                        "type": p.type,
                        "required": p.required,
                        "enum": list(p.enum),
                    }
                    for p in spec.parameters
                ],
            }
        )
    return out


def to_openai_function_schema(spec: ToolSpec) -> dict[str, Any]:
    """OpenAI / Ollama function-calling schema for a single ToolSpec."""
    properties: dict[str, Any] = {}
    required: list[str] = []
    for param in spec.parameters:
        prop: dict[str, Any] = {
            "type": param.type or "string",
            "description": param.description_en or param.name,
        }
        if param.enum:
            prop["enum"] = list(param.enum)
        if param.type == "array" and param.items_type:
            prop["items"] = {"type": param.items_type}
        properties[param.name] = prop
        if param.required:
            required.append(param.name)
    return {
        "type": "function",
        "function": {
            "name": spec.id,
            "description": (spec.description_en or spec.id).strip(),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def openai_tools_schema(
    registry: dict[str, ToolSpec],
    *,
    tool_ids: set[str] | frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    """OpenAI-style tools array suitable for Ollama / LangGraph bind_tools."""
    out: list[dict[str, Any]] = []
    for spec in registry.values():
        if tool_ids is not None and spec.id not in tool_ids:
            continue
        out.append(to_openai_function_schema(spec))
    return out


def openai_tool_calls_to_ir(
    raw_tool_calls: list[dict[str, Any]] | None,
    *,
    raw_text: str = "",
    source_lang: str = "en",
) -> list[ToolCall]:
    """Map an OpenAI ``message.tool_calls`` array onto Dana's ``ToolCall`` IR.

    Malformed ``function.arguments`` JSON degrades to an empty-args
    ``ToolCall`` rather than raising — a broken cloud tool call should fail
    that tool's own argument validation downstream, not crash the turn.
    """
    calls: list[ToolCall] = []
    for raw in raw_tool_calls or []:
        fn = (raw or {}).get("function") or {}
        name = str(fn.get("name") or "").strip()
        if not name:
            continue
        raw_args = fn.get("arguments")
        args: dict[str, Any] = {}
        if isinstance(raw_args, dict):
            args = raw_args
        elif isinstance(raw_args, str) and raw_args.strip():
            try:
                parsed = json.loads(raw_args)
                if isinstance(parsed, dict):
                    args = parsed
            except (json.JSONDecodeError, ValueError):
                args = {}
        calls.append(
            ToolCall(
                tool_id=name,
                arguments=args,
                source_lang=source_lang,
                raw_text=raw_text,
                confidence=1.0,
            )
        )
    return calls
