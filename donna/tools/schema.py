"""Language-agnostic tool Intermediate Representation (IR) for Donna."""

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
