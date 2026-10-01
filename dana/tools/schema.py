"""Language-agnostic tool Intermediate Representation (IR) for Dana."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError, create_model


@dataclass(frozen=True)
class ToolItemPropertySpec:
    """One field of a ``type == "array"``, ``items_type == "object"``
    parameter's item schema — e.g. ``create_plan``'s own
    ``tasks[i].expected_tools``. A deliberately flatter sibling of
    ``ToolParameterSpec`` (no ``items_type``/nested ``item_properties`` of
    its own beyond one scalar-array level) — one level of object nesting
    covers every structured-array need this codebase has so far; a second
    level would round-trip through this same shape recursively with no
    real use yet to justify the extra complexity.
    """

    name: str
    type: str
    required: bool = True
    items_type: str = ""
    description_en: str = ""


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
    # Nested item schema, populated ONLY when items_type == "object" (e.g.
    # create_plan's own `tasks`: each element is a {"description": str,
    # "expected_tools": [str]} object, not a bare scalar) — empty tuple for
    # every scalar-item array, which is every array param before this field
    # existed, so their to_openai_function_schema output is byte-for-byte
    # unchanged.
    item_properties: tuple[ToolItemPropertySpec, ...] = ()
    description_en: str = ""
    description_fa: str = ""
    # Only for ``type == "array"``: also accept a single bare item (e.g. one
    # feature name where a list of names is also allowed). The LLM-facing
    # schema still advertises the array; this only keeps argument validation
    # from rejecting a model that sends the one-element case as a plain value.
    accepts_single: bool = False


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
    # Opaque provider-specific wire metadata that must round-trip verbatim
    # onto the SAME function call when this turn's assistant message is
    # replayed in a later request — e.g. Gemini's OpenAI-compat endpoint
    # attaches {"google": {"thought_signature": "..."}} to each tool_calls[N]
    # entry and 400s on the next turn if it isn't echoed back exactly where
    # it was received. None for every provider that doesn't use this
    # (OpenAI, Groq, local Ollama) — see openai_tool_calls_to_ir below and
    # dana.core.react_dispatch.build_assistant_tool_call_message.
    provider_extra: dict[str, Any] | None = None


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
                item_properties=tuple(
                    ToolItemPropertySpec(
                        name=str(ip["name"]),
                        type=str(ip.get("type", "string")),
                        required=bool(ip.get("required", True)),
                        items_type=str(ip.get("items_type") or ""),
                        description_en=str(ip.get("description_en") or ""),
                    )
                    for ip in (p.get("items_properties") or [])
                ),
                description_en=str(p.get("description_en") or ""),
                description_fa=str(p.get("description_fa") or ""),
                accepts_single=bool(p.get("accepts_single", False)),
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
        if param.type == "array" and param.items_type == "object" and param.item_properties:
            item_props: dict[str, Any] = {}
            item_required: list[str] = []
            for ip in param.item_properties:
                item_prop: dict[str, Any] = {"type": ip.type or "string", "description": ip.description_en or ip.name}
                if ip.type == "array" and ip.items_type:
                    item_prop["items"] = {"type": ip.items_type}
                item_props[ip.name] = item_prop
                if ip.required:
                    item_required.append(ip.name)
            prop["items"] = {"type": "object", "properties": item_props, "required": item_required}
        elif param.type == "array" and param.items_type:
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
                # Silent Parameter Dropping fix: a provider whose own
                # function-calling implementation enforces JSON Schema
                # (e.g. OpenAI strict mode) now rejects an invented/
                # hallucinated extra parameter before Dana ever sees the
                # call at all — dana.tools.schema.validate_tool_arguments
                # is the second, ALWAYS-enforced layer of this same rule
                # for every provider that doesn't validate this itself.
                "additionalProperties": False,
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


# Silent Parameter Dropping fix: to_openai_function_schema (above) already
# tells the LLM/provider each tool's real properties/required/enum, but
# nothing anywhere in this pipeline ever enforced it — a call carrying a key
# that ISN'T one of those properties (an invented/hallucinated parameter,
# e.g. an early build of create_freecad_helix's own angle_offset before its
# schema entry existed) used to just sit unread in `args`, since every
# `_tool_*` handler in dana.core.react_dispatch only ever reads the SPECIFIC
# keys it knows about via `args.get(...)` — never raising on anything else.
# The two functions below are the enforcement half of that same schema:
# `tool_argument_model` mirrors a ToolSpec into a real pydantic model with
# `extra="forbid"`, and `validate_tool_arguments` is what
# dana.core.react_dispatch.dispatch_tool_call calls, for every registered
# tool, before its handler ever runs.
_JSON_TYPE_TO_PY: dict[str, type] = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
}

_ARGUMENT_MODEL_CACHE: dict[str, type[BaseModel]] = {}


# extra="forbid" is this whole feature's actual point; coerce_numbers_to_str
# is a deliberate, narrow escape hatch alongside it — several existing
# "string"-typed params (e.g. modify_freecad_parameter's new_value, which
# also accepts a "[x, y, z]" vector string, so it can't just be declared
# "number") are routinely sent as a bare JSON number by a real LLM/test
# caller, and every hand-written `_tool_*` handler already does its own
# `str(args.get(...))` coercion on read — rejecting that same value one
# layer earlier, here, would be a stricter-than-intended false positive,
# not a real "invented parameter" catch. A dict/list is NEVER coerced into
# a scalar string field regardless (confirmed: still raises), so a
# genuinely wrong-SHAPED value is still caught.
_MODEL_CONFIG = ConfigDict(extra="forbid", coerce_numbers_to_str=True)


def _item_annotation(param: ToolParameterSpec) -> Any:
    """The element type for one ``type == "array"`` parameter.

    Deliberately ``Any`` for ``items_type == "object"`` (e.g. ``create_plan``'s
    own ``tasks``) rather than a nested ``extra="forbid"`` model of
    ``item_properties`` — confirmed live that ``_tool_create_plan`` itself
    documents and accepts a BARE STRING per task ("a bare string is still
    accepted ... treated as a task with no declared expected_tools") as a
    deliberate alternative to the full ``{"description", "expected_tools"}``
    object shape, so a strict nested model would reject that already-
    supported, already-tested calling convention as if it were the exact
    "invented parameter" problem this whole module exists to catch. This
    keeps enforcement where the reported problem actually is — a bogus
    TOP-LEVEL parameter name, or the wrong JSON type for one that's a plain
    scalar — without guessing at how permissive a specific tool's own
    nested-array business logic is allowed to be.
    """
    return _JSON_TYPE_TO_PY.get(param.items_type, Any)


def tool_argument_model(spec: ToolSpec) -> type[BaseModel]:
    """A pydantic model for ``spec``'s own declared parameters, with
    ``extra="forbid"``. Every field is Optional here REGARDLESS of the
    tool's own ``required`` flag — this model's only job is catching a
    parameter the LLM invented that isn't declared at all (or the wrong
    JSON *shape* for one that is); a genuinely MISSING required field
    still falls through to that tool's own hand-written ``_tool_*``
    handler, which already raises its own clearer, tool-specific "X is
    required" message, and an ``enum``-declared param's VALUE (e.g.
    ``operation: "bogus"``) is deliberately left to that same handler too
    — it already rejects an unknown enum value with a friendlier,
    tool-specific message than a generic pydantic one would, and that
    value was never the thing silently vanishing; only an entirely
    undeclared key was. Cached per tool_id: tools.json (and any
    manifest.json-derived ToolSpec) is static for the life of the process,
    so this is built at most once per tool ever dispatched.
    """
    cached = _ARGUMENT_MODEL_CACHE.get(spec.id)
    if cached is not None:
        return cached
    fields: dict[str, Any] = {}
    for param in spec.parameters:
        if param.type == "array":
            item = _item_annotation(param)
            py_type = list[item] | item if param.accepts_single else list[item]
        else:
            py_type = _JSON_TYPE_TO_PY.get(param.type, Any)
        fields[param.name] = (py_type | None, None)
    model = create_model(f"_{spec.id}_args", __config__=_MODEL_CONFIG, **fields)
    _ARGUMENT_MODEL_CACHE[spec.id] = model
    return model


def validate_tool_arguments(spec: ToolSpec, arguments: dict[str, Any]) -> str | None:
    """``None`` when ``arguments`` matches ``spec``'s own schema; otherwise
    a plain-English message — worded for the Honest Error Handler to hand
    straight back to the LLM as this call's own failure reason, so it
    rewrites its NEXT call using only real, declared properties instead of
    an extra key silently vanishing with no error at all — naming exactly
    which key(s) don't belong or which are the wrong type.
    """
    try:
        tool_argument_model(spec).model_validate(arguments)
    except ValidationError as exc:
        problems: list[str] = []
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"]) or "(top level)"
            if err["type"] == "extra_forbidden":
                problems.append(f"unexpected parameter '{loc}' is not part of {spec.id}'s schema")
            else:
                problems.append(f"'{loc}': {err['msg']}")
        return f"{spec.id} received invalid arguments — " + "; ".join(problems)
    return None


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
        # Gemini's OpenAI-compat endpoint (only) rides its thought_signature
        # here — see ToolCall.provider_extra's own docstring. Absent for
        # OpenAI/Groq/Ollama responses, so this is a no-op for them.
        raw_extra = raw.get("extra_content")
        provider_extra = raw_extra if isinstance(raw_extra, dict) else None
        calls.append(
            ToolCall(
                tool_id=name,
                arguments=args,
                source_lang=source_lang,
                raw_text=raw_text,
                confidence=1.0,
                provider_extra=provider_extra,
            )
        )
    return calls
