"""Unit tests for dana.tools.schema_minify: minify_tool_schema condenses
every description (function + parameters); strip_tool_schema goes further
for local models but — per the Minification Compromise — keeps a condensed
(not dropped) function-level description, since a 7B model needs that
physical-shape context to pick the right tool instead of fixating on one
already-loaded tool. Pure-function tests: no LLM, no network.
"""

from __future__ import annotations

import json

import pytest

from dana.tools.schema_minify import (
    COMPACTION_LEVELS,
    compact_tool_schema,
    minify_tool_schema,
    minify_tool_schemas,
    should_strip_tool_schemas,
    strip_tool_schema,
    strip_tool_schemas,
)

_LONG_DESCRIPTION = (
    "Create a parametric box in FreeCAD. This tool builds a rectangular cuboid solid "
    "from length, width, and height parameters, optionally placed at a target position "
    "with a given normal vector, and always recomputes the document afterward."
)


def _schema(*, description: str, param_description: str = "The box length in millimeters.") -> dict:
    return {
        "type": "function",
        "function": {
            "name": "create_freecad_box",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "length": {"type": "number", "description": param_description},
                    "mode": {"type": "string", "enum": ["solid", "wireframe"]},
                },
                "required": ["length"],
            },
        },
    }


def test_minify_tool_schema_condenses_both_function_and_parameter_descriptions() -> None:
    schema = _schema(description=_LONG_DESCRIPTION)
    minified = minify_tool_schema(schema)
    fn = minified["function"]
    assert len(fn["description"]) <= len(_LONG_DESCRIPTION)
    assert fn["description"]  # not dropped
    assert fn["parameters"]["properties"]["length"]["description"]  # not dropped
    # Structural keys pass through byte-identical.
    assert fn["name"] == "create_freecad_box"
    assert fn["parameters"]["required"] == ["length"]
    assert fn["parameters"]["properties"]["mode"]["enum"] == ["solid", "wireframe"]


def test_minify_tool_schema_never_mutates_input() -> None:
    schema = _schema(description=_LONG_DESCRIPTION)
    original_description = schema["function"]["description"]
    minify_tool_schema(schema)
    assert schema["function"]["description"] == original_description


def test_strip_tool_schema_drops_parameter_descriptions_entirely() -> None:
    schema = _schema(description=_LONG_DESCRIPTION)
    stripped = strip_tool_schema(schema)
    length_prop = stripped["function"]["parameters"]["properties"]["length"]
    assert "description" not in length_prop
    assert length_prop["type"] == "number"


def test_strip_tool_schema_keeps_a_condensed_function_description() -> None:
    """The Minification Compromise: strip mode is more aggressive than
    minify mode everywhere else, but the function's OWN description
    survives, condensed to <=100 chars — the physical-shape context a 7B
    model needs to stop fixating on one already-loaded tool."""
    schema = _schema(description=_LONG_DESCRIPTION)
    stripped = strip_tool_schema(schema)
    fn = stripped["function"]
    assert "description" in fn
    assert 0 < len(fn["description"]) <= 100
    assert fn["description"] in _LONG_DESCRIPTION or _LONG_DESCRIPTION.startswith(fn["description"].rstrip("…."))


def test_strip_tool_schema_short_description_passes_through_unchanged() -> None:
    schema = _schema(description="Creates a box.")
    stripped = strip_tool_schema(schema)
    assert stripped["function"]["description"] == "Creates a box."


def test_strip_tool_schema_missing_description_omits_the_key() -> None:
    schema = _schema(description="")
    del schema["function"]["description"]
    stripped = strip_tool_schema(schema)
    assert "description" not in stripped["function"]


def test_strip_tool_schema_preserves_structural_parameter_keys() -> None:
    schema = _schema(description=_LONG_DESCRIPTION)
    stripped = strip_tool_schema(schema)
    fn = stripped["function"]
    assert fn["name"] == "create_freecad_box"
    assert fn["parameters"]["type"] == "object"
    assert fn["parameters"]["required"] == ["length"]
    assert fn["parameters"]["properties"]["length"]["type"] == "number"
    assert fn["parameters"]["properties"]["mode"]["enum"] == ["solid", "wireframe"]


def test_strip_tool_schema_never_mutates_input() -> None:
    schema = _schema(description=_LONG_DESCRIPTION)
    import copy

    before = copy.deepcopy(schema)
    strip_tool_schema(schema)
    assert schema == before


def test_strip_tool_schemas_batches_over_a_list() -> None:
    schemas = [_schema(description=_LONG_DESCRIPTION), _schema(description="Creates a cylinder.")]
    stripped = strip_tool_schemas(schemas)
    assert len(stripped) == 2
    assert all("description" in s["function"] for s in stripped)


def test_minify_tool_schemas_respects_disable_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DANA_MINIFY_SCHEMAS", "0")
    schemas = [_schema(description=_LONG_DESCRIPTION)]
    assert minify_tool_schemas(schemas) == schemas


def test_should_strip_tool_schemas_true_for_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DANA_MINIFY_TOOL_SCHEMAS", raising=False)
    assert should_strip_tool_schemas("ollama") is True
    assert should_strip_tool_schemas("openrouter") is False


def test_should_strip_tool_schemas_forced_via_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DANA_MINIFY_TOOL_SCHEMAS", "true")
    assert should_strip_tool_schemas("openrouter") is True


def test_minify_does_not_cut_a_description_at_an_abbreviation() -> None:
    """The first-sentence cut used to stop at "e.g.", sending create_plan's
    description as "Breaks a complex, long-horizon goal (e.g." on the wire."""
    description = (
        "Breaks a complex goal (e.g. 'build a full web app from scratch') down into an ordered checklist. "
        "Use this before starting a multi-step project, not partway through one, and only once per goal."
    )
    out = minify_tool_schema(_schema(description=description))
    assert out["function"]["description"] == (
        "Breaks a complex goal (e.g. 'build a full web app from scratch') down into an ordered checklist."
    )


def _placement_schema() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "create_freecad_box",
            "description": "Create a parametric rectangular box (Part::Box) solid in FreeCAD, saved as a .FCStd document.",
            "parameters": {
                "type": "object",
                "properties": {
                    "length": {"type": "number", "description": "Box length in mm, measured along X."},
                    "mode": {"type": "string", "enum": ["solid", "shell"], "description": "Whether to hollow it out."},
                    "placement_x": {
                        "type": "number",
                        "description": "X of the box's minimum corner in mm (the box spans x to x + length).",
                    },
                },
                "required": ["length"],
            },
        },
    }


@pytest.mark.parametrize("level", COMPACTION_LEVELS)
def test_compaction_keeps_the_call_contract_at_every_level(level: int) -> None:
    original = _placement_schema()
    compacted = compact_tool_schema(original, level)
    params, before = compacted["function"]["parameters"], original["function"]["parameters"]
    assert compacted["function"]["name"] == "create_freecad_box"
    assert params["required"] == ["length"]
    assert params["properties"]["mode"]["enum"] == ["solid", "shell"]
    assert {k: v["type"] for k, v in params["properties"].items()} == {
        k: v["type"] for k, v in before["properties"].items()
    }
    assert params["properties"]["placement_x"]["description"] == "X of the box's minimum corner in mm"
    assert original == _placement_schema()  # never mutated


def test_compaction_levels_only_get_smaller() -> None:
    sizes = [len(json.dumps(compact_tool_schema(_placement_schema(), level))) for level in COMPACTION_LEVELS]
    assert len(json.dumps(_placement_schema())) > sizes[0] >= sizes[1] >= sizes[2]
    level2 = compact_tool_schema(_placement_schema(), 2)["function"]["parameters"]["properties"]
    assert "description" not in level2["length"]  # non-placement prose goes at level 2
