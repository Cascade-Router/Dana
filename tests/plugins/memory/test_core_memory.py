"""Tests for dana.plugins.memory.core_memory — the persistent Core Memory
store behind Dana's "session amnesia" fix, plus its injection into
dana.core.react_dispatch.build_system_prompt. Every test redirects
CORE_MEMORY_PATH to a throwaway temp file (see the autouse `_memory_file`
fixture) — none of these ever touch the real on-disk agent_workspace.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dana.core import react_dispatch as rd
from dana.plugins.memory import core_memory


@pytest.fixture(autouse=True)
def _memory_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "data" / "core_memory.json"
    monkeypatch.setattr(core_memory, "CORE_MEMORY_PATH", path)
    return path


# --------------------------------------------------------------------------
# read_core_memory / write_core_memory
# --------------------------------------------------------------------------


def test_read_core_memory_returns_empty_dict_when_file_missing() -> None:
    assert core_memory.read_core_memory() == {}


def test_write_core_memory_creates_file_and_parent_dirs(_memory_file: Path) -> None:
    assert not _memory_file.parent.exists()
    result = core_memory.write_core_memory("user_preferences", "prefers metric units")
    assert result == {
        "ok": True,
        "section": "user_preferences",
        "content": "prefers metric units",
        "memory": {"user_preferences": "prefers metric units"},
    }
    assert _memory_file.is_file()


def test_write_then_read_round_trips(_memory_file: Path) -> None:
    core_memory.write_core_memory("active_project", "60x40x20mm enclosure, aluminum")
    assert core_memory.read_core_memory() == {"active_project": "60x40x20mm enclosure, aluminum"}


def test_writing_a_second_section_does_not_clobber_the_first(_memory_file: Path) -> None:
    core_memory.write_core_memory("user_preferences", "prefers metric units")
    core_memory.write_core_memory("active_project", "60x40x20mm enclosure")
    assert core_memory.read_core_memory() == {
        "user_preferences": "prefers metric units",
        "active_project": "60x40x20mm enclosure",
    }


def test_rewriting_the_same_section_overwrites_it(_memory_file: Path) -> None:
    core_memory.write_core_memory("active_project", "first draft")
    core_memory.write_core_memory("active_project", "revised spec")
    assert core_memory.read_core_memory() == {"active_project": "revised spec"}


def test_write_core_memory_rejects_empty_section(_memory_file: Path) -> None:
    result = core_memory.write_core_memory("   ", "some content")
    assert result["ok"] is False
    assert core_memory.read_core_memory() == {}


def test_read_core_memory_degrades_gracefully_on_corrupt_json(_memory_file: Path) -> None:
    _memory_file.parent.mkdir(parents=True)
    _memory_file.write_text("{not valid json", encoding="utf-8")
    assert core_memory.read_core_memory() == {}


def test_read_core_memory_degrades_gracefully_on_non_dict_json(_memory_file: Path) -> None:
    _memory_file.parent.mkdir(parents=True)
    _memory_file.write_text("[1, 2, 3]", encoding="utf-8")
    assert core_memory.read_core_memory() == {}


def test_read_core_memory_drops_non_string_values() -> None:
    core_memory.write_core_memory("valid_section", "text content")
    # Simulate a foreign/hand-edited file with a non-string value mixed in.
    import json

    data = json.loads(core_memory.CORE_MEMORY_PATH.read_text(encoding="utf-8"))
    data["bad_section"] = {"nested": "object"}
    core_memory.CORE_MEMORY_PATH.write_text(json.dumps(data), encoding="utf-8")
    assert core_memory.read_core_memory() == {"valid_section": "text content"}


# --------------------------------------------------------------------------
# format_core_memory_for_prompt
# --------------------------------------------------------------------------


def test_format_core_memory_for_prompt_empty_returns_empty_string() -> None:
    assert core_memory.format_core_memory_for_prompt({}) == ""
    assert core_memory.format_core_memory_for_prompt() == ""  # reads the (missing) file itself


def test_format_core_memory_for_prompt_renders_heading_and_sections() -> None:
    text = core_memory.format_core_memory_for_prompt(
        {"user_preferences": "prefers metric units", "active_project": "60x40x20mm enclosure"}
    )
    assert text.startswith("## Persistent Core Memory")
    assert "- active_project: 60x40x20mm enclosure" in text
    assert "- user_preferences: prefers metric units" in text


# --------------------------------------------------------------------------
# System-prompt injection (dana.core.react_dispatch.build_system_prompt)
# --------------------------------------------------------------------------


def test_build_system_prompt_omits_memory_section_when_empty() -> None:
    prompt = rd.build_system_prompt(None)
    assert "Persistent Core Memory" not in prompt


def test_build_system_prompt_appends_memory_section_when_present() -> None:
    core_memory.write_core_memory("user_preferences", "prefers metric units")
    prompt = rd.build_system_prompt(None)
    assert prompt.rstrip().endswith("- user_preferences: prefers metric units")
    assert "## Persistent Core Memory" in prompt


def test_build_system_prompt_memory_section_survives_active_selection_text() -> None:
    core_memory.write_core_memory("active_project", "60x40x20mm enclosure")
    selection = {"centroid": [1.0, 2.0, 3.0], "normal": [0.0, 1.0, 0.0]}
    prompt = rd.build_system_prompt(selection)
    assert "[1.0, 2.0, 3.0]" in prompt
    assert "## Persistent Core Memory" in prompt
    # Memory section is the last block appended, after the selection note.
    assert prompt.index("Persistent Core Memory") > prompt.index("[1.0, 2.0, 3.0]")


# --------------------------------------------------------------------------
# update_core_memory tool wiring
# --------------------------------------------------------------------------


def test_update_core_memory_tool_is_always_core_available() -> None:
    assert "update_core_memory" in rd._CORE_TOOL_IDS
    assert "update_core_memory" in rd.TOOL_HANDLERS
    assert rd.is_mutating_tool("update_core_memory") is False


def test_update_core_memory_tool_handler_writes_through(_memory_file: Path) -> None:
    handler = rd.TOOL_HANDLERS["update_core_memory"]
    result = handler({"section": "user_preferences", "content": "likes dark mode"}, None, None)
    assert result["ok"] is True
    assert core_memory.read_core_memory() == {"user_preferences": "likes dark mode"}
