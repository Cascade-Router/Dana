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
# _MAX_MEMORY_CHARS eviction (oldest-written section first)
# --------------------------------------------------------------------------

# ~600 chars per section: three fit under the 2000-char cap, four don't.
_BIG = "x" * 600


def _on_disk(path: Path) -> dict:
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict) and all(isinstance(v, str) for v in data.values())
    return data


def test_write_over_cap_drops_oldest_section_from_disk(_memory_file: Path) -> None:
    # Written in reverse-alphabetical order, so oldest-first and alphabetical
    # eviction would drop different sections.
    for section in ("zeta", "gamma", "beta"):
        core_memory.write_core_memory(section, _BIG)
    result = core_memory.write_core_memory("alpha", _BIG)

    assert result["ok"] is True
    assert list(result["memory"]) == ["gamma", "beta", "alpha"]
    data = _on_disk(_memory_file)
    assert list(data) == ["gamma", "beta", "alpha"]
    assert core_memory._rendered_size(data) <= core_memory._MAX_MEMORY_CHARS


def test_rewriting_a_section_refreshes_its_recency(_memory_file: Path) -> None:
    for section in ("zeta", "gamma", "beta"):
        core_memory.write_core_memory(section, _BIG)
    core_memory.write_core_memory("zeta", _BIG)  # zeta is now the newest
    core_memory.write_core_memory("alpha", _BIG)

    assert list(_on_disk(_memory_file)) == ["beta", "zeta", "alpha"]


def test_write_order_survives_a_reload(_memory_file: Path) -> None:
    for section in ("zeta", "gamma"):
        core_memory.write_core_memory(section, _BIG)
    # A restart re-reads the file; the order on disk is the only record of
    # write recency, so it must come back unchanged (not sorted).
    assert list(core_memory.read_core_memory()) == ["zeta", "gamma"]
    core_memory.write_core_memory("beta", _BIG)
    core_memory.write_core_memory("alpha", _BIG)

    assert list(_on_disk(_memory_file)) == ["gamma", "beta", "alpha"]


def test_single_oversize_section_is_kept_not_evicted_to_empty(_memory_file: Path) -> None:
    core_memory.write_core_memory("old", "short note")
    result = core_memory.write_core_memory("huge", "y" * (core_memory._MAX_MEMORY_CHARS + 500))

    assert result["content"] == "y" * (core_memory._MAX_MEMORY_CHARS + 500)
    assert list(_on_disk(_memory_file)) == ["huge"]


def test_replace_core_memory_evicts_oldest_when_over_cap(_memory_file: Path) -> None:
    result = core_memory.replace_core_memory({"zeta": _BIG, "gamma": _BIG, "beta": _BIG, "alpha": _BIG})

    assert result["ok"] is True
    assert list(_on_disk(_memory_file)) == ["gamma", "beta", "alpha"]


def test_format_recaps_an_oversize_file_without_rewriting_it(_memory_file: Path) -> None:
    import json

    # A file written before the cap existed, or hand-edited on disk.
    original = {"zeta": _BIG, "gamma": _BIG, "beta": _BIG, "alpha": _BIG}
    _memory_file.parent.mkdir(parents=True)
    _memory_file.write_text(json.dumps(original), encoding="utf-8")

    rendered = core_memory.format_core_memory_for_prompt()
    assert "- zeta:" not in rendered
    assert all(f"- {s}:" in rendered for s in ("gamma", "beta", "alpha"))
    assert _on_disk(_memory_file) == original


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
