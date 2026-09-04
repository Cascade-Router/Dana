"""Unit tests for dana.api.server's ``_generate_code_diff`` — the optional
unified-diff preview attached to the ``hitl_approval_required`` payload for
``write_file``/``edit_file`` calls. Every other mutating tool_id has no
before/after text to diff, so it must return ``None`` rather than guessing.
"""

from __future__ import annotations

from pathlib import Path

from dana.api import server as server_module
from dana.tools.schema import ToolCall


def test_write_file_generates_diff(tmp_path: Path) -> None:
    target = tmp_path / "hello.py"
    target.write_text("print('old')\n", encoding="utf-8")

    call = ToolCall(tool_id="write_file", arguments={"path": str(target), "content": "print('new')\n"})

    diff = server_module._generate_code_diff(call)

    assert diff is not None
    assert "-print('old')" in diff
    assert "+print('new')" in diff


def test_edit_file_generates_diff_on_match(tmp_path: Path) -> None:
    target = tmp_path / "hello.py"
    target.write_text("def foo():\n    return 1\n", encoding="utf-8")

    call = ToolCall(
        tool_id="edit_file",
        arguments={"path": str(target), "search_block": "return 1", "replace_block": "return 2"},
    )

    diff = server_module._generate_code_diff(call)

    assert diff is not None
    assert "-    return 1" in diff
    assert "+    return 2" in diff


def test_edit_file_returns_none_when_search_block_missing(tmp_path: Path) -> None:
    target = tmp_path / "hello.py"
    target.write_text("def foo():\n    return 1\n", encoding="utf-8")

    call = ToolCall(
        tool_id="edit_file",
        arguments={"path": str(target), "search_block": "does not exist", "replace_block": "return 2"},
    )

    assert server_module._generate_code_diff(call) is None


def test_irrelevant_tool_returns_none() -> None:
    call = ToolCall(tool_id="create_freecad_box", arguments={"length": 40, "width": 25, "height": 15})

    assert server_module._generate_code_diff(call) is None
