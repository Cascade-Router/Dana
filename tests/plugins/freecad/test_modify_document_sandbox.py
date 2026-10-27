"""modify_existing_freecad_document takes an LLM-chosen path and saves over
it, so it may only open .FCStd documents inside Dana's own workspace."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dana.plugins.freecad import engine


@pytest.fixture
def workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    root = tmp_path / "freecad_output"
    root.mkdir()
    monkeypatch.setattr(engine, "_OUTPUT_DIR", root)
    return root


def _modify(path: Path | str) -> dict:
    return json.loads(engine.modify_existing_document(str(path), "pass"))


def test_document_outside_the_workspace_is_refused(workspace: Path, tmp_path: Path) -> None:
    outside = tmp_path / "elsewhere" / "user_part.FCStd"
    outside.parent.mkdir()
    outside.write_bytes(b"x")
    result = _modify(outside)
    assert result["ok"] is False
    assert "not a .FCStd document inside Dana's workspace" in result["error"]


def test_dot_dot_traversal_out_of_the_workspace_is_refused(workspace: Path, tmp_path: Path) -> None:
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere" / "user_part.FCStd").write_bytes(b"x")
    assert _modify(workspace / ".." / "elsewhere" / "user_part.FCStd")["ok"] is False


def test_non_fcstd_file_inside_the_workspace_is_refused(workspace: Path) -> None:
    (workspace / "notes.txt").write_text("x", encoding="utf-8")
    assert _modify(workspace / "notes.txt")["ok"] is False


def test_session_document_inside_the_workspace_is_allowed(workspace: Path) -> None:
    inside = workspace / "sessions" / "s1" / "Session_Active.FCStd"
    inside.parent.mkdir(parents=True)
    inside.write_bytes(b"x")
    assert engine._workspace_document_path(str(inside)) == inside.resolve()
