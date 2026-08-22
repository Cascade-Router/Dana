"""Tests for the /api/skills REST endpoints backing the frontend's
SkillsPlugin — GET must reflect exactly what's currently loaded/dispatchable
(dana.core.react_dispatch.list_user_skills), and DELETE must remove the
skill's file and hot-reload the registry through the SAME
resolve_sandboxed_path traversal check the agent's own delete_skill tool
uses. The os_tools sandbox root and react_dispatch's skill registry are
both reset by tests/conftest.py's global autouse fixtures — no test here
ever touches the real AGENT_WORKSPACE_DIR, and nothing leaks between tests.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

import dana.core.react_dispatch as rd
import dana.core.skill_loader as skill_loader
from dana.api import server as server_module
from dana.platform.mock import MockControlPlane, MockFreeCADEngine
from dana.tools.schema import ToolCall


@pytest.fixture(autouse=True)
def _mock_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server_module, "get_cad_engine", lambda: MockFreeCADEngine())
    monkeypatch.setattr(server_module, "get_control_plane", lambda: MockControlPlane())


@pytest.fixture
def client() -> TestClient:
    return TestClient(server_module.app)


_DOUBLE_CODE = "def run(args):\n    return {\"ok\": True, \"result\": args[\"n\"] * 2}\n"


def _double_schema(name: str = "double_number") -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Doubles a number.",
            "parameters": {
                "type": "object",
                "properties": {"n": {"type": "number", "description": "Number to double."}},
                "required": ["n"],
            },
        },
    }


def _save_and_load_double_number() -> None:
    result = skill_loader.save_skill("double_number", _DOUBLE_CODE, _double_schema())
    assert result["ok"] is True
    rd.refresh_user_skills()
    assert "double_number" in rd._USER_SKILL_TOOL_IDS


# --------------------------------------------------------------------------
# GET /api/skills
# --------------------------------------------------------------------------


def test_get_skills_empty_when_none_loaded(client: TestClient) -> None:
    resp = client.get("/api/skills")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "skills": []}


def test_get_skills_reflects_a_currently_loaded_skill(client: TestClient) -> None:
    _save_and_load_double_number()

    resp = client.get("/api/skills")

    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert len(body["skills"]) == 1
    skill = body["skills"][0]
    assert skill["name"] == "double_number"
    assert skill["description"] == "Doubles a number."
    assert "def run(args):" in skill["code"]
    assert 'TOOL_SCHEMA = {' in skill["code"]


def test_get_skills_only_lists_what_is_actually_loaded_not_stale_disk_files(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GET reflects list_user_skills() (the live registry), not an
    independent re-scan — a broken .py file sitting on disk that never
    successfully loaded must not appear."""
    from dana.plugins.os import file_system

    skills_dir = file_system._SANDBOX_ROOT / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    (skills_dir / "broken.py").write_text("this is not : valid python(", encoding="utf-8")
    rd.refresh_user_skills()

    resp = client.get("/api/skills")
    assert resp.json() == {"ok": True, "skills": []}


# --------------------------------------------------------------------------
# DELETE /api/skills/{skill_name}
# --------------------------------------------------------------------------


def test_delete_skill_removes_it_and_hot_reloads(client: TestClient) -> None:
    _save_and_load_double_number()

    resp = client.delete("/api/skills/double_number")

    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "deleted": True}
    assert "double_number" not in rd.TOOL_HANDLERS
    assert "double_number" not in rd._CAPABILITY_TOOL_IDS["user_skills"]

    # And it's gone from the GET listing too.
    assert client.get("/api/skills").json() == {"ok": True, "skills": []}


def test_delete_skill_is_idempotent_for_a_missing_skill(client: TestClient) -> None:
    resp = client.delete("/api/skills/never_existed")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "deleted": False}


def test_delete_skill_rejects_path_traversal_attempt(client: TestClient) -> None:
    """The skill_name path parameter is validated the SAME way
    dana.core.skill_loader.is_valid_skill_name (and resolve_sandboxed_path
    underneath delete_skill) would reject it — a crafted name must never
    reach disk deletion logic at all."""
    resp = client.delete("/api/skills/bad$name!")
    assert resp.status_code == 400


def test_delete_skill_does_not_touch_files_outside_the_skills_directory(client: TestClient) -> None:
    """Defense-in-depth: even if a segment somehow slipped past the
    is_valid_skill_name filter, resolve_sandboxed_path underneath
    dana.core.skill_loader.delete_skill still confines every delete to
    skills/ — verified here by placing a REAL file directly in the
    sandbox root (one level above skills/) and confirming it survives
    any skill deletion attempt naming it."""
    from dana.plugins.os import file_system

    sentinel = file_system._SANDBOX_ROOT / "important.txt"
    sentinel.write_text("do not delete me", encoding="utf-8")

    client.delete("/api/skills/important")  # valid name, but no skills/important.py exists

    assert sentinel.is_file()
    assert sentinel.read_text(encoding="utf-8") == "do not delete me"


# --------------------------------------------------------------------------
# PUT /api/skills/{skill_name} — the frontend SkillsPlugin's manual-edit path
# --------------------------------------------------------------------------


def test_put_skill_overwrites_source_and_hot_reloads(client: TestClient) -> None:
    """PUT writes the given code VERBATIM (no TOOL_SCHEMA re-wrapping) —
    so a realistic edit starts from the ACTUAL current file content (what
    the frontend's textarea would be pre-filled with via GET) and tweaks
    it, keeping the existing TOOL_SCHEMA assignment intact."""
    _save_and_load_double_number()
    original_code = client.get("/api/skills").json()["skills"][0]["code"]
    triple_code = original_code.replace("* 2", "* 3")

    resp = client.put("/api/skills/double_number", json={"code": triple_code})

    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "skill_name": "double_number"}

    # The hot-reloaded handler reflects the NEW code, not the original —
    # dispatch through react_dispatch proves the registry actually swapped.
    result = rd.dispatch_tool_call(ToolCall(tool_id="double_number", arguments={"n": 10}), None, None)
    assert result.ok is True
    assert result.payload["result"] == 30

    # And GET's source view reflects the edit too.
    skill = client.get("/api/skills").json()["skills"][0]
    assert "* 3" in skill["code"]


def test_put_skill_rejects_syntax_error_with_specific_detail(client: TestClient) -> None:
    _save_and_load_double_number()

    resp = client.put("/api/skills/double_number", json={"code": "def run(args:\n    not valid python"})

    assert resp.status_code == 400
    assert "SyntaxError" in resp.json()["detail"]
    # The now-broken skill must be dropped from the registry, not left
    # dispatchable with its STALE (pre-edit) working handler.
    assert "double_number" not in rd.TOOL_HANDLERS
    assert "double_number" not in rd._CAPABILITY_TOOL_IDS["user_skills"]


def test_put_skill_rejects_missing_run_function(client: TestClient) -> None:
    _save_and_load_double_number()

    resp = client.put("/api/skills/double_number", json={"code": "x = 1\n"})

    assert resp.status_code == 400
    assert "run" in resp.json()["detail"]
    assert "double_number" not in rd.TOOL_HANDLERS


def test_put_skill_rejects_invalid_skill_name(client: TestClient) -> None:
    resp = client.put("/api/skills/bad$name!", json={"code": "def run(args):\n    return {'ok': True}\n"})
    assert resp.status_code == 400


def test_put_skill_rejects_empty_code(client: TestClient) -> None:
    _save_and_load_double_number()
    resp = client.put("/api/skills/double_number", json={"code": "   "})
    assert resp.status_code == 400
    # An empty-code PUT never even reached write_skill_source's file write —
    # the previously-working skill must be untouched.
    assert "double_number" in rd.TOOL_HANDLERS
