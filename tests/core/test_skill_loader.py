"""Tests for Autonomous Skill Acquisition: dana.core.skill_loader's pure
save/load helpers, plus dana.core.react_dispatch's hot-reloading registry
integration (save_new_skill -> refresh_user_skills -> dispatchable in the
next ReAct turn).

The real os_tools sandbox root (dana.plugins.os.file_system._SANDBOX_ROOT)
is already redirected to a throwaway per-test tmp_path by tests/conftest.py's
global `_isolate_os_tools_sandbox` autouse fixture — skill_loader reuses
that SAME sandbox (skills/ lives under it), so no test here ever touches
the real AGENT_WORKSPACE_DIR. dana.core.react_dispatch's own process-wide
skill registry state (TOOL_HANDLERS/_USER_SKILL_TOOL_IDS/
_CAPABILITY_TOOL_IDS["user_skills"]) is reset after every test by
tests/conftest.py's global `_reset_user_skills_registry` autouse fixture —
not repeated here, since tests/api/test_skills_api.py needs the exact same
cleanup and duplicating it would risk the two copies drifting apart.
"""

from __future__ import annotations

import asyncio
import textwrap
from typing import Any

import pytest

import dana.core.react_dispatch as rd
import dana.core.skill_loader as skill_loader
from dana.plugins.os import file_system
from dana.tools.schema import ToolCall


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


_DOUBLE_CODE = textwrap.dedent(
    """\
    def run(args):
        return {"ok": True, "result": args["n"] * 2}
    """
)


class _FakeProvider:
    def __init__(self, tool_calls: list[ToolCall] | None = None, content: str = "") -> None:
        self._tool_calls = tool_calls or []
        self._content = content

    def complete_with_tool_calls(self, messages: Any, *, tools: Any, provider: Any = None, **kwargs: Any) -> dict:
        return {"content": self._content, "tool_calls": self._tool_calls, "provider": "test"}


def _mock_llm(monkeypatch: pytest.MonkeyPatch, tool_calls: list[ToolCall]) -> None:
    fake = _FakeProvider(tool_calls=tool_calls)
    monkeypatch.setattr(rd, "ModelProvider", lambda **_kwargs: fake)


# --------------------------------------------------------------------------
# skill_loader — pure validation helpers
# --------------------------------------------------------------------------


def test_is_valid_skill_name() -> None:
    assert skill_loader.is_valid_skill_name("convert_csv_to_json") is True
    assert skill_loader.is_valid_skill_name("a") is True
    assert skill_loader.is_valid_skill_name("") is False
    assert skill_loader.is_valid_skill_name("Convert") is False  # uppercase
    assert skill_loader.is_valid_skill_name("1skill") is False  # leading digit
    assert skill_loader.is_valid_skill_name("../evil") is False
    assert skill_loader.is_valid_skill_name("has space") is False


def test_validate_tool_schema_accepts_well_formed_schema() -> None:
    assert skill_loader.validate_tool_schema("double_number", _double_schema()) is None


def test_validate_tool_schema_rejects_wrong_type() -> None:
    schema = _double_schema()
    schema["type"] = "not_a_function"
    assert skill_loader.validate_tool_schema("double_number", schema) is not None


def test_validate_tool_schema_rejects_name_mismatch() -> None:
    schema = _double_schema(name="wrong_name")
    error = skill_loader.validate_tool_schema("double_number", schema)
    assert error is not None
    assert "wrong_name" in error


def test_validate_tool_schema_rejects_missing_parameters() -> None:
    schema = _double_schema()
    del schema["function"]["parameters"]
    assert skill_loader.validate_tool_schema("double_number", schema) is not None


# --------------------------------------------------------------------------
# skill_loader.save_skill
# --------------------------------------------------------------------------


def test_save_skill_writes_file_with_schema_and_code() -> None:
    result = skill_loader.save_skill("double_number", _DOUBLE_CODE, _double_schema())
    assert result == {"ok": True, "skill_name": "double_number", "path": "skills/double_number.py"}

    written = (file_system._SANDBOX_ROOT / "skills" / "double_number.py").read_text(encoding="utf-8")
    assert "TOOL_SCHEMA = {" in written
    assert '"name": "double_number"' in written
    assert "def run(args):" in written


def test_save_skill_rejects_invalid_skill_name() -> None:
    result = skill_loader.save_skill("Not Valid!", _DOUBLE_CODE, _double_schema("Not Valid!"))
    assert result["ok"] is False


def test_save_skill_rejects_schema_name_mismatch() -> None:
    result = skill_loader.save_skill("double_number", _DOUBLE_CODE, _double_schema(name="other_name"))
    assert result["ok"] is False


def test_save_skill_rejects_missing_run_function() -> None:
    result = skill_loader.save_skill("double_number", "x = 1\n", _double_schema())
    assert result["ok"] is False
    assert "run" in result["error"]


def test_save_skill_rejects_empty_code() -> None:
    result = skill_loader.save_skill("double_number", "   ", _double_schema())
    assert result["ok"] is False


# --------------------------------------------------------------------------
# skill_loader.load_user_skills
# --------------------------------------------------------------------------


def test_load_user_skills_loads_and_executes_a_valid_skill() -> None:
    skill_loader.save_skill("double_number", _DOUBLE_CODE, _double_schema())

    result = skill_loader.load_user_skills()
    assert result["skipped"] == []
    assert "double_number" in result["skills"]

    entry = result["skills"]["double_number"]
    assert entry["schema"] == _double_schema()
    assert entry["handler"]({"n": 21}) == {"ok": True, "result": 42}


def test_load_user_skills_skips_broken_syntax_without_crashing() -> None:
    skills_dir = file_system._SANDBOX_ROOT / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    (skills_dir / "broken_skill.py").write_text("def run(args:\n    this is not valid python", encoding="utf-8")

    # A perfectly good skill sits alongside the broken one.
    skill_loader.save_skill("double_number", _DOUBLE_CODE, _double_schema())

    result = skill_loader.load_user_skills()
    assert "double_number" in result["skills"]
    skipped_files = {s["file"] for s in result["skipped"]}
    assert "broken_skill.py" in skipped_files


def test_load_user_skills_skips_missing_tool_schema() -> None:
    skills_dir = file_system._SANDBOX_ROOT / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    (skills_dir / "no_schema.py").write_text("def run(args):\n    return {'ok': True}\n", encoding="utf-8")

    result = skill_loader.load_user_skills()
    assert "no_schema" not in result["skills"]
    assert any(s["file"] == "no_schema.py" for s in result["skipped"])


def test_load_user_skills_empty_when_no_skills_dir() -> None:
    result = skill_loader.load_user_skills()
    assert result == {"skills": {}, "skipped": []}


# --------------------------------------------------------------------------
# react_dispatch integration — save_new_skill, hot-reload, dispatch
# --------------------------------------------------------------------------


def test_save_new_skill_tool_writes_hot_reloads_and_registers() -> None:
    handler = rd.TOOL_HANDLERS["save_new_skill"]
    args = {"skill_name": "double_number", "python_code": _DOUBLE_CODE, "schema": _double_schema()}

    result = handler(args, None, None)

    assert result["ok"] is True
    assert "double_number" in rd.TOOL_HANDLERS
    assert rd.is_mutating_tool("double_number") is True
    assert "double_number" in rd._CAPABILITY_TOOL_IDS["user_skills"]


def test_save_new_skill_is_itself_mutating_and_core() -> None:
    assert "save_new_skill" in rd._CORE_TOOL_IDS
    assert rd.is_mutating_tool("save_new_skill") is True


def test_save_new_skill_rejects_collision_with_existing_tool_id() -> None:
    handler = rd.TOOL_HANDLERS["save_new_skill"]
    args = {"skill_name": "search_web", "python_code": _DOUBLE_CODE, "schema": _double_schema(name="search_web")}

    result = handler(args, None, None)

    assert result["ok"] is False
    assert "search_web" not in rd._USER_SKILL_TOOL_IDS


def test_saved_skill_is_dispatchable_via_dispatch_tool_call() -> None:
    rd.TOOL_HANDLERS["save_new_skill"](
        {"skill_name": "double_number", "python_code": _DOUBLE_CODE, "schema": _double_schema()}, None, None
    )

    result = rd.dispatch_tool_call(ToolCall(tool_id="double_number", arguments={"n": 21}), None, None)

    assert result.ok is True
    assert result.payload["result"] == 42


def test_saved_skill_dispatchable_in_the_next_react_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    """The core end-to-end promise: writes the file, hot-reloads the
    registry, and makes the new tool immediately dispatchable in the next
    ReAct turn — no server restart, and gated the SAME way every other
    capability domain is (must be active for the LLM to even be offered
    it, and for a proposed call to survive next_react_turn's routing
    guard rather than being silently downgraded to "final")."""
    rd.TOOL_HANDLERS["save_new_skill"](
        {"skill_name": "double_number", "python_code": _DOUBLE_CODE, "schema": _double_schema()}, None, None
    )

    _mock_llm(monkeypatch, [ToolCall(tool_id="double_number", arguments={"n": 10})])
    messages = [
        {"role": "system", "content": rd.build_system_prompt(None, active_plugins=frozenset({"user_skills"}))},
        {"role": "user", "content": "double 10"},
    ]
    turn = asyncio.run(
        rd.next_react_turn(messages, None, raw_text="double 10", active_plugins=frozenset({"user_skills"}))
    )

    assert turn.kind == "tool_call"
    assert turn.call.tool_id == "double_number"
    assert turn.call.arguments == {"n": 10}


def test_saved_skill_not_offered_when_user_skills_domain_inactive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Loaded skills are still capability-gated like every other domain —
    a session that never activated (or load_capability'd) "user_skills"
    must not see the tool, even though it's saved and hot-reloaded."""
    rd.TOOL_HANDLERS["save_new_skill"](
        {"skill_name": "double_number", "python_code": _DOUBLE_CODE, "schema": _double_schema()}, None, None
    )

    _mock_llm(monkeypatch, [ToolCall(tool_id="double_number", arguments={"n": 10})])
    messages = [
        {"role": "system", "content": rd.build_system_prompt(None, active_plugins=frozenset())},
        {"role": "user", "content": "double 10"},
    ]
    turn = asyncio.run(rd.next_react_turn(messages, None, raw_text="double 10", active_plugins=frozenset()))

    # Downgraded to "final" — the model named a tool_id this turn's schema
    # never offered, same as any other out-of-scope tool_id.
    assert turn.kind == "final"


def test_load_capability_user_skills_domain_lists_loaded_skills() -> None:
    rd.TOOL_HANDLERS["save_new_skill"](
        {"skill_name": "double_number", "python_code": _DOUBLE_CODE, "schema": _double_schema()}, None, None
    )

    result = rd._tool_load_capability({"domain": "user_skills"}, None, None)

    assert result["ok"] is True
    assert result["unlocked_tools"] == ["double_number"]


def test_refresh_user_skills_drops_tool_id_whose_file_was_deleted() -> None:
    rd.TOOL_HANDLERS["save_new_skill"](
        {"skill_name": "double_number", "python_code": _DOUBLE_CODE, "schema": _double_schema()}, None, None
    )
    assert "double_number" in rd.TOOL_HANDLERS

    (file_system._SANDBOX_ROOT / "skills" / "double_number.py").unlink()
    rd.refresh_user_skills()

    assert "double_number" not in rd.TOOL_HANDLERS
    assert "double_number" not in rd._CAPABILITY_TOOL_IDS["user_skills"]
    assert rd.is_mutating_tool("double_number") is False


def test_save_new_skill_reports_load_failure_after_writing_broken_code() -> None:
    handler = rd.TOOL_HANDLERS["save_new_skill"]
    args = {
        "skill_name": "double_number",
        "python_code": "def run(args:\n    this is not valid python",
        "schema": _double_schema(),
    }

    result = handler(args, None, None)

    assert result["ok"] is False
    assert "double_number" not in rd._USER_SKILL_TOOL_IDS
    # The file is still written (so the agent/user can inspect and fix it
    # in a follow-up save_new_skill call) even though it failed to load.
    assert (file_system._SANDBOX_ROOT / "skills" / "double_number.py").is_file()


# --------------------------------------------------------------------------
# skill_loader.delete_skill / react_dispatch's delete_skill tool
# --------------------------------------------------------------------------


def test_delete_skill_removes_file() -> None:
    skill_loader.save_skill("double_number", _DOUBLE_CODE, _double_schema())
    path = file_system._SANDBOX_ROOT / "skills" / "double_number.py"
    assert path.is_file()

    result = skill_loader.delete_skill("double_number")

    assert result == {"ok": True, "skill_name": "double_number", "deleted": True}
    assert not path.exists()


def test_delete_skill_is_idempotent_for_a_missing_skill() -> None:
    result = skill_loader.delete_skill("never_existed")
    assert result == {"ok": True, "skill_name": "never_existed", "deleted": False}


def test_delete_skill_rejects_invalid_skill_name() -> None:
    result = skill_loader.delete_skill("../evil")
    assert result["ok"] is False


def test_delete_skill_tool_is_core_and_mutating() -> None:
    assert "delete_skill" in rd._CORE_TOOL_IDS
    assert rd.is_mutating_tool("delete_skill") is True


def test_delete_skill_tool_deletes_and_hot_reloads_registry() -> None:
    rd.TOOL_HANDLERS["save_new_skill"](
        {"skill_name": "double_number", "python_code": _DOUBLE_CODE, "schema": _double_schema()}, None, None
    )
    assert "double_number" in rd.TOOL_HANDLERS
    assert "double_number" in rd._CAPABILITY_TOOL_IDS["user_skills"]

    delete_handler = rd.TOOL_HANDLERS["delete_skill"]
    result = delete_handler({"skill_name": "double_number"}, None, None)

    assert result["ok"] is True
    assert "double_number" not in rd.TOOL_HANDLERS
    assert "double_number" not in rd._CAPABILITY_TOOL_IDS["user_skills"]
    assert rd.is_mutating_tool("double_number") is False
    # Dispatching the now-deleted tool_id must fail cleanly, not crash.
    dispatch_result = rd.dispatch_tool_call(ToolCall(tool_id="double_number", arguments={"n": 1}), None, None)
    assert dispatch_result.ok is False


def test_delete_skill_tool_reports_nothing_to_delete_but_still_ok() -> None:
    delete_handler = rd.TOOL_HANDLERS["delete_skill"]
    result = delete_handler({"skill_name": "never_existed"}, None, None)
    assert result["ok"] is True
    assert "nothing to delete" in result["message"].lower()


def test_deleted_skill_no_longer_dispatchable_in_the_next_react_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mirrors test_saved_skill_dispatchable_in_the_next_react_turn but for
    the deletion path — once delete_skill runs, the very next ReAct turn's
    routing guard must downgrade a model-proposed call to the now-gone
    tool_id to "final", exactly like any other unrecognized tool_id."""
    rd.TOOL_HANDLERS["save_new_skill"](
        {"skill_name": "double_number", "python_code": _DOUBLE_CODE, "schema": _double_schema()}, None, None
    )
    rd.TOOL_HANDLERS["delete_skill"]({"skill_name": "double_number"}, None, None)

    _mock_llm(monkeypatch, [ToolCall(tool_id="double_number", arguments={"n": 10})])
    messages = [
        {"role": "system", "content": rd.build_system_prompt(None, active_plugins=frozenset({"user_skills"}))},
        {"role": "user", "content": "double 10"},
    ]
    turn = asyncio.run(
        rd.next_react_turn(messages, None, raw_text="double 10", active_plugins=frozenset({"user_skills"}))
    )

    assert turn.kind == "final"


# --------------------------------------------------------------------------
# read_skill_source tool — the agent's own debugging companion
# --------------------------------------------------------------------------


def test_read_skill_source_tool_is_core_and_not_mutating() -> None:
    assert "read_skill_source" in rd._CORE_TOOL_IDS
    assert rd.is_mutating_tool("read_skill_source") is False


def test_read_skill_source_tool_returns_the_file_content() -> None:
    rd.TOOL_HANDLERS["save_new_skill"](
        {"skill_name": "double_number", "python_code": _DOUBLE_CODE, "schema": _double_schema()}, None, None
    )

    result = rd.TOOL_HANDLERS["read_skill_source"]({"skill_name": "double_number"}, None, None)

    assert result["ok"] is True
    assert "def run(args):" in result["code"]
    assert "TOOL_SCHEMA = {" in result["code"]


def test_read_skill_source_tool_reports_missing_skill_cleanly() -> None:
    result = rd.TOOL_HANDLERS["read_skill_source"]({"skill_name": "never_existed"}, None, None)
    assert result["ok"] is False


def test_system_prompt_includes_skill_debugging_guidance() -> None:
    prompt = rd.build_system_prompt(None)
    assert "read_skill_source" in prompt
    assert "save_new_skill" in prompt
    assert "traceback" in prompt


# --------------------------------------------------------------------------
# Skill Exception Tracebacks — dispatch_tool_call attaches
# traceback.format_exc() for a raising user_skills-domain tool, never for
# a built-in tool.
# --------------------------------------------------------------------------


def test_dispatch_tool_call_attaches_traceback_for_a_buggy_skill() -> None:
    buggy_code = textwrap.dedent(
        """\
        def run(args):
            return args["missing_key"] * 2
        """
    )
    rd.TOOL_HANDLERS["save_new_skill"](
        {"skill_name": "double_number", "python_code": buggy_code, "schema": _double_schema()}, None, None
    )

    result = rd.dispatch_tool_call(ToolCall(tool_id="double_number", arguments={"n": 5}), None, None)

    assert result.ok is False
    assert "traceback" in result.payload
    assert "KeyError" in result.payload["traceback"]
    assert "missing_key" in result.payload["traceback"]


def test_dispatch_tool_call_omits_traceback_for_a_built_in_tool_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """The traceback attachment is user_skills-only — a built-in tool's
    internal exception must never leak server implementation details to
    the model, only a digested {status, reason, suggestion} triple."""

    def _boom(args: dict, engine: Any, cp: Any) -> dict:
        raise RuntimeError("boom")

    monkeypatch.setitem(rd.TOOL_HANDLERS, "system_state", _boom)

    result = rd.dispatch_tool_call(ToolCall(tool_id="system_state", arguments={}), None, None)

    assert result.ok is False
    assert "traceback" not in result.payload


def test_dispatch_tool_call_omits_traceback_when_skill_fails_without_raising() -> None:
    """A skill that returns {"ok": False, ...} on its own (no exception)
    has nothing to attach a traceback FOR — must not fabricate one."""
    clean_failure_code = textwrap.dedent(
        """\
        def run(args):
            return {"ok": False, "error": "not implemented yet"}
        """
    )
    rd.TOOL_HANDLERS["save_new_skill"](
        {"skill_name": "double_number", "python_code": clean_failure_code, "schema": _double_schema()}, None, None
    )

    result = rd.dispatch_tool_call(ToolCall(tool_id="double_number", arguments={"n": 5}), None, None)

    assert result.ok is False
    assert "traceback" not in result.payload
