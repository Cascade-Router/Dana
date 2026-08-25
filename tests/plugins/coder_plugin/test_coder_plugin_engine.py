"""Unit tests for dana.plugins.coder_plugin.engine — the software_engineering
domain plugin wrapping ``aider`` as a subprocess. Every ``execute_code_task``
(aider) ``subprocess.run`` call is mocked: no network call, no live Gemini
API key ever touched (per this plugin's own verification session — real
aider runs cost real, if tiny, money and shouldn't be spent on every test
run). ``search_codebase``/``run_verification_command`` each get exactly one
real, unmocked call too (a plain local ``git grep``/``pytest`` — free, no
network, no API key) specifically to prove the fixed argv actually works
against a live process, not just against a mocked one.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import dana.core.react_dispatch as rd
from dana.paths import PROJECT_ROOT
from dana.plugins.coder_plugin import engine


# ---------------------------------------------------------------------------
# search_codebase — the context-compressing regex grep
# ---------------------------------------------------------------------------


@patch("dana.plugins.coder_plugin.engine.subprocess.run")
def test_search_codebase_runs_fixed_git_grep_argv(mock_run: MagicMock) -> None:
    mock_run.return_value = MagicMock(returncode=0, stdout="dana/foo.py:12:def bar():\n", stderr="")
    result = engine.search_codebase({"regex_pattern": "def bar"})
    assert result["ok"] is True
    assert "def bar" in result["matches"]
    called_command = mock_run.call_args.args[0]
    assert called_command == ["git", "grep", "--untracked", "-n", "-E", "def bar"]
    assert mock_run.call_args.kwargs["shell"] is False
    assert mock_run.call_args.kwargs["cwd"] == str(PROJECT_ROOT)


@patch("dana.plugins.coder_plugin.engine.subprocess.run")
def test_search_codebase_scopes_by_file_extension(mock_run: MagicMock) -> None:
    mock_run.return_value = MagicMock(returncode=0, stdout="dana/foo.py:1:class Foo:\n", stderr="")
    result = engine.search_codebase({"regex_pattern": "class Foo", "file_extension": ".py"})
    assert result["ok"] is True
    called_command = mock_run.call_args.args[0]
    assert called_command == ["git", "grep", "--untracked", "-n", "-E", "class Foo", "--", "*.py"]


@patch("dana.plugins.coder_plugin.engine.subprocess.run")
def test_search_codebase_no_matches_returns_clean_message_not_a_crash(mock_run: MagicMock) -> None:
    # git grep exits 1 (not an error) when it simply finds nothing.
    mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")
    result = engine.search_codebase({"regex_pattern": "definitely_not_in_the_repo_xyz"})
    assert result["ok"] is True
    assert result["matches"] == "No matches found."


@patch("dana.plugins.coder_plugin.engine.subprocess.run")
def test_search_codebase_real_git_error_is_reported(mock_run: MagicMock) -> None:
    mock_run.return_value = MagicMock(returncode=128, stdout="", stderr="fatal: not a git repository")
    result = engine.search_codebase({"regex_pattern": "x"})
    assert result["ok"] is False
    assert "not a git repository" in result["error"]


@patch("dana.plugins.coder_plugin.engine.subprocess.run")
def test_search_codebase_timeout_reported_cleanly(mock_run: MagicMock) -> None:
    import subprocess

    mock_run.side_effect = subprocess.TimeoutExpired(cmd="git grep", timeout=20.0)
    result = engine.search_codebase({"regex_pattern": "x"})
    assert result["ok"] is False
    assert "timed out" in result["error"]


def test_search_codebase_requires_regex_pattern() -> None:
    result = engine.search_codebase({})
    assert result["ok"] is False
    assert "regex_pattern" in result["error"]


def test_search_codebase_finds_a_real_known_symbol_in_this_repo() -> None:
    """One real, unmocked call — proves the fixed argv actually works
    against a live git checkout, not just against a mocked subprocess."""
    result = engine.search_codebase({"regex_pattern": r"def search_codebase", "file_extension": "py"})
    assert result["ok"] is True
    assert "coder_plugin/engine.py" in result["matches"] or "coder_plugin\\engine.py" in result["matches"]


# ---------------------------------------------------------------------------
# analyze_codebase — read-only file reads only (search_codebase now owns grep)
# ---------------------------------------------------------------------------


def test_analyze_codebase_reads_an_existing_file() -> None:
    # manifest.json is small, stable, and always present — no fixture needed.
    result = engine.analyze_codebase({"files": ["dana/plugins/coder_plugin/manifest.json"]})
    assert result["ok"] is True
    assert "dana/plugins/coder_plugin/manifest.json" in result["files"]
    assert "software_engineering" in result["files"]["dana/plugins/coder_plugin/manifest.json"]


def test_analyze_codebase_reports_missing_file_without_crashing() -> None:
    result = engine.analyze_codebase({"files": ["dana/plugins/coder_plugin/does_not_exist.py"]})
    assert result["ok"] is False
    assert "does not exist" in result["error"]


def test_analyze_codebase_rejects_path_traversal() -> None:
    result = engine.analyze_codebase({"files": ["../../../../etc/passwd"]})
    assert result["ok"] is False
    assert "escapes the project root" in result["error"]


def test_analyze_codebase_rejects_denylisted_env_file() -> None:
    result = engine.analyze_codebase({"files": [".env"]})
    assert result["ok"] is False
    assert "denylisted" in result["error"]


def test_analyze_codebase_rejects_git_internals() -> None:
    result = engine.analyze_codebase({"files": [".git/config"]})
    assert result["ok"] is False
    assert "denylisted" in result["error"]


def test_analyze_codebase_requires_nonempty_files() -> None:
    result = engine.analyze_codebase({})
    assert result["ok"] is False
    assert "files" in result["error"]

    result = engine.analyze_codebase({"files": []})
    assert result["ok"] is False
    assert "files" in result["error"]


# ---------------------------------------------------------------------------
# run_verification_command — read-only, whitelist-only verification runner
# (the Self-Correction Loop's "Verify" step)
# ---------------------------------------------------------------------------


@patch("dana.plugins.coder_plugin.engine.subprocess.run")
def test_run_verification_command_allows_pytest(mock_run: MagicMock) -> None:
    mock_run.return_value = MagicMock(returncode=0, stdout="1 passed", stderr="")
    result = engine.run_verification_command({"command": "pytest tests/plugins/coder_plugin/"})
    assert result["ok"] is True
    assert "1 passed" in result["output"]
    called_command = mock_run.call_args.args[0]
    assert called_command == ["pytest", "tests/plugins/coder_plugin/"]
    assert mock_run.call_args.kwargs["shell"] is False
    assert mock_run.call_args.kwargs["cwd"] == str(PROJECT_ROOT)


@patch("dana.plugins.coder_plugin.engine.subprocess.run")
def test_run_verification_command_allows_flake8_and_mypy(mock_run: MagicMock) -> None:
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
    assert engine.run_verification_command({"command": "flake8 dana/plugins/coder_plugin"})["ok"] is True
    assert engine.run_verification_command({"command": "mypy dana/plugins/coder_plugin"})["ok"] is True


@patch("dana.plugins.coder_plugin.engine.subprocess.run")
def test_run_verification_command_allows_black_check_only(mock_run: MagicMock) -> None:
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
    result = engine.run_verification_command({"command": "black --check dana/plugins/coder_plugin/engine.py"})
    assert result["ok"] is True

    result = engine.run_verification_command({"command": "black dana/plugins/coder_plugin/engine.py"})
    assert result["ok"] is False
    assert "--check" in result["error"]
    mock_run.assert_called_once()  # only the --check invocation above ever actually ran


@patch("dana.plugins.coder_plugin.engine.subprocess.run")
def test_run_verification_command_rejects_rm_rf(mock_run: MagicMock) -> None:
    result = engine.run_verification_command({"command": "rm -rf /"})
    assert result["ok"] is False
    assert "not a whitelisted verification command" in result["error"]
    mock_run.assert_not_called()


@patch("dana.plugins.coder_plugin.engine.subprocess.run")
def test_run_verification_command_rejects_pip_install(mock_run: MagicMock) -> None:
    result = engine.run_verification_command({"command": "pip install something-malicious"})
    assert result["ok"] is False
    assert "not a whitelisted verification command" in result["error"]
    mock_run.assert_not_called()


@patch("dana.plugins.coder_plugin.engine.subprocess.run")
def test_run_verification_command_shell_operators_never_reach_a_shell(mock_run: MagicMock) -> None:
    """A shell-metacharacter injection attempt through an allowed base
    command's arguments must never chain a second process — shell=False
    means '&&'/'rm' land as literal, harmless argv elements to pytest
    itself, never as shell syntax."""
    mock_run.return_value = MagicMock(returncode=2, stdout="", stderr="usage error")
    result = engine.run_verification_command({"command": "pytest && rm -rf /"})
    called_command = mock_run.call_args.args[0]
    assert called_command == ["pytest", "&&", "rm", "-rf", "/"]
    assert mock_run.call_args.kwargs["shell"] is False


def test_run_verification_command_requires_nonempty_command() -> None:
    result = engine.run_verification_command({})
    assert result["ok"] is False
    assert "command" in result["error"]


@patch("dana.plugins.coder_plugin.engine.subprocess.run")
def test_run_verification_command_timeout_reported_cleanly(mock_run: MagicMock) -> None:
    import subprocess

    mock_run.side_effect = subprocess.TimeoutExpired(cmd="pytest", timeout=120.0)
    result = engine.run_verification_command({"command": "pytest"})
    assert result["ok"] is False
    assert "timed out" in result["error"]


@patch("dana.plugins.coder_plugin.engine.subprocess.run")
def test_run_verification_command_nonzero_exit_reports_traceback_for_self_correction(mock_run: MagicMock) -> None:
    mock_run.return_value = MagicMock(
        returncode=1, stdout="", stderr="AssertionError: expected 2 got 3\nFAILED tests/test_foo.py"
    )
    result = engine.run_verification_command({"command": "pytest tests/test_foo.py"})
    assert result["ok"] is False
    assert "AssertionError" in result["error"]
    assert "AssertionError" in result["output"]


@patch("dana.plugins.coder_plugin.engine.subprocess.run")
def test_run_verification_command_stdout_only_traceback_is_not_swallowed(mock_run: MagicMock) -> None:
    """Regression for the exact reported bug: pytest writes its failure
    report/traceback to STDOUT (stderr is typically empty), and "error"
    used to fall back to a bare "pytest exited with code 1" whenever
    stderr was empty — silently discarding the one thing the LLM needs to
    self-correct, even though it survived elsewhere in the payload."""
    traceback_text = (
        "FAILED tests/test_foo.py::test_bar - AssertionError: expected 2, got 3\n"
        "1 failed in 0.12s"
    )
    mock_run.return_value = MagicMock(returncode=1, stdout=traceback_text, stderr="")
    result = engine.run_verification_command({"command": "pytest tests/test_foo.py"})
    assert result["ok"] is False
    assert "AssertionError" in result["error"]
    assert result["error"] != "pytest exited with code 1"


@patch("dana.plugins.coder_plugin.engine.subprocess.run")
def test_run_verification_command_traceback_survives_dispatch_tool_call_end_to_end(
    mock_run: MagicMock,
) -> None:
    """The actual reported failure mode happened one layer up: dana.core.
    react_dispatch.dispatch_tool_call replaces a failed tool's ENTIRE
    payload with {"ok": False, **digest_error(tool_id, payload["error"])}
    — "stdout"/"stderr"/"output" never survive that, only whatever is in
    "error". This drives run_verification_command through the real
    dispatch_tool_call/digest_error pipeline to confirm the traceback
    text — not a generic exit-code message, and not misclassified as a
    FreeCAD failure — is what the LLM actually receives."""
    import dana.core.react_dispatch as rd
    from dana.tools.schema import ToolCall

    mock_run.return_value = MagicMock(
        returncode=1, stdout="AssertionError: expected 2 got 3\nFAILED tests/test_foo.py", stderr=""
    )
    call = ToolCall(tool_id="run_verification_command", arguments={"command": "pytest tests/test_foo.py"})
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)

    assert result.ok is False
    assert "AssertionError" in result.payload["raw_error"]
    assert "FreeCAD" not in result.payload["reason"]


def test_run_verification_command_pytest_actually_runs_against_a_real_trivial_case() -> None:
    """One real, unmocked call against this repo's own git working tree —
    proves the fixed argv actually invokes a real pytest process, not just
    a mocked subprocess."""
    result = engine.run_verification_command({"command": "pytest --collect-only tests/plugins/coder_plugin/"})
    assert result["ok"] is True
    assert "run_verification_command" in result["output"] or "collected" in result["output"].lower()


# ---------------------------------------------------------------------------
# execute_code_task — mutating, always mocked (never a real aider/API call)
# ---------------------------------------------------------------------------


def test_execute_code_task_requires_task_description() -> None:
    result = engine.execute_code_task({"files": ["aider_test.txt"]})
    assert result["ok"] is False
    assert "task_description" in result["error"]


def test_execute_code_task_requires_nonempty_files() -> None:
    result = engine.execute_code_task({"task_description": "do something", "files": []})
    assert result["ok"] is False
    assert "files" in result["error"]


@patch("dana.plugins.coder_plugin.engine.subprocess.run")
def test_execute_code_task_rejects_path_traversal_without_ever_calling_subprocess(mock_run: MagicMock) -> None:
    result = engine.execute_code_task(
        {"task_description": "do something", "files": ["../../../../etc/passwd"]}
    )
    assert result["ok"] is False
    assert "escapes the project root" in result["error"]
    mock_run.assert_not_called()


@patch("dana.plugins.coder_plugin.engine.subprocess.run")
def test_execute_code_task_rejects_env_file_without_ever_calling_subprocess(mock_run: MagicMock) -> None:
    result = engine.execute_code_task({"task_description": "leak the key", "files": [".env"]})
    assert result["ok"] is False
    mock_run.assert_not_called()


@patch("dana.plugins.coder_plugin.engine.subprocess.run")
def test_execute_code_task_builds_the_exact_verified_command(mock_run: MagicMock) -> None:
    mock_run.return_value = MagicMock(returncode=0, stdout="Applied edit.", stderr="")
    result = engine.execute_code_task(
        {"task_description": "Change the text to 'Hello Dana'", "files": ["aider_test.txt"]}
    )
    assert result["ok"] is True
    assert result["stdout"] == "Applied edit."
    command = mock_run.call_args.args[0]
    assert command[0] == "aider"
    assert command[1:7] == [
        "--model", "gemini/gemini-3.6-flash",
        "--edit-format", "diff",
        "--yes", "--no-stream",
    ]
    assert command[7] == "--message"
    assert command[8] == "Change the text to 'Hello Dana'"
    assert command[9].endswith("aider_test.txt")
    assert mock_run.call_args.kwargs["shell"] is False
    assert mock_run.call_args.kwargs["cwd"] == str(PROJECT_ROOT)


@patch("dana.plugins.coder_plugin.engine.subprocess.run")
def test_execute_code_task_nonzero_exit_reports_stderr(mock_run: MagicMock) -> None:
    mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="litellm.NotFoundError: model not found")
    result = engine.execute_code_task({"task_description": "x", "files": ["aider_test.txt"]})
    assert result["ok"] is False
    assert "model not found" in result["error"]
    assert result["returncode"] == 1


@patch("dana.plugins.coder_plugin.engine.subprocess.run")
def test_execute_code_task_timeout_reported_cleanly(mock_run: MagicMock) -> None:
    import subprocess

    mock_run.side_effect = subprocess.TimeoutExpired(cmd="aider", timeout=180.0, output="partial", stderr="")
    result = engine.execute_code_task({"task_description": "x", "files": ["aider_test.txt"]})
    assert result["ok"] is False
    assert "timed out" in result["error"]


@patch("dana.plugins.coder_plugin.engine.subprocess.run")
def test_execute_code_task_missing_aider_binary_reported_cleanly(mock_run: MagicMock) -> None:
    mock_run.side_effect = FileNotFoundError("aider not found")
    result = engine.execute_code_task({"task_description": "x", "files": ["aider_test.txt"]})
    assert result["ok"] is False
    assert "not installed" in result["error"]


# ---------------------------------------------------------------------------
# execute_code_task's optional test_command — native aider --test-cmd/
# --auto-test wiring, validated against the exact same allowlist as
# run_verification_command (via _validate_verify_command)
# ---------------------------------------------------------------------------


@patch("dana.plugins.coder_plugin.engine.subprocess.run")
def test_execute_code_task_appends_test_cmd_and_auto_test_when_provided(mock_run: MagicMock) -> None:
    mock_run.return_value = MagicMock(returncode=0, stdout="Applied edit. 1 passed", stderr="")
    result = engine.execute_code_task(
        {
            "task_description": "Fix the bug",
            "files": ["aider_test.txt"],
            "test_command": "pytest tests/test_foo.py",
        }
    )
    assert result["ok"] is True
    command = mock_run.call_args.args[0]
    assert "--test-cmd" in command
    idx = command.index("--test-cmd")
    assert command[idx + 1] == "pytest tests/test_foo.py"
    assert command[idx + 2] == "--auto-test"
    # --test-cmd/--auto-test must come BEFORE --message so aider parses the
    # flags rather than swallowing them into the free-form task text.
    assert idx < command.index("--message")


@patch("dana.plugins.coder_plugin.engine.subprocess.run")
def test_execute_code_task_omits_test_cmd_when_not_provided(mock_run: MagicMock) -> None:
    mock_run.return_value = MagicMock(returncode=0, stdout="Applied edit.", stderr="")
    engine.execute_code_task({"task_description": "x", "files": ["aider_test.txt"]})
    command = mock_run.call_args.args[0]
    assert "--test-cmd" not in command
    assert "--auto-test" not in command


@patch("dana.plugins.coder_plugin.engine.subprocess.run")
def test_execute_code_task_rejects_non_whitelisted_test_command_without_calling_subprocess(
    mock_run: MagicMock,
) -> None:
    result = engine.execute_code_task(
        {"task_description": "x", "files": ["aider_test.txt"], "test_command": "rm -rf /"}
    )
    assert result["ok"] is False
    assert "not a whitelisted verification command" in result["error"]
    mock_run.assert_not_called()


@patch("dana.plugins.coder_plugin.engine.subprocess.run")
def test_execute_code_task_rejects_bare_black_test_command(mock_run: MagicMock) -> None:
    result = engine.execute_code_task(
        {"task_description": "x", "files": ["aider_test.txt"], "test_command": "black dana/foo.py"}
    )
    assert result["ok"] is False
    assert "--check" in result["error"]
    mock_run.assert_not_called()


@patch("dana.plugins.coder_plugin.engine.subprocess.run")
def test_execute_code_task_allows_black_check_test_command(mock_run: MagicMock) -> None:
    mock_run.return_value = MagicMock(returncode=0, stdout="Applied edit.", stderr="")
    result = engine.execute_code_task(
        {
            "task_description": "x",
            "files": ["aider_test.txt"],
            "test_command": "black --check dana/foo.py",
        }
    )
    assert result["ok"] is True
    command = mock_run.call_args.args[0]
    assert command[command.index("--test-cmd") + 1] == "black --check dana/foo.py"


# ---------------------------------------------------------------------------
# Generic plugin dispatch wiring (dana.core.react_dispatch.refresh_plugin_tools)
# — confirms manifest.json/engine.py are enough on their own, with zero
# react_dispatch.py edits, for the tools to actually be reachable.
# ---------------------------------------------------------------------------


def test_coder_plugin_tools_registered_in_tool_handlers() -> None:
    assert "search_codebase" in rd.TOOL_HANDLERS
    assert "analyze_codebase" in rd.TOOL_HANDLERS
    assert "run_verification_command" in rd.TOOL_HANDLERS
    assert "execute_code_task" in rd.TOOL_HANDLERS


def test_coder_plugin_domain_discovered_automatically() -> None:
    assert rd._CAPABILITY_TOOL_IDS.get("software_engineering") == frozenset(
        {"search_codebase", "analyze_codebase", "run_verification_command", "execute_code_task"}
    )


def test_execute_code_task_is_mutating_others_are_not() -> None:
    assert rd.is_mutating_tool("execute_code_task") is True
    assert rd.is_mutating_tool("analyze_codebase") is False
    assert rd.is_mutating_tool("search_codebase") is False
    assert rd.is_mutating_tool("run_verification_command") is False


def test_software_engineering_domain_tools_appear_in_llm_schema() -> None:
    schema = rd._llm_tools_schema(frozenset({"software_engineering"}))
    names = {t["function"]["name"] for t in schema}
    assert {"search_codebase", "analyze_codebase", "run_verification_command", "execute_code_task"} <= names


def test_software_engineering_domain_absent_when_not_active() -> None:
    schema = rd._llm_tools_schema(frozenset())
    names = {t["function"]["name"] for t in schema}
    assert "search_codebase" not in names
    assert "analyze_codebase" not in names
    assert "run_verification_command" not in names
    assert "execute_code_task" not in names


def test_execute_code_task_schema_declares_files_as_string_array() -> None:
    schema = rd._llm_tools_schema(frozenset({"software_engineering"}))
    (spec,) = [t for t in schema if t["function"]["name"] == "execute_code_task"]
    files_param = spec["function"]["parameters"]["properties"]["files"]
    assert files_param["type"] == "array"
    assert files_param["items"] == {"type": "string"}


def test_search_codebase_schema_declares_regex_pattern_required() -> None:
    schema = rd._llm_tools_schema(frozenset({"software_engineering"}))
    (spec,) = [t for t in schema if t["function"]["name"] == "search_codebase"]
    assert "regex_pattern" in spec["function"]["parameters"]["required"]
    assert "file_extension" not in spec["function"]["parameters"]["required"]
