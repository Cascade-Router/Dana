"""Tests for dana.plugins.os.process_manager — time-boxed, sandboxed Python
script execution backing the real "os_tools" capability domain
(dana.core.react_dispatch's _OS_TOOLS_TOOL_IDS). Every test redirects the
sandbox root to a throwaway temp directory (see the autouse `_sandbox`
fixture) — none of these ever touch the real AGENT_WORKSPACE_DIR on disk.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from dana.plugins.os import file_system, process_manager


@pytest.fixture(autouse=True)
def _sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "agent_workspace"
    # exist_ok=True: tests/conftest.py's global _isolate_os_tools_sandbox
    # autouse fixture already creates this same tmp_path/agent_workspace
    # directory first — tolerate it already existing rather than raising.
    root.mkdir(exist_ok=True)
    monkeypatch.setattr(file_system, "_SANDBOX_ROOT", root)
    return root


def _write_script(sandbox: Path, name: str, source: str) -> None:
    (sandbox / name).write_text(source)


@pytest.fixture
def _mount(tmp_path: Path) -> Path:
    mount_dir = tmp_path / "mounted_project"
    mount_dir.mkdir()
    return mount_dir


def _python_command(script_name: str) -> str:
    """A shell command line invoking THIS test run's own interpreter on a
    script already written into the target working directory — quoted so
    it survives cmd.exe's parsing even when sys.executable contains spaces
    (e.g. "C:\\Program Files\\..."), same as any real caller's command
    string would need to be."""
    return f'"{sys.executable}" {script_name}'


# --------------------------------------------------------------------------
# Path traversal rejection
# --------------------------------------------------------------------------


def test_rejects_parent_traversal(_sandbox: Path) -> None:
    result = process_manager.run_python_script("../outside.py")
    assert result["ok"] is False
    assert "outside the sandbox" in result["error"]


def test_rejects_absolute_path(_sandbox: Path, tmp_path: Path) -> None:
    outside = tmp_path / "elsewhere.py"
    outside.write_text("print('should never run')")
    result = process_manager.run_python_script(str(outside))
    assert result["ok"] is False
    assert "resolves outside the sandbox" in result["error"] or "absolute paths are not allowed" in result["error"]


def test_rejects_non_py_extension(_sandbox: Path) -> None:
    _write_script(_sandbox, "not_a_script.txt", "print('hi')")
    result = process_manager.run_python_script("not_a_script.txt")
    assert result["ok"] is False
    assert "only .py files" in result["error"]


def test_rejects_missing_script(_sandbox: Path) -> None:
    result = process_manager.run_python_script("does_not_exist.py")
    assert result["ok"] is False
    assert "does not exist" in result["error"]


def test_rejects_directory_target(_sandbox: Path) -> None:
    (_sandbox / "adir.py").mkdir()
    result = process_manager.run_python_script("adir.py")
    assert result["ok"] is False
    assert "not a file" in result["error"]


# --------------------------------------------------------------------------
# Timeout handling
# --------------------------------------------------------------------------


def test_infinite_loop_times_out(_sandbox: Path) -> None:
    _write_script(_sandbox, "loop.py", "while True:\n    pass\n")
    result = process_manager.run_python_script("loop.py", timeout_s=1.0)
    assert result["ok"] is False
    assert result["timed_out"] is True
    assert "timed out after 1.0s" in result["error"]


def test_fast_script_does_not_time_out(_sandbox: Path) -> None:
    _write_script(_sandbox, "fast.py", "print('done')")
    result = process_manager.run_python_script("fast.py", timeout_s=1.0)
    assert result["ok"] is True
    assert "timed_out" not in result


# --------------------------------------------------------------------------
# Successful stdout capture, and non-zero exit / traceback propagation
# --------------------------------------------------------------------------


def test_successful_script_captures_stdout(_sandbox: Path) -> None:
    _write_script(_sandbox, "hello.py", "print('hello from sandbox')")
    result = process_manager.run_python_script("hello.py")
    assert result["ok"] is True
    assert result["stdout"].strip() == "hello from sandbox"
    assert result["returncode"] == 0


def test_script_arguments_are_passed_through(_sandbox: Path) -> None:
    _write_script(_sandbox, "echo_args.py", "import sys\nprint(' '.join(sys.argv[1:]))\n")
    result = process_manager.run_python_script("echo_args.py", args=["foo", "bar"])
    assert result["ok"] is True
    assert result["stdout"].strip() == "foo bar"


def test_nonzero_exit_reports_stderr_for_self_correction(_sandbox: Path) -> None:
    _write_script(_sandbox, "broken.py", "raise ValueError('deliberately broken')")
    result = process_manager.run_python_script("broken.py")
    assert result["ok"] is False
    assert result["returncode"] != 0
    assert "ValueError" in result["stderr"]
    assert "deliberately broken" in result["error"]


def test_nonzero_exit_code_without_stderr_still_reports_clearly(_sandbox: Path) -> None:
    _write_script(_sandbox, "exit_code.py", "import sys\nsys.exit(7)\n")
    result = process_manager.run_python_script("exit_code.py")
    assert result["ok"] is False
    assert "7" in result["error"]


def test_script_cannot_write_outside_sandbox_via_relative_cwd(_sandbox: Path, tmp_path: Path) -> None:
    """The subprocess's cwd is the sandbox root — a script using a plain
    relative path stays confined to it by default."""
    _write_script(_sandbox, "write_relative.py", "open('produced.txt', 'w').write('x')")
    result = process_manager.run_python_script("write_relative.py")
    assert result["ok"] is True
    assert (_sandbox / "produced.txt").is_file()
    assert not (tmp_path / "produced.txt").exists()


# --------------------------------------------------------------------------
# dispatch_tool_call end-to-end + HITL wiring
# --------------------------------------------------------------------------


def test_dispatch_tool_call_run_python_script_end_to_end(_sandbox: Path) -> None:
    import dana.core.react_dispatch as rd
    from dana.tools.schema import ToolCall

    _write_script(_sandbox, "greet.py", "print('greetings from dispatch')")
    call = ToolCall(tool_id="run_python_script", arguments={"script_path": "greet.py"})
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is True
    assert "greetings from dispatch" in result.payload["stdout"]


def test_dispatch_tool_call_traversal_is_digested_not_crashed(_sandbox: Path) -> None:
    import dana.core.react_dispatch as rd
    from dana.tools.schema import ToolCall

    call = ToolCall(tool_id="run_python_script", arguments={"script_path": "../escape.py"})
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "outside the sandbox" in result.payload.get("raw_error", "")


def test_run_python_script_is_mutating_and_requires_hitl() -> None:
    import dana.core.react_dispatch as rd

    assert rd.is_mutating_tool("run_python_script") is True


def test_run_python_script_registered_and_in_os_tools_domain() -> None:
    import dana.core.react_dispatch as rd

    assert "run_python_script" in rd.TOOL_HANDLERS
    assert "run_python_script" in rd._OS_TOOLS_TOOL_IDS


# --------------------------------------------------------------------------
# Generalized Terminal Execution — execute_terminal_command. Genuinely
# shell=True (unlike run_python_script's fixed argv list), so this section
# covers its own path-traversal, timeout, and output-truncation behavior,
# plus the security-critical is_mutating_tool gate.
# --------------------------------------------------------------------------


def test_execute_terminal_command_captures_stdout(_sandbox: Path) -> None:
    _write_script(_sandbox, "hello.py", "print('hello from terminal')")
    result = process_manager.execute_terminal_command(_python_command("hello.py"))
    assert result["ok"] is True
    assert result["stdout"].strip() == "hello from terminal"
    assert result["exit_code"] == 0


def test_execute_terminal_command_nonzero_exit_reports_error(_sandbox: Path) -> None:
    _write_script(_sandbox, "broken.py", "import sys\nsys.exit(3)\n")
    result = process_manager.execute_terminal_command(_python_command("broken.py"))
    assert result["ok"] is False
    assert result["exit_code"] == 3
    assert "3" in result["error"]


def test_execute_terminal_command_stderr_reported_for_self_correction(_sandbox: Path) -> None:
    _write_script(_sandbox, "broken.py", "raise ValueError('deliberately broken')")
    result = process_manager.execute_terminal_command(_python_command("broken.py"))
    assert result["ok"] is False
    assert "ValueError" in result["stderr"]
    assert "deliberately broken" in result["error"]


def test_execute_terminal_command_rejects_empty_command(_sandbox: Path) -> None:
    result = process_manager.execute_terminal_command("   ")
    assert result["ok"] is False
    assert "must not be empty" in result["error"]


# --- cwd (working_dir) path traversal rejection ---


def test_execute_terminal_command_rejects_working_dir_traversal(_sandbox: Path) -> None:
    result = process_manager.execute_terminal_command(_python_command("x.py"), working_dir="../../etc")
    assert result["ok"] is False
    assert "outside the sandbox" in result["error"]


def test_execute_terminal_command_rejects_absolute_working_dir_with_no_mount(
    _sandbox: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    result = process_manager.execute_terminal_command(_python_command("x.py"), working_dir=str(outside))
    assert result["ok"] is False
    assert "outside the sandbox" in result["error"]


def test_execute_terminal_command_rejects_missing_working_dir(_sandbox: Path) -> None:
    result = process_manager.execute_terminal_command("echo hi", working_dir="nope")
    assert result["ok"] is False
    assert "does not exist" in result["error"]


def test_execute_terminal_command_rejects_a_file_as_working_dir(_sandbox: Path) -> None:
    (_sandbox / "a_file.txt").write_text("x")
    result = process_manager.execute_terminal_command("echo hi", working_dir="a_file.txt")
    assert result["ok"] is False
    assert "not a directory" in result["error"]


def test_execute_terminal_command_cannot_write_outside_sandbox_via_default_cwd(
    _sandbox: Path, tmp_path: Path
) -> None:
    """The subprocess's cwd defaults to the sandbox root — a command using
    a plain relative path stays confined to it by default."""
    _write_script(_sandbox, "write_relative.py", "open('produced.txt', 'w').write('x')")
    result = process_manager.execute_terminal_command(_python_command("write_relative.py"))
    assert result["ok"] is True
    assert (_sandbox / "produced.txt").is_file()
    assert not (tmp_path / "produced.txt").exists()


# --- Dynamic Workspace Mounting ---


def test_execute_terminal_command_runs_inside_a_mounted_directory(_sandbox: Path, _mount: Path) -> None:
    (_mount / "hello.py").write_text("print('hello from the mount')")
    result = process_manager.execute_terminal_command(
        _python_command("hello.py"), working_dir=str(_mount), allowed_mounts=[str(_mount)]
    )
    assert result["ok"] is True
    assert result["stdout"].strip() == "hello from the mount"


def test_execute_terminal_command_in_mount_rejected_without_the_mount_registered(
    _sandbox: Path, _mount: Path
) -> None:
    (_mount / "hello.py").write_text("print('should never run')")
    result = process_manager.execute_terminal_command(
        _python_command("hello.py"), working_dir=str(_mount), allowed_mounts=None
    )
    assert result["ok"] is False
    assert "outside the sandbox" in result["error"]


# --- Timeout handling ---


def test_execute_terminal_command_times_out(_sandbox: Path) -> None:
    # A finite (5s) sleep, not `while True: pass` — even in the worst case
    # where killing the shell doesn't reliably kill this grandchild process
    # too (a known shell=True-on-Windows quirk), it still exits on its own
    # a few seconds later instead of leaking a truly infinite runaway
    # process onto the machine running this test.
    _write_script(_sandbox, "slow.py", "import time\ntime.sleep(5)\n")
    result = process_manager.execute_terminal_command(_python_command("slow.py"), timeout_s=1.0)
    assert result["ok"] is False
    assert result["timed_out"] is True
    assert "timed out after 1.0s" in result["error"]


def test_execute_terminal_command_fast_command_does_not_time_out(_sandbox: Path) -> None:
    _write_script(_sandbox, "fast.py", "print('done')")
    result = process_manager.execute_terminal_command(_python_command("fast.py"), timeout_s=10.0)
    assert result["ok"] is True
    assert "timed_out" not in result


# --- Output truncation ---


def test_execute_terminal_command_truncates_huge_stdout(_sandbox: Path) -> None:
    _write_script(_sandbox, "wall_of_text.py", "print('x' * 20000)")
    result = process_manager.execute_terminal_command(_python_command("wall_of_text.py"))
    assert result["ok"] is True
    assert len(result["stdout"]) <= process_manager._MAX_OUTPUT_CHARS + len("\n…[truncated]")
    assert result["stdout"].endswith("[truncated]")


# --- dispatch_tool_call end-to-end + HITL wiring ---


def test_dispatch_tool_call_execute_terminal_command_end_to_end(_sandbox: Path) -> None:
    """dispatch_tool_call itself (not just the handler) must produce a
    successful ToolResult for a real command — proving the wiring, not just
    the underlying process_manager function, works."""
    import dana.core.react_dispatch as rd
    from dana.tools.schema import ToolCall

    _write_script(_sandbox, "greet.py", "print('greetings from dispatch')")
    call = ToolCall(
        tool_id="execute_terminal_command", arguments={"command": _python_command("greet.py")}
    )
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is True
    assert "greetings from dispatch" in result.payload["stdout"]


def test_dispatch_tool_call_execute_terminal_command_traversal_is_digested_not_crashed(
    _sandbox: Path,
) -> None:
    import dana.core.react_dispatch as rd
    from dana.tools.schema import ToolCall

    call = ToolCall(
        tool_id="execute_terminal_command",
        arguments={"command": "echo hi", "working_dir": "../../etc"},
    )
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "outside the sandbox" in result.payload.get("raw_error", "")


def test_dispatch_tool_call_execute_terminal_command_reaches_a_mounted_directory(
    _sandbox: Path, _mount: Path
) -> None:
    """End-to-end through react_dispatch.dispatch_tool_call's own
    allowed_mounts param, mirroring the file_system tools' equivalent tests
    — execute_terminal_command must be threaded through
    _TOOLS_NEEDING_MOUNTS the same way."""
    import dana.core.react_dispatch as rd
    from dana.tools.schema import ToolCall

    (_mount / "hello.py").write_text("print('hello from the mount')")
    call = ToolCall(
        tool_id="execute_terminal_command",
        arguments={"command": _python_command("hello.py"), "working_dir": str(_mount)},
    )
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None, allowed_mounts=[str(_mount)])
    assert result.ok is True
    assert "hello from the mount" in result.payload["stdout"]


def test_execute_terminal_command_is_mutating_and_requires_hitl() -> None:
    import dana.core.react_dispatch as rd

    assert rd.is_mutating_tool("execute_terminal_command") is True


def test_execute_terminal_command_registered_and_in_os_tools_domain() -> None:
    import dana.core.react_dispatch as rd

    assert "execute_terminal_command" in rd.TOOL_HANDLERS
    assert "execute_terminal_command" in rd._OS_TOOLS_TOOL_IDS


def test_execute_terminal_command_is_threaded_through_tools_needing_mounts() -> None:
    import dana.core.react_dispatch as rd

    assert "execute_terminal_command" in rd._TOOLS_NEEDING_MOUNTS
