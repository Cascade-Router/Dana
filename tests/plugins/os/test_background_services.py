"""Tests for dana.plugins.os.background_services — Background Process
Management: non-blocking service spawn/stop/list backing the real
"os_tools" capability domain (dana.core.react_dispatch's
_OS_TOOLS_TOOL_IDS/is_mutating_tool). Every test redirects the sandbox root
to a throwaway temp directory (see the autouse `_sandbox` fixture) — none
of these ever touch the real AGENT_WORKSPACE_DIR on disk.

These tests spawn REAL OS processes (that's the whole point of this
module) — the autouse `_cleanup_active_processes` fixture force-kills
anything still tracked in `_ACTIVE_PROCESSES` after every test, so a
failing assertion mid-test can never leak a live process onto the machine
running this suite.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import psutil
import pytest

from dana.plugins.os import background_services, file_system


@pytest.fixture(autouse=True)
def _sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "agent_workspace"
    # exist_ok=True: tests/conftest.py's own global _isolate_os_tools_sandbox
    # autouse fixture already creates this same tmp_path/agent_workspace
    # directory first — tolerate it already existing rather than raising.
    root.mkdir(exist_ok=True)
    monkeypatch.setattr(file_system, "_SANDBOX_ROOT", root)
    return root


@pytest.fixture(autouse=True)
def _cleanup_active_processes():
    """Safety net, not the thing under test: force-kills every process this
    test session left tracked in _ACTIVE_PROCESSES, regardless of whether
    the test itself called stop_background_service or asserted anything
    about it — a real subprocess (and its own real children) must never
    outlive the test that spawned it, pass or fail.
    """
    yield
    for alias, process in list(background_services._ACTIVE_PROCESSES.items()):
        try:
            if process.poll() is None:
                background_services.stop_background_service(alias)
        except Exception:  # noqa: BLE001 — best-effort cleanup, never fail teardown over it
            pass
    background_services._ACTIVE_PROCESSES.clear()


@pytest.fixture
def _mount(tmp_path: Path) -> Path:
    mount_dir = tmp_path / "mounted_project"
    mount_dir.mkdir()
    return mount_dir


def _write_script(sandbox: Path, name: str, source: str) -> None:
    (sandbox / name).write_text(source)


def _python_command(script_name: str) -> str:
    """A shell command line invoking THIS test run's own interpreter on a
    script already written into the target working directory — quoted so
    it survives cmd.exe's parsing even when sys.executable contains spaces
    (e.g. "C:\\Program Files\\...")."""
    return f'"{sys.executable}" {script_name}'


def _wait_for(predicate, timeout_s: float = 5.0, interval_s: float = 0.1) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return predicate()


# --------------------------------------------------------------------------
# Non-blocking execution
# --------------------------------------------------------------------------


def test_start_background_service_returns_immediately(_sandbox: Path) -> None:
    _write_script(_sandbox, "slow.py", "import time\ntime.sleep(5)\n")
    start = time.perf_counter()
    result = background_services.start_background_service(_python_command("slow.py"), alias="svc-immediate")
    elapsed = time.perf_counter() - start
    assert result["ok"] is True
    assert elapsed < 2.0  # well under the 5s the service itself sleeps for


def test_start_background_service_returns_pid_and_ok(_sandbox: Path) -> None:
    _write_script(_sandbox, "slow.py", "import time\ntime.sleep(5)\n")
    result = background_services.start_background_service(_python_command("slow.py"), alias="svc-pid")
    assert result["ok"] is True
    assert isinstance(result["pid"], int)
    assert result["pid"] > 0


# --------------------------------------------------------------------------
# Log file creation
# --------------------------------------------------------------------------


def test_start_background_service_creates_log_file_and_captures_output(_sandbox: Path) -> None:
    _write_script(_sandbox, "hello.py", "print('hello from background')")
    result = background_services.start_background_service(_python_command("hello.py"), alias="svc-log")
    assert result["ok"] is True
    assert result["log_path"] == "data/logs/svc-log.log"

    process = background_services._ACTIVE_PROCESSES["svc-log"]
    process.wait(timeout=10.0)  # this one exits quickly on its own

    log_file = _sandbox / "data" / "logs" / "svc-log.log"
    assert log_file.is_file()
    assert "hello from background" in log_file.read_text()


def test_start_background_service_log_captures_stderr_too(_sandbox: Path) -> None:
    _write_script(_sandbox, "broken.py", "raise ValueError('background crash')")
    result = background_services.start_background_service(_python_command("broken.py"), alias="svc-crash")
    assert result["ok"] is True
    process = background_services._ACTIVE_PROCESSES["svc-crash"]
    process.wait(timeout=10.0)

    log_file = _sandbox / "data" / "logs" / "svc-crash.log"
    assert "ValueError" in log_file.read_text()
    assert "background crash" in log_file.read_text()


# --------------------------------------------------------------------------
# list_background_services
# --------------------------------------------------------------------------


def test_list_background_services_reports_running_status(_sandbox: Path) -> None:
    _write_script(_sandbox, "slow.py", "import time\ntime.sleep(5)\n")
    background_services.start_background_service(_python_command("slow.py"), alias="svc-listed")

    result = background_services.list_background_services()
    assert result["ok"] is True
    entry = next(s for s in result["services"] if s["alias"] == "svc-listed")
    assert entry["running"] is True
    assert entry["pid"] > 0


def test_list_background_services_reports_false_after_natural_exit(_sandbox: Path) -> None:
    _write_script(_sandbox, "hello.py", "print('done')")
    background_services.start_background_service(_python_command("hello.py"), alias="svc-finished")
    process = background_services._ACTIVE_PROCESSES["svc-finished"]
    process.wait(timeout=10.0)

    result = background_services.list_background_services()
    entry = next(s for s in result["services"] if s["alias"] == "svc-finished")
    assert entry["running"] is False


def test_list_background_services_empty_when_nothing_started(_sandbox: Path) -> None:
    result = background_services.list_background_services()
    assert result == {"ok": True, "services": []}


# --------------------------------------------------------------------------
# stop_background_service — including the CRITICAL cross-platform,
# whole-tree kill guarantee (no orphaned grandchildren).
# --------------------------------------------------------------------------


def test_stop_background_service_terminates_the_process(_sandbox: Path) -> None:
    _write_script(_sandbox, "slow.py", "import time\ntime.sleep(30)\n")
    start_result = background_services.start_background_service(_python_command("slow.py"), alias="svc-stop")
    pid = start_result["pid"]
    assert psutil.pid_exists(pid)

    stop_result = background_services.stop_background_service("svc-stop")
    assert stop_result["ok"] is True
    assert stop_result["already_stopped"] is False

    assert _wait_for(lambda: not psutil.pid_exists(pid))


def test_stop_background_service_kills_the_entire_process_tree_no_orphans(_sandbox: Path) -> None:
    """The critical cross-platform guarantee this whole module exists to
    provide: killing only the Popen's own top-level pid (on Windows,
    shell=True's cmd.exe wrapper process) must never leave the ACTUAL
    long-running grandchild (the real "server") alive and orphaned,
    holding its port forever."""
    _write_script(
        _sandbox,
        "server.py",
        "import os, time\n"
        "with open('child_pid.txt', 'w') as f:\n"
        "    f.write(str(os.getpid()))\n"
        "time.sleep(30)\n",
    )
    result = background_services.start_background_service(_python_command("server.py"), alias="svc-tree")
    assert result["ok"] is True

    pid_file = _sandbox / "child_pid.txt"
    assert _wait_for(pid_file.is_file), "grandchild never started / never wrote its own pid"
    grandchild_pid = int(pid_file.read_text().strip())
    assert psutil.pid_exists(grandchild_pid)
    # Prove the grandchild is genuinely a DIFFERENT OS process than
    # Popen's own top-level pid — otherwise this test would trivially pass
    # even without real process-GROUP tree-kill logic actually working.
    assert grandchild_pid != result["pid"]

    stop_result = background_services.stop_background_service("svc-tree")
    assert stop_result["ok"] is True

    assert _wait_for(lambda: not psutil.pid_exists(grandchild_pid)), "grandchild process was orphaned, not killed"


def test_stop_background_service_on_already_exited_process_reports_already_stopped(_sandbox: Path) -> None:
    _write_script(_sandbox, "hello.py", "print('done')")
    background_services.start_background_service(_python_command("hello.py"), alias="svc-already-done")
    process = background_services._ACTIVE_PROCESSES["svc-already-done"]
    process.wait(timeout=10.0)

    result = background_services.stop_background_service("svc-already-done")
    assert result["ok"] is True
    assert result["already_stopped"] is True
    assert "svc-already-done" not in background_services._ACTIVE_PROCESSES


def test_stop_background_service_unknown_alias_reports_error(_sandbox: Path) -> None:
    result = background_services.stop_background_service("never-started")
    assert result["ok"] is False
    assert "no active service" in result["error"]


def test_stop_background_service_rejects_empty_alias(_sandbox: Path) -> None:
    result = background_services.stop_background_service("   ")
    assert result["ok"] is False
    assert "must not be empty" in result["error"]


# --------------------------------------------------------------------------
# Validation — command/alias rejection, working_dir traversal/existence
# --------------------------------------------------------------------------


def test_start_background_service_rejects_empty_command(_sandbox: Path) -> None:
    result = background_services.start_background_service("", alias="svc-empty-cmd")
    assert result["ok"] is False
    assert "command must not be empty" in result["error"]


def test_start_background_service_rejects_empty_alias(_sandbox: Path) -> None:
    result = background_services.start_background_service("echo hi", alias="")
    assert result["ok"] is False
    assert "alias must not be empty" in result["error"]


def test_start_background_service_rejects_alias_with_path_separators(_sandbox: Path) -> None:
    result = background_services.start_background_service("echo hi", alias="../escape")
    assert result["ok"] is False
    assert "alias must contain only" in result["error"]


def test_start_background_service_rejects_duplicate_alias_while_running(_sandbox: Path) -> None:
    _write_script(_sandbox, "slow.py", "import time\ntime.sleep(5)\n")
    first = background_services.start_background_service(_python_command("slow.py"), alias="svc-dup")
    assert first["ok"] is True

    second = background_services.start_background_service(_python_command("slow.py"), alias="svc-dup")
    assert second["ok"] is False
    assert "already running" in second["error"]


def test_start_background_service_rejects_working_dir_traversal(_sandbox: Path) -> None:
    result = background_services.start_background_service(
        "echo hi", alias="svc-traversal", working_dir="../../etc"
    )
    assert result["ok"] is False
    assert "outside the sandbox" in result["error"]


def test_start_background_service_rejects_missing_working_dir(_sandbox: Path) -> None:
    result = background_services.start_background_service("echo hi", alias="svc-missing-dir", working_dir="nope")
    assert result["ok"] is False
    assert "does not exist" in result["error"]


def test_start_background_service_rejects_a_file_as_working_dir(_sandbox: Path) -> None:
    (_sandbox / "a_file.txt").write_text("x")
    result = background_services.start_background_service(
        "echo hi", alias="svc-file-dir", working_dir="a_file.txt"
    )
    assert result["ok"] is False
    assert "not a directory" in result["error"]


# --------------------------------------------------------------------------
# Dynamic Workspace Mounting
# --------------------------------------------------------------------------


def test_start_background_service_runs_inside_a_mounted_directory(_sandbox: Path, _mount: Path) -> None:
    (_mount / "hello.py").write_text("print('hello from the mount')")
    result = background_services.start_background_service(
        _python_command("hello.py"), alias="svc-mount", working_dir=str(_mount), allowed_mounts=[str(_mount)]
    )
    assert result["ok"] is True
    process = background_services._ACTIVE_PROCESSES["svc-mount"]
    process.wait(timeout=10.0)
    # The log itself still lives in the SANDBOX's own data/logs, never the
    # mount — a service's log location is fixed regardless of working_dir.
    assert (_sandbox / "data" / "logs" / "svc-mount.log").read_text().strip() == "hello from the mount"


def test_start_background_service_in_mount_rejected_without_the_mount_registered(
    _sandbox: Path, _mount: Path
) -> None:
    (_mount / "hello.py").write_text("print('should never run')")
    result = background_services.start_background_service(
        _python_command("hello.py"), alias="svc-mount-rejected", working_dir=str(_mount), allowed_mounts=None
    )
    assert result["ok"] is False
    assert "outside the sandbox" in result["error"]


# --------------------------------------------------------------------------
# dispatch_tool_call end-to-end + registry/HITL wiring
# --------------------------------------------------------------------------


def test_dispatch_tool_call_start_background_service_end_to_end(_sandbox: Path) -> None:
    import dana.core.react_dispatch as rd
    from dana.tools.schema import ToolCall

    _write_script(_sandbox, "slow.py", "import time\ntime.sleep(5)\n")
    call = ToolCall(
        tool_id="start_background_service",
        arguments={"command": _python_command("slow.py"), "alias": "svc-dispatch"},
    )
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is True
    assert result.payload["alias"] == "svc-dispatch"


def test_dispatch_tool_call_start_background_service_traversal_is_digested_not_crashed(_sandbox: Path) -> None:
    import dana.core.react_dispatch as rd
    from dana.tools.schema import ToolCall

    call = ToolCall(
        tool_id="start_background_service",
        arguments={"command": "echo hi", "alias": "svc-dispatch-escape", "working_dir": "../../etc"},
    )
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "outside the sandbox" in result.payload.get("raw_error", "")


def test_dispatch_tool_call_start_background_service_reaches_a_mounted_directory(
    _sandbox: Path, _mount: Path
) -> None:
    import dana.core.react_dispatch as rd
    from dana.tools.schema import ToolCall

    (_mount / "hello.py").write_text("print('hello from the mount')")
    call = ToolCall(
        tool_id="start_background_service",
        arguments={"command": _python_command("hello.py"), "alias": "svc-dispatch-mount", "working_dir": str(_mount)},
    )
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None, allowed_mounts=[str(_mount)])
    assert result.ok is True


def test_dispatch_tool_call_stop_and_list_background_service_end_to_end(_sandbox: Path) -> None:
    import dana.core.react_dispatch as rd
    from dana.tools.schema import ToolCall

    _write_script(_sandbox, "slow.py", "import time\ntime.sleep(30)\n")
    start_call = ToolCall(
        tool_id="start_background_service",
        arguments={"command": _python_command("slow.py"), "alias": "svc-dispatch-stop"},
    )
    rd.dispatch_tool_call(start_call, engine=None, control_plane=None)

    list_result = rd.dispatch_tool_call(
        ToolCall(tool_id="list_background_services", arguments={}), engine=None, control_plane=None
    )
    assert list_result.ok is True
    assert any(s["alias"] == "svc-dispatch-stop" and s["running"] for s in list_result.payload["services"])

    stop_result = rd.dispatch_tool_call(
        ToolCall(tool_id="stop_background_service", arguments={"alias": "svc-dispatch-stop"}),
        engine=None,
        control_plane=None,
    )
    assert stop_result.ok is True


def test_start_and_stop_background_service_are_mutating_and_require_hitl() -> None:
    import dana.core.react_dispatch as rd

    assert rd.is_mutating_tool("start_background_service") is True
    assert rd.is_mutating_tool("stop_background_service") is True


def test_list_background_services_is_not_mutating() -> None:
    import dana.core.react_dispatch as rd

    assert rd.is_mutating_tool("list_background_services") is False


def test_all_three_tools_registered_and_in_os_tools_domain() -> None:
    import dana.core.react_dispatch as rd

    for tool_id in ("start_background_service", "stop_background_service", "list_background_services"):
        assert tool_id in rd.TOOL_HANDLERS, tool_id
        assert tool_id in rd._OS_TOOLS_TOOL_IDS, tool_id


def test_only_start_background_service_is_threaded_through_tools_needing_mounts() -> None:
    import dana.core.react_dispatch as rd

    assert "start_background_service" in rd._TOOLS_NEEDING_MOUNTS
    assert "stop_background_service" not in rd._TOOLS_NEEDING_MOUNTS
    assert "list_background_services" not in rd._TOOLS_NEEDING_MOUNTS
