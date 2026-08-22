"""Tests for dana.api.services — the REST API backing the frontend's
Services Manager plugin. GET /api/services and DELETE /api/services/{alias}
delegate directly to dana.plugins.os.background_services's
list_background_services/stop_background_service — these tests spawn REAL
short-lived processes via that SAME module (not a mock) to prove the REST
layer genuinely interfaces with it, including the cross-platform tree-kill
behavior, rather than re-implementing any of its own logic.

Every test redirects the sandbox root to a throwaway temp directory (see
the autouse `_sandbox` fixture) — none of these ever touch the real
AGENT_WORKSPACE_DIR on disk. The autouse `_cleanup_active_processes`
fixture force-kills anything still tracked in `_ACTIVE_PROCESSES` after
every test, so a failing assertion mid-test can never leak a live process
onto the machine running this suite.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import psutil
import pytest
from fastapi.testclient import TestClient

from dana.api import server as server_module
from dana.api import services as services_module
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
    test session left tracked in _ACTIVE_PROCESSES, regardless of what the
    test itself did — a real subprocess must never outlive the test that
    spawned it, pass or fail."""
    yield
    for alias, process in list(background_services._ACTIVE_PROCESSES.items()):
        try:
            if process.poll() is None:
                background_services.stop_background_service(alias)
        except Exception:  # noqa: BLE001 — best-effort cleanup, never fail teardown over it
            pass
    background_services._ACTIVE_PROCESSES.clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(server_module.app)


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
# GET /api/services
# --------------------------------------------------------------------------


def test_get_services_empty_when_nothing_running(client: TestClient, _sandbox: Path) -> None:
    resp = client.get("/api/services")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "services": []}


def test_get_services_lists_a_real_running_service(client: TestClient, _sandbox: Path) -> None:
    _write_script(_sandbox, "slow.py", "import time\ntime.sleep(5)\n")
    start_result = background_services.start_background_service(_python_command("slow.py"), alias="svc-api")
    assert start_result["ok"] is True

    resp = client.get("/api/services")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    entry = next(s for s in body["services"] if s["alias"] == "svc-api")
    assert entry["running"] is True
    assert entry["pid"] == start_result["pid"]


def test_get_services_reflects_a_service_that_exited_on_its_own(client: TestClient, _sandbox: Path) -> None:
    _write_script(_sandbox, "hello.py", "print('done')")
    background_services.start_background_service(_python_command("hello.py"), alias="svc-finished")
    background_services._ACTIVE_PROCESSES["svc-finished"].wait(timeout=10.0)

    resp = client.get("/api/services")
    body = resp.json()
    entry = next(s for s in body["services"] if s["alias"] == "svc-finished")
    assert entry["running"] is False


# --------------------------------------------------------------------------
# GET /api/services/{alias}/logs
# --------------------------------------------------------------------------


def test_get_service_logs_missing_log_file_is_graceful(client: TestClient, _sandbox: Path) -> None:
    resp = client.get("/api/services/never-started/logs")
    assert resp.status_code == 200
    assert resp.json() == {
        "ok": True,
        "alias": "never-started",
        "log_path": "data/logs/never-started.log",
        "exists": False,
        "lines": [],
    }


def test_get_service_logs_returns_the_tail_of_a_log_file(client: TestClient, _sandbox: Path) -> None:
    log_dir = _sandbox / "data" / "logs"
    log_dir.mkdir(parents=True)
    lines = [f"line {i}" for i in range(1, 11)]
    (log_dir / "svc-log.log").write_text("\n".join(lines) + "\n")

    resp = client.get("/api/services/svc-log/logs?lines=3")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["exists"] is True
    assert body["lines"] == ["line 8", "line 9", "line 10"]


def test_get_service_logs_default_lines_returns_a_whole_short_file(client: TestClient, _sandbox: Path) -> None:
    log_dir = _sandbox / "data" / "logs"
    log_dir.mkdir(parents=True)
    (log_dir / "svc-short.log").write_text("hello\nworld\n")

    resp = client.get("/api/services/svc-short/logs")
    body = resp.json()
    assert body["lines"] == ["hello", "world"]


def test_get_service_logs_lines_param_is_capped(client: TestClient, _sandbox: Path) -> None:
    log_dir = _sandbox / "data" / "logs"
    log_dir.mkdir(parents=True)
    content = "\n".join(f"line {i}" for i in range(1, 1500)) + "\n"
    (log_dir / "svc-huge.log").write_text(content)

    resp = client.get(f"/api/services/svc-huge/logs?lines={services_module._MAX_LOG_LINES + 500}")
    body = resp.json()
    assert len(body["lines"]) == services_module._MAX_LOG_LINES
    assert body["lines"][-1] == "line 1499"


def test_get_service_logs_from_a_real_started_service_end_to_end(client: TestClient, _sandbox: Path) -> None:
    """The log content itself comes from an ACTUAL start_background_service
    call, not a hand-authored fixture file — proving the REST endpoint
    reads exactly what the background process really wrote."""
    _write_script(_sandbox, "hello.py", "print('hello from a real service')")
    background_services.start_background_service(_python_command("hello.py"), alias="svc-real-log")
    background_services._ACTIVE_PROCESSES["svc-real-log"].wait(timeout=10.0)

    resp = client.get("/api/services/svc-real-log/logs")
    assert resp.status_code == 200
    body = resp.json()
    assert body["exists"] is True
    assert "hello from a real service" in "\n".join(body["lines"])


# --------------------------------------------------------------------------
# DELETE /api/services/{alias}
# --------------------------------------------------------------------------


def test_delete_service_stops_a_real_running_service_end_to_end(client: TestClient, _sandbox: Path) -> None:
    """Proves DELETE genuinely delegates to
    background_services.stop_background_service's real cross-platform
    tree-kill (not a reimplementation): the underlying OS process id is
    actually dead afterward, not just removed from bookkeeping."""
    _write_script(_sandbox, "slow.py", "import time\ntime.sleep(30)\n")
    start_result = background_services.start_background_service(_python_command("slow.py"), alias="svc-delete")
    pid = start_result["pid"]
    assert psutil.pid_exists(pid)

    resp = client.delete("/api/services/svc-delete")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["alias"] == "svc-delete"
    assert body["already_stopped"] is False

    assert _wait_for(lambda: not psutil.pid_exists(pid)), "process was not actually killed by DELETE"
    assert "svc-delete" not in background_services._ACTIVE_PROCESSES


def test_delete_service_on_already_exited_service_reports_already_stopped(client: TestClient, _sandbox: Path) -> None:
    _write_script(_sandbox, "hello.py", "print('done')")
    background_services.start_background_service(_python_command("hello.py"), alias="svc-already-done")
    background_services._ACTIVE_PROCESSES["svc-already-done"].wait(timeout=10.0)

    resp = client.delete("/api/services/svc-already-done")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["already_stopped"] is True


def test_delete_service_missing_alias_returns_404(client: TestClient, _sandbox: Path) -> None:
    resp = client.delete("/api/services/never-existed")
    assert resp.status_code == 404
    assert "no active service" in resp.json()["detail"]


def test_get_services_no_longer_lists_alias_after_delete(client: TestClient, _sandbox: Path) -> None:
    _write_script(_sandbox, "slow.py", "import time\ntime.sleep(30)\n")
    background_services.start_background_service(_python_command("slow.py"), alias="svc-cycle")

    client.delete("/api/services/svc-cycle")

    resp = client.get("/api/services")
    assert not any(s["alias"] == "svc-cycle" for s in resp.json()["services"])
