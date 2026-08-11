"""Watchdog LangGraph + Template Method + AST gate + Titan (mocked Ollama)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from dana_jason_loop.jason_critic import (
    parse_titan_verdict,
    review_watchdog_code,
    static_code_safety_reject,
)
from dana.swarm.watchdog_graph import (
    WatchdogState,
    _route_after_ast,
    _route_after_titan,
    analyze_watchdog_ast,
    ast_static_analyzer,
    build_watchdog_graph,
    dana_coder,
    preflight_watchdog_write,
    titan_supervisor,
    repl_executor,
    terminal_failure,
)
from dana.swarm.watchdog_template import (
    assemble_watchdog_script,
    parse_coder_payload,
)


def _state(**overrides: object) -> WatchdogState:
    base: WatchdogState = {
        "task": "",
        "code": "",
        "feedback": "",
        "lint_errors": "",
        "status": "pending",
        "revisions": 0,
        "history": [],
    }
    base.update(overrides)  # type: ignore[typeddict-item]
    return base


def test_static_safety_rejects_destructive() -> None:
    bad = (
        "import subprocess\n"
        "print('__DANA_TTS__: wipe')\n"
        "subprocess.run(['rm', '-rf', '/'])\n"
    )
    reason = static_code_safety_reject(bad)
    assert reason and "Forbidden import" in reason
    assert review_watchdog_code(bad, task="clean disk").startswith("REJECTED")

    missing_tts = "assert True\nprint('alert')\n"
    assert "Missing mandatory TTS" in (static_code_safety_reject(missing_tts) or "")
    print("[PASS] static safety rejects destructive code")


def test_parse_titan_verdict() -> None:
    assert parse_titan_verdict("APPROVED") == "APPROVED"
    assert parse_titan_verdict("REJECTED: missing self-test").startswith("REJECTED")
    print("[PASS] parse_titan_verdict")


def test_assemble_watchdog_script_topology() -> None:
    code = assemble_watchdog_script(
        run_self_test="assert True",
        monitor_loop="self.alert('Notepad detected')",
        extra_imports=["time", "os"],  # os must be filtered out
    )
    assert "class BaseWatchdog" in code
    assert "class GeneratedWatchdog(BaseWatchdog)" in code
    assert "def run_self_test" in code
    assert "def monitor_loop" in code
    assert "import time" in code
    assert "import os" not in code
    assert analyze_watchdog_ast(code) == []
    print("[PASS] assemble_watchdog_script topology")


def test_analyze_watchdog_ast_forbidden_import() -> None:
    evil = assemble_watchdog_script(
        run_self_test="assert True",
        monitor_loop="self.alert('x')",
    )
    evil = "import os\n" + evil
    errs = analyze_watchdog_ast(evil)
    assert errs
    assert any("ast.Import detected 'os'" in e for e in errs)
    print("[PASS] AST rejects forbidden import")


def test_ast_static_analyzer_bypasses_titan_on_fail() -> None:
    code = "import subprocess\nprint('nope')\n"
    out = ast_static_analyzer(_state(code=code, status="drafting"))
    assert out["status"] == "LINT_FAIL"
    assert "FATAL" in (out.get("lint_errors") or "")
    assert out["revisions"] == 1
    assert out["history"][-1]["stage"] == "ast_lint"
    print("[PASS] ast_static_analyzer LINT_FAIL")


def test_dana_coder_assembles_from_json(monkeypatch) -> None:
    class _Msg:
        content = (
            '{"extra_imports": ["time"], '
            '"run_self_test": "assert True", '
            '"monitor_loop": "self.alert(\'alert\')"}'
        )

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = _Msg()
    monkeypatch.setattr(
        "dana.swarm.watchdog_graph._chat_ollama",
        lambda **_k: fake_llm,
    )
    out = dana_coder(
        _state(
            task="Watch for Notepad",
            lint_errors="FATAL: ast.Import detected 'os'. Remove immediately.",
        )
    )
    assert out["status"] == "drafting"
    assert "GeneratedWatchdog" in (out.get("code") or "")
    assert "assert True" in (out.get("code") or "")
    user_blob = fake_llm.invoke.call_args[0][0][1]["content"]
    assert "FATAL AST LINT ERRORS" in user_blob
    print("[PASS] dana_coder assembles from JSON + prefers lint_errors")


def test_parse_coder_payload_json_and_python_fallback() -> None:
    payload = parse_coder_payload(
        '{"run_self_test": "assert 1", "monitor_loop": "self.alert(\'x\')"}'
    )
    assert payload["run_self_test"] == "assert 1"
    py = (
        "class X:\n"
        "    def run_self_test(self):\n"
        "        assert True\n"
        "    def monitor_loop(self):\n"
        "        self.alert('hi')\n"
    )
    payload2 = parse_coder_payload(py)
    assert "assert True" in payload2["run_self_test"]
    assert "alert" in payload2["monitor_loop"]
    print("[PASS] parse_coder_payload")


def test_titan_supervisor_approve_path(monkeypatch) -> None:
    safe = assemble_watchdog_script(
        run_self_test="assert True",
        monitor_loop="self.alert('monitor')",
    )
    monkeypatch.setattr(
        "dana_jason_loop.jason_critic.static_code_safety_reject",
        lambda _c: None,
    )
    with patch(
        "dana_jason_loop.jason_critic.review_watchdog_code",
        return_value="APPROVED",
    ):
        out = titan_supervisor(
            _state(
                task="monitor notepad",
                code=safe,
                status="LINT_OK",
            )
        )
    assert out["status"] == "APPROVED"
    assert out["feedback"] == "APPROVED"
    assert out["history"] and out["history"][-1]["stage"] == "titan_eval"
    print("[PASS] titan_supervisor APPROVED path")


def test_route_after_ast_and_titan() -> None:
    assert (
        _route_after_ast(_state(status="LINT_OK", revisions=0)) == "titan_supervisor"
    )
    assert (
        _route_after_ast(_state(status="LINT_FAIL", revisions=1)) == "dana_coder"
    )
    assert (
        _route_after_ast(_state(status="LINT_FAIL", revisions=3))
        == "terminal_failure"
    )
    assert (
        _route_after_titan(_state(status="APPROVED", revisions=1)) == "repl_executor"
    )
    assert (
        _route_after_titan(
            _state(status="REJECTED: missing tts", revisions=1)
        )
        == "dana_coder"
    )
    assert (
        _route_after_titan(
            _state(status="REJECTED: still bad", revisions=3)
        )
        == "terminal_failure"
    )
    print("[PASS] conditional routes after AST + Titan")


def test_repl_executor_runs_safe_script_and_forwards_tts(monkeypatch) -> None:
    spoken: list[str] = []

    def _capture(phrase: str) -> None:
        spoken.append(phrase)

    monkeypatch.setattr("dana.audio.tts_manager.enqueue_speech_impl", _capture)

    code = assemble_watchdog_script(
        run_self_test="assert True\nprint('self-test ok')",
        monitor_loop="self.alert('Notepad detected')",
    )
    out = repl_executor(
        _state(
            task="demo",
            code=code,
            feedback="APPROVED",
            status="APPROVED",
            revisions=1,
        ),
        timeout_s=15.0,
    )
    assert out["status"] == "executed", out
    assert spoken == ["Notepad detected"]
    assert "self-test ok" in (out.get("feedback") or "")
    assert out["history"] and out["history"][-1]["stage"] == "execution"
    print("[PASS] repl_executor + TTS bridge")


def test_repl_executor_uses_execution_jail_cwd(monkeypatch) -> None:
    """Guardrail: temp script + subprocess cwd must stay under execution_jail."""
    import io
    import os
    from pathlib import Path

    import dana.swarm.watchdog_graph as wg

    captured: dict[str, object] = {}

    class _FakeProc:
        returncode = 0

        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            self.stdout = io.StringIO("ok\n")
            self.stderr = io.StringIO("")

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0

        def kill(self):
            return None

    monkeypatch.setattr(wg.subprocess, "Popen", _FakeProc)

    sandbox = wg.ensure_watchdog_sandbox().resolve()
    monkeypatch.setattr(
        "dana_jason_loop.jason_critic.static_code_safety_reject",
        lambda _c: None,
    )
    code = assemble_watchdog_script(
        run_self_test="pass",
        monitor_loop="self.alert('ok')",
    )
    out = repl_executor(
        _state(
            task="cwd check",
            code=code,
            feedback="APPROVED",
            status="APPROVED",
            revisions=1,
        ),
        timeout_s=5.0,
    )
    assert out["status"] == "executed", out
    kwargs = captured["kwargs"]
    cwd = Path(os.path.abspath(str(kwargs["cwd"]))).resolve()
    assert cwd == sandbox
    assert cwd.name == "execution_jail"
    assert kwargs.get("bufsize") == 1
    assert captured["kwargs"]["env"]["PYTHONUNBUFFERED"] == "1"

    argv = captured["args"][0]
    assert isinstance(argv, (list, tuple)) and len(argv) >= 2
    script_path = Path(argv[1]).resolve()
    assert script_path.parent == sandbox
    assert script_path.name.endswith("_dana_watchdog.py")
    print("[PASS] repl_executor sandboxed cwd + script path")


def test_build_watchdog_graph_compiles() -> None:
    app = build_watchdog_graph()
    assert app is not None
    print("[PASS] watchdog graph compiles")


def test_preflight_rejects_malformed_script() -> None:
    import pytest

    sandbox = preflight_watchdog_write(require_code=False)
    assert sandbox.is_dir()

    with pytest.raises(RuntimeError, match="empty"):
        preflight_watchdog_write("", require_code=True)
    with pytest.raises(RuntimeError, match="malformed"):
        preflight_watchdog_write("not a script at all!!!", require_code=True)
    ok = preflight_watchdog_write(
        assemble_watchdog_script(
            run_self_test="assert True",
            monitor_loop="self.alert('hi')",
        ),
        require_code=True,
    )
    assert ok == sandbox
    print("[PASS] preflight_watchdog_write integrity checks")


def test_terminal_failure_logs_root_cause(monkeypatch) -> None:
    import sys
    import types

    spoken: list[str] = []
    logged: list[tuple[str, str]] = []

    monkeypatch.setattr("dana.audio.tts_manager.enqueue_speech_impl", lambda p: spoken.append(p))

    fake_log = types.ModuleType("dana.logging")

    def _lex(thread: str, message: str, *, exc=None):  # noqa: ANN001
        logged.append((thread, message))
        assert exc is not None
        assert "aborted" in str(exc).lower() or "revisions" in str(exc).lower()

    fake_log.log_exception = _lex  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "dana.logging", fake_log)

    out = terminal_failure(
        _state(
            task="watch notes",
            code="print('x')",
            feedback="REJECTED: missing self-test",
            lint_errors="FATAL: ast.Import detected 'os'. Remove immediately.",
            status="LINT_FAIL",
            revisions=3,
            history=[
                {
                    "stage": "ast_lint",
                    "revision": 3,
                    "code": "print('x')",
                    "feedback": "FATAL: ast.Import detected 'os'.",
                    "status": "LINT_FAIL",
                }
            ],
        )
    )
    assert out["status"] == "error"
    assert "ast.Import" in (out.get("feedback") or "")
    assert spoken and "abort" in spoken[0].lower()
    assert logged and logged[0][0] == "Watchdog"
    print("[PASS] terminal_failure logs RuntimeError root cause")


if __name__ == "__main__":
    test_static_safety_rejects_destructive()
    test_parse_titan_verdict()
    test_assemble_watchdog_script_topology()
    print("OK")
