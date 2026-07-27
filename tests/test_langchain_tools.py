"""LangChain tool bridge smoke tests (no live Ollama required)."""

from __future__ import annotations

from test_support_react import patch_scripted_llm
from donna.agentic import REACT_MAX_ITERS, run_react_loop
from donna.tools.broker import IntentBroker
from donna.tools.langchain_tools import build_langchain_tools
from donna.tools.schema import ToolCall


def test_build_langchain_tools_from_registry() -> None:
    calls: list[ToolCall] = []

    def execute(tc: ToolCall) -> str:
        calls.append(tc)
        return f"OK: {tc.tool_id}"

    tools = build_langchain_tools(execute)
    names = {t.name for t in tools}
    assert "open_application" in names
    assert "read_local_file" in names
    assert "web_search" in names
    assert "dispatch_watchdog" in names
    assert "kill_watchdog" in names
    assert "save_script_to_library" in names
    assert len(tools) >= 10

    # Invoke the structured tool → Donna ToolCall IR.
    open_tool = next(t for t in tools if t.name == "open_application")
    result = open_tool.invoke({"app_name": "notepad"})
    assert result == "OK: open_application"
    assert calls and calls[0].tool_id == "open_application"
    assert calls[0].arguments.get("app_name") == "notepad"
    print(f"[PASS] built {len(tools)} LangChain tools; open_application OK")


def test_dispatch_watchdog_is_fire_and_forget(monkeypatch) -> None:
    import re
    import threading
    import time

    from donna.tools import langchain_tools as lt

    started = threading.Event()
    release = threading.Event()

    def _slow(task_id: str, _task: str) -> None:
        started.set()
        release.wait(timeout=2.0)
        with lt._watchdog_lock:
            lt.active_watchdogs.pop(task_id, None)

    monkeypatch.setattr(lt, "_watchdog_worker", _slow)

    t0 = time.perf_counter()
    result = lt.dispatch_watchdog_impl("Alert when Notepad opens")
    elapsed = time.perf_counter() - t0
    assert result.startswith("OK: Watchdog deployed with ID:")
    assert elapsed < 0.5, f"dispatch blocked too long: {elapsed:.3f}s"
    assert started.wait(timeout=1.0)
    m = re.search(r"ID:\s*(\S+)", result)
    assert m
    assert m.group(1) in lt.active_watchdogs
    release.set()

    tool_result = lt.dispatch_watchdog.invoke({"task": "Watch for toast"})
    assert "Watchdog deployed with ID:" in tool_result
    assert lt._WATCHDOG_TOOL_DESCRIPTION in (lt.dispatch_watchdog.description or "")
    print("[PASS] dispatch_watchdog fire-and-forget")


def test_kill_watchdog_stops_registered_job(monkeypatch) -> None:
    import re
    import threading

    from donna.tools import langchain_tools as lt

    release = threading.Event()

    def _slow(task_id: str, _task: str) -> None:
        entry = lt.active_watchdogs.get(task_id) or {}
        stop = entry.get("stop")
        while stop is not None and not stop.is_set():
            if release.wait(timeout=0.05):
                break
        with lt._watchdog_lock:
            lt.active_watchdogs.pop(task_id, None)

    monkeypatch.setattr(lt, "_watchdog_worker", _slow)
    result = lt.dispatch_watchdog_impl("Watch forever")
    tid = re.search(r"ID:\s*(\S+)", result).group(1)
    assert tid in lt.active_watchdogs

    killed = lt.kill_watchdog_impl(tid)
    assert killed.startswith("OK: Watchdog")
    assert tid not in lt.active_watchdogs
    release.set()
    print("[PASS] kill_watchdog")


def test_save_script_to_library_stays_in_sandbox(tmp_path, monkeypatch) -> None:
    from donna.tools import langchain_tools as lt

    lib = tmp_path / "execution_jail" / "library"
    lib.mkdir(parents=True)
    monkeypatch.setattr(lt, "_SANDBOX_LIBRARY", lib.resolve())
    monkeypatch.setattr(lt, "_REPO_ROOT", tmp_path.resolve())
    # Path-jail unit test — bypass Watchdog TTS policy (tested elsewhere).
    monkeypatch.setattr(
        "donna_jason_loop.jason_critic.static_code_safety_reject",
        lambda _c: None,
    )

    ok = lt.save_script_to_library_impl(
        "notepad_watch",
        "def main():\n    assert True\n",
    )
    assert ok.startswith("OK: saved script to")
    assert (lib / "notepad_watch.py").is_file()

    # Path separators are stripped to a basename (still lands in library/).
    nested = lt.save_script_to_library_impl("../escape", "x=1")
    assert nested.startswith("OK:")
    assert (lib / "escape.py").is_file()
    assert not (tmp_path / "escape.py").exists()

    bad = lt.save_script_to_library_impl("bad name!", "x=1")
    assert bad.startswith("ERROR:")
    print("[PASS] save_script_to_library sandbox jail")


def test_active_watchdogs_xml_in_recency_block(monkeypatch) -> None:
    from donna.tools import langchain_tools as lt
    from donna.prompts.spatial_synthesis import format_recency_context_block

    with lt._watchdog_lock:
        lt.active_watchdogs.clear()
        lt.active_watchdogs["42"] = {
            "thread": None,
            "task": "Alert when Notepad opens",
            "stop": None,
            "process": None,
        }
    try:
        block = format_recency_context_block(vision_line="", prior_turn_count=0)
        assert "<active_watchdogs>" in block
        assert "42: Alert when Notepad opens" in block
    finally:
        with lt._watchdog_lock:
            lt.active_watchdogs.clear()
    print("[PASS] active_watchdogs recency XML")


def test_spoken_answer_path_with_scripted_llm(monkeypatch) -> None:
    """Native bind_tools path: conversational AIMessage with no tool_calls."""
    patch_scripted_llm(monkeypatch, ["FINAL: Hello from the native path."])

    def execute(_tc: ToolCall) -> str:
        raise AssertionError("execute should not run on spoken-only turn")

    result = run_react_loop(
        user_text="hi",
        system_prompt="You are Donna.",
        execute_fn=execute,
        max_iters=REACT_MAX_ITERS,
        broker=IntentBroker(),
        enable_reflection=False,
    )
    assert "Hello" in (result.final_text or "")
    print(f"[PASS] native spoken path: {result.final_text}")


def test_langchain_loop_with_mocked_llm(monkeypatch) -> None:
    """Native bind_tools path: mocked ChatOllama returns a tool_call then FINAL."""
    from langchain_core.messages import AIMessage

    class _FakeBound:
        def __init__(self) -> None:
            self.n = 0

        def invoke(self, _messages):
            self.n += 1
            if self.n == 1:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "open_application",
                            "args": {"app_name": "notepad"},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                )
            return AIMessage(content="FINAL: Opened Notepad.")

    class _FakeLLM:
        def bind_tools(self, _tools, **_kwargs):
            return _FakeBound()

    monkeypatch.setattr(
        "langchain_ollama.ChatOllama",
        lambda **_kwargs: _FakeLLM(),
    )

    seen: list[str] = []

    def execute(tc: ToolCall) -> str:
        seen.append(tc.tool_id)
        return "OK: Launched notepad."

    result = run_react_loop(
        user_text="Open Notepad",
        system_prompt="You are Donna.",
        execute_fn=execute,
        max_iters=REACT_MAX_ITERS,
        broker=IntentBroker(),
        enable_reflection=False,
        model="llama3.2",
    )
    assert seen == ["open_application"]
    assert "Notepad" in (result.final_text or "") or "Opened" in (result.final_text or "")
    print(f"[PASS] langchain native tools: {result.final_text} trace={result.tool_trace}")


if __name__ == "__main__":
    test_build_langchain_tools_from_registry()
    print("OK (run pytest for mocked LangChain loop)")
