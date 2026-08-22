"""LangChain tool bridge smoke tests (no live Ollama required)."""

from __future__ import annotations

from dana.tools.langchain_tools import build_langchain_tools
from dana.tools.schema import ToolCall


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

    # Invoke the structured tool → Dana ToolCall IR.
    open_tool = next(t for t in tools if t.name == "open_application")
    result = open_tool.invoke({"app_name": "notepad"})
    assert result == "OK: open_application"
    assert calls and calls[0].tool_id == "open_application"
    assert calls[0].arguments.get("app_name") == "notepad"
    print(f"[PASS] built {len(tools)} LangChain tools; open_application OK")


def test_save_script_to_library_stays_in_sandbox(tmp_path, monkeypatch) -> None:
    from dana.tools import langchain_tools as lt

    lib = tmp_path / "execution_jail" / "library"
    lib.mkdir(parents=True)
    monkeypatch.setattr(lt, "_SANDBOX_LIBRARY", lib.resolve())
    monkeypatch.setattr(lt, "_REPO_ROOT", tmp_path.resolve())
    # Path-jail unit test — bypass Watchdog TTS policy (tested elsewhere).
    monkeypatch.setattr(
        "dana_jason_loop.jason_critic.static_code_safety_reject",
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
    from dana.tools import langchain_tools as lt
    from dana.prompts.spatial_synthesis import format_recency_context_block

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


if __name__ == "__main__":
    test_build_langchain_tools_from_registry()
    print("OK (run pytest for mocked LangChain loop)")
