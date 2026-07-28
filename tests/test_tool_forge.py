"""Semantic Tool Registry + Tool Forge pipeline tests (no live Ollama required)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from dana.tools.registry import (
    ToolRegistry,
    get_tool_registry,
    hash_embed,
    load_security_policy,
)
from dana.tools.sandbox_io import SandboxReadError, sandbox_read, sandbox_read_root
from dana.swarm.tool_forge_graph import (
    analyze_tool_ast,
    ast_gatekeeper_forge,
    build_tool_forge_graph,
    hot_load_forged_tool,
    security_reviewer_agent,
)
from dana.tools.schema import ToolSpec


def test_hash_embed_stable() -> None:
    a = hash_embed("web search the news")
    b = hash_embed("web search the news")
    assert a.shape == b.shape
    assert (a == b).all()
    print("[PASS] hash_embed stable")


def test_tool_registry_o1_and_retrieve() -> None:
    reg = ToolRegistry()
    reg.register(
        ToolSpec(
            id="web_search",
            description_en="Search the live web for news and facts",
            description_fa="",
        )
    )
    reg.register(
        ToolSpec(
            id="open_application",
            description_en="Open a desktop application by name",
            description_fa="",
        )
    )
    assert "web_search" in reg.tools  # O(1) dict
    hits = reg.retrieve("look up latest news online", k=1)
    assert hits and hits[0].name == "web_search"
    specs = reg.retrieve_specs("open application launch notepad desktop app", k=2)
    assert "open_application" in specs
    print("[PASS] ToolRegistry O(1) + semantic retrieve")


def test_get_tool_registry_loads_tools_json() -> None:
    reg = get_tool_registry(reload=True)
    assert len(reg.tools) >= 5
    assert "web_search" in reg.tools or "open_application" in reg.tools
    print(f"[PASS] get_tool_registry loaded {len(reg.tools)} tools")


def test_security_policy_tiers() -> None:
    policy = load_security_policy()
    assert "math" in policy["tier1_allowed"]
    assert "os" in policy["tier3_forbidden"]
    assert "open" in policy["forbidden_builtins"]
    print("[PASS] security_policy.json tiers")


def test_analyze_tool_ast_rejects_os_and_open() -> None:
    bad = "import os\ndef foo(text):\n    return os.getcwd()\n"
    errs = analyze_tool_ast(bad)
    assert errs and any("Tier-3" in e or "forbidden" in e.lower() for e in errs)

    bad_open = "def foo(text):\n    return open('x').read()\n"
    errs2 = analyze_tool_ast(bad_open)
    assert errs2 and any("open" in e.lower() for e in errs2)

    good = (
        "from dana.tools.sandbox_io import sandbox_read\n"
        "import math\n"
        "def count_pi(text):\n"
        "    return str(math.pi)\n"
    )
    assert analyze_tool_ast(good) == []
    print("[PASS] analyze_tool_ast policy")


def test_assemble_forged_tool_topology() -> None:
    from dana.swarm.tool_forge_template import assemble_forged_tool, extract_coder_json

    src = assemble_forged_tool(
        tool_name="reverse_string",
        docstring="Reverse text.",
        python_code="return (text or '')[::-1]",
    )
    assert "@tool" in src
    assert "def reverse_string(" in src
    assert "return (text or '')[::-1]" in src
    assert analyze_tool_ast(src) == []

    data = extract_coder_json(
        '{"tool_name":"x","description":"d","docstring":"d","python_code":"return 1"}'
    )
    assert data and data["python_code"] == "return 1"
    print("[PASS] assemble_forged_tool topology")


def test_ast_gatekeeper_json_brace_feedback() -> None:
    from dana.swarm.tool_forge_template import JSON_SCHEMA_FAILURE

    out = ast_gatekeeper_forge(
        {
            "query": "x",
            "tool_name": "bad",
            "code": '{"tool_name": "x", "python_code": "return 1"}',
            "lint_errors": "",
            "security_feedback": "",
            "security_review": {},
            "feedback": "",
            "status": "drafting",
            "revisions": 0,
            "history": [],
            "loaded_tool": "",
        }
    )
    assert out["status"] == "LINT_FAIL"
    assert "not valid JSON" in (out.get("lint_errors") or "")
    assert JSON_SCHEMA_FAILURE.split(".")[0] in (out.get("lint_errors") or "")
    print("[PASS] ast_gatekeeper JSON brace feedback")


def test_ast_gatekeeper_forge_bounce() -> None:
    out = ast_gatekeeper_forge(
        {
            "query": "x",
            "tool_name": "bad",
            "code": "import subprocess\ndef bad(text): return 1\n",
            "lint_errors": "",
            "security_feedback": "",
            "security_review": {},
            "feedback": "",
            "status": "drafting",
            "revisions": 0,
            "history": [],
            "loaded_tool": "",
        }
    )
    assert out["status"] == "LINT_FAIL"
    assert "FATAL" in (out.get("lint_errors") or "")
    print("[PASS] ast_gatekeeper_forge LINT_FAIL")


def test_security_reviewer_json_schema(monkeypatch) -> None:
    class _Msg:
        content = (
            '{"status":"REJECTED","threat_assessment":"exfil risk",'
            '"violations":["network"],"required_remediation":"remove sockets"}'
        )

    fake = MagicMock()
    fake.invoke.return_value = _Msg()
    monkeypatch.setattr(
        "dana.swarm.tool_forge_graph._chat_ollama",
        lambda **_k: fake,
    )
    out = security_reviewer_agent(
        {
            "query": "x",
            "tool_name": "t",
            "code": "def t(text): return text\n",
            "lint_errors": "",
            "security_feedback": "",
            "security_review": {},
            "feedback": "",
            "status": "LINT_OK",
            "revisions": 0,
            "history": [],
            "loaded_tool": "",
        }
    )
    assert out["status"] == "SEC_REJECTED"
    assert "remove sockets" in (out.get("security_feedback") or "")
    assert out["security_review"]["status"] == "REJECTED"
    print("[PASS] security_reviewer rigid JSON")


def test_hot_load_respects_synthesis_lock(monkeypatch) -> None:
    monkeypatch.setattr(
        "dana.settings.is_dynamic_tool_synthesis_enabled",
        lambda: False,
    )
    out = hot_load_forged_tool(
        {
            "query": "reverse text",
            "tool_name": "reverse_string",
            "code": "def reverse_string(text):\n    return (text or '')[::-1]\n",
            "lint_errors": "",
            "security_feedback": "",
            "security_review": {"status": "APPROVED"},
            "feedback": "reverse a string",
            "status": "APPROVED",
            "revisions": 1,
            "history": [],
            "loaded_tool": "",
        }
    )
    assert out["status"] == "error"
    assert "locked" in (out.get("feedback") or "").lower()
    print("[PASS] hot_load synthesis lock")


def test_hot_load_registers_when_unlocked(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "dana.settings.is_dynamic_tool_synthesis_enabled",
        lambda: True,
    )
    # Isolate generated tools dir.
    gen = tmp_path / "generated_tools"
    gen.mkdir()
    (gen / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(
        "dana.swarm.tool_forge_graph.CUSTOM_TOOLS_DIR",
        gen,
    )
    monkeypatch.setattr(
        "dana.tools.registry.CUSTOM_TOOLS_DIR",
        gen,
    )
    # Avoid mutating real tools.json / broker.
    monkeypatch.setattr(
        "donna_security.register_tool_schema",
        lambda *a, **k: {"id": "reverse_string"},
        raising=False,
    )
    with patch("dana.tools.broker.reload_broker_registry", lambda: None):
        # register_tool_schema import is inside hot_load — patch donna_security module attr
        import donna_security

        monkeypatch.setattr(
            donna_security,
            "register_tool_schema",
            lambda *a, **k: {"id": "reverse_string"},
        )
        out = hot_load_forged_tool(
            {
                "query": "reverse text",
                "tool_name": "reverse_string",
                "code": "def reverse_string(text):\n    return (text or '')[::-1]\n",
                "lint_errors": "",
                "security_feedback": "",
                "security_review": {"status": "APPROVED"},
                "feedback": "reverse a string",
                "status": "APPROVED",
                "revisions": 1,
                "history": [],
                "loaded_tool": "",
            }
        )
    assert out["status"] == "loaded", out
    assert out["loaded_tool"] == "reverse_string"
    assert (gen / "reverse_string.py").is_file()
    reg = get_tool_registry()
    assert "reverse_string" in reg.tools
    print("[PASS] hot_load registers unlocked tool")


def test_sandbox_read_jail(tmp_path, monkeypatch) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "note.txt").write_text("hello jail", encoding="utf-8")
    monkeypatch.setattr("dana.tools.sandbox_io._SANDBOX_READ_ROOT", docs.resolve())
    assert sandbox_read("note.txt") == "hello jail"
    with pytest.raises(SandboxReadError):
        sandbox_read("../../etc/passwd")
    print("[PASS] sandbox_read jail")


def test_build_tool_forge_graph_compiles() -> None:
    app = build_tool_forge_graph()
    assert app is not None
    print("[PASS] tool forge graph compiles")


if __name__ == "__main__":
    test_hash_embed_stable()
    test_tool_registry_o1_and_retrieve()
    test_security_policy_tiers()
    test_analyze_tool_ast_rejects_os_and_open()
    test_assemble_forged_tool_topology()
    test_ast_gatekeeper_json_brace_feedback()
    print("OK")
