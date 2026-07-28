"""ReAct tool-failure handling without live network or vault disk I/O.

Uses ``unittest.mock.patch`` so ``web_search`` and ``read_vault_memory`` never
hit the search API or the encrypted vault on disk.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from test_support_react import patch_scripted_llm
from dana.agentic import REACT_MAX_ITERS, run_react_loop
from dana.tools.broker import IntentBroker
from dana.tools.schema import ToolCall
from dana.prompts.spatial_synthesis import build_agent_system_prompt

NETWORK_ERROR = "Error: Network timeout. Cannot reach search API."
KEY_NOT_FOUND = "KeyNotFound"


def _system_prompt() -> str:
    return build_agent_system_prompt(
        spatial_block="vis=screen|ui=idle|dom=none@center|scene=[]|intent=",
        labels_csv="",
        profile_summary="{}",
        reply_lang="en",
    )


def test_offline_mode_web_search_network_timeout(monkeypatch) -> None:
    """Mocked web_search failure → graceful offline reply, no crash/hallucination."""
    broker = IntentBroker()
    query = "Research the latest news on AI"

    with patch("dana.web_search.web_search") as mock_search:
        mock_search.return_value = {
            "ok": False,
            "error": NETWORK_ERROR,
            "query": query,
            "results": [],
        }

        def execute_fn(tc: ToolCall) -> str:
            if tc.tool_id != "web_search":
                return f"ERROR: unexpected tool {tc.tool_id}"
            from dana.web_search import format_search_observation, web_search

            payload = web_search(str(tc.arguments.get("query") or ""))
            err = str(payload.get("error") or NETWORK_ERROR)
            obs = format_search_observation(payload)
            return f"ERROR: {err} | {obs}"

        patch_scripted_llm(
            monkeypatch,
            [
                "TOOL: web_search(query=latest AI news)",
                "FINAL: I can't connect to the internet right now — "
                "the search API timed out, so I don't have live news to share.",
            ],
        )
        result = run_react_loop(
            user_text=query,
            system_prompt=_system_prompt(),
            execute_fn=execute_fn,
            max_iters=REACT_MAX_ITERS,
            broker=broker,
            enable_reflection=False,
        )

        mock_search.assert_called()
        assert any(t.get("tool") == "web_search" for t in result.tool_trace)
        obs = " ".join(str(t.get("observation") or "") for t in result.tool_trace)
        assert (
            "timeout" in obs.lower()
            or "cannot reach" in obs.lower()
            or "network" in obs.lower()
        )

        reply = (result.final_text or "").lower()
        assert any(
            marker in reply
            for marker in (
                "can't connect",
                "cannot connect",
                "timed out",
                "timeout",
                "internet",
                "offline",
                "reach",
            )
        ), f"expected offline apology, got: {result.final_text!r}"
        assert "according to" not in reply
        assert result.had_errors or "error" in obs.lower()
        print(f"[PASS] Offline Mode: {result.final_text}")


def test_empty_vault_partner_name_unknown(monkeypatch) -> None:
    """Mocked read_vault_memory miss → polite 'not in vault' reply."""
    broker = IntentBroker()
    query = "What is my partner's name?"

    mock_vault = MagicMock()
    mock_vault.session_token = "test-session"
    mock_vault.profile = {}
    mock_vault.read_memory.side_effect = KeyError(KEY_NOT_FOUND)

    with patch("dana.core_agent.vault_client", mock_vault):

        def execute_fn(tc: ToolCall) -> str:
            if tc.tool_id != "read_vault_memory":
                return f"ERROR: unexpected tool {tc.tool_id}"
            key = str(tc.arguments.get("key") or "").strip() or "partner_name"
            try:
                value = mock_vault.read_memory(key)
            except KeyError:
                return f"OK: key '{key}' not found"
            if value is None:
                return f"OK: key '{key}' not found"
            return f"OK: {key}={value!r}"

        patch_scripted_llm(
            monkeypatch,
            [
                "TOOL: read_vault_memory(key=partner_name)",
                "FINAL: I don't have your partner's name in my vault — "
                "that information isn't saved yet.",
            ],
        )
        result = run_react_loop(
            user_text=query,
            system_prompt=_system_prompt(),
            execute_fn=execute_fn,
            max_iters=REACT_MAX_ITERS,
            broker=broker,
            enable_reflection=False,
        )

        mock_vault.read_memory.assert_called()
        assert any(t.get("tool") == "read_vault_memory" for t in result.tool_trace)
        obs = " ".join(str(t.get("observation") or "") for t in result.tool_trace)
        assert "not found" in obs.lower() or KEY_NOT_FOUND.lower() in obs.lower()

        reply = (result.final_text or "").lower()
        assert any(
            marker in reply
            for marker in (
                "don't have",
                "do not have",
                "not in my vault",
                "isn't saved",
                "not saved",
                "vault",
                "don't know",
            )
        ), f"expected polite vault miss, got: {result.final_text!r}"
        for invented in ("narges", "sarah", "alex"):
            assert invented not in reply
        print(f"[PASS] Empty Vault: {result.final_text}")


if __name__ == "__main__":
    test_offline_mode_web_search_network_timeout()
    test_empty_vault_partner_name_unknown()
    print("OK")
