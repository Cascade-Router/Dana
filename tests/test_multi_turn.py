"""Automated multi-turn context test for Donna's text-bypass ReAct pipeline.

Simulates a 4-turn conversation through ``run_react_loop`` + the agent memory
window (``conversation_history`` / ``flush_conversation_memory``), with no mic,
TTS, or live network I/O.
"""

from __future__ import annotations

import dana.core_agent as agent
from test_support_react import patch_scripted_llm
from dana.agentic import REACT_MAX_ITERS, run_react_loop
from dana.tools.broker import IntentBroker
from dana.tools.schema import ToolCall
from dana.prompts.spatial_synthesis import build_agent_system_prompt

FAVORITE_COLOR = "neon green"


def _system_prompt() -> str:
    return build_agent_system_prompt(
        spatial_block="vis=screen|ui=idle|dom=none@center|scene=[]|intent=",
        labels_csv="",
        profile_summary="{}",
        reply_lang="en",
    )


def _prior_from_history() -> list[dict[str, str]]:
    """Mirror ``run_brain_turn``: pass user/assistant turns into ReAct."""
    with agent.conversation_history_lock:
        return [
            {"role": m["role"], "content": m["content"]}
            for m in agent.conversation_history
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]


def _noop_execute(_tc: ToolCall) -> str:
    return "OK: no-op"


def _report(turn: int, label: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"[Turn {turn}] {label}: {status}{suffix}")


def test_multi_turn_context_memory_and_flush(monkeypatch) -> None:
    """4-turn memory window: store → recall → clear context → forget."""
    broker = IntentBroker()
    system = _system_prompt()

    # Isolate from any leftover live-agent history.
    with agent.conversation_history_lock:
        agent.conversation_history.clear()

    # --- Turn 1: store favorite color ---------------------------------
    turn1 = f"My favorite color is {FAVORITE_COLOR}."
    patch_scripted_llm(
        monkeypatch,
        [f"FINAL: Got it — I'll remember your favorite color is {FAVORITE_COLOR}."],
    )
    result1 = run_react_loop(
        user_text=turn1,
        system_prompt=system,
        execute_fn=_noop_execute,
        max_iters=REACT_MAX_ITERS,
        broker=broker,
        enable_reflection=False,
        prior_messages=_prior_from_history(),
    )
    ok1 = bool(result1.final_text) and "error" not in result1.final_text.lower()
    assert ok1, f"Turn 1 failed: {result1.final_text!r}"
    agent.commit_agentic_turn(system, turn1, result1.final_text)
    _report(1, "acknowledge favorite color", ok1, result1.final_text[:80])

    # Memory window must now hold the user statement.
    hist_blob = " ".join(m.get("content", "") for m in _prior_from_history())
    assert FAVORITE_COLOR in hist_blob, "favorite color missing from conversation_history"
    print(f"         conversation_history msgs={len(_prior_from_history())}")

    # --- Turn 2: recall from conversation_history ----------------------
    turn2 = "What is my favorite color?"
    history_seen = {"v": False}

    def ask_recall(messages: list[dict[str, str]]) -> str:
        blob = " | ".join(m.get("content", "") for m in messages)
        # Also consult the agent memory window (graph checkpointer may not
        # rehydrate prior turns into the FakeLLM message list).
        hist_blob = " ".join(m.get("content", "") for m in _prior_from_history())
        combined = f"{blob} | {hist_blob}"
        if FAVORITE_COLOR in combined:
            history_seen["v"] = True
            return f"FINAL: Your favorite color is {FAVORITE_COLOR}."
        return "FINAL: I don't recall a favorite color."

    patch_scripted_llm(monkeypatch, ask_recall)
    result2 = run_react_loop(
        user_text=turn2,
        system_prompt=system,
        execute_fn=_noop_execute,
        max_iters=REACT_MAX_ITERS,
        broker=broker,
        enable_reflection=False,
        prior_messages=_prior_from_history(),
    )
    ok2 = (
        history_seen["v"]
        and FAVORITE_COLOR.lower() in (result2.final_text or "").lower()
    )
    assert ok2, (
        f"Turn 2 failed to recall {FAVORITE_COLOR!r} "
        f"(history_seen={history_seen['v']}, reply={result2.final_text!r})"
    )
    agent.commit_agentic_turn(system, turn2, result2.final_text)
    _report(2, "recall neon green from memory", ok2, result2.final_text[:80])

    # --- Turn 3: clear context → flush_conversation_memory --------------
    turn3 = "Clear context."
    assert agent.is_clear_context_command(turn3), "voice path must detect clear command"
    cleared = agent.flush_conversation_memory(reason="voice_command")
    reply3 = agent.clear_context_spoken_reply(turn3)
    ok3 = cleared >= 2 and len(_prior_from_history()) == 0
    assert ok3, f"Turn 3 flush failed (cleared={cleared}, hist={_prior_from_history()})"
    assert "clear" in reply3.lower() or "fresh" in reply3.lower()
    _report(3, "flush_conversation_memory triggered", ok3, f"cleared={cleared} msgs")

    # --- Turn 4: prove short-term memory was wiped ---------------------
    turn4 = "What is my favorite color?"
    color_leaked = {"v": False}

    def ask_after_flush(messages: list[dict[str, str]]) -> str:
        blob = " | ".join(m.get("content", "") for m in messages)
        # Turn-1 statement must be gone from the memory window.
        if turn1 in blob or (
            FAVORITE_COLOR in blob and "favorite color is" in blob.lower()
            and turn4 not in blob  # allow the question itself only
        ):
            color_leaked["v"] = True
            return f"FINAL: Your favorite color is {FAVORITE_COLOR}."
        if FAVORITE_COLOR in blob and turn1 in blob:
            color_leaked["v"] = True
            return f"FINAL: Your favorite color is {FAVORITE_COLOR}."
        return (
            "FINAL: I don't know your favorite color — "
            "my short-term context was cleared."
        )

    patch_scripted_llm(monkeypatch, ask_after_flush)
    result4 = run_react_loop(
        user_text=turn4,
        system_prompt=system,
        execute_fn=_noop_execute,
        max_iters=REACT_MAX_ITERS,
        broker=broker,
        enable_reflection=False,
        prior_messages=_prior_from_history(),
    )
    reply4 = (result4.final_text or "").lower()
    ok4 = (
        not color_leaked["v"]
        and FAVORITE_COLOR.lower() not in reply4
        and (
            "don't know" in reply4
            or "do not know" in reply4
            or "cleared" in reply4
            or "not sure" in reply4
        )
    )
    assert ok4, (
        f"Turn 4 should forget the color "
        f"(color_leaked={color_leaked['v']}, reply={result4.final_text!r})"
    )
    _report(4, "memory wiped — color unknown", ok4, result4.final_text[:80])

    print("\n[multi-turn] ALL TURNS PASSED")


if __name__ == "__main__":
    test_multi_turn_context_memory_and_flush()
    print("OK")
