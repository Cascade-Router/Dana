"""TTS / spoken-reply sanitization — strip few-shot dialog leaks before Piper."""

from __future__ import annotations

from dana.agentic import sanitize_spoken_reply, strip_simulated_dialog_leaks
from dana.tools.broker import IntentBroker


def test_strip_user_prefix_and_arrow_speak() -> None:
    leaked = (
        'User: "What time is it?"\n'
        "→ speak: It's 3:05 PM."
    )
    cleaned = strip_simulated_dialog_leaks(leaked)
    assert "User:" not in cleaned
    assert "→" not in cleaned
    assert "3:05" in cleaned
    spoken = sanitize_spoken_reply(leaked, reply_lang="en")
    assert spoken == "It's 3:05 PM."
    print("[PASS] strip User: / arrow speak leaks")


def test_strip_me_and_answer_labels() -> None:
    leaked = "Me: hello\nAnswer: The vault is locked."
    spoken = sanitize_spoken_reply(leaked, reply_lang="en")
    assert "Me:" not in spoken
    assert spoken == "The vault is locked."
    print("[PASS] strip Me: / Answer: labels")


def test_strip_raw_json_tool_call_speech() -> None:
    from dana.agentic import looks_like_raw_json_speech, sanitize_spoken_reply

    leak = '{"name": "read_local_file", "parameters": {"path": "tools.json"}}'
    assert looks_like_raw_json_speech(leak)
    spoken = sanitize_spoken_reply(leak, reply_lang="en", last_obs="")
    assert "{" not in spoken
    assert "read_local_file" not in spoken
    assert "Sorry" in spoken or "ask me again" in spoken.lower()
    print("[PASS] strip raw JSON tool-call speech")


def test_memory_key_miss_fallback_is_natural() -> None:
    from dana.agentic import _obs_fallback

    spoken = _obs_fallback("Error: Memory key not found in vault.", "en")
    assert "don't know" in spoken.lower() or "referring" in spoken.lower()
    assert "couldn't finish" not in spoken.lower()
    print("[PASS] memory key miss -> natural reply")


def test_research_latest_updates_forces_web_search() -> None:
    broker = IntentBroker()
    call = broker.parse_utterance("research the latest updates on LangGraph")
    assert call is not None
    assert call.tool_id == "web_search"
    assert str(call.arguments.get("query") or "").strip()
    print("[PASS] research the latest updates -> web_search")


def test_write_a_report_forces_swarm() -> None:
    broker = IntentBroker()
    call = broker.parse_utterance("Donna, write a report on robots navigating a maze")
    assert call is not None
    assert call.tool_id == "dispatch_research_swarm"
    assert str(call.arguments.get("query") or "").strip()
    print("[PASS] write a report -> dispatch_research_swarm")


def test_generic_greeting_not_forced_for_chat() -> None:
    # Sanitizer leaves a normal greeting alone when there is no tool obs.
    assert sanitize_spoken_reply("Hi there!", reply_lang="en") == "Hi there!"
    print("[PASS] bare greeting preserved for true chat turns")


if __name__ == "__main__":
    test_strip_user_prefix_and_arrow_speak()
    test_strip_me_and_answer_labels()
    test_strip_raw_json_tool_call_speech()
    test_memory_key_miss_fallback_is_natural()
    test_research_latest_updates_forces_web_search()
    test_write_a_report_forces_swarm()
    test_generic_greeting_not_forced_for_chat()
    print("OK")
