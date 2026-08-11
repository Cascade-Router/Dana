"""Guards against Project Omega / Whisper-bias conversational leaks."""

from __future__ import annotations

from dana.agentic import looks_like_confidential_fixture_leak, sanitize_spoken_reply
from dana.audio.stt import is_whisper_hallucination, is_whisper_prompt_echo
from dana.tools.broker import IntentBroker


def test_whisper_omega_bias_echo_is_hallucination() -> None:
    echo = "read the file project_omega_status.txt, file_jail_enforcer"
    assert is_whisper_prompt_echo(echo)
    assert is_whisper_hallucination(echo)
    assert IntentBroker().parse_utterance(echo) is None
    print("[PASS] whisper omega bias echo blocked")


def test_forge_cpu_not_routed_to_file() -> None:
    call = IntentBroker().parse_utterance(
        "build a tool that checks my current CPU and RAM usage"
    )
    assert call is not None
    assert call.tool_id == "architect_new_tool"
    print("[PASS] forge CPU routes to architect_new_tool")


def test_sanitize_blocks_omega_dump() -> None:
    dump = (
        "CONFIDENTIAL STATUS REPORT - PROJECT OMEGA\n"
        "Lead Engineer: Narges\n"
        "The multi-agent swarm deployment has encountered a latency bottleneck."
    )
    assert looks_like_confidential_fixture_leak(dump)
    out = sanitize_spoken_reply(dump, reply_lang="en", last_obs="")
    assert "OMEGA" not in out.upper()
    assert "Sorry" in out or "ask" in out.lower()
    print("[PASS] sanitize blocks Omega dump")


if __name__ == "__main__":
    test_whisper_omega_bias_echo_is_hallucination()
    test_forge_cpu_not_routed_to_file()
    test_sanitize_blocks_omega_dump()
    print("OK")
