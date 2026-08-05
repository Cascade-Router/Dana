"""Verify Phase 2 idle monitor + keep_alive wiring (no engine required)."""

from __future__ import annotations

import inspect

from dana.middleware.idle_monitor import (
    DEFAULT_RESEARCH_TOPICS,
    USER_ACTIVE,
    get_idle_state,
    ollama_keep_alive,
)
from dana.tools.broker import IntentBroker


def main() -> None:
    assert get_idle_state() == USER_ACTIVE
    assert ollama_keep_alive() in (0, "5m")

    broker = IntentBroker()
    for topic in DEFAULT_RESEARCH_TOPICS:
        call = broker.parse_utterance(topic)
        tool = None if call is None else call.tool_id
        print(f"{tool}\t{topic[:72]}...")

    import dana.cascade_router as cr
    import dana.core_agent as ca

    src = inspect.getsource(cr.resolve_chat_model)
    assert 'keep_alive="-1"' not in src
    assert "keep_alive='-1'" not in src
    src2 = inspect.getsource(ca._ask_ollama_messages_unlocked)
    assert "keep_alive\": -1" not in src2
    assert "ollama_keep_alive" in src and "ollama_keep_alive" in src2
    print("ok keep_alive=", ollama_keep_alive(), "state=", get_idle_state())


if __name__ == "__main__":
    main()
