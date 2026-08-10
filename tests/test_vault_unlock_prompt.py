"""Vault-unlock GUI prompt bridge (dana.core.shared_state).

Covers the request/response handoff ``unlock_dana_memory()`` uses to ask a
registered GUI for a passcode instead of silently ``SystemExit``-ing the
AgentLoop thread when no env/keyring credential is available.
"""

from __future__ import annotations

import pytest

# dana.core.shared_state imports dana.audio.tts_worker at module level, whose
# own module-level canned-UX table construction lazily imports shared_state
# back — importing dana.core_agent first (as every other test in this suite
# ends up doing transitively) fully warms that cycle so this file is safe to
# run standalone, not just as part of the full suite's alphabetical order.
import dana.core_agent  # noqa: F401
from dana.core import shared_state as state


@pytest.fixture(autouse=True)
def _clean_vault_listeners():
    """Reset before AND after: other test modules instantiate ``DanaGUI()``
    directly (e.g. test_stage810_silent_text_chat.py) without ever calling
    ``unregister_vault_prompt_listener`` — same pre-existing gap as the
    transcript-listener registry — so a prior test in the full suite can
    leave a stale listener registered here.
    """

    def _reset() -> None:
        with state._vault_prompt_listeners_lock:
            state._vault_prompt_listeners.clear()
        state.vault_unlock_response = None
        state.vault_unlock_response_event.clear()

    _reset()
    yield
    _reset()


def test_request_vault_unlock_returns_none_without_listener() -> None:
    assert state.has_vault_prompt_listener() is False
    assert state.request_vault_unlock("locked", timeout=1.0) is None


def test_register_and_unregister_listener() -> None:
    def _listener(reason: str) -> None:
        pass

    state.register_vault_prompt_listener(_listener)
    assert state.has_vault_prompt_listener() is True
    state.register_vault_prompt_listener(_listener)  # idempotent, no dup
    assert len(state._vault_prompt_listeners) == 1

    state.unregister_vault_prompt_listener(_listener)
    assert state.has_vault_prompt_listener() is False


def test_request_vault_unlock_returns_password_supplied_by_listener() -> None:
    seen_reasons: list[str] = []

    def _listener(reason: str) -> None:
        seen_reasons.append(reason)
        state.supply_vault_unlock_response("typed-secret")

    state.register_vault_prompt_listener(_listener)
    try:
        result = state.request_vault_unlock("vault needs a passcode", timeout=2.0)
    finally:
        state.unregister_vault_prompt_listener(_listener)

    assert result == "typed-secret"
    assert seen_reasons == ["vault needs a passcode"]


def test_request_vault_unlock_cancel_returns_none() -> None:
    def _listener(reason: str) -> None:
        state.supply_vault_unlock_response(None)

    state.register_vault_prompt_listener(_listener)
    try:
        result = state.request_vault_unlock("vault needs a passcode", timeout=2.0)
    finally:
        state.unregister_vault_prompt_listener(_listener)

    assert result is None


def test_request_vault_unlock_timeout_returns_none() -> None:
    def _listener(reason: str) -> None:
        pass  # never responds

    state.register_vault_prompt_listener(_listener)
    try:
        result = state.request_vault_unlock("vault needs a passcode", timeout=0.2)
    finally:
        state.unregister_vault_prompt_listener(_listener)

    assert result is None


def test_notify_vault_unlocked_sends_empty_reason() -> None:
    seen_reasons: list[str] = []
    state.register_vault_prompt_listener(seen_reasons.append)
    try:
        state.notify_vault_unlocked()
    finally:
        state.unregister_vault_prompt_listener(seen_reasons.append)

    assert seen_reasons == [""]


def test_listener_exception_does_not_propagate() -> None:
    def _boom(reason: str) -> None:
        raise RuntimeError("listener blew up")

    state.register_vault_prompt_listener(_boom)
    try:
        # Must not raise, and must fall through to the timeout (no response).
        result = state.request_vault_unlock("vault needs a passcode", timeout=0.2)
    finally:
        state.unregister_vault_prompt_listener(_boom)

    assert result is None
