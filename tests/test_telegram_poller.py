"""Unit tests for the Telegram long-polling bridge (Milestone 4).

Mocks the HTTP layer directly (``urllib.request.urlopen`` at its
in-module import path, matching ``tests/test_github_issue_reporter.py``'s
convention) for the two real-backend methods, and injects
``get_updates_fn``/``send_message_fn``/``on_message`` stubs for everything
else so the polling/filtering/offset/injection logic runs with no real
network access and no real file I/O.

Resets the module-wide start/stop singletons between tests (same rationale
as the actuator test files clearing ``DANA_OS_DRY_RUN``/rate-limiter
state — these are process-wide globals that would otherwise leak across
tests in the same pytest session).
"""
from __future__ import annotations

import json
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

import dana.middleware.telegram_poller as telegram_poller
from dana.middleware.telegram_poller import (
    TelegramPoller,
    relay_reply,
    start_telegram_poller,
    stop_telegram_poller,
)


@pytest.fixture(autouse=True)
def _reset_module_state(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_ALLOWED_CHAT_ID", raising=False)
    monkeypatch.delenv("DANA_DISABLE_TELEGRAM_POLLER", raising=False)
    telegram_poller._stop_event.clear()
    telegram_poller._started = False
    telegram_poller._thread = None
    telegram_poller._active_poller = None
    yield
    # Best-effort: stop any real thread a test left running before the next test.
    telegram_poller._stop_event.set()
    telegram_poller._started = False
    telegram_poller._thread = None
    telegram_poller._active_poller = None
    telegram_poller._stop_event.clear()


def _mock_urlopen_response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.read.return_value = json.dumps(payload).encode("utf-8")
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


# --------------------------------------------------------------------------
# poll_once — filtering / offset logic (no network, no file I/O)
# --------------------------------------------------------------------------


def test_poll_once_routes_message_from_allowed_chat_and_advances_offset() -> None:
    delivered: list[str] = []
    updates = [
        {"update_id": 10, "message": {"chat": {"id": 999}, "text": "from a stranger"}},
        {"update_id": 11, "message": {"chat": {"id": 42}, "text": "hello dana"}},
    ]
    poller = TelegramPoller(
        bot_token="tok",
        allowed_chat_id="42",
        on_message=delivered.append,
        get_updates_fn=lambda offset: updates,
    )
    processed = poller.poll_once()
    assert processed == 1
    assert delivered == ["hello dana"]
    assert poller.offset == 12


def test_poll_once_ignores_updates_without_text() -> None:
    delivered: list[str] = []
    updates = [{"update_id": 5, "message": {"chat": {"id": 42}, "sticker": {}}}]
    poller = TelegramPoller(
        bot_token="tok",
        allowed_chat_id="42",
        on_message=delivered.append,
        get_updates_fn=lambda offset: updates,
    )
    processed = poller.poll_once()
    assert processed == 0
    assert delivered == []
    assert poller.offset == 6


def test_poll_once_advances_offset_for_disallowed_chat_without_delivering() -> None:
    delivered: list[str] = []
    poller = TelegramPoller(
        bot_token="tok",
        allowed_chat_id="42",
        on_message=delivered.append,
        get_updates_fn=lambda offset: [
            {"update_id": 7, "message": {"chat": {"id": 1}, "text": "hi"}}
        ],
    )
    poller.poll_once()
    assert delivered == []
    assert poller.offset == 8


def test_poll_once_passes_current_offset_to_backend() -> None:
    seen_offsets: list[int] = []

    def fake_get_updates(offset):
        seen_offsets.append(offset)
        return []

    poller = TelegramPoller(bot_token="tok", allowed_chat_id="42", get_updates_fn=fake_get_updates)
    poller.offset = 99
    poller.poll_once()
    assert seen_offsets == [99]


def test_poll_once_processes_multiple_allowed_messages_in_one_batch() -> None:
    delivered: list[str] = []
    updates = [
        {"update_id": 1, "message": {"chat": {"id": 42}, "text": "first"}},
        {"update_id": 2, "message": {"chat": {"id": 42}, "text": "second"}},
    ]
    poller = TelegramPoller(
        bot_token="tok",
        allowed_chat_id="42",
        on_message=delivered.append,
        get_updates_fn=lambda offset: updates,
    )
    processed = poller.poll_once()
    assert processed == 2
    assert delivered == ["first", "second"]


# --------------------------------------------------------------------------
# send_message — reply hook (no network)
# --------------------------------------------------------------------------


def test_send_message_success() -> None:
    sent: list[str] = []
    poller = TelegramPoller(bot_token="tok", allowed_chat_id="42", send_message_fn=sent.append)
    out = poller.send_message("hi there")
    assert out == "OK: telegram message sent"
    assert sent == ["hi there"]


def test_send_message_rejects_empty_text_without_calling_backend() -> None:
    calls: list[str] = []
    poller = TelegramPoller(bot_token="tok", allowed_chat_id="42", send_message_fn=calls.append)
    out = poller.send_message("   ")
    assert out == "ERROR: send_message requires non-empty text"
    assert calls == []


def test_send_message_wraps_backend_failure_as_error_string() -> None:
    def failing_send(text):
        raise RuntimeError("Telegram sendMessage rejected: chat not found")

    poller = TelegramPoller(bot_token="tok", allowed_chat_id="42", send_message_fn=failing_send)
    out = poller.send_message("hi")
    assert out.startswith("ERROR: telegram send_message failed")
    assert "chat not found" in out


# --------------------------------------------------------------------------
# Real HTTP backend (mocked urlopen) — request shape + response parsing
# --------------------------------------------------------------------------


def test_get_updates_via_api_builds_expected_request_and_parses_result() -> None:
    poller = TelegramPoller(bot_token="tok123", allowed_chat_id="42")
    captured: dict = {}

    def fake_urlopen(req, timeout=None):  # noqa: ANN001
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        return _mock_urlopen_response({"ok": True, "result": [{"update_id": 1}]})

    with patch(
        "dana.middleware.telegram_poller.urllib.request.urlopen", side_effect=fake_urlopen
    ):
        result = poller._get_updates_via_api(5)

    assert result == [{"update_id": 1}]
    assert captured["method"] == "GET"
    assert captured["url"].startswith("https://api.telegram.org/bottok123/getUpdates?")
    assert "offset=5" in captured["url"]


def test_get_updates_via_api_raises_on_rejected_response() -> None:
    poller = TelegramPoller(bot_token="tok", allowed_chat_id="42")

    with patch(
        "dana.middleware.telegram_poller.urllib.request.urlopen",
        return_value=_mock_urlopen_response({"ok": False, "description": "Unauthorized"}),
    ):
        with pytest.raises(RuntimeError, match="Unauthorized"):
            poller._get_updates_via_api(0)


def test_send_message_via_api_posts_expected_body() -> None:
    poller = TelegramPoller(bot_token="tok123", allowed_chat_id="42")
    captured: dict = {}

    def fake_urlopen(req, timeout=None):  # noqa: ANN001
        import urllib.parse

        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["body"] = dict(urllib.parse.parse_qsl(req.data.decode("utf-8")))
        return _mock_urlopen_response({"ok": True})

    with patch(
        "dana.middleware.telegram_poller.urllib.request.urlopen", side_effect=fake_urlopen
    ):
        poller.send_message("hello there")

    assert captured["method"] == "POST"
    assert captured["url"] == "https://api.telegram.org/bottok123/sendMessage"
    assert captured["body"] == {"chat_id": "42", "text": "hello there"}


def test_send_message_via_api_raises_on_rejected_response() -> None:
    poller = TelegramPoller(bot_token="tok", allowed_chat_id="42")

    with patch(
        "dana.middleware.telegram_poller.urllib.request.urlopen",
        return_value=_mock_urlopen_response({"ok": False, "description": "chat not found"}),
    ):
        out = poller.send_message("hi")

    assert "chat not found" in out
    assert out.startswith("ERROR:")


# --------------------------------------------------------------------------
# Default input-injection pathway (real file I/O, redirected to tmp_path)
# --------------------------------------------------------------------------


def test_default_inject_message_writes_input_txt_and_touches_trigger(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    injection_path = tmp_path / "input.txt"
    trigger_path = tmp_path / ".trigger_ask"
    monkeypatch.setattr("dana.paths.TEXT_INJECTION_PATH", injection_path)
    monkeypatch.setattr("dana.paths.TRIGGER_ASK_PATH", trigger_path)

    telegram_poller._default_inject_message("hello from telegram")

    assert injection_path.read_text(encoding="utf-8") == "hello from telegram\n\n"
    assert trigger_path.exists()
    assert trigger_path.read_text(encoding="utf-8") == ""


def test_default_inject_message_ignores_empty_text(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    injection_path = tmp_path / "input.txt"
    trigger_path = tmp_path / ".trigger_ask"
    monkeypatch.setattr("dana.paths.TEXT_INJECTION_PATH", injection_path)
    monkeypatch.setattr("dana.paths.TRIGGER_ASK_PATH", trigger_path)

    telegram_poller._default_inject_message("   ")

    assert not injection_path.exists()
    assert not trigger_path.exists()


def test_poll_once_uses_default_injection_when_no_on_message_given(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    injection_path = tmp_path / "input.txt"
    trigger_path = tmp_path / ".trigger_ask"
    monkeypatch.setattr("dana.paths.TEXT_INJECTION_PATH", injection_path)
    monkeypatch.setattr("dana.paths.TRIGGER_ASK_PATH", trigger_path)

    poller = TelegramPoller(
        bot_token="tok",
        allowed_chat_id="42",
        get_updates_fn=lambda offset: [
            {"update_id": 1, "message": {"chat": {"id": 42}, "text": "hello dana"}}
        ],
    )
    poller.poll_once()

    assert injection_path.read_text(encoding="utf-8") == "hello dana\n\n"
    assert trigger_path.exists()


# --------------------------------------------------------------------------
# run_forever — credential guard + interruptible loop
# --------------------------------------------------------------------------


def test_run_forever_raises_if_bot_token_missing() -> None:
    poller = TelegramPoller(bot_token="", allowed_chat_id="42")
    with pytest.raises(RuntimeError, match="TELEGRAM_BOT_TOKEN"):
        poller.run_forever(threading.Event())


def test_run_forever_raises_if_allowed_chat_id_missing() -> None:
    poller = TelegramPoller(bot_token="tok", allowed_chat_id="")
    with pytest.raises(RuntimeError, match="TELEGRAM_ALLOWED_CHAT_ID"):
        poller.run_forever(threading.Event())


def test_run_forever_stops_promptly_when_stop_event_is_set() -> None:
    poller = TelegramPoller(bot_token="tok", allowed_chat_id="42", get_updates_fn=lambda offset: [])
    stop_event = threading.Event()
    t = threading.Thread(target=poller.run_forever, args=(stop_event,), daemon=True)
    t.start()
    time.sleep(0.05)
    stop_event.set()
    t.join(timeout=2.0)
    assert not t.is_alive()


def test_run_forever_backs_off_on_backend_error_without_crashing_the_loop() -> None:
    calls: list[int] = []

    def flaky_get_updates(offset):
        calls.append(offset)
        raise RuntimeError("network blip")

    poller = TelegramPoller(bot_token="tok", allowed_chat_id="42", get_updates_fn=flaky_get_updates)
    stop_event = threading.Event()
    t = threading.Thread(target=poller.run_forever, args=(stop_event,), daemon=True)
    t.start()
    time.sleep(0.05)
    stop_event.set()
    # _ERROR_BACKOFF_S is a `stop_event.wait(...)`, which returns immediately
    # once the event is set — so this join should be fast, not wait 5s.
    t.join(timeout=2.0)
    assert not t.is_alive()
    assert len(calls) >= 1


# --------------------------------------------------------------------------
# Module-level start/stop/relay singletons
# --------------------------------------------------------------------------


def test_start_telegram_poller_returns_none_when_credentials_missing() -> None:
    poller = TelegramPoller(bot_token="", allowed_chat_id="")
    assert start_telegram_poller(poller=poller) is None


def test_start_telegram_poller_returns_none_when_disabled_by_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DANA_DISABLE_TELEGRAM_POLLER", "1")
    poller = TelegramPoller(bot_token="tok", allowed_chat_id="42", get_updates_fn=lambda o: [])
    assert start_telegram_poller(poller=poller) is None


def test_start_telegram_poller_starts_daemon_thread_and_is_idempotent() -> None:
    def slow_empty_updates(offset):
        time.sleep(0.01)
        return []

    poller = TelegramPoller(bot_token="tok", allowed_chat_id="42", get_updates_fn=slow_empty_updates)
    t1 = start_telegram_poller(poller=poller)
    try:
        assert t1 is not None
        assert t1.daemon is True
        assert t1.is_alive()

        t2 = start_telegram_poller(poller=poller)
        assert t2 is t1  # idempotent — second call returns the already-running thread
    finally:
        stop_telegram_poller()
        t1.join(timeout=2.0)
    assert not t1.is_alive()


def test_relay_reply_without_running_poller_returns_error() -> None:
    assert relay_reply("hi") == "ERROR: telegram poller is not running"


def test_relay_reply_routes_through_the_active_poller() -> None:
    sent: list[str] = []

    def slow_empty_updates(offset):
        time.sleep(0.01)
        return []

    poller = TelegramPoller(
        bot_token="tok",
        allowed_chat_id="42",
        get_updates_fn=slow_empty_updates,
        send_message_fn=sent.append,
    )
    t = start_telegram_poller(poller=poller)
    try:
        out = relay_reply("the answer")
        assert out == "OK: telegram message sent"
        assert sent == ["the answer"]
    finally:
        stop_telegram_poller()
        t.join(timeout=2.0)
