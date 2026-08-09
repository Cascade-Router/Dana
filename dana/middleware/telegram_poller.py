"""Telegram long-polling bridge — Milestone 4 (Persistent Background Autonomy).

Once Dānā runs unattended as a background Windows process, the user needs
a remote 2-way channel: send a command from a phone, get Dānā's answer
back, without touching the desktop. This polls the Telegram Bot API's
``getUpdates`` long-poll endpoint on a daemon thread, and for every
message from the one allowed chat, injects it into Dānā's existing
"as if the user typed this" input pathway — the same flat-file +
``.trigger_ask`` mechanism ``dana.middleware.idle_monitor`` already uses
to feed background-research topics into ``InputIngest`` →
``task_queue.json`` → ``drain_structured_task_queue`` → ``run_react_loop``
(see ``idle_monitor._inject_research_via_input_txt``). Unlike that
background-research path, a Telegram message is NOT tagged
``"[BACKGROUND TASK]"`` — that tag forces deep-research routing and
vision-blindfolding elsewhere in the codebase, which is the opposite of
"treat it exactly like a user typing in the UI".

No third-party Telegram SDK — plain ``urllib.request`` against the HTTP
Bot API, matching the ``dana.tools.general.github_issue_reporter``
convention (explicit timeout, catch every failure mode, never raise).

Security: ``TELEGRAM_ALLOWED_CHAT_ID`` is a hard allowlist. A message from
any other chat id is silently dropped — never injected into the input
pipeline, never replied to. This is a personal remote-control channel, not
a public bot.

Isolation: ``start_telegram_poller()``/``stop_telegram_poller()`` follow
this codebase's established daemon-thread convention (see
``dana.middleware.idle_monitor``/``dana.middleware.sidekick_supervisor``,
which this module lives alongside for that reason): a module-wide
start-lock guards against double-starting, a ``threading.Event`` is the
cooperative stop signal, and the thread itself is created with
``daemon=True`` so the process can exit immediately even if a long-poll
request is still in flight — shutdown does not block on the network. An
env-var kill switch (``DANA_DISABLE_TELEGRAM_POLLER``) lets the whole
feature be disabled without touching code, matching every other
background-poller subsystem in this repo.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

TELEGRAM_API_BASE = "https://api.telegram.org"
_LONG_POLL_TIMEOUT_S = 25
# Client-side socket timeout must exceed the server-side long-poll window,
# or every idle poll (no new messages) looks like a network failure.
_REQUEST_TIMEOUT_S = _LONG_POLL_TIMEOUT_S + 10
_SEND_TIMEOUT_S = 20
_ERROR_BACKOFF_S = 5.0
_MAX_MESSAGE_CHARS = 4096  # Telegram's own hard limit on sendMessage text.

_start_lock = threading.Lock()
_started = False
_stop_event = threading.Event()
_thread: threading.Thread | None = None
_active_poller: "TelegramPoller | None" = None


def _disabled_by_env() -> bool:
    return os.environ.get("DANA_DISABLE_TELEGRAM_POLLER", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _default_inject_message(text: str) -> None:
    """Feed ``text`` into Dānā's input pipeline exactly like typed UI input.

    Appends to the flat input-ingest file and touches the empty
    ``.trigger_ask`` wake file — the same mechanism
    ``idle_monitor._inject_research_via_input_txt`` uses, minus the
    ``[BACKGROUND TASK]`` tag (see module docstring for why).
    """
    from dana.paths import TEXT_INJECTION_PATH, TRIGGER_ASK_PATH

    body = str(text or "").strip()
    if not body:
        return
    target = Path(TEXT_INJECTION_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(body.rstrip() + "\n\n")
    Path(TRIGGER_ASK_PATH).write_text("", encoding="utf-8")


@dataclass
class TelegramPoller:
    """Long-polls Telegram ``getUpdates`` and routes allowed messages in.

    ``bot_token``/``allowed_chat_id`` default to the
    ``TELEGRAM_BOT_TOKEN``/``TELEGRAM_ALLOWED_CHAT_ID`` environment
    variables when not passed explicitly. ``on_message`` defaults to the
    real input-injection pathway (writes to Dānā's input-ingest file);
    tests inject a stub to capture calls with no real file I/O.
    ``get_updates_fn``/``send_message_fn`` default to real
    ``urllib``-based Telegram Bot API calls; tests inject stubs so the
    polling/filtering/offset logic runs with no real network access.
    """

    bot_token: str = ""
    allowed_chat_id: str = ""
    on_message: Callable[[str], None] | None = None
    get_updates_fn: Callable[[int], list[dict[str, Any]]] | None = None
    send_message_fn: Callable[[str], None] | None = None
    offset: int = field(default=0)

    def __post_init__(self) -> None:
        if not self.bot_token:
            self.bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        if not self.allowed_chat_id:
            self.allowed_chat_id = os.environ.get("TELEGRAM_ALLOWED_CHAT_ID", "").strip()

    # -- backend (real Telegram Bot API via urllib; injectable for tests) --

    def _get_updates_via_api(self, offset: int) -> list[dict[str, Any]]:
        if self.get_updates_fn is not None:
            return self.get_updates_fn(offset)
        params = urllib.parse.urlencode(
            {
                "offset": offset,
                "timeout": _LONG_POLL_TIMEOUT_S,
                "allowed_updates": json.dumps(["message"]),
            }
        )
        url = f"{TELEGRAM_API_BASE}/bot{self.bot_token}/getUpdates?{params}"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_S) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        data = json.loads(raw) if raw else {}
        if not data.get("ok"):
            raise RuntimeError(f"Telegram getUpdates rejected: {data.get('description')}")
        return list(data.get("result") or [])

    def _send_message_via_api(self, text: str) -> None:
        if self.send_message_fn is not None:
            self.send_message_fn(text)
            return
        if not self.bot_token or not self.allowed_chat_id:
            raise RuntimeError("missing TELEGRAM_BOT_TOKEN/TELEGRAM_ALLOWED_CHAT_ID")
        payload = urllib.parse.urlencode(
            {"chat_id": self.allowed_chat_id, "text": str(text or "")[:_MAX_MESSAGE_CHARS]}
        ).encode("utf-8")
        url = f"{TELEGRAM_API_BASE}/bot{self.bot_token}/sendMessage"
        req = urllib.request.Request(url, data=payload, method="POST")
        with urllib.request.urlopen(req, timeout=_SEND_TIMEOUT_S) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        data = json.loads(raw) if raw else {}
        if not data.get("ok"):
            raise RuntimeError(f"Telegram sendMessage rejected: {data.get('description')}")

    # -- injection (real input-pipeline write; injectable for tests) --

    def _deliver(self, text: str) -> None:
        if self.on_message is not None:
            self.on_message(text)
            return
        _default_inject_message(text)

    # -- public API --

    def poll_once(self) -> int:
        """Fetch and process one batch of updates.

        Advances ``self.offset`` past every update seen (matched chat or
        not — an unmatched message must never be redelivered by Telegram
        on the next poll). Returns how many messages were actually routed
        into the input pipeline (i.e. passed the chat-id allowlist).
        """
        updates = self._get_updates_via_api(self.offset)
        processed = 0
        for update in updates:
            update_id = update.get("update_id")
            if isinstance(update_id, int):
                self.offset = max(self.offset, update_id + 1)
            message = update.get("message") or {}
            text = message.get("text")
            if not text:
                continue
            chat = message.get("chat") or {}
            chat_id = str(chat.get("id", ""))
            if not self.allowed_chat_id or chat_id != str(self.allowed_chat_id):
                continue  # security: silently drop messages from any other chat
            self._deliver(str(text))
            processed += 1
        return processed

    def send_message(self, text: str) -> str:
        """Reply hook: relay ``text`` back to the allowed Telegram chat.

        This is the "callback or mechanism" other Dānā code calls to route
        a generated answer back out to Telegram. Returns an
        ``"OK: ..."``/``"ERROR: ..."`` string like every other tool in
        this codebase — never raises.
        """
        body = str(text or "").strip()
        if not body:
            return "ERROR: send_message requires non-empty text"
        try:
            self._send_message_via_api(body)
        except Exception as exc:  # noqa: BLE001
            return f"ERROR: telegram send_message failed: {exc}"
        return "OK: telegram message sent"

    def run_forever(self, stop_event: threading.Event) -> None:
        """Poll in a loop until ``stop_event`` is set.

        A failed poll backs off ``_ERROR_BACKOFF_S`` (interruptible via
        ``stop_event.wait``) rather than retrying immediately — matches
        the "never crash the loop, degrade gracefully" convention used
        throughout this codebase's tool/backend error handling.
        """
        if not self.bot_token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN is required")
        if not self.allowed_chat_id:
            raise RuntimeError("TELEGRAM_ALLOWED_CHAT_ID is required")
        while not stop_event.is_set():
            try:
                self.poll_once()
            except Exception:  # noqa: BLE001
                stop_event.wait(timeout=_ERROR_BACKOFF_S)


def start_telegram_poller(*, poller: TelegramPoller | None = None) -> threading.Thread | None:
    """Idempotent start of the background long-polling thread.

    Returns ``None`` (and starts nothing) if ``DANA_DISABLE_TELEGRAM_POLLER``
    is set, or if ``TELEGRAM_BOT_TOKEN``/``TELEGRAM_ALLOWED_CHAT_ID`` aren't
    configured — missing credentials fail closed rather than spinning and
    logging errors forever. Returns the existing thread if already running.
    """
    global _started, _thread, _active_poller
    with _start_lock:
        if _started and _thread is not None and _thread.is_alive():
            return _thread
        if _disabled_by_env():
            return None
        p = poller or TelegramPoller()
        if not p.bot_token or not p.allowed_chat_id:
            return None

        _stop_event.clear()
        _active_poller = p

        def _run() -> None:
            p.run_forever(_stop_event)

        t = threading.Thread(target=_run, name="TelegramPoller", daemon=True)
        t.start()
        _thread = t
        _started = True
        return t


def stop_telegram_poller() -> None:
    """Signal the polling thread to stop.

    Does not block on the thread exiting — ``daemon=True`` already
    guarantees the process can exit immediately regardless of an in-flight
    long-poll request (which may take up to ``_REQUEST_TIMEOUT_S``).
    """
    global _started, _active_poller
    _stop_event.set()
    _started = False
    _active_poller = None


def relay_reply(text: str) -> str:
    """Send ``text`` back to the allowed chat via the currently running poller.

    The integration point other Dānā code (e.g. the main turn-taking loop)
    calls to relay an answer generated in response to a Telegram-originated
    message. Returns ``"ERROR: ..."`` if no poller is currently running.
    """
    if _active_poller is None:
        return "ERROR: telegram poller is not running"
    return _active_poller.send_message(text)


__all__ = (
    "TelegramPoller",
    "start_telegram_poller",
    "stop_telegram_poller",
    "relay_reply",
)
