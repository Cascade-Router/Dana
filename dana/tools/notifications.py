"""Remote push notifications — Pushover.

Milestone 4 (Persistent Background Autonomy): once Dānā runs unattended as
a background process, the user needs an out-of-band channel to be told
something happened (a background research task finished, a watchdog
fired, a long-running action completed) without watching the desktop UI.
Pushover is a simple REST push-notification service; this posts to it via
stdlib ``urllib.request`` (no new dependency), matching the existing
``dana.tools.general.github_issue_reporter`` convention: read credentials
from environment variables, POST with an explicit timeout, and always
return a string rather than raising, so a tool-call failure degrades
gracefully in the agent loop instead of crashing it.

Safety:
  - Reads ``PUSHOVER_USER_KEY``/``PUSHOVER_API_TOKEN`` first, falling back
    to the shorter ``PUSHOVER_USER``/``PUSHOVER_TOKEN`` names already
    present in this repo's ``.env`` — supports either naming without
    forcing a rename of existing secrets.
  - Missing credentials, network failure, and a non-success Pushover
    response are all caught and reported as an ``"ERROR: ..."`` string.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request

PUSHOVER_API_URL = "https://api.pushover.net/1/messages.json"
_REQUEST_TIMEOUT_S = 20
_MAX_MESSAGE_CHARS = 1024  # Pushover's own hard limit on the "message" field.


def _pushover_credentials() -> tuple[str, str]:
    user_key = (
        os.environ.get("PUSHOVER_USER_KEY") or os.environ.get("PUSHOVER_USER") or ""
    ).strip()
    api_token = (
        os.environ.get("PUSHOVER_API_TOKEN") or os.environ.get("PUSHOVER_TOKEN") or ""
    ).strip()
    return user_key, api_token


def send_pushover_notification(message: str, title: str = "Dānā") -> str:
    """POST ``message`` to the Pushover API as a push notification.

    Returns ``"OK: ..."`` on a confirmed send (Pushover ``status == 1``),
    or ``"ERROR: ..."`` if credentials are missing, the request fails, or
    Pushover rejects the payload. Never raises.
    """
    body = str(message or "").strip()
    if not body:
        return "ERROR: send_pushover_notification requires a non-empty message"

    user_key, api_token = _pushover_credentials()
    if not user_key or not api_token:
        return (
            "ERROR: send_pushover_notification missing credentials "
            "(set PUSHOVER_USER_KEY/PUSHOVER_API_TOKEN, or PUSHOVER_USER/PUSHOVER_TOKEN)"
        )

    payload = urllib.parse.urlencode(
        {
            "token": api_token,
            "user": user_key,
            "title": str(title or "Dānā").strip() or "Dānā",
            "message": body[:_MAX_MESSAGE_CHARS],
        }
    ).encode("utf-8")

    req = urllib.request.Request(PUSHOVER_API_URL, data=payload, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_S) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        data = json.loads(raw) if raw else {}
        if int(data.get("status", 0)) == 1:
            return f"OK: send_pushover_notification delivered (request={data.get('request')})"
        errors = data.get("errors") or ["unknown error"]
        return f"ERROR: send_pushover_notification rejected: {', '.join(str(e) for e in errors)}"
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
        return f"ERROR: send_pushover_notification HTTP {exc.code}: {detail or exc.reason}"
    except urllib.error.URLError as exc:
        return f"ERROR: send_pushover_notification network error: {exc.reason}"
    except TimeoutError:
        return "ERROR: send_pushover_notification timed out"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: send_pushover_notification failed: {exc}"


def send_notification(message: str) -> str:
    """Tool entry point: send ``message`` to the user's phone via Pushover."""
    return send_pushover_notification(message)


__all__ = ("send_pushover_notification", "send_notification")
