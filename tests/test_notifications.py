"""Unit tests for the Pushover notification tool (Milestone 4).

Mocks the HTTP layer directly (``urllib.request.urlopen`` at its
in-module import path), matching the convention established in
``tests/test_github_issue_reporter.py`` for other raw-``urllib`` tool
modules in this codebase.
"""
from __future__ import annotations

import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from dana.tools.notifications import send_notification, send_pushover_notification


def _mock_urlopen_response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.read.return_value = json.dumps(payload).encode("utf-8")
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


def _clear_pushover_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("PUSHOVER_USER_KEY", "PUSHOVER_API_TOKEN", "PUSHOVER_USER", "PUSHOVER_TOKEN"):
        monkeypatch.delenv(key, raising=False)


def test_missing_credentials_returns_error_without_network_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_pushover_env(monkeypatch)
    with patch("dana.tools.notifications.urllib.request.urlopen") as mock_urlopen:
        out = send_pushover_notification("hello")
    assert out.startswith("ERROR: send_pushover_notification missing credentials")
    mock_urlopen.assert_not_called()


def test_empty_message_rejected_without_credentials_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_pushover_env(monkeypatch)
    out = send_pushover_notification("   ")
    assert out == "ERROR: send_pushover_notification requires a non-empty message"


def test_success_with_new_env_var_names(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_pushover_env(monkeypatch)
    monkeypatch.setenv("PUSHOVER_USER_KEY", "u123")
    monkeypatch.setenv("PUSHOVER_API_TOKEN", "t456")

    captured: dict = {}

    def fake_urlopen(req, timeout=20):  # noqa: ANN001
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        import urllib.parse

        captured["body"] = dict(urllib.parse.parse_qsl(req.data.decode("utf-8")))
        return _mock_urlopen_response({"status": 1, "request": "req-abc"})

    with patch(
        "dana.tools.notifications.urllib.request.urlopen", side_effect=fake_urlopen
    ):
        out = send_pushover_notification("hello world", title="Test Title")

    assert out == "OK: send_pushover_notification delivered (request=req-abc)"
    assert captured["method"] == "POST"
    assert captured["url"] == "https://api.pushover.net/1/messages.json"
    assert captured["body"] == {
        "token": "t456",
        "user": "u123",
        "title": "Test Title",
        "message": "hello world",
    }


def test_success_falls_back_to_legacy_env_var_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_pushover_env(monkeypatch)
    monkeypatch.setenv("PUSHOVER_USER", "legacy-user")
    monkeypatch.setenv("PUSHOVER_TOKEN", "legacy-token")

    with patch(
        "dana.tools.notifications.urllib.request.urlopen",
        return_value=_mock_urlopen_response({"status": 1, "request": "req-1"}),
    ):
        out = send_pushover_notification("hello")

    assert out == "OK: send_pushover_notification delivered (request=req-1)"


def test_new_env_var_names_take_priority_over_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_pushover_env(monkeypatch)
    monkeypatch.setenv("PUSHOVER_USER_KEY", "new-user")
    monkeypatch.setenv("PUSHOVER_USER", "legacy-user")
    monkeypatch.setenv("PUSHOVER_API_TOKEN", "new-token")
    monkeypatch.setenv("PUSHOVER_TOKEN", "legacy-token")

    captured: dict = {}

    def fake_urlopen(req, timeout=20):  # noqa: ANN001
        import urllib.parse

        captured["body"] = dict(urllib.parse.parse_qsl(req.data.decode("utf-8")))
        return _mock_urlopen_response({"status": 1, "request": "req-2"})

    with patch(
        "dana.tools.notifications.urllib.request.urlopen", side_effect=fake_urlopen
    ):
        send_pushover_notification("hello")

    assert captured["body"]["user"] == "new-user"
    assert captured["body"]["token"] == "new-token"


def test_pushover_rejection_reported_as_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUSHOVER_USER_KEY", "u")
    monkeypatch.setenv("PUSHOVER_API_TOKEN", "t")

    with patch(
        "dana.tools.notifications.urllib.request.urlopen",
        return_value=_mock_urlopen_response(
            {"status": 0, "errors": ["invalid user key"]}
        ),
    ):
        out = send_pushover_notification("hello")

    assert out == "ERROR: send_pushover_notification rejected: invalid user key"


def test_http_error_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUSHOVER_USER_KEY", "u")
    monkeypatch.setenv("PUSHOVER_API_TOKEN", "t")

    err = urllib.error.HTTPError(
        url="https://api.pushover.net/1/messages.json",
        code=401,
        msg="Unauthorized",
        hdrs=None,
        fp=None,
    )
    with patch(
        "dana.tools.notifications.urllib.request.urlopen", side_effect=err
    ):
        out = send_pushover_notification("hello")

    assert out.startswith("ERROR: send_pushover_notification HTTP 401")


def test_network_error_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUSHOVER_USER_KEY", "u")
    monkeypatch.setenv("PUSHOVER_API_TOKEN", "t")

    with patch(
        "dana.tools.notifications.urllib.request.urlopen",
        side_effect=urllib.error.URLError("Connection refused"),
    ):
        out = send_pushover_notification("hello")

    assert out.startswith("ERROR: send_pushover_notification network error")


def test_send_notification_tool_entry_point_delegates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PUSHOVER_USER_KEY", "u")
    monkeypatch.setenv("PUSHOVER_API_TOKEN", "t")

    with patch(
        "dana.tools.notifications.urllib.request.urlopen",
        return_value=_mock_urlopen_response({"status": 1, "request": "req-3"}),
    ):
        out = send_notification("background task finished")

    assert out == "OK: send_pushover_notification delivered (request=req-3)"


def test_registered_in_tool_registry_with_required_param() -> None:
    from dana.tools.registry import get_tool_registry

    entry = get_tool_registry(reload=True).get("send_notification")
    assert entry is not None
    param_names = {(p.name, p.required) for p in entry.spec.parameters}
    assert ("message", True) in param_names


def test_default_args_for_forced_send_notification() -> None:
    from dana.agentic_react_graph import _default_args_for_forced_tool

    args = _default_args_for_forced_tool("send_notification", "let me know when it's done")
    assert args == {"message": "let me know when it's done"}
