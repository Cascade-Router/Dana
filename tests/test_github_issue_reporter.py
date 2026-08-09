"""Unit tests for the permanent github_issue_reporter general tool."""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch

import pytest

from dana.tools.general.github_issue_reporter import (
    _MISSING_CONFIG_MSG,
    _format_issue_body,
    report_github_issue,
)
from dana.tools.registry import get_tool_registry, load_general_tools_from_disk
from dana.tools.schema import ToolCall
from dana.tools.broker import IntentBroker


def test_format_issue_body_markdown() -> None:
    body = _format_issue_body("Steps:\n1. Open app\n2. Crash")
    assert body.startswith("## Bug report\n")
    assert "Steps:" in body
    assert "*Submitted via Dana community reporter*" in body


def test_missing_config_returns_friendly_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_REPO_OWNER", raising=False)
    monkeypatch.delenv("GITHUB_REPO_NAME", raising=False)
    with patch(
        "dana.tools.general.github_issue_reporter._load_local_config",
        return_value={},
    ):
        msg = report_github_issue("Crash on launch", "It dies immediately")
    assert msg == _MISSING_CONFIG_MSG


def test_empty_title_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_ACCESS_TOKEN", "tok")
    monkeypatch.setenv("GITHUB_REPO_OWNER", "acme")
    monkeypatch.setenv("GITHUB_REPO_NAME", "dana")
    assert report_github_issue("  ", "body").startswith("ERROR:")


def _mock_urlopen_response(payload: dict, status: int = 201) -> MagicMock:
    raw = json.dumps(payload).encode("utf-8")
    resp = MagicMock()
    resp.status = status
    resp.read.return_value = raw
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


def test_successful_201_payload_and_tts_string(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_ACCESS_TOKEN", "test-token")
    monkeypatch.setenv("GITHUB_REPO_OWNER", "camgrasper")
    monkeypatch.setenv("GITHUB_REPO_NAME", "dana")

    captured: dict = {}

    def fake_urlopen(req, timeout=20):  # noqa: ANN001
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["headers"] = {k: v for k, v in req.header_items()}
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _mock_urlopen_response(
            {
                "number": 42,
                "html_url": "https://github.com/camgrasper/dana/issues/42",
            }
        )

    with patch(
        "dana.tools.general.github_issue_reporter.urllib.request.urlopen",
        side_effect=fake_urlopen,
    ):
        msg = report_github_issue(
            title="Wake word false positive",
            body="Heard Dana when TV was on",
            labels=["bug", "voice"],
        )

    assert "Issue #42" in msg
    assert "successfully submitted" in msg
    assert "https://github.com/camgrasper/dana/issues/42" in msg
    assert captured["method"] == "POST"
    assert captured["url"].endswith("/repos/camgrasper/dana/issues")
    assert captured["body"]["title"] == "Wake word false positive"
    assert captured["body"]["body"].startswith("## Bug report\n")
    assert "Heard Dana when TV was on" in captured["body"]["body"]
    assert captured["body"]["labels"] == ["bug", "voice"]
    assert "Bearer test-token" in captured["headers"].get("Authorization", "")


def test_connection_error_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_ACCESS_TOKEN", "tok")
    monkeypatch.setenv("GITHUB_REPO_OWNER", "acme")
    monkeypatch.setenv("GITHUB_REPO_NAME", "dana")

    import urllib.error

    with patch(
        "dana.tools.general.github_issue_reporter.urllib.request.urlopen",
        side_effect=urllib.error.URLError("Connection refused"),
    ):
        msg = report_github_issue("Title", "Body")
    assert msg.startswith("ERROR: Could not reach GitHub")


def test_bad_credentials_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_ACCESS_TOKEN", "bad")
    monkeypatch.setenv("GITHUB_REPO_OWNER", "acme")
    monkeypatch.setenv("GITHUB_REPO_NAME", "dana")

    import urllib.error

    err = urllib.error.HTTPError(
        url="https://api.github.com/repos/acme/dana/issues",
        code=401,
        msg="Unauthorized",
        hdrs=None,
        fp=io.BytesIO(b'{"message":"Bad credentials"}'),
    )
    with patch(
        "dana.tools.general.github_issue_reporter.urllib.request.urlopen",
        side_effect=err,
    ):
        msg = report_github_issue("Title", "Body")
    assert "credentials" in msg.lower()
    assert msg.startswith("ERROR:")


def test_registry_loads_non_ephemeral_general_tool() -> None:
    reg = get_tool_registry(reload=True)
    loaded = load_general_tools_from_disk()
    assert "github_issue_reporter" in loaded
    entry = reg.get("github_issue_reporter")
    assert entry is not None
    assert entry.source == "general"
    assert entry.ephemeral is False
    assert entry.is_ephemeral is False
    assert entry.callable is not None
    param_names = {p.name for p in entry.spec.parameters}
    assert "title" in param_names
    assert "body" in param_names


def test_broker_dispatches_via_registry_not_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_ACCESS_TOKEN", "tok")
    monkeypatch.setenv("GITHUB_REPO_OWNER", "acme")
    monkeypatch.setenv("GITHUB_REPO_NAME", "dana")

    reg = get_tool_registry(reload=True)
    load_general_tools_from_disk()
    assert reg.get("github_issue_reporter") is not None

    with patch(
        "dana.tools.general.github_issue_reporter.urllib.request.urlopen",
        return_value=_mock_urlopen_response(
            {"number": 7, "html_url": "https://github.com/acme/dana/issues/7"}
        ),
    ):
        broker = IntentBroker()
        # Ensure broker knows the tool id from tools.json
        assert "github_issue_reporter" in broker.registry
        result = broker.dispatch(
            ToolCall(
                tool_id="github_issue_reporter",
                arguments={
                    "title": "Broker path",
                    "body": "via registry",
                    "labels": "bug",
                },
            ),
            handlers={},  # no dedicated handler / no __dynamic__
        )
    assert "Issue #7" in str(result)
