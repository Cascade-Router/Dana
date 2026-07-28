"""Stage 8.9.3 — consecutive HITL denials → GitHub escalate button / URL."""

from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import pytest

from dana.middleware import hitl_ticket as hitl
from dana.ui.github_escalation import (
    build_github_issue_url,
    github_issues_new_base_url,
    open_github_issue,
)


@pytest.fixture(autouse=True)
def _reset_denials() -> None:
    hitl.reset_consecutive_denials(reason="test")
    hitl.clear_pending()
    # Clear active fingerprint via begin on empty then reset again.
    hitl.begin_ticket_hitl({"objective": "", "context": ""})
    hitl.reset_consecutive_denials(reason="test-fp")
    yield
    hitl.reset_consecutive_denials(reason="teardown")
    hitl.clear_pending()


def test_denials_increment_and_reset_on_approve() -> None:
    hitl.publish_pending(
        {"objective": "O1", "context": "C1", "jason_critique": "x"},
        thread_id="t",
    )
    assert hitl.get_consecutive_denials() == 0
    hitl.submit_decision(False, action="deny")
    assert hitl.get_consecutive_denials() == 1
    hitl.clear_pending()
    hitl.publish_pending(
        {"objective": "O1", "context": "C1", "jason_critique": "x"},
        thread_id="t",
    )
    # Same task fingerprint — counter preserved.
    assert hitl.get_consecutive_denials() == 1
    hitl.submit_decision(False, action="deny")
    assert hitl.get_consecutive_denials() == 2
    hitl.clear_pending()
    hitl.publish_pending(
        {"objective": "O1", "context": "C1", "jason_critique": "x"},
        thread_id="t",
    )
    pending = hitl.get_pending() or {}
    assert int(pending.get("consecutive_denials") or 0) >= 2
    assert "ESCALATION" in hitl.format_ticket_payload(pending)
    hitl.submit_decision(True, action="approve")
    assert hitl.get_consecutive_denials() == 0


def test_new_distinct_task_resets_denials() -> None:
    hitl.publish_pending({"objective": "A", "context": "1"}, thread_id="t")
    hitl.submit_decision(False, action="deny")
    hitl.clear_pending()
    hitl.publish_pending({"objective": "A", "context": "1"}, thread_id="t")
    hitl.submit_decision(False, action="deny")
    assert hitl.get_consecutive_denials() == 2
    hitl.clear_pending()
    # New distinct ticket → reset.
    hitl.publish_pending({"objective": "B", "context": "2"}, thread_id="t")
    assert hitl.get_consecutive_denials() == 0


def test_github_issue_url_encodes_critique_and_ticket(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_REPO_OWNER", "acme")
    monkeypatch.setenv("GITHUB_REPO_NAME", "dana")
    base = github_issues_new_base_url()
    assert base == "https://github.com/acme/dana/issues/new"
    url = build_github_issue_url(
        {
            "objective": "Fix OCR bounds",
            "context": "Need region boxes",
            "tool": "draft_cursor_prompt",
            "consecutive_denials": 2,
        },
        "Missing visual bounds — deny.",
    )
    parsed = urlparse(url)
    assert parsed.netloc == "github.com"
    qs = parse_qs(parsed.query)
    assert qs["title"][0] == "Agent Failure: Task Escalation"
    body = qs["body"][0]
    assert "Missing visual bounds" in body
    assert "Fix OCR bounds" in body
    assert "Consecutive HITL denials" in body


def test_open_github_issue_calls_webbrowser(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []
    monkeypatch.setenv("GITHUB_REPO_OWNER", "acme")
    monkeypatch.setenv("GITHUB_REPO_NAME", "dana")
    monkeypatch.setattr(
        "dana.ui.github_escalation.webbrowser.open",
        lambda url: opened.append(url) or True,
    )
    url = open_github_issue({"objective": "O", "context": "C"}, "Critique here")
    assert opened and opened[0] == url
    from urllib.parse import unquote_plus

    assert "Critique here" in unquote_plus(url)


def test_hitl_bar_shows_github_when_denials_ge_2() -> None:
    import customtkinter as ctk

    from dana.schema import TraceEvent
    from dana.ui.trace_window import LiveTracePanel

    hitl.reset_consecutive_denials()
    # Seed two denials on a stable fingerprint.
    hitl.publish_pending({"objective": "Z", "context": "z"}, thread_id="g")
    hitl.submit_decision(False, action="deny")
    hitl.clear_pending()
    hitl.publish_pending({"objective": "Z", "context": "z"}, thread_id="g")
    hitl.submit_decision(False, action="deny")
    hitl.clear_pending()
    assert hitl.get_consecutive_denials() == 2

    hitl.publish_pending(
        {
            "objective": "Z",
            "context": "z",
            "jason_critique": "Still missing bounds.",
            "consecutive_denials": 2,
        },
        thread_id="g",
    )

    root = ctk.CTk()
    root.withdraw()
    panel = LiveTracePanel(root)
    root.update_idletasks()
    panel._handle_event(
        TraceEvent(
            event_type="status",
            node="ticket_approval",
            message="HITL_PENDING_APPROVAL",
            payload="pending",
        )
    )
    root.update_idletasks()
    assert panel._hitl_visible is True
    assert panel._hitl_github_visible is True
    assert panel._hitl_github_btn.pack_info()
    root.destroy()
    hitl.clear_pending()
