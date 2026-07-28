"""Stage 8.9.3 — Pre-filled GitHub issue escalation (local browser only)."""

from __future__ import annotations

import os
import webbrowser
from typing import Any
from urllib.parse import urlencode


def github_issues_new_base_url() -> str:
    """Base ``…/issues/new`` URL for the configured repo.

    Prefer ``GITHUB_REPO_OWNER`` / ``GITHUB_REPO_NAME`` from the environment.
    Otherwise use the placeholder host path — replace YOUR_USERNAME/YOUR_REPO
    before relying on this in production.
    """
    owner = (os.environ.get("GITHUB_REPO_OWNER") or "").strip()
    repo = (os.environ.get("GITHUB_REPO_NAME") or "").strip()
    if not owner or not repo:
        # TODO: replace YOUR_USERNAME/YOUR_REPO with your GitHub org/repo slug.
        owner = owner or "YOUR_USERNAME"
        repo = repo or "YOUR_REPO"
    return f"https://github.com/{owner}/{repo}/issues/new"


def build_github_issue_url(
    ticket_content: dict[str, Any] | str | None,
    jason_critique: str,
    *,
    title: str = "Agent Failure: Task Escalation",
) -> str:
    """Return a pre-filled GitHub new-issue URL (does not open the browser)."""
    if isinstance(ticket_content, dict):
        objective = str(ticket_content.get("objective") or "").strip()
        context = str(ticket_content.get("context") or "").strip()
        tool = str(ticket_content.get("tool") or "draft_cursor_prompt").strip()
        denials = ticket_content.get("consecutive_denials")
        ticket_block = (
            f"**Tool:** `{tool}`\n\n"
            f"**Objective:**\n{objective or '(empty)'}\n\n"
            f"**Context:**\n{context or '(empty)'}\n"
        )
        if denials is not None:
            ticket_block += f"\n**Consecutive HITL denials:** {denials}\n"
    else:
        ticket_block = str(ticket_content or "(no ticket content)")

    critique = str(jason_critique or "").strip() or "(no Jason critique)"
    body = (
        "## Escalation\n"
        "Donna HITL denied this drafted ticket repeatedly. "
        "Please review and advise.\n\n"
        "## Jason Review\n"
        f"{critique}\n\n"
        "## Drafted Ticket\n"
        f"{ticket_block}\n"
    )
    query = urlencode({"title": title, "body": body})
    return f"{github_issues_new_base_url()}?{query}"


def open_github_issue(
    ticket_content: dict[str, Any] | str | None,
    jason_critique: str,
    *,
    title: str = "Agent Failure: Task Escalation",
) -> str:
    """Open the default browser to a pre-filled GitHub issue. Returns the URL."""
    full_url = build_github_issue_url(
        ticket_content, jason_critique, title=title
    )
    try:
        webbrowser.open(full_url)
    except Exception:  # noqa: BLE001
        pass
    try:
        from dana.logging import log

        log("GitHubEscalation", f"opened issue draft url_chars={len(full_url)}")
    except Exception:  # noqa: BLE001
        pass
    return full_url
