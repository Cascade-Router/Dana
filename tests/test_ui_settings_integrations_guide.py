"""dana.ui.settings — Integrations Setup guide content (Milestone 4).

Checks the guide covers the concrete steps a user needs (where to get
each credential, and the exact .env variable names the corresponding
tool/poller actually reads) rather than rendering, since the desktop UI
has no Markdown renderer to exercise here.
"""
from __future__ import annotations

from dana.ui.settings import (
    get_integrations_setup_markdown,
    get_integrations_setup_text,
)


def test_markdown_guide_covers_pushover_setup_steps() -> None:
    md = get_integrations_setup_markdown()
    assert "pushover.net" in md
    assert "User Key" in md
    assert "API Token" in md
    assert "PUSHOVER_USER_KEY" in md
    assert "PUSHOVER_API_TOKEN" in md


def test_markdown_guide_covers_telegram_setup_steps() -> None:
    md = get_integrations_setup_markdown()
    assert "@BotFather" in md
    assert "/newbot" in md
    assert "TELEGRAM_BOT_TOKEN" in md
    assert "TELEGRAM_ALLOWED_CHAT_ID" in md
    assert "Chat ID" in md or "chat id" in md.lower()


def test_env_var_names_match_the_actual_tool_and_poller() -> None:
    # The guide's instructed env var names must be the ones the real code
    # reads, or a user follows the guide and nothing works.
    from dana.middleware.telegram_poller import TelegramPoller
    from dana.tools.notifications import _pushover_credentials

    md = get_integrations_setup_markdown()
    import inspect

    assert "PUSHOVER_USER_KEY" in inspect.getsource(_pushover_credentials)
    assert "PUSHOVER_API_TOKEN" in inspect.getsource(_pushover_credentials)
    assert "TELEGRAM_BOT_TOKEN" in inspect.getsource(TelegramPoller.__post_init__)
    assert "TELEGRAM_ALLOWED_CHAT_ID" in inspect.getsource(TelegramPoller.__post_init__)
    assert md  # keep the import used


def test_plaintext_rendering_strips_markdown_markup() -> None:
    text = get_integrations_setup_text()
    assert "##" not in text
    assert "**" not in text
    assert "```" not in text
    assert "pushover.net" in text
    assert "@BotFather" in text
