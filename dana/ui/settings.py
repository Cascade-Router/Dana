"""Integrations Setup guide — Milestone 4 (Persistent Background Autonomy).

Dānā now runs unattended as a background process and can reach the user
remotely via Pushover push notifications (``dana.tools.notifications``)
and a Telegram bot (``dana.middleware.telegram_poller``). Both need
API keys the user has to go fetch themselves; this module is the single
source of truth for the setup instructions shown in the desktop UI's
Settings tab, so the wording only needs to be maintained in one place.

The desktop UI (CustomTkinter, ``dana.core_agent.DanaGUI``) has no
Markdown renderer — ``dana.ui.chat_view`` actively strips Markdown down to
plain text for its chat bubbles, since ``CTkLabel``/``CTkTextbox`` have no
rich-text mode. So this module keeps the guide as plain Markdown text (for
anywhere that *can* render Markdown — docs, a future web/Gradio surface)
and additionally exposes a plain-text rendering for the current
Tkinter-based Settings tab, produced by stripping the same
``**bold**``/`` `code` `` markers ``chat_view`` already strips.
"""

from __future__ import annotations

import re

PUSHOVER_SETUP_MARKDOWN = """\
### Pushover (push notifications to your phone)

1. Go to **pushover.net** and create a free account, then install the \
Pushover app on your phone (iOS/Android).
2. On the Pushover dashboard, copy your **User Key** — it's shown right \
on the main page after you log in.
3. Scroll down to **"Your Applications"** and click **"Create an \
Application/API Token"**. Give it any name (e.g. "Dana"), submit, and \
copy the **API Token/Key** it generates.
4. Open this project's `.env` file (create it from `.env.example` if it \
doesn't exist yet) and add:

   ```
   PUSHOVER_USER_KEY=<your User Key>
   PUSHOVER_API_TOKEN=<your API Token>
   ```

5. Restart Dana. It can now push you a notification with the \
`send_notification` tool any time it needs to reach you.
"""

TELEGRAM_SETUP_MARKDOWN = """\
### Telegram (2-way remote chat)

1. Open Telegram and message **@BotFather**.
2. Send `/newbot` and follow the prompts (pick a name and a username \
ending in `bot`). BotFather replies with your bot's **token** — copy it.
3. Message your new bot at least once (search for its username and send \
`/start`) — Telegram won't deliver messages to a bot that's never been \
messaged.
4. Find your personal **Chat ID** so only you can command Dana: message \
**@userinfobot** (or **@getidsbot**) and it will reply with your numeric \
chat id.
5. Open this project's `.env` file and add:

   ```
   TELEGRAM_BOT_TOKEN=<the token from BotFather>
   TELEGRAM_ALLOWED_CHAT_ID=<your numeric chat id>
   ```

6. Restart Dana. Message your bot from Telegram and it's routed into Dana \
exactly like typing in the desktop window; Dana's reply comes back to \
that same chat. Messages from any other chat id are silently ignored — \
this locks the bot down to you.
"""

INTEGRATIONS_SETUP_MARKDOWN = (
    "## Integrations Setup\n\n"
    "Connect Dana to Pushover and/or Telegram so it can reach you (or take "
    "commands from you) even when you're away from the desktop.\n\n"
    f"{PUSHOVER_SETUP_MARKDOWN}\n"
    f"{TELEGRAM_SETUP_MARKDOWN}"
)

_MD_HEADING_RE = re.compile(r"^#{1,6}\s*", re.MULTILINE)
_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_MD_CODE_FENCE_RE = re.compile(r"```(?:\w+)?\n?")
_MD_INLINE_CODE_RE = re.compile(r"`([^`]+)`")


def get_integrations_setup_markdown() -> str:
    """Full Markdown guide (Pushover + Telegram setup instructions)."""
    return INTEGRATIONS_SETUP_MARKDOWN


def get_integrations_setup_text() -> str:
    """Plain-text rendering for surfaces with no Markdown renderer.

    Strips headings/bold/code-fence markup the same way
    ``dana.ui.chat_view._strip_simple_markdown`` does for chat bubbles, so
    it reads cleanly inside a plain ``CTkTextbox``.
    """
    text = INTEGRATIONS_SETUP_MARKDOWN
    text = _MD_HEADING_RE.sub("", text)
    text = _MD_BOLD_RE.sub(r"\1", text)
    text = _MD_CODE_FENCE_RE.sub("", text)
    text = _MD_INLINE_CODE_RE.sub(r"\1", text)
    return text


__all__ = (
    "PUSHOVER_SETUP_MARKDOWN",
    "TELEGRAM_SETUP_MARKDOWN",
    "INTEGRATIONS_SETUP_MARKDOWN",
    "get_integrations_setup_markdown",
    "get_integrations_setup_text",
)
