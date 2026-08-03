"""Headless Chromium webpage fetch actuator for the LangGraph ReAct agent."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse


def _scrub_stale_playwright_browsers_path() -> None:
    """Drop ``PLAYWRIGHT_BROWSERS_PATH`` when it points at an empty/incomplete install.

    Cursor/sandbox hosts sometimes point this env var at a cache directory that
    no longer contains Chromium; Playwright then fails with
    ``Executable doesn't exist`` even when a normal user install is available.
    """
    configured = (os.environ.get("PLAYWRIGHT_BROWSERS_PATH") or "").strip()
    if not configured:
        return
    root = Path(configured)
    if not root.is_dir():
        os.environ.pop("PLAYWRIGHT_BROWSERS_PATH", None)
        return
    has_browser = any(root.glob("chromium-*/**/chrome.exe")) or any(
        root.glob("chromium_headless_shell-*/**/chrome-headless-shell.exe")
    )
    if not has_browser:
        os.environ.pop("PLAYWRIGHT_BROWSERS_PATH", None)


def fetch_webpage(
    url: str,
    *,
    selector: str | None = None,
    limit: int | None = None,
    extract_hn_titles: bool = False,
) -> str:
    """Fetch visible page text from a fully qualified http(s) URL via headless Chromium.

    ReAct agent contract
    --------------------
    - ``url`` **must** be a fully qualified ``http://`` or ``https://`` URL
      (scheme + host required). Relative paths, ``www.`` bare hosts, and
      ``file:`` / other schemes are rejected with an ``ERROR:`` observation.
    - Returns the page body's visible text (``inner_text``) as the Observation
      for the LLM — not HTML source.
    - Optional ``selector`` extracts matching elements' text (one per line).
      ``extract_hn_titles=True`` is a convenience for Hacker News story titles
      (``.titleline > a``). ``limit`` caps how many matches are returned.
    - Navigation timeouts and launch failures return a clean ``ERROR:`` string
      suitable for ReAct self-correction; this actuator does not raise.

    Returns
    -------
    Page body text, selector match lines, or an ``ERROR:`` observation string.
    """
    raw = (url or "").strip()
    if not raw:
        return "ERROR: empty url"

    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return (
            "ERROR: url must be a fully qualified http:// or https:// URL "
            f"(got {raw!r})"
        )

    sel = (selector or "").strip()
    if extract_hn_titles and not sel:
        sel = ".titleline > a"

    max_items: int | None = None
    if limit is not None:
        try:
            max_items = max(1, int(limit))
        except (TypeError, ValueError):
            max_items = None
    elif extract_hn_titles:
        max_items = 3

    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        return f"ERROR: fetch_webpage failed: playwright not installed ({exc})"

    _scrub_stale_playwright_browsers_path()

    try:
        with sync_playwright() as p:
            with p.chromium.launch(headless=True) as browser:
                page = browser.new_page()
                page.goto(raw, wait_until="domcontentloaded")
                if sel:
                    texts = [
                        (t or "").strip()
                        for t in page.locator(sel).all_inner_texts()
                        if (t or "").strip()
                    ]
                    if max_items is not None:
                        texts = texts[:max_items]
                    if not texts:
                        return f"ERROR: no elements matched selector {sel!r}"
                    return "\n".join(texts)
                text = page.locator("body").inner_text()
                return (text or "").strip()
    except PlaywrightTimeoutError:
        return f"ERROR: fetch_webpage navigation timed out for {raw}"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: fetch_webpage failed: {exc}"
