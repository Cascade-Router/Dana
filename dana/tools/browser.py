"""Headless Chromium webpage fetch actuator for the LangGraph ReAct agent."""

from __future__ import annotations

from urllib.parse import urlparse


def fetch_webpage(url: str) -> str:
    """Fetch visible page text from a fully qualified http(s) URL via headless Chromium.

    ReAct agent contract
    --------------------
    - ``url`` **must** be a fully qualified ``http://`` or ``https://`` URL
      (scheme + host required). Relative paths, ``www.`` bare hosts, and
      ``file:`` / other schemes are rejected with an ``ERROR:`` observation.
    - Returns the page body's visible text (``inner_text``) as the Observation
      for the LLM — not HTML source.
    - Navigation timeouts and launch failures return a clean ``ERROR:`` string
      suitable for ReAct self-correction; this actuator does not raise.

    Returns
    -------
    Page body text, or an ``ERROR:`` observation string.
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

    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        return f"ERROR: fetch_webpage failed: playwright not installed ({exc})"

    try:
        with sync_playwright() as p:
            with p.chromium.launch(headless=True) as browser:
                page = browser.new_page()
                page.goto(raw, wait_until="domcontentloaded")
                text = page.locator("body").inner_text()
                return (text or "").strip()
    except PlaywrightTimeoutError:
        return f"ERROR: fetch_webpage navigation timed out for {raw}"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: fetch_webpage failed: {exc}"
