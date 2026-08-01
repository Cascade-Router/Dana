"""Hermetic + live checks for the headless Playwright fetch_webpage actuator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from dana.tools.browser import fetch_webpage


def _chromium_available() -> bool:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            browser.close()
        return True
    except Exception:  # noqa: BLE001
        return False


def _mock_playwright_stack(*, inner_text: str = "", goto_side_effect: Exception | None = None):
    fake_body = MagicMock()
    fake_body.inner_text.return_value = inner_text

    fake_page = MagicMock()
    fake_page.locator.return_value = fake_body
    if goto_side_effect is not None:
        fake_page.goto.side_effect = goto_side_effect

    fake_browser = MagicMock()
    fake_browser.new_page.return_value = fake_page
    fake_browser.__enter__ = MagicMock(return_value=fake_browser)
    fake_browser.__exit__ = MagicMock(return_value=False)

    fake_chromium = MagicMock()
    fake_chromium.launch.return_value = fake_browser

    fake_p = MagicMock()
    fake_p.chromium = fake_chromium

    fake_cm = MagicMock()
    fake_cm.__enter__ = MagicMock(return_value=fake_p)
    fake_cm.__exit__ = MagicMock(return_value=False)
    return fake_cm, fake_page


def test_fetch_webpage_rejects_empty_and_relative() -> None:
    assert fetch_webpage("").startswith("ERROR:")
    assert fetch_webpage("   ").startswith("ERROR:")
    assert fetch_webpage("example.com").startswith("ERROR:")
    assert fetch_webpage("/relative/path").startswith("ERROR:")


def test_fetch_webpage_hermetic_mock() -> None:
    """Offline CI path: mock Playwright so Chromium need not be installed."""
    fake_cm, fake_page = _mock_playwright_stack(
        inner_text="Example Domain\nThis domain is for use in documentation."
    )

    with patch("playwright.sync_api.sync_playwright", return_value=fake_cm):
        out = fetch_webpage("https://example.com")

    assert "Example Domain" in out
    fake_page.goto.assert_called_with(
        "https://example.com", wait_until="domcontentloaded"
    )
    fake_page.locator.assert_called_with("body")


def test_fetch_webpage_navigation_timeout() -> None:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

    fake_cm, _fake_page = _mock_playwright_stack(
        goto_side_effect=PlaywrightTimeoutError("Timeout 30000ms exceeded")
    )

    with patch("playwright.sync_api.sync_playwright", return_value=fake_cm):
        out = fetch_webpage("https://example.com/slow")

    assert out.startswith("ERROR:")
    assert "timed out" in out.lower()


@pytest.mark.skipif(
    not _chromium_available(),
    reason="Live fetch requires Playwright Chromium installed",
)
def test_fetch_webpage_example_domain_live() -> None:
    out = fetch_webpage("https://example.com")
    assert "Example Domain" in out
