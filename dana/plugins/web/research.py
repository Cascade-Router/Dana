"""Web research tools backing Dana's "web_tools" capability domain (see
dana.core.react_dispatch's _WEB_TOOLS_TOOL_IDS/_CAPABILITY_TOOL_IDS) —
search_web, read_webpage.

Naming note: dana/tools/tools.json already defines "web_search" and
"fetch_webpage" — a DIFFERENT, live subsystem's tools (dana/web_search.py +
dana/tools/browser.py, dispatched by dana.core.agent_loop's own separate
regex/voice broker, not dana.core.react_dispatch). That fetch_webpage uses
headless-Chromium Playwright; this module deliberately uses a lighter
httpx + BeautifulSoup approach instead, per this task's requirements — a
genuinely different implementation, not a drop-in replacement. Reusing
those exact ids would have silently rewritten the schema the OTHER live
subsystem's LLM calls are shown (dana.tools.schema.load_tool_registry keys
tools by id with last-write-wins, no error on a duplicate), while that
subsystem's own handlers kept running the old Playwright code underneath
— a real mismatch between what a model is told and what actually runs.
So this domain uses distinct ids (search_web, read_webpage) instead.

Both DDGS and httpx are imported at MODULE level (not lazily inside the
functions, unlike dana/web_search.py's pattern) specifically so tests can
monkeypatch/mock them directly — see tests/plugins/web/test_research.py.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

try:
    from ddgs import DDGS
except ImportError:  # pragma: no cover — ddgs is a pinned requirement; this is a graceful floor
    DDGS = None  # type: ignore[assignment,misc]

try:
    import httpx
except ImportError:  # pragma: no cover — httpx is a pinned requirement; this is a graceful floor
    httpx = None  # type: ignore[assignment]

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover — beautifulsoup4 is a pinned requirement; this is a graceful floor
    BeautifulSoup = None  # type: ignore[assignment,misc]

_SEARCH_TIMEOUT_S = 10
_FETCH_TIMEOUT_S = 10.0
_MAX_RESULTS_CAP = 20
_MAX_CONTENT_CHARS = 10_000
_USER_AGENT = "Mozilla/5.0 (compatible; DanaResearchBot/1.0; +https://github.com/)"

# Stripped before text extraction — script/style are never readable content;
# the rest are heavy boilerplate/non-text elements that just add noise.
_STRIP_TAGS = ("script", "style", "noscript", "template", "svg", "iframe", "nav", "footer", "header")


def search_web(query: str, max_results: int = 5) -> dict[str, Any]:
    """DuckDuckGo (keyless) text search, timeout-bounded. Read-only.

    Returns ``{"ok": True, "query": ..., "results": [{"title", "href",
    "body"}, ...]}`` on success, ``{"ok": False, "error": ...}`` on any
    failure (no results, network/DNS failure, timeout, missing dependency)
    — never raises.
    """
    q = (query or "").strip()
    if not q:
        return {"ok": False, "error": "query must not be empty"}
    if DDGS is None:
        return {"ok": False, "error": "ddgs package not installed"}

    n = max(1, min(int(max_results or 5), _MAX_RESULTS_CAP))

    try:
        with DDGS(timeout=_SEARCH_TIMEOUT_S) as ddgs:
            raw = list(ddgs.text(q, max_results=n))
    except Exception as exc:  # noqa: BLE001 — network/DNS/timeout/parse errors all land here
        return {"ok": False, "error": str(exc)}

    results: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        href = str(item.get("href") or item.get("link") or "").strip()
        body = str(item.get("body") or item.get("snippet") or "").strip()
        if not (title or body):
            continue
        results.append({"title": title, "href": href, "body": body})

    if not results:
        return {"ok": False, "error": "no results", "query": q}
    return {"ok": True, "query": q, "results": results}


def read_webpage(url: str) -> dict[str, Any]:
    """Fetches ``url`` and extracts clean, readable text (script/style/nav/
    footer stripped), truncated to ``_MAX_CONTENT_CHARS``. Timeout-bounded.
    Read-only.

    Returns ``{"ok": True, "url", "content", "truncated", "length"}`` on
    success, ``{"ok": False, "error": ...}`` on any failure (bad URL,
    timeout, DNS failure, non-2xx status, unparseable content) — never
    raises.
    """
    target = (url or "").strip()
    if not target:
        return {"ok": False, "error": "url must not be empty"}
    parsed = urlparse(target)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return {"ok": False, "error": f"url must be a fully-qualified http(s) URL, got: {url!r}"}
    if httpx is None:
        return {"ok": False, "error": "httpx package not installed"}
    if BeautifulSoup is None:
        return {"ok": False, "error": "beautifulsoup4 package not installed"}

    try:
        response = httpx.get(
            target,
            timeout=_FETCH_TIMEOUT_S,
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT},
        )
        response.raise_for_status()
    except httpx.TimeoutException:
        return {"ok": False, "error": f"request to {url!r} timed out after {_FETCH_TIMEOUT_S}s"}
    except httpx.HTTPStatusError as exc:
        return {"ok": False, "error": f"HTTP {exc.response.status_code} fetching {url!r}"}
    except httpx.HTTPError as exc:
        # Covers DNS failures, connection refused, malformed responses, etc.
        return {"ok": False, "error": f"could not fetch {url!r}: {exc}"}

    try:
        soup = BeautifulSoup(response.text, "html.parser")
    except Exception as exc:  # noqa: BLE001 — malformed HTML must never crash the tool
        return {"ok": False, "error": f"could not parse page content: {exc}"}

    for tag in soup(_STRIP_TAGS):
        tag.decompose()

    raw_text = soup.get_text(separator="\n")
    lines = (line.strip() for line in raw_text.splitlines())
    cleaned = "\n".join(line for line in lines if line)

    truncated = len(cleaned) > _MAX_CONTENT_CHARS
    content = cleaned[:_MAX_CONTENT_CHARS]

    return {"ok": True, "url": target, "content": content, "truncated": truncated, "length": len(cleaned)}


__all__ = ("search_web", "read_webpage")
