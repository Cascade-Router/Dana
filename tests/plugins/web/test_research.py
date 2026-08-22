"""Tests for dana.plugins.web.research — the real "web_tools" capability
domain (dana.core.react_dispatch's _WEB_TOOLS_TOOL_IDS): search_web,
read_webpage. Every network call (ddgs.DDGS, httpx.get) is mocked — these
tests never touch the real internet.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from dana.plugins.web import research


# --------------------------------------------------------------------------
# search_web
# --------------------------------------------------------------------------


class _FakeDDGS:
    """Stands in for ddgs.DDGS — a context manager whose .text() returns
    canned hits or raises, exactly like the real thing would on a network
    failure/timeout."""

    def __init__(self, hits: list[dict[str, Any]] | None = None, raises: Exception | None = None) -> None:
        self._hits = hits or []
        self._raises = raises
        self.received_kwargs: dict[str, Any] = {}

    def __enter__(self) -> "_FakeDDGS":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def text(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        self.received_kwargs = {"query": query, **kwargs}
        if self._raises is not None:
            raise self._raises
        return self._hits


def _mock_ddgs(monkeypatch: pytest.MonkeyPatch, fake: _FakeDDGS) -> None:
    monkeypatch.setattr(research, "DDGS", lambda **_kwargs: fake)


def test_search_web_rejects_empty_query() -> None:
    result = research.search_web("")
    assert result["ok"] is False
    assert "empty" in result["error"]


def test_search_web_reports_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(research, "DDGS", None)
    result = research.search_web("anything")
    assert result["ok"] is False
    assert "ddgs" in result["error"]


def test_search_web_success_returns_structured_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeDDGS(
        hits=[
            {"title": "Result One", "href": "https://example.com/1", "body": "First snippet"},
            {"title": "Result Two", "href": "https://example.com/2", "body": "Second snippet"},
        ]
    )
    _mock_ddgs(monkeypatch, fake)

    result = research.search_web("dana ai assistant")

    assert result["ok"] is True
    assert result["query"] == "dana ai assistant"
    assert result["results"] == [
        {"title": "Result One", "href": "https://example.com/1", "body": "First snippet"},
        {"title": "Result Two", "href": "https://example.com/2", "body": "Second snippet"},
    ]


def test_search_web_no_results_reports_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_ddgs(monkeypatch, _FakeDDGS(hits=[]))
    result = research.search_web("something with truly no hits")
    assert result["ok"] is False
    assert "no results" in result["error"]


def test_search_web_timeout_or_network_failure_reports_clean_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_ddgs(monkeypatch, _FakeDDGS(raises=TimeoutError("search timed out")))
    result = research.search_web("anything")
    assert result["ok"] is False
    assert "timed out" in result["error"]


def test_search_web_max_results_is_capped_and_passed_through(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeDDGS(hits=[{"title": "x", "href": "https://x", "body": "y"}])
    _mock_ddgs(monkeypatch, fake)

    research.search_web("query", max_results=500)

    assert fake.received_kwargs["max_results"] == research._MAX_RESULTS_CAP


def test_search_web_default_max_results_is_five(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeDDGS(hits=[{"title": "x", "href": "https://x", "body": "y"}])
    _mock_ddgs(monkeypatch, fake)

    research.search_web("query")

    assert fake.received_kwargs["max_results"] == 5


# --------------------------------------------------------------------------
# read_webpage
# --------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, text: str = "", status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://example.com")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError(f"HTTP {self.status_code}", request=request, response=response)


def _mock_httpx_get(monkeypatch: pytest.MonkeyPatch, *, response: _FakeResponse | None = None, raises: Exception | None = None):
    def fake_get(url: str, **kwargs: Any):
        if raises is not None:
            raise raises
        return response

    monkeypatch.setattr(research.httpx, "get", fake_get)


def test_read_webpage_rejects_empty_url() -> None:
    result = research.read_webpage("")
    assert result["ok"] is False
    assert "empty" in result["error"]


def test_read_webpage_rejects_non_http_scheme() -> None:
    result = research.read_webpage("ftp://example.com/file")
    assert result["ok"] is False
    assert "http" in result["error"]


def test_read_webpage_rejects_url_missing_host() -> None:
    result = research.read_webpage("http://")
    assert result["ok"] is False


def test_read_webpage_reports_missing_httpx_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(research, "httpx", None)
    result = research.read_webpage("https://example.com")
    assert result["ok"] is False
    assert "httpx" in result["error"]


def test_read_webpage_reports_missing_bs4_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_httpx_get(monkeypatch, response=_FakeResponse("<html></html>"))
    monkeypatch.setattr(research, "BeautifulSoup", None)
    result = research.read_webpage("https://example.com")
    assert result["ok"] is False
    assert "beautifulsoup4" in result["error"]


def test_read_webpage_timeout_reports_clean_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_httpx_get(monkeypatch, raises=httpx.TimeoutException("timed out"))
    result = research.read_webpage("https://slow.example.com")
    assert result["ok"] is False
    assert "timed out" in result["error"]


def test_read_webpage_dns_or_connection_failure_reports_clean_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_httpx_get(monkeypatch, raises=httpx.ConnectError("name resolution failed"))
    result = research.read_webpage("https://does-not-resolve.invalid")
    assert result["ok"] is False
    assert "could not fetch" in result["error"]


def test_read_webpage_non_2xx_status_reports_clean_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_httpx_get(monkeypatch, response=_FakeResponse("not found", status_code=404))
    result = research.read_webpage("https://example.com/missing")
    assert result["ok"] is False
    assert "404" in result["error"]


def test_read_webpage_strips_script_and_style_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    html = """
    <html>
      <head><style>body { color: red; }</style></head>
      <body>
        <script>alert('should not appear');</script>
        <h1>Real Title</h1>
        <p>Real paragraph content.</p>
      </body>
    </html>
    """
    _mock_httpx_get(monkeypatch, response=_FakeResponse(html))

    result = research.read_webpage("https://example.com/article")

    assert result["ok"] is True
    assert "Real Title" in result["content"]
    assert "Real paragraph content." in result["content"]
    assert "should not appear" not in result["content"]
    assert "color: red" not in result["content"]


def test_read_webpage_truncates_long_content(monkeypatch: pytest.MonkeyPatch) -> None:
    long_paragraph = "word " * 5000  # well over _MAX_CONTENT_CHARS
    html = f"<html><body><p>{long_paragraph}</p></body></html>"
    _mock_httpx_get(monkeypatch, response=_FakeResponse(html))

    result = research.read_webpage("https://example.com/long")

    assert result["ok"] is True
    assert result["truncated"] is True
    assert len(result["content"]) == research._MAX_CONTENT_CHARS
    assert result["length"] > research._MAX_CONTENT_CHARS


def test_read_webpage_short_content_is_not_truncated(monkeypatch: pytest.MonkeyPatch) -> None:
    html = "<html><body><p>Short page.</p></body></html>"
    _mock_httpx_get(monkeypatch, response=_FakeResponse(html))

    result = research.read_webpage("https://example.com/short")

    assert result["ok"] is True
    assert result["truncated"] is False
    assert result["content"] == "Short page."


# --------------------------------------------------------------------------
# Registry / routing wiring — not mutating, correctly registered.
# --------------------------------------------------------------------------


def test_search_web_and_read_webpage_are_not_mutating() -> None:
    import dana.core.react_dispatch as rd

    assert rd.is_mutating_tool("search_web") is False
    assert rd.is_mutating_tool("read_webpage") is False


def test_search_web_and_read_webpage_registered_in_web_tools_domain() -> None:
    import dana.core.react_dispatch as rd

    for tool_id in ("search_web", "read_webpage"):
        assert tool_id in rd.TOOL_HANDLERS, tool_id
        assert tool_id in rd._WEB_TOOLS_TOOL_IDS, tool_id
    assert rd._CAPABILITY_TOOL_IDS["web_tools"] == rd._WEB_TOOLS_TOOL_IDS


def test_dispatch_tool_call_search_web_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    import dana.core.react_dispatch as rd
    from dana.tools.schema import ToolCall

    _mock_ddgs(monkeypatch, _FakeDDGS(hits=[{"title": "Hit", "href": "https://x", "body": "y"}]))
    call = ToolCall(tool_id="search_web", arguments={"query": "test query"})
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is True
    assert result.payload["results"][0]["title"] == "Hit"


def test_dispatch_tool_call_read_webpage_failure_is_digested_not_crashed(monkeypatch: pytest.MonkeyPatch) -> None:
    import dana.core.react_dispatch as rd
    from dana.tools.schema import ToolCall

    _mock_httpx_get(monkeypatch, raises=httpx.TimeoutException("timed out"))
    call = ToolCall(tool_id="read_webpage", arguments={"url": "https://slow.example.com"})
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "timed out" in result.payload.get("raw_error", "")
