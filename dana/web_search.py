"""Lightweight web search for Donna (ddgs / DuckDuckGo — no API key)."""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any

_MAX_RESULTS = 5
_MAX_SNIPPET = 360

_CLOCK_SPECIFIC_RE = re.compile(
    r"\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)\b"
    r"|\b\d{1,2}\s*(?:AM|PM|am|pm)\b"
    r"|\b\d{1,2}:\d{2}\s*(?:ET|PT|CT|MT|UTC|GMT)\b"
    r"|\b\d{1,2}\s*(?:a\.m\.|p\.m\.)",
    re.I,
)


def _hit_dict(item: dict[str, Any]) -> dict[str, str] | None:
    title = str(item.get("title") or "").strip()
    body = str(item.get("body") or item.get("snippet") or "").strip()
    href = str(item.get("href") or item.get("link") or "").strip()
    if not (title or body):
        return None
    return {
        "title": title[:160],
        "snippet": body[:_MAX_SNIPPET],
        "url": href,
        "source": "ddgs",
    }


def _blob_has_specific_kickoff(items: list[dict[str, str]]) -> bool:
    """True only for usable spoken kickoffs (not bare 17:00–05:00 ranges)."""
    blob = " ".join(f"{x.get('title', '')} {x.get('snippet', '')}" for x in items)
    return bool(_CLOCK_SPECIFIC_RE.search(blob))


def _ddg_text(query: str, max_results: int) -> list[dict[str, str]]:
    from ddgs import DDGS

    out: list[dict[str, str]] = []
    with DDGS() as ddgs:
        raw = list(ddgs.text(query, max_results=max_results))
    for item in raw:
        if not isinstance(item, dict):
            continue
        hit = _hit_dict(item)
        if hit:
            out.append(hit)
    return out


def web_search(query: str, *, max_results: int = _MAX_RESULTS) -> dict[str, Any]:
    """Search the public web; returns structured hits for the ReAct observation."""
    q = enrich_schedule_query((query or "").strip())
    if not q:
        return {"ok": False, "error": "empty query", "results": []}
    max_results = max(1, min(int(max_results), 8))

    try:
        from ddgs import DDGS  # noqa: F401
    except ImportError as exc:
        return {
            "ok": False,
            "error": f"ddgs package not installed: {exc}",
            "query": q,
            "results": [],
        }

    try:
        results = _ddg_text(q, max_results)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "query": q, "results": []}

    if not results:
        return {"ok": False, "error": "no results", "query": q, "results": []}

    # FIFA/schedule: always try a near-term kickoff query if snippets lack AM/PM/ET.
    if re.search(r"fifa|world\s*cup|match|fixture|kickoff", q, re.I) and (
        not _blob_has_specific_kickoff(results)
    ):
        today = date.today()
        tomorrow = today + timedelta(days=1)
        day_a = today.strftime("%B %d")
        day_b = tomorrow.strftime("%B %d")
        retry_queries = [
            f'World Cup 2026 {day_b} 2026 kickoff "p.m. ET" OR "pm ET" OR "5 p.m." OR "9 p.m."',
            f'World Cup 2026 {day_a} OR {day_b} 2026 schedule kickoff times ET',
            f"FIFA World Cup 2026 semi-final quarter-final {today.strftime('%B %Y')} "
            f"kickoff ET pm",
        ]
        try:
            for retry_q in retry_queries:
                extra = _ddg_text(retry_q, max_results)
                if not extra:
                    continue
                if _blob_has_specific_kickoff(extra):
                    results = extra
                    q = retry_q
                    break
                # Merge: prefer retry titles first.
                seen = {r["title"] for r in extra}
                for r in results:
                    if r["title"] not in seen:
                        extra.append(r)
                results = extra[:max_results]
                q = retry_q
        except Exception:
            pass

    return {
        "ok": True,
        "query": q,
        "results": results[:max_results],
        "today": date.today().isoformat(),
    }


def format_search_observation(payload: dict[str, Any]) -> str:
    """Compact observation string for the LLM (not for raw TTS)."""
    if not payload.get("ok"):
        return f"ERROR: web_search failed: {payload.get('error')}"
    today = payload.get("today") or date.today().isoformat()
    try:
        from dana.settings import local_now_context

        loc = local_now_context()
        local_hint = (
            f" User local timezone={loc['timezone']} ({loc['tz_abbr']})"
            + (f", place={loc['place']}" if loc.get("place") else "")
            + "."
        )
    except Exception:
        local_hint = ""
    lines = [
        f"OK: web_search q={payload.get('query')!r} "
        f"hits={len(payload.get('results') or [])} today={today}.{local_hint}"
    ]
    for i, hit in enumerate(payload.get("results") or [], 1):
        lines.append(f"{i}. {hit.get('title', '')} — {hit.get('snippet', '')}")
    return "\n".join(lines)


def enrich_schedule_query(query: str) -> str:
    """Bias sports/schedule searches toward upcoming kickoffs (date-aware)."""
    q = (query or "").strip()
    if not q:
        return q
    if not re.search(
        r"\b(fifa|world\s*cup|match|matches|fixture|kickoff|soccer|football)\b",
        q,
        re.I,
    ):
        return q

    today = date.today()
    tomorrow = today + timedelta(days=1)
    day_a = today.strftime("%B %d")
    day_b = tomorrow.strftime("%B %d")
    # Vague STT ("FIFA match") → near-term kickoff query with ET times.
    if len(re.findall(r"[A-Za-z0-9]+", q)) <= 4:
        return (
            f"World Cup 2026 {day_a} OR {day_b} 2026 kickoff times "
            f'"p.m. ET" OR "pm ET" schedule'
        )

    if not re.search(
        r"kickoff|kick-off|what time|time of day|\bAM\b|\bPM\b|o'?clock|p\.m\.|a\.m\.",
        q,
        re.I,
    ):
        q = f"{q} kickoff time ET"
    if str(today.year) not in q:
        q = f"{q} {today.strftime('%B %d %Y')}"
    if re.search(r"\b(next|upcoming|when)\b", q, re.I) and "after" not in q.lower():
        q = f"{q} upcoming after {day_a}"
    return q
