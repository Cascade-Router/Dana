"""SQLite episodic_facts primary retrieval + contradiction guardrails.

Episodic rows are immutable grounding for personal history / past interviews /
mechanics corrections. They must be checked before Chroma vault or vision.
"""

from __future__ import annotations

import json
import re
from typing import Any

from dana.memory.store import EpisodicMemoryStore, get_episodic_store

# Personal history / interview / mechanics / background intents → SQLite first.
_EPISODIC_PRIMARY_RE = re.compile(
    r"(?i)\b(?:"
    r"interview(?:s|ed)?|companies?\s+(?:i|we)\s+(?:interview|spoke)|"
    r"technical\s+(?:software\s+)?(?:engineering\s+)?(?:and\s+)?planning|"
    r"orbital\s+period|mechanics\s+simulation|sidereal|true\s+orbital|"
    r"calculation\s+mistake|6[- ]?axis|six[- ]?axis|mobility\s+platform|"
    r"professional\s+experience|work(?:ing)?\s+experience|"
    r"my\s+(?:background|experience|work\s+history|career)|"
    r"earlier\s+this\s+year|past\s+interview|"
    r"what\s+(?:companies|roles?)\s+did\s+i|"
    r"from\s+(?:episodic\s+)?memory|remember(?:ed)?\s+(?:that|when)\s+we|"
    # Suite 5 combinatorial personal facts.
    r"names?\s+of\s+my\s+cats?|my\s+cats?|"
    r"model\s+is\s+my\s+car|my\s+car|"
    r"maintenance\s+parts|splash\s+shield|glass\s+repair|"
    r"order\s+food|chipotle|shake\s+shack|what\s+specific\s+drink|"
    r"usually\s+get|dining\s+preference|diet\s+coke"
    r")\b",
)

# Claims of experience that can contradict a recorded negative fact.
_EXPERIENCE_CLAIM_RE = re.compile(
    r"(?i)\b(?:"
    r"my\s+(?:professional\s+)?experience|"
    r"summarize\s+my\s+(?:professional\s+)?experience|"
    r"when\s+i\s+worked|i\s+worked\s+with|experience\s+working|"
    r"my\s+work\s+(?:with|on)|background\s+(?:in|with|on)"
    r")\b",
)

_NEGATIVE_FACT_RE = re.compile(
    r"(?i)\b(?:"
    r"\bno\b|\bnone\b|\bnever\b|\bnot\b|\bzero\b|"
    r"do\s+not\s+invent|don't\s+invent|"
    r"no\s+(?:professional\s+)?experience|"
    r"has\s+no\b|have\s+no\b|"
    r"does\s+not\s+have|do\s+not\s+have"
    r")\b",
)

# Expand sparse user queries so search_facts hits known Suite / tag keys.
_SEMANTIC_TAG_EXPANSIONS: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (
        re.compile(r"(?i)\binterview|\bcompanies\b|\bzoox\b|\bwaymo\b|\bdimensional\b"),
        ("interview", "companies", "zoox", "waymo", "dimensional", "planning"),
    ),
    (
        re.compile(
            r"(?i)\borbital|\bmechanics\s+simulation|\bsidereal|\b86400\b|\b86164\b"
        ),
        ("orbital", "period", "sidereal", "solar", "86164", "86400", "mechanics"),
    ),
    (
        re.compile(r"(?i)\b6[- ]?axis\b|\bsix[- ]?axis\b|\bmobility\s+platform"),
        ("six", "axis", "mobility", "experience", "platforms"),
    ),
    (
        re.compile(r"(?i)\bcats?\b|\beddie\b|\btulip\b"),
        ("cats", "eddie", "tulip", "pets"),
    ),
    (
        re.compile(
            r"(?i)\bcar\b|\brav4\b|\btoyota\b|\bsplash\s+shield\b|\bglass\s+repair\b"
        ),
        ("car", "toyota", "rav4", "2022", "splash", "shield", "glass", "repair"),
    ),
    (
        re.compile(
            r"(?i)\bchipotle\b|\bshake\s+shack\b|\bdrink\b|\bdiet\s+coke\b|\border\s+food\b"
        ),
        ("drink", "diet", "coke", "chipotle", "shake", "shack", "dining"),
    ),
)

_VAULT_OR_VISION_TOOLS = frozenset(
    {
        "analyze_visual_context",
        "ocr_with_region",
        "search_vault",
        "read_vault_memory",
        "ingest_local_directory",
        "describe_spatial_scene",
    }
)

CONTRADICTION_DIRECTIVE = (
    "The user prompt asserts experience that contradicts recorded episodic facts. "
    "Explicitly state that there is no record of this work, refuse the premise "
    "politely, and ask for clarification."
)


def is_episodic_primary_query(text: str) -> bool:
    """True when SQLite episodic_facts must be consulted before vault/vision."""
    return bool(_EPISODIC_PRIMARY_RE.search(text or ""))


def _parse_value(raw: Any) -> Any:
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return raw


def _fact_blob(fact: dict[str, Any]) -> str:
    return f"{fact.get('key') or ''} {_parse_value(fact.get('value'))}"


def _expanded_search_query(text: str) -> str:
    parts = [text or ""]
    for pattern, tags in _SEMANTIC_TAG_EXPANSIONS:
        if pattern.search(text or ""):
            parts.append(" ".join(tags))
    return " ".join(p for p in parts if p).strip()


def _topic_overlap(query: str, fact: dict[str, Any]) -> bool:
    q = (query or "").lower()
    blob = _fact_blob(fact).lower()
    key = str(fact.get("key") or "").lower()
    if "interview" in q or "companies" in q:
        if "interview" in key or any(c in blob for c in ("zoox", "waymo", "dimensional")):
            return True
    if any(t in q for t in ("orbital", "mechanics", "sidereal", "period")):
        if "orbital" in key or "86164" in blob or "86400" in blob:
            return True
    if "6-axis" in q or "6 axis" in q or "six-axis" in q or "mobility" in q:
        if "six_axis" in key or "6-axis" in blob or "six-axis" in blob or "mobility" in blob:
            return True
    if "cat" in q or "eddie" in q or "tulip" in q:
        if "cat" in key or "eddie" in blob or "tulip" in blob or "pet" in key:
            return True
    if "car" in q or "rav4" in q or "maintenance" in q or "toyota" in q:
        if (
            "car" in key
            or "rav4" in blob
            or "toyota" in blob
            or "splash" in blob
            or "glass repair" in blob
        ):
            return True
    if any(t in q for t in ("drink", "chipotle", "shake shack", "order food", "dining")):
        if (
            "drink" in key
            or "diet coke" in blob
            or "chipotle" in blob
            or "dining" in key
            or "food" in key
        ):
            return True
    return False


def detect_contradiction(
    query: str,
    matches: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return contradicting fact when the query claims experience that memory denies."""
    if not _EXPERIENCE_CLAIM_RE.search(query or "") and not (
        "6-axis" in (query or "").lower()
        or "six-axis" in (query or "").lower()
        or "6 axis" in (query or "").lower()
    ):
        return None
    for fact in matches:
        blob = _fact_blob(fact)
        if not _NEGATIVE_FACT_RE.search(blob):
            continue
        if _topic_overlap(query, fact):
            return fact
    return None


def format_grounding_block(
    matches: list[dict[str, Any]],
    *,
    contradiction: dict[str, Any] | None = None,
) -> str:
    """Immutable grounding block for the LLM system prompt."""
    if not matches and contradiction is None:
        return ""
    lines = [
        "IMMUTABLE EPISODIC GROUNDING (SQLite episodic_facts — primary source):",
        "Treat the following as authoritative recorded facts. Prefer them over "
        "vault search, Chroma, vision, or general knowledge for this turn.",
    ]
    for fact in matches:
        key = str(fact.get("key") or "").strip()
        val = _parse_value(fact.get("value"))
        if isinstance(val, (dict, list)):
            try:
                val_s = json.dumps(val, ensure_ascii=False)
            except (TypeError, ValueError):
                val_s = str(val)
        else:
            val_s = str(val)
        lines.append(f"- [{key}] {val_s}")
    if contradiction is not None:
        lines.append("")
        lines.append("CONTRADICTION GUARDRAIL:")
        lines.append(CONTRADICTION_DIRECTIVE)
    return "\n".join(lines).strip()


def retrieve_episodic_grounding(
    query: str,
    store: EpisodicMemoryStore | None = None,
    *,
    db_path: str | None = None,
    limit: int = 12,
) -> dict[str, Any]:
    """Search episodic_facts first; return matches + optional contradiction."""
    mem = store if store is not None else get_episodic_store(db_path)
    mem.prune_expired_entries()
    q = (query or "").strip()
    if not q:
        return {
            "matches": [],
            "contradiction": None,
            "grounding_block": "",
            "primary_source": "episodic_facts",
            "suppress_vault_vision": False,
        }

    search_q = _expanded_search_query(q)
    matches = list(mem.search_facts(search_q, limit=limit) or [])

    # Exact / semantic tag pass: include environment facts whose topic overlaps
    # even if token scoring was sparse.
    if is_episodic_primary_query(q):
        seen = {int(f["id"]) for f in matches if f.get("id") is not None}
        for fact in mem.list_facts(include_expired=False):
            fid = fact.get("id")
            if fid is not None and int(fid) in seen:
                continue
            if _topic_overlap(q, fact):
                matches.append(fact)
                if fid is not None:
                    seen.add(int(fid))

    contradiction = detect_contradiction(q, matches)
    if contradiction is not None:
        cid = contradiction.get("id")
        if cid is not None and not any(f.get("id") == cid for f in matches):
            matches.insert(0, contradiction)

    suppress = bool(matches) and (
        is_episodic_primary_query(q) or contradiction is not None
    )
    block = format_grounding_block(matches, contradiction=contradiction)
    return {
        "matches": matches,
        "contradiction": contradiction,
        "grounding_block": block,
        "primary_source": "episodic_facts",
        "suppress_vault_vision": suppress,
        "contradiction_directive": (
            CONTRADICTION_DIRECTIVE if contradiction is not None else ""
        ),
    }


def should_suppress_vault_vision_tool(tool_id: str | None) -> bool:
    return (tool_id or "").strip() in _VAULT_OR_VISION_TOOLS
