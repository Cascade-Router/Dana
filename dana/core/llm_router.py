"""Dynamic LLM Router — config-driven fleet selection + fallback chain.

Reads the optional fleet defined in ``routing_config.yaml`` (see
``dana.core.routing_config``) and, given a turn's ``messages``/``tools``,
returns an ordered chain of candidate models to try. Entirely additive:
when no routing config is present, ``resolve_chain`` returns ``None`` and
``dana.core.model_provider`` falls back to its existing ``.env``-toggle
resolution, unchanged.

Token estimate is a plain char-count heuristic (``len(text) // 4``), not a
real tokenizer — exact tokenization differs per vendor anyway, so
provider-exactness is unachievable regardless, and this matches the
codebase's existing char-based conservatism (``dana.core.context_manager``'s
own truncation thresholds) rather than adding a new dependency for
approximate numbers.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from dana.core.routing_config import cloud_allowed, load_routing_config

_DEFAULT_SAFETY_MARGIN = 0.85

# dana.core.openai_tool_bridge's HTTPError branch raises
# RuntimeError(f"cloud HTTP {exc.code}: ...") — no structured status code is
# threaded through, so report_fleet_entry_failure below recovers it from the
# message text. 0 (not None) when a failure has no HTTP status at all (a
# TimeoutError, a resolve_openai_endpoint config error, ...) since the
# registry's status_code field is typed as a plain int for the frontend.
_HTTP_STATUS_RE = re.compile(r"HTTP (\d{3})")


@dataclass(frozen=True)
class FleetEntry:
    id: str
    provider: str
    model: str
    context_window: int
    priority: int
    api_key_env: str | None = None
    cost_per_1m_prompt: float = 0.0
    cost_per_1m_completion: float = 0.0
    tpm_limit: int | None = None


def _parse_fleet(raw: dict[str, Any]) -> tuple[list[FleetEntry], dict[str, Any]]:
    entries: list[FleetEntry] = []
    for raw_entry in raw.get("fleet") or []:
        entries.append(
            FleetEntry(
                id=str(raw_entry["id"]),
                provider=str(raw_entry["provider"]).strip().lower(),
                model=str(raw_entry["model"]),
                context_window=int(raw_entry["context_window"]),
                priority=int(raw_entry["priority"]),
                api_key_env=(str(raw_entry["api_key_env"]) if raw_entry.get("api_key_env") else None),
                cost_per_1m_prompt=float(raw_entry.get("cost_per_1m_prompt") or 0.0),
                cost_per_1m_completion=float(raw_entry.get("cost_per_1m_completion") or 0.0),
                tpm_limit=(int(raw_entry["tpm_limit"]) if raw_entry.get("tpm_limit") else None),
            )
        )
    defaults = raw.get("defaults") or {}
    return entries, defaults


def estimate_tokens(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> int:
    """Char-count / 4 heuristic across ``messages`` + ``tools`` — see this
    module's own docstring for why this isn't a real tokenizer."""
    chars = 0
    for message in messages or []:
        content = message.get("content")
        if isinstance(content, str):
            chars += len(content)
        elif isinstance(content, list):
            chars += len(json.dumps(content))
        tool_calls = message.get("tool_calls")
        if tool_calls:
            chars += len(json.dumps(tool_calls))
    if tools:
        chars += len(json.dumps(tools))
    return chars // 4


def select_fleet_chain(
    entries: list[FleetEntry],
    defaults: dict[str, Any],
    *,
    estimated_tokens: int,
    prefer_id: str | None = None,
) -> list[FleetEntry]:
    """Order ``entries`` into the chain to try, cheapest-fitting-first.

    Entries whose ``context_window * safety_margin`` fits ``estimated_tokens``
    are tried first, ordered by (priority, cost_per_1m_prompt). Entries that
    don't fit are appended afterward, ordered by largest context_window
    first — still attempted as a last resort (never return an empty chain)
    rather than giving up before making a single request. The configured
    ``terminal_fallback`` entry (if any) is guaranteed to be the final entry
    in the chain regardless of fit, so there's always a last resort —
    UNLESS that same entry is also ``prefer_id`` (Admin/Introspection
    Local-First Preference, see ``_turn_is_admin_only`` below), in which
    case it keeps whatever position its own ``(priority, cost)`` sort gave
    it instead of being pushed back to last: ``prefer_id`` sorts ahead of
    every entry's own configured ``priority`` for this one call only,
    without mutating ``FleetEntry.priority`` itself, so a config-driven
    fleet order stays the single source of truth for every other turn.
    """
    margin = float(defaults.get("context_window_safety_margin") or _DEFAULT_SAFETY_MARGIN)
    fitting = [e for e in entries if e.context_window * margin >= estimated_tokens]
    overflowing = [e for e in entries if e not in fitting]

    fitting.sort(key=lambda e: (e.id != prefer_id, e.priority, e.cost_per_1m_prompt))
    overflowing.sort(key=lambda e: -e.context_window)

    chain = fitting + overflowing

    terminal_id = str(defaults.get("terminal_fallback") or "").strip()
    if terminal_id and terminal_id != prefer_id:
        terminal = next((e for e in entries if e.id == terminal_id), None)
        if terminal is not None:
            chain = [e for e in chain if e.id != terminal_id] + [terminal]
    return chain


# CAD Geometry Precision Gate: create_freecad_*/perform_freecad_* are NOT
# in dana.core.react_dispatch._CORE_TOOL_IDS — they're only offered once a
# CAD capability domain is actually unlocked for THIS turn, unlike
# create_plan/mark_task_completed (always-core, offered on literally every
# turn regardless of anything — see that set's own comments). That means
# geometry tools are a genuine per-turn signal a fleet selector can act on,
# where create_plan's own presence isn't (excluding Ollama whenever
# create_plan is offered would exclude it from every turn ever, permanently
# — see report_ollama_tool_call_failure below for how create_plan's own
# failure mode is instead handled reactively, after the fact, rather than
# by pre-filtering on a signal that's always true).
_GEOMETRY_TOOL_PREFIXES = ("create_freecad_", "perform_freecad_")

# Admin/Introspection Local-First Preference: hand-maintained mirror of
# dana.core.react_dispatch._CORE_TOOL_IDS (NOT imported from there —
# llm_router is deliberately import-light and lazily loaded by
# model_provider specifically to dodge dana.core's own partial-init race,
# see that call site's comment; pulling in react_dispatch, a ~7000-line
# module with its own heavy import graph, would defeat that). Keep in sync
# by hand if react_dispatch's set changes.
#
# NOT schema-presence-based (checking whether these are the ONLY tools
# OFFERED this turn) — that was this feature's first cut, and it was dead
# on arrival: dana.core.react_dispatch._FREECAD_ESSENTIAL_TOOL_IDS gets
# pre-seeded into `tools` for the WHOLE session the instant a CAD-shaped
# request is detected (or the CAD tab is open), including several
# create_freecad_*/perform_freecad_* names. That means a pure
# search_tool_catalog/mark_task_completed turn in a CAD session has those
# geometry tools sitting in its `tools` schema right alongside the admin
# ones regardless of what's actually about to happen — so "every offered
# tool is admin-only" was never true after turn one, and worse,
# _turn_needs_geometry_precision below (also schema-presence-based, but
# checking the OPPOSITE thing) was ALSO true on every one of those same
# turns, hard-excluding Ollama before this preference ever got a chance to
# apply it. Confirmed live: dana_performance.log session 7a326b94 (and the
# follow-up b1b5f61f) show 100% nemotron-3-super-120b-a12b:free across
# search_tool_catalog/system_state/mark_task_completed calls in a CAD
# session, ending in the same empty-completion crash both times.
#
# Fixed by keying off STATE instead of the offered schema: what tool did
# the immediately-preceding assistant turn actually dispatch (see
# _last_dispatched_tool_names below), not what's merely unlocked/allowed.
# That's a genuine per-turn signal — a session can have FreeCAD unlocked
# for its whole lifetime and still spend individual turns on nothing but
# catalog lookups and bookkeeping in between real geometry calls.
#
# create_plan is deliberately part of this mirrored set (it's core, always
# offered) but is NEVER actually the live risk here: the one turn it could
# plausibly be dispatched — no plan yet — is already hard-routed to cloud
# before this module is ever reached (is_planning_phase / Planning-Phase
# Cloud Lock in react_dispatch._call_llm_once); every turn that reaches
# resolve_chain at all already has a plan, so a model offered create_plan
# here has no real reason to call it again. On the off chance it tries
# anyway against local Ollama, the create_plan Schema-Adherence Gate in
# ModelProvider._complete_with_tool_calls_via_router below already detects
# that exact failure shape and advances to the next (cloud) chain entry —
# this preference reorders the chain, it never shortens it, so that same
# safety net still applies.
_ADMIN_ONLY_TOOL_IDS = frozenset(
    {
        "take_canvas_screenshot",
        "system_state",
        "check_plugin_registry",
        "read_system_architecture",
        "load_capability",
        "unload_capability",
        "search_tool_catalog",
        "load_specific_tool",
        "update_core_memory",
        "save_new_skill",
        "delete_skill",
        "read_skill_source",
        "create_plan",
        "mark_task_completed",
        "insert_task",  # Dynamic FSM Replanning — hand-mirrored here same as create_plan/
        # mark_task_completed above; pure plan bookkeeping, no geometry schema risk.
        "cancel_pending_task",
        "cancel_active_task",  # FSM Recovery — same reasoning as insert_task/cancel_pending_task
        # directly above: pure plan bookkeeping, no geometry schema risk.
        "compile_plan_as_skill",
    }
)


def _last_dispatched_tool_names(messages: list[dict[str, Any]]) -> list[str]:
    """Name(s) of the tool call(s) the immediately-preceding assistant turn
    made, found by walking back from the end of ``messages`` past any
    ``role: "tool"`` RESULT messages (OpenAI wire shape puts one or more of
    these right after the assistant message that triggered them — same
    shape ``dana.core.react_dispatch``'s own tool_result_msg construction
    always produces, so this never has to guess at a different format)
    until the assistant message with ``tool_calls`` that produced them is
    found. Returns ``[]`` for a fresh chain (this turn's most recent
    message is a plain user/system turn, not a tool result) — there's no
    "last dispatched tool" to key off yet, so callers correctly treat that
    as "unknown", not "admin"."""
    for message in reversed(messages or []):
        role = message.get("role") if isinstance(message, dict) else None
        if role == "tool":
            continue
        if role == "assistant":
            names: list[str] = []
            for call in message.get("tool_calls") or []:
                fn = call.get("function") if isinstance(call, dict) else None
                name = fn.get("name") if isinstance(fn, dict) else None
                if isinstance(name, str):
                    names.append(name)
            return names
        break
    return []


def _turn_is_admin_only(messages: list[dict[str, Any]]) -> bool:
    """True when the immediately-preceding assistant turn dispatched ONLY
    admin/introspection tool calls — see this module's own state-vs-schema
    comment above for why this checks history instead of ``tools``. A fresh
    chain (nothing dispatched yet this turn) is NOT admin-only: there's no
    evidence either way, so this falls through to normal priority
    ordering rather than guessing."""
    names = _last_dispatched_tool_names(messages)
    return bool(names) and all(n in _ADMIN_ONLY_TOOL_IDS for n in names)


def _turn_needs_geometry_precision(tools: list[dict[str, Any]] | None) -> bool:
    """True when ``tools`` (this turn's offered OpenAI-shape function
    schema) includes at least one CAD geometry tool. Confirmed live
    (dana_runtime.log, a LiDAR-mounting-assembly session) that a 7B local
    model reliably fails to produce a valid nested tool-call schema for
    ``create_plan``; there's no reason to expect a geometry tool's own
    schema (equally nested — placement tuples, enum-typed operations,
    array-of-name parameters) to fare any better, so local Ollama is
    excluded from the fleet chain entirely for these turns rather than
    only detected-and-retried after already failing once.
    """
    for tool in tools or []:
        name = (tool.get("function") or {}).get("name") if isinstance(tool, dict) else None
        if isinstance(name, str) and name.startswith(_GEOMETRY_TOOL_PREFIXES):
            return True
    return False


def resolve_chain(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> list[FleetEntry] | None:
    """Return the ordered fallback chain for this turn, or ``None`` when no
    (valid) ``routing_config.yaml`` exists — the signal callers use to fall
    back to legacy ``.env``-toggle resolution unchanged."""
    raw = load_routing_config()
    if raw is None:
        return None
    entries, defaults = _parse_fleet(raw)
    if not entries:
        return None

    # Full-Local Override (``defaults.allow_cloud: false``) — checked
    # before, and short-circuiting, every other signal below (planning
    # phase, geometry precision, admin-only): when the operator has
    # explicitly opted out of cloud entirely, none of those signals should
    # get a vote, because every one of them exists ONLY to pick which
    # provider handles a turn, and there is no longer a choice to make.
    # Returns local entries UNORDERED-BY-FIT (no context-window/token-count
    # filtering — a full-local policy means "try local regardless", not
    # "try local unless it looks too big") and WITHOUT any cloud entry
    # appended as a fallback, per that policy's whole point: a turn that
    # exceeds local Ollama's real context or that Ollama can't complete
    # fails outright rather than silently escaping to cloud. If the fleet
    # happens to define no Ollama entry at all (not this project's shipped
    # routing_config.yaml, but a hand-edited one could), there's nothing
    # local to return — falls through to the normal signals below rather
    # than returning an empty chain, since ``resolve_chain``'s only two
    # states callers distinguish are "a chain" and "falsy -> legacy path",
    # and a misconfigured allow_cloud:false should not silently reroute to
    # the legacy path's own cloud default either.
    #
    # NOTE this alone does not cover Turn 0 (create_plan, no plan yet):
    # that turn is forced to "openrouter" by literal string in
    # dana.core.react_dispatch._call_llm_once's Planning-Phase Cloud Lock
    # BEFORE resolve_chain is ever reached — resolve_chain has no
    # visibility into is_planning_phase at all, so it cannot short-circuit
    # a check it never sees. That lock now calls
    # dana.core.routing_config.cloud_allowed() itself and skips the
    # override when it's False — see that function's own docstring for why
    # it's a standalone helper rather than routed through this module.
    local_entries = [e for e in entries if e.provider == "ollama"]
    if local_entries and not cloud_allowed():
        return sorted(local_entries, key=lambda e: e.priority)

    # Admin/Introspection Local-First Preference — only meaningful when the
    # configured terminal_fallback is itself an Ollama entry (the whole
    # point is routing bookkeeping/discovery turns to a LOCAL model instead
    # of burning shared free-tier cloud quota on them); a fleet with no
    # Ollama terminal_fallback at all has nothing local to prefer.
    terminal_id = str(defaults.get("terminal_fallback") or "").strip()
    terminal_entry = next((e for e in entries if e.id == terminal_id), None)
    admin_only = (
        terminal_entry is not None and terminal_entry.provider == "ollama" and _turn_is_admin_only(messages)
    )

    # Deliberate precedence: admin_only (state-based — what the model
    # actually JUST did) is checked and can win BEFORE the geometry gate
    # (schema-based — what's merely allowed this turn) gets a veto. Once
    # dana.core.react_dispatch._FREECAD_ESSENTIAL_TOOL_IDS is pre-seeded for
    # a session, _turn_needs_geometry_precision(tools) is true on nearly
    # every turn for the rest of that session regardless of what's actually
    # happening — if it ran unconditionally first, it would silently negate
    # this preference for the exact CAD-session turns it exists to help
    # (see _ADMIN_ONLY_TOOL_IDS' own comment for the confirmed-live
    # incident). Positive state evidence ("the last thing that happened was
    # a catalog lookup") is a more specific signal than "a geometry tool
    # happens to be in this turn's allow-list" for classifying what THIS
    # turn is likely to do. The residual risk — Ollama gets tried first on
    # a turn we called admin_only, and the model pivots to a geometry call
    # anyway — is bounded and self-correcting: that one local attempt can
    # fail its schema same as always, the chain still falls through to
    # cloud on that exception, and the VERY NEXT turn's own
    # _last_dispatched_tool_names would then see a geometry tool as the
    # last-dispatched call, so the hard exclusion below applies correctly
    # from that point on.
    if admin_only:
        prefer_id = terminal_id
    else:
        prefer_id = None
        if _turn_needs_geometry_precision(tools):
            entries = [e for e in entries if e.provider != "ollama"]
            if not entries:
                return None

    estimated = estimate_tokens(messages, tools)
    return select_fleet_chain(entries, defaults, estimated_tokens=estimated, prefer_id=prefer_id)


def report_fleet_entry_failure(entry: FleetEntry, exc: Exception) -> None:
    """Feed a fleet entry's live failure (429/404/503/timeout/... caught while
    ``_complete_with_tool_calls_via_router`` walks the fallback chain) into
    the Model Registry's circuit-breaker state, so the control panel shows
    exactly why a model most recently failed rather than just its static
    key-presence check. Never raises — telemetry must not break the fallback
    loop that's already in the middle of recovering from a real failure.
    """
    try:
        from dana.core.model_registry import get_registry_service

        match = _HTTP_STATUS_RE.search(str(exc))
        status_code = int(match.group(1)) if match else 0
        get_registry_service().report_runtime_error(entry.model, status_code, str(exc))
    except Exception:  # noqa: BLE001 — see docstring
        pass


def api_key_for(entry: FleetEntry, *, fallback_key: str) -> str:
    """Resolve ``entry``'s API key from its configured env var, falling back
    to whatever ``_resolve_openai_endpoint`` would have picked for
    ``entry.provider`` on its own (so an entry can omit ``api_key_env`` and
    still work via the provider's normal default env var)."""
    if entry.api_key_env:
        key = (os.environ.get(entry.api_key_env) or "").strip()
        if key:
            return key
    return fallback_key


__all__ = (
    "FleetEntry",
    "estimate_tokens",
    "select_fleet_chain",
    "resolve_chain",
    "api_key_for",
    "report_fleet_entry_failure",
)
