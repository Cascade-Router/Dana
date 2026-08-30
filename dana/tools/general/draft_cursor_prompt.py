"""Permanent general tool: append async improvement tickets to the patch ledger.

USE THIS TOOL EXCLUSIVELY FOR ALL SYSTEM MODIFICATIONS, BUG FIXES,
ARCHITECTURAL TICKETS, AND SELF-IMPROVEMENT REQUESTS. This is the only
authorized tool for logging code changes.

Writes structured tickets for the human to execute in Cursor.
Does **not** mutate Dana's live core memory or apply patches in-process.

Ledger path (repo-rooted): ``CAMGRASPER/docs/patch_ledger.md``
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

_SUCCESS_MSG = "Ticket added to patch_ledger.md. Status is PENDING."
_LEDGER_HEADER = "# Dana Patch Ledger\n"
_WRITE_ERROR_MSG = "Error writing to ledger"
_REJECT_AMBIGUOUS_MSG = (
    "Error: Draft rejected. Context must include concrete target files or symbols "
    "plus step-by-step logic / acceptance criteria (not an intent-echo)."
)
_MIN_CONTEXT_CHARS = 120


def _context_is_sufficiently_specific(context: str) -> bool:
    """Accept expanded TPM tickets; reject empty, short, or intent-echo-only bodies."""
    ctx = (context or "").strip()
    if len(ctx) <= _MIN_CONTEXT_CHARS:
        return False

    # Intent-echo: only Technical intent + Target Files lines (no real body).
    substantive = [
        ln.strip()
        for ln in ctx.splitlines()
        if ln.strip()
        and not re.match(r"(?i)^\*{0,2}\s*Technical intent:", ln.strip())
        and not re.match(r"(?i)^\*{0,2}\s*Target Files:", ln.strip())
    ]
    if not substantive:
        return False

    body = "\n".join(substantive)
    # Reject when the only "detail" restates the technical intent phrase.
    intent_m = re.search(
        r"(?im)^\s*\*{0,2}\s*Technical intent:\*{0,2}\s*(.+?)\s*$",
        ctx,
    )
    if intent_m:
        intent = re.sub(r"\s+", " ", intent_m.group(1)).strip().lower()
        body_l = re.sub(r"\s+", " ", body).strip().lower()
        if intent and (body_l == intent or body_l in intent or intent in body_l):
            if len(body_l) < len(intent) + 40 and not re.search(
                r"(?i)\b(root\s+cause|step[- ]?by[- ]?step|acceptance\s+criteria)\b",
                body,
            ):
                return False

    has_path_or_symbol = bool(
        re.search(
            r"(?i)\b(?:dana/|[\w.-]+\.(?:py|md|json|txt|yml|yaml)\b|[A-Za-z_][\w]*\()",
            ctx,
        )
    )
    has_steps = bool(
        re.search(
            r"(?is)\b(?:root\s+cause|step[- ]?by[- ]?step|acceptance\s+criteria|"
            r"refactoring steps|\d+\.\s+\w+)",
            ctx,
        )
    )
    return bool(has_path_or_symbol and has_steps)


class DraftCursorPromptArgs(BaseModel):
    """Strict argument schema for ``draft_cursor_prompt`` (LLM + runtime).

    TECHNICAL PRODUCT MANAGER RULE: When the user gives a high-level or casual
    voice command for a code change, you must act as a Technical Product Manager.
    Translate their vague request into a highly detailed technical prompt for the
    Cursor IDE. If the user does not provide file paths, use your reasoning to
    outline clear architectural goals, logic steps, and acceptance criteria in the
    ``context`` argument. Do not ask the user for more details—expand their intent
    into a usable developer ticket.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    objective: str = Field(
        ...,
        min_length=1,
        description=(
            "Summarize the goal in ONE sentence. MUST NOT be a copy-paste of "
            "the user's prompt — rewrite as a concise technical objective."
        ),
    )
    context: str = Field(
        ...,
        min_length=1,
        description=(
            "TECHNICAL PRODUCT MANAGER RULE: When the user gives a high-level or "
            "casual voice command for a code change, you must act as a Technical "
            "Product Manager. Translate their vague request into a highly detailed "
            "technical prompt for the Cursor IDE. If the user does not provide file "
            "paths, use your reasoning to outline clear architectural goals, logic "
            "steps, and acceptance criteria in the context argument. Do not ask the "
            "user for more details—expand their intent into a usable developer ticket."
        ),
    )


def _ledger_path() -> Path:
    """Resolve ``docs/patch_ledger.md`` under the CAMGRASPER project root."""
    try:
        from dana.paths import PATCH_LEDGER_PATH, PROJECT_ROOT

        path = Path(PATCH_LEDGER_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        root = Path(PROJECT_ROOT).resolve()
        try:
            path.resolve().relative_to(root)
        except ValueError:
            return root / "docs" / "patch_ledger.md"
        return path
    except Exception:  # noqa: BLE001
        return Path(__file__).resolve().parents[3] / "docs" / "patch_ledger.md"


def _new_ticket_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"ticket_{stamp}_{uuid.uuid4().hex[:8]}"


def _ticket_title(objective: str) -> str:
    """Generate a concise title from the objective string."""
    text = re.sub(r"\s+", " ", (objective or "").strip())
    if not text:
        return "Untitled improvement"
    text = re.split(r"[.!?\n]", text, maxsplit=1)[0].strip() or text
    if len(text) > 72:
        text = text[:69].rstrip() + "..."
    return text


def _format_ticket(
    objective: str,
    context: str,
    *,
    ticket_id: str,
    target_files: str = "",
) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    title = _ticket_title(objective)
    obj = (objective or "").strip() or "_No objective provided._"
    ctx = (context or "").strip() or (
        "_No architectural context provided. Reject and re-draft with root cause, "
        "affected symbols, and step-by-step logic changes._"
    )
    targets = (target_files or "").strip()
    if not targets:
        # Pull from enriched context when present.
        m = re.search(
            r"(?im)^\s*\*\*Target Files:\*\*\s*(.+)$",
            ctx,
        )
        if m:
            targets = m.group(1).strip()
    if not targets:
        targets = "*(Cursor to determine based on context)*"

    lines = [
        "---",
        f"### Ticket: {title}",
        f"**ID:** `{ticket_id}`",
        "**Status:** `[PENDING]`",
        f"**Date Drafted:** {stamp}",
        f"**Objective:** {obj}",
        f"**Target Files:** {targets}",
        "**Context & Instructions:** ",
        ctx,
        "",
        "**Security & Guardrails:** Keep diffs minimal. Do not modify offline routing constraints or ToolForge security gates.",
        "**Cursor Receipt:** ",
        "*(Awaiting compilation...)*",
    ]
    return "\n" + "\n".join(lines) + "\n"


def _append_ticket_to_ledger(ticket: str) -> Path:
    """Create ``docs/patch_ledger.md`` + header if needed, then append ``ticket``."""
    dest = _ledger_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        try:
            from dana.tools.task_queue import shadow_backup_before_write

            shadow_backup_before_write(dest)
        except Exception:  # noqa: BLE001
            pass
    if not dest.is_file() or dest.stat().st_size == 0:
        dest.write_text(_LEDGER_HEADER + "\n", encoding="utf-8")
    with dest.open("a", encoding="utf-8") as fh:
        fh.write(ticket)
    return dest


def draft_cursor_prompt(
    objective: str,
    context: str = "",
    **kwargs: object,
) -> str:
    """USE THIS TOOL EXCLUSIVELY FOR ALL SYSTEM MODIFICATIONS, BUG FIXES, ARCHITECTURAL TICKETS, AND SELF-IMPROVEMENT REQUESTS. This is the only authorized tool for logging code changes.

    Appends a PENDING ticket to ``docs/patch_ledger.md``.

    TECHNICAL PRODUCT MANAGER RULE: When the user gives a high-level or casual
    voice command for a code change, expand intent into a detailed Cursor ticket
    (architectural goals, logic steps, acceptance criteria) in ``context``.

    Args:
        objective: One-sentence technical goal. MUST NOT copy-paste the user prompt.
        context: Expanded developer ticket body (>50 chars). File paths optional.

    Returns:
        A short status string for Dana TTS / MoA. Never raises into the
        parent agent runtime graph.
    """
    # Ignore legacy target_files if a stale caller still passes it — schema forbids it.
    kwargs.pop("target_files", None)
    if kwargs:
        return (
            "ERROR: draft_cursor_prompt only accepts objective and context "
            "(unknown arguments rejected)."
        )

    try:
        args = DraftCursorPromptArgs(
            objective=objective or "",
            context=context or "",
        )
    except ValidationError:
        return (
            "ERROR: draft_cursor_prompt requires objective (one-sentence summary) "
            "and context (deep architectural step-by-step logic)."
        )

    # Voice sanitizer + topic→file auto-map (concrete targets / refactor steps).
    target_files = ""
    try:
        from dana.agentic import enrich_draft_cursor_args

        enriched = enrich_draft_cursor_args(
            raw_text="",
            objective=args.objective,
            context=args.context,
        )
        args = DraftCursorPromptArgs(
            objective=enriched["objective"],
            context=enriched["context"],
        )
        target_files = str(enriched.get("target_files") or "")
    except Exception:  # noqa: BLE001
        pass

    # Module 4 strict guard — truncated / intent-echo / missing structure.
    try:
        from dana.tools.guards import DraftCursorTicketPayload, format_validation_bounce

        DraftCursorTicketPayload.model_validate(
            {"objective": args.objective, "context": args.context}
        )
    except ValidationError as exc:
        return f"ERROR: {format_validation_bounce(exc)}"

    if not _context_is_sufficiently_specific(args.context):
        return _REJECT_AMBIGUOUS_MSG

    ticket_id = _new_ticket_id()
    ticket = _format_ticket(
        args.objective,
        args.context,
        ticket_id=ticket_id,
        target_files=target_files,
    )

    try:
        dest = _append_ticket_to_ledger(ticket)
    except Exception as exc:  # noqa: BLE001 — never crash the ReAct loop
        try:
            from dana.logging import log

            log(
                "Ledger",
                f"ERROR writing ticket id={ticket_id}: "
                f"{type(exc).__name__}: {exc}",
            )
        except Exception:  # noqa: BLE001
            print(
                f"[Ledger] ERROR writing ticket id={ticket_id}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
        return f"{_WRITE_ERROR_MSG}: {type(exc).__name__}: {exc}"

    try:
        from dana.logging import log

        log("Ledger", f"Ticket written ID={ticket_id} path={dest}")
    except Exception:  # noqa: BLE001
        pass
    return f"{_SUCCESS_MSG} ID={ticket_id} path={dest}"


# Disk loader binds by module stem name (`draft_cursor_prompt`).
