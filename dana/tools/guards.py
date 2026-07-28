"""Module 4 — Pydantic tool guards + localized ValidationError bounce prompts."""

from __future__ import annotations

import re
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

# Truncation / intent-echo heuristics (shared with draft_cursor ledger gate).
_INTENT_ECHO_RE = re.compile(
    r"(?is)^\s*\*{0,2}\s*Technical intent:\*{0,2}\s*.+?\n\s*\*{0,2}\s*Target Files:",
)


class DraftCursorTicketPayload(BaseModel):
    """Strict payload for ``draft_cursor_prompt`` (raises ``ValidationError``)."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    objective: str = Field(..., min_length=8)
    context: str = Field(..., min_length=120)

    @field_validator("objective")
    @classmethod
    def _objective_not_truncated(cls, value: str) -> str:
        text = (value or "").strip()
        if not text:
            raise ValueError("objective must be a non-empty string")
        # Forbid mid-word ellipsis truncation (e.g. "enhancing cursor pro...")
        if re.search(r"[A-Za-z0-9]\.\.\.\s*$", text):
            raise ValueError(
                "objective looks mid-word truncated (ends with '...'); "
                "provide a complete sentence"
            )
        return text

    @model_validator(mode="after")
    def _context_must_be_structured(self) -> DraftCursorTicketPayload:
        ctx = self.context or ""
        if _INTENT_ECHO_RE.match(ctx) and not re.search(
            r"(?is)\b(root\s+cause|step[- ]?by[- ]?step|acceptance\s+criteria)\b",
            ctx,
        ):
            raise ValueError(
                "context is an intent-echo payload; require root cause, "
                "step-by-step changes, and acceptance criteria"
            )
        has_targets = bool(
            re.search(r"(?is)\btarget\s+files?\b", ctx)
            and re.search(r"(?i)\b[\w./\\-]+\.(?:py|md|json|txt|yml|yaml)\b", ctx)
        )
        has_root = bool(re.search(r"(?is)\broot\s+cause\b", ctx))
        has_steps = bool(
            re.search(
                r"(?is)\bstep[- ]?by[- ]?step(?:\s+changes?)?\b|\bsteps?\s*:",
                ctx,
            )
        )
        has_accept = bool(re.search(r"(?is)\bacceptance\s+criteria\b", ctx))
        missing: list[str] = []
        if not has_targets:
            missing.append("target_files")
        if not has_root:
            missing.append("root_cause")
        if not has_steps:
            missing.append("step_by_step")
        if not has_accept:
            missing.append("acceptance_criteria")
        if missing:
            raise ValueError(
                "context missing required sections: " + ", ".join(missing)
            )
        return self


class GenericToolPayload(BaseModel):
    """Loose-but-strict guard: every provided string arg must be non-empty."""

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="before")
    @classmethod
    def _no_empty_strings(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        for key, value in list(data.items()):
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                raise ValueError(f"'{key}' field missing or empty")
        return data


_TOOL_MODELS: dict[str, type[BaseModel]] = {
    "draft_cursor_prompt": DraftCursorTicketPayload,
}


def guard_tool_call(tool_id: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
    """Validate tool arguments with the registered Pydantic model (raises ValidationError)."""
    tid = (tool_id or "").strip()
    args = dict(arguments or {})
    model = _TOOL_MODELS.get(tid)
    if model is None:
        # Universal minimum: no empty strings on provided keys.
        GenericToolPayload.model_validate(args)
        return args
    validated = model.model_validate(args)
    return validated.model_dump()


def format_validation_bounce(exc: ValidationError | Exception) -> str:
    """Concise bounce prompt for the generating agent (no supervisor LLM)."""
    if isinstance(exc, ValidationError):
        parts: list[str] = []
        for err in exc.errors():
            loc = ".".join(str(x) for x in (err.get("loc") or ()))
            msg = str(err.get("msg") or "invalid")
            if loc:
                parts.append(f"'{loc}' {msg}")
            else:
                parts.append(msg)
        detail = "; ".join(parts) if parts else str(exc)
    else:
        detail = str(exc)
    detail = re.sub(r"\s+", " ", detail).strip()
    if len(detail) > 280:
        detail = detail[:277].rstrip() + "..."
    return (
        f"Validation Error: {detail}. "
        "Fix the tool arguments and retry the same tool once with complete fields."
    )
