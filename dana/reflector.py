"""Execution Critic — distill ReAct failures into durable behavioral lessons."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

LESSONS_DOMAIN = "lessons_learned"
LESSONS_VAULT_KEY = "lessons_learned"
MAX_LESSONS = 32

REFLECTION_PROMPT = """You are Donna's Execution Critic (offline meta-learning pass).
Analyze the failed agentic turn and distill EXACTLY ONE concise behavioral rule
that would prevent repeating this mistake.

Output format (strict):
Rule: <one sentence imperative rule>

Constraints:
- Reference the tool id when relevant.
- No stack traces, no apologies, no multi-rule lists.
- Prefer concrete argument hygiene / validation advice.
""".strip()


@dataclass
class Lesson:
    rule: str
    tool_id: str = ""
    domain: str = LESSONS_DOMAIN
    error_signature: str = ""
    last_updated: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "tool_id": self.tool_id,
            "domain": self.domain,
            "error_signature": self.error_signature,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Lesson:
        return cls(
            rule=str(data.get("rule") or "").strip(),
            tool_id=str(data.get("tool_id") or "").strip(),
            domain=str(data.get("domain") or LESSONS_DOMAIN),
            error_signature=str(data.get("error_signature") or "").strip(),
            last_updated=float(data.get("last_updated") or time.time()),
        )


@dataclass
class ReflectionResult:
    lesson: Lesson | None
    latency_ms: float
    triggered: bool
    raw_critique: str = ""
    persisted: bool = False
    error: str = ""


def trace_has_failure(tool_trace: list[dict[str, Any]]) -> bool:
    """True when the ReAct trace contains ERROR observations or LLM failures."""
    for row in tool_trace or []:
        if row.get("error"):
            return True
        obs = str(row.get("observation") or "")
        # Soft memory misses are not agent crashes.
        if re.search(r"(?i)memory key not found", obs):
            continue
        if obs.upper().startswith("ERROR") or " failed:" in obs.lower():
            return True
    return False


def _error_rows(tool_trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in tool_trace or []:
        if row.get("error"):
            rows.append(row)
            continue
        obs = str(row.get("observation") or "")
        if re.search(r"(?i)memory key not found", obs):
            continue
        if obs.upper().startswith("ERROR") or " failed:" in obs.lower():
            rows.append(row)
    return rows


def build_reflection_user_payload(
    *,
    user_text: str,
    tool_trace: list[dict[str, Any]],
) -> str:
    failures = _error_rows(tool_trace)
    payload = {
        "user_intent": user_text,
        "failures": [
            {
                "tool": r.get("tool") or r.get("error"),
                "args": r.get("args"),
                "observation": r.get("observation") or r.get("error"),
            }
            for r in failures
        ],
        "full_trace": tool_trace,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


_RULE_RE = re.compile(r"(?im)^\s*Rule\s*:\s*(.+?)\s*$")


def extract_rule(critique: str) -> str | None:
    raw = (critique or "").strip()
    if not raw:
        return None
    m = _RULE_RE.search(raw)
    if m:
        rule = m.group(1).strip().strip('"').strip("'")
        if rule and not rule.lower().startswith("rule:"):
            return f"Rule: {rule}" if not rule.lower().startswith("rule") else rule
        return rule if rule.lower().startswith("rule:") else f"Rule: {rule}"
    # Fallback: first non-empty line if it looks imperative.
    for line in raw.splitlines():
        line = line.strip()
        if line:
            return line if line.lower().startswith("rule:") else f"Rule: {line}"
    return None


def distill_lesson_offline(
    *,
    user_text: str,
    tool_trace: list[dict[str, Any]],
) -> Lesson:
    """Deterministic critic used when no LLM reflect_fn is provided (tests / offline)."""
    failures = _error_rows(tool_trace)
    primary = failures[-1] if failures else {}
    tool_id = str(primary.get("tool") or "unknown")
    obs = str(primary.get("observation") or primary.get("error") or "unknown error")
    args = primary.get("args") or {}
    # Heuristic distillation from common failure modes.
    obs_l = obs.lower()
    if "missing" in obs_l or "required" in obs_l:
        rule = (
            f"Rule: Before calling `{tool_id}`, ensure all required arguments "
            f"are present and non-empty (saw: {obs[:120]})."
        )
    elif "invalid" in obs_l or "type" in obs_l or "enum" in obs_l:
        rule = (
            f"Rule: When calling `{tool_id}`, coerce arguments to the schema types/"
            f"enums before dispatch (failure: {obs[:120]})."
        )
    elif "timeout" in obs_l or "max" in obs_l:
        rule = (
            f"Rule: Keep `{tool_id}` calls concise and avoid retrying the same "
            "failing call within one ReAct turn; FINAL with best effort after one error."
        )
    elif tool_id == "write_vault_memory":
        rule = (
            "Rule: When extracting values for `write_vault_memory`, strip trailing "
            "whitespace and validate the key is a non-empty identifier before writing."
        )
    else:
        arg_hint = ", ".join(f"{k}={v!r}" for k, v in list(args.items())[:3]) or "no args"
        rule = (
            f"Rule: After an ERROR from `{tool_id}` ({arg_hint}), repair arguments "
            "from the Observation before retrying; do not repeat the identical call."
        )
    return Lesson(
        rule=rule,
        tool_id=tool_id if tool_id != "unknown" else "",
        domain=LESSONS_DOMAIN,
        error_signature=obs[:200],
    )


def run_reflection(
    *,
    user_text: str,
    tool_trace: list[dict[str, Any]],
    reflect_fn: Callable[[list[dict[str, str]]], str] | None = None,
) -> ReflectionResult:
    """Secondary critique pass. Uses reflect_fn (LLM) when provided, else offline heuristic."""
    if not trace_has_failure(tool_trace):
        return ReflectionResult(lesson=None, latency_ms=0.0, triggered=False)

    t0 = time.perf_counter()
    try:
        if reflect_fn is not None:
            messages = [
                {"role": "system", "content": REFLECTION_PROMPT},
                {
                    "role": "user",
                    "content": build_reflection_user_payload(
                        user_text=user_text, tool_trace=tool_trace
                    ),
                },
            ]
            raw = (reflect_fn(messages) or "").strip()
            rule = extract_rule(raw)
            if not rule:
                lesson = distill_lesson_offline(user_text=user_text, tool_trace=tool_trace)
                raw = lesson.rule
            else:
                failures = _error_rows(tool_trace)
                tool_id = str((failures[-1] if failures else {}).get("tool") or "")
                lesson = Lesson(
                    rule=rule,
                    tool_id=tool_id,
                    error_signature=str(
                        (failures[-1] if failures else {}).get("observation") or ""
                    )[:200],
                )
        else:
            lesson = distill_lesson_offline(user_text=user_text, tool_trace=tool_trace)
            raw = lesson.rule
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return ReflectionResult(
            lesson=lesson,
            latency_ms=latency_ms,
            triggered=True,
            raw_critique=raw,
        )
    except Exception as exc:  # noqa: BLE001
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return ReflectionResult(
            lesson=None,
            latency_ms=latency_ms,
            triggered=True,
            error=str(exc),
        )


def _unwrap_lessons_blob(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, dict) and "value" in raw:
        raw = raw.get("value")
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if isinstance(raw, dict) and "rule" in raw:
        return [raw]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return _unwrap_lessons_blob(parsed)
        except json.JSONDecodeError:
            return [{"rule": raw, "domain": LESSONS_DOMAIN}]
    return []


def load_lessons(vault_client: Any) -> list[Lesson]:
    """Load lessons_learned from the vault (empty list if missing)."""
    try:
        raw = vault_client.read_memory(LESSONS_VAULT_KEY)
    except KeyError:
        return []
    except Exception:
        return []
    return [Lesson.from_dict(d) for d in _unwrap_lessons_blob(raw) if d.get("rule")]


def persist_lesson(vault_client: Any, lesson: Lesson) -> bool:
    """Append a lesson under domain=lessons_learned (capped, deduped by rule text)."""
    if not lesson or not lesson.rule:
        return False
    existing = load_lessons(vault_client)
    # Deduplicate identical rules.
    if any(e.rule.strip().lower() == lesson.rule.strip().lower() for e in existing):
        return False
    existing.append(lesson)
    existing = existing[-MAX_LESSONS:]
    blob = [e.to_dict() for e in existing]
    vault_client.write_memory(LESSONS_VAULT_KEY, blob)
    return True


def match_lessons_for_intent(
    user_text: str,
    lessons: list[Lesson],
    *,
    tool_hint: str | None = None,
) -> list[Lesson]:
    """Select lessons relevant to the current utterance / tool domain."""
    if not lessons:
        return []
    hay = (user_text or "").lower()
    matched: list[Lesson] = []
    for lesson in lessons:
        tid = (lesson.tool_id or "").lower()
        if tool_hint and tid and tid == tool_hint.lower():
            matched.append(lesson)
            continue
        if tid and tid.replace("_", " ") in hay:
            matched.append(lesson)
            continue
        if tid and tid in hay:
            matched.append(lesson)
            continue
        # Domain-wide lessons always eligible when any prior failure domain matches.
        if lesson.domain == LESSONS_DOMAIN and tid:
            # Soft match: vault / memory / type / clipboard keywords.
            keywords = {
                "write_vault_memory": ("remember", "save", "vault", "ip", "", ""),
                "read_vault_memory": ("recall", "saved", "memory", "", ""),
                "inject_keystrokes": ("type", "typing", ""),
                "read_clipboard_context": ("clipboard", "copied", ""),
                "architect_new_tool": ("create tool", "new tool", "architect", ""),
            }
            for kw in keywords.get(tid, ()):
                if kw in hay:
                    matched.append(lesson)
                    break
    # If nothing matched but we have lessons and a tool hint failed recently, return those.
    if not matched and tool_hint:
        matched = [L for L in lessons if L.tool_id == tool_hint]
    return matched[:8]


def format_lessons_block(lessons: list[Lesson]) -> str:
    if not lessons:
        return ""
    lines = ["## Lessons learned (do not repeat these mistakes)"]
    for lesson in lessons:
        lines.append(f"- {lesson.rule}")
    return "\n".join(lines)


def inject_lessons_into_prompt(system_prompt: str, lessons: list[Lesson]) -> str:
    block = format_lessons_block(lessons)
    if not block:
        return system_prompt
    return f"{block}\n\n{system_prompt}"
