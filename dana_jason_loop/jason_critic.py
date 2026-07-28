"""Titan critic — multi-dimensional evaluation + scarce Green Flag gate.

Internal package path remains ``dana_jason_loop`` for import stability; the
spoken / prompt codename is **Titan** (not Jason — Whisper confuses Jason with JSON).
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from dana_jason_loop.ollama_client import ask_ollama, extract_json_payload

GREEN_FLAG_THRESHOLD = 85
DIMENSIONS = ("safety", "feasibility", "resource_cost", "ecosystem_alignment")

# Deterministic tripwires — complement the LLM matrix when outlines are overtly hostile.
_HOSTILE_RE = re.compile(
    r"("
    r"reverse[_\s-]?shell|cmd\.exe|/bin/sh|socket\.connect|"
    r"commercial[-_]?c2|c2\.example|evil-c2|credential\s+harvest|"
    r"exfiltrat|meterpreter|bind[_\s-]?shell"
    r")",
    re.I,
)
_HOSTILE_DEPS = {
    "commercial-c2-sdk",
    "closed-binary-agent",
    "metasploit",
}

TITAN_SYSTEM_PROMPT = """
You are Titan, the adversarial architecture critic for an offline Windows voice agent.
You receive a batch of five capability pitches and must grade each one with clinical coldness.

Scoring matrix — assign an integer from 0 to 25 on each axis:

• safety
  Reward proposals that stay inside sandboxed local transforms of user data,
  never escape the host process, and never perform remote shell or outbound
  control-plane tricks. Punish anything that reaches into shells, sockets for
  command channels, credential harvesting, or silent host mutation.

• feasibility
  Reward proposals that fit a single Python module plus stdlib, or at most one
  small pinned pip package already realistic on this stack. Punish anything that
  demands extra foundation-model VRAM, multi-gigabyte weight downloads, exotic
  Windows driver APIs, or a second always-on inference stack.

• resource_cost
  Reward fully offline processing with zero billed tokens or SaaS seats.
  Punish metered commercial endpoints, paid cloud LLM calls, and subscription APIs.

• ecosystem_alignment
  Reward auditable open-source Python libraries the team can inspect and pin.
  Punish opaque proprietary binaries, closed SDKs, and license-hostile blobs.

Aggregate score is the sum of the four axes (max 100).

Flag policy (hard scarcity):
- A Green Flag is extraordinarily rare. Award it ONLY when a proposal's
  aggregate score is strictly greater than 85.
- Even then, at most ONE Green Flag may exist in the entire batch — give it
  solely to the single highest aggregate. Ties go to the earlier list index.
- Every other proposal must be Amber Flag (Deferred) if it is promising but
  did not win the Green slot, or Red Flag (Rejected) if it is unsafe, infeasible,
  costly, proprietary, or simply weak. If nothing exceeds 85, every item is
  Red Flag (Rejected) — never force a Green Flag.

Output EXACTLY one JSON object (no markdown, no commentary) with this shape:
{
  "evaluations": [
    {
      "proposed_id": "<id>",
      "scores": {
        "safety": <0-25>,
        "feasibility": <0-25>,
        "resource_cost": <0-25>,
        "ecosystem_alignment": <0-25>
      },
      "total": <sum>,
      "flag": "green" | "amber" | "red",
      "rationale": "one terse sentence"
    }
  ]
}
Evaluate ALL supplied proposals. Do not invent new proposed_id values.
""".strip()

# Backward-compatible alias (legacy imports / tests).
JASON_SYSTEM_PROMPT = TITAN_SYSTEM_PROMPT


def _clamp(score: Any) -> int:
    try:
        n = int(round(float(score)))
    except (TypeError, ValueError):
        n = 0
    return max(0, min(25, n))


def _proposal_is_hostile(proposal: dict[str, Any] | None) -> bool:
    if not proposal:
        return False
    blob = " ".join(
        [
            str(proposal.get("proposed_id") or ""),
            str(proposal.get("problem_statement") or ""),
            str(proposal.get("proposed_solution_code_outline") or ""),
            " ".join(str(d) for d in (proposal.get("dependencies") or [])),
        ]
    )
    if _HOSTILE_RE.search(blob):
        return True
    deps = {str(d).strip().lower() for d in (proposal.get("dependencies") or [])}
    return bool(deps & _HOSTILE_DEPS)


def _normalize_eval_row(
    row: dict[str, Any],
    proposed_id: str,
    proposal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scores_in = row.get("scores") if isinstance(row.get("scores"), dict) else {}
    scores = {dim: _clamp(scores_in.get(dim, 0)) for dim in DIMENSIONS}
    rationale = str(row.get("rationale") or "").strip() or "No rationale."
    if _proposal_is_hostile(proposal):
        # Floor hostile pitches: cannot clear Green Flag scarcity gate.
        scores["safety"] = min(scores["safety"], 2)
        scores["resource_cost"] = min(scores["resource_cost"], 5)
        scores["ecosystem_alignment"] = min(scores["ecosystem_alignment"], 5)
        rationale = (
            "Hostile control-plane / non-auditable commercial channel detected; "
            + rationale
        )
    total = sum(scores.values())
    return {
        "proposed_id": proposed_id,
        "scores": scores,
        "total": total,
        "flag": "red",  # filled by enforce_green_flag_gate
        "rationale": rationale,
        "_model_flag": str(row.get("flag") or "").strip().lower(),
    }


def enforce_green_flag_gate(evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministic post-filter: at most one Green Flag, only if total > 85."""
    if not evaluations:
        return evaluations

    # Provisional labels from totals (ignore model's flag for scarcity rules).
    eligible = [
        (i, e)
        for i, e in enumerate(evaluations)
        if int(e.get("total") or 0) > GREEN_FLAG_THRESHOLD
    ]
    winner_idx: int | None = None
    if eligible:
        winner_idx = max(
            eligible,
            key=lambda pair: (int(pair[1].get("total") or 0), -pair[0]),
        )[0]

    out: list[dict[str, Any]] = []
    for i, e in enumerate(evaluations):
        total = int(e.get("total") or 0)
        row = {
            "proposed_id": e["proposed_id"],
            "scores": dict(e["scores"]),
            "total": total,
            "rationale": e.get("rationale") or "",
        }
        if winner_idx is not None and i == winner_idx:
            row["flag"] = "green"
        elif total >= 70:
            row["flag"] = "amber"
        else:
            row["flag"] = "red"
        out.append(row)
    return out


def evaluate_proposals(
    proposals: list[dict[str, Any]],
    *,
    model: str = "llama3.2",
    temperature: float = 0.1,
) -> dict[str, Any]:
    """Score a Donna batch; return matrix + enforced flag policy."""
    if not proposals:
        return {"evaluations": [], "green_flag": None, "threshold": GREEN_FLAG_THRESHOLD}

    user_prompt = (
        "Grade this batch of capability pitches against your matrix.\n\n"
        f"PROPOSALS_JSON=\n{json.dumps(proposals, ensure_ascii=False, indent=2)}\n"
    )
    raw = ask_ollama(
        TITAN_SYSTEM_PROMPT,
        user_prompt,
        model=model,
        temperature=temperature,
    )
    payload = extract_json_payload(raw)
    if isinstance(payload, list):
        eval_rows = payload
    elif isinstance(payload, dict):
        eval_rows = payload.get("evaluations") or payload.get("results") or []
    else:
        raise ValueError("Titan reply was not a JSON object/array")

    by_id: dict[str, dict[str, Any]] = {}
    for row in eval_rows:
        if isinstance(row, dict) and row.get("proposed_id"):
            by_id[str(row["proposed_id"])] = row

    normalized: list[dict[str, Any]] = []
    for idx, prop in enumerate(proposals):
        pid = str(prop["proposed_id"])
        src = by_id.get(pid) or {}
        # Fallback: match by index position if id drift.
        if not src and len(eval_rows) == len(proposals):
            if isinstance(eval_rows[idx], dict):
                src = eval_rows[idx]
        normalized.append(_normalize_eval_row(src, pid, prop))

    evaluations = enforce_green_flag_gate(normalized)
    green = next((e for e in evaluations if e["flag"] == "green"), None)
    return {
        "evaluations": evaluations,
        "green_flag": green,
        "threshold": GREEN_FLAG_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# Watchdog supervisor — Titan reviews generated monitor scripts
# ---------------------------------------------------------------------------

TITAN_SUPERVISOR_PROMPT = """
You are Titan, the adversarial supervisor for Donna's asynchronous Watchdog agent.
You receive a monitoring task and a pure-Python script Donna assembled via the
BaseWatchdog Template Method (GeneratedWatchdog with run_self_test + monitor_loop).

Deterministic AST safety (forbidden imports, required methods) already passed.
Focus on subjective quality only:
1. Logic — the script must plausibly fulfill the stated monitoring task.
2. Validation — run_self_test must meaningfully probe assumptions (not a no-op).
3. TTS alert — monitor_loop should call self.alert(...) (or print '__DONNA_TTS__: …')
   when the watch condition is met.

If the script is acceptable, reply with EXACTLY:
APPROVED

Otherwise reply with EXACTLY:
REJECTED: <one terse sentence of reasoning>

No markdown. No extra lines.
""".strip()

WATCHDOG_SUPERVISOR_PROMPT = TITAN_SUPERVISOR_PROMPT


def static_code_safety_reject(code: str) -> str | None:
    """Deterministic AST tripwire and TTS check before Titan's LLM review."""
    blob = (code or "").strip()
    if not blob:
        return "empty script — nothing to supervise"

    # 1. Pre-flight TTS check
    if "__DONNA_TTS__" not in blob:
        return "Missing mandatory TTS alert. You must include exactly: print('__DONNA_TTS__: <message>')"

    # 2. AST parsing for hostile operations
    try:
        tree = ast.parse(blob)
    except SyntaxError as e:
        return f"Syntax error in generated code: {e}"

    forbidden_modules = {"os", "subprocess", "shutil", "socket", "sys", "pty", "requests", "urllib"}
    forbidden_builtins = {"eval", "exec", "compile", "globals", "locals", "memoryview"}

    for node in ast.walk(tree):
        # Block forbidden imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split('.')[0] in forbidden_modules:
                    return f"Forbidden import detected: {alias.name}. Keep execution sandboxed."
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split('.')[0] in forbidden_modules:
                return f"Forbidden import detected: {node.module}. Keep execution sandboxed."

        # Block malicious built-in calls (eval/exec)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in forbidden_builtins:
                    return f"Forbidden built-in call detected: {node.func.id}() is not permitted."

    return None


def parse_titan_verdict(raw: str) -> str:
    """Normalize model text to ``APPROVED`` or ``REJECTED: ...``."""
    text = (raw or "").strip()
    if not text:
        return "REJECTED: empty supervisor response"
    # Prefer first meaningful line.
    for line in text.splitlines():
        line = line.strip().strip("`").strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("APPROVED"):
            return "APPROVED"
        if upper.startswith("REJECTED"):
            # Keep original reasoning after the colon when present.
            if ":" in line:
                reason = line.split(":", 1)[1].strip() or "rejected by Titan"
                return f"REJECTED: {reason}"
            return "REJECTED: rejected by Titan"
    upper = text.upper()
    if "APPROVED" in upper and "REJECTED" not in upper:
        return "APPROVED"
    return f"REJECTED: unparseable supervisor verdict — {text[:160]}"


parse_jason_verdict = parse_titan_verdict


def review_watchdog_code(
    code: str,
    task: str = "",
    *,
    model: str = "llama3.2",
    temperature: float = 0.1,
    ask_fn: Any | None = None,
) -> str:
    """Titan Watchdog review → ``APPROVED`` or ``REJECTED: <reason>``.

    ``ask_fn`` (optional) is ``(system, user) -> str`` for tests / ChatOllama adapters.
    """
    static = static_code_safety_reject(code)
    if static:
        return f"REJECTED: {static}"

    user_prompt = (
        f"TASK:\n{(task or '').strip() or '(unspecified monitoring task)'}\n\n"
        f"PYTHON_SCRIPT:\n```python\n{(code or '').strip()}\n```\n"
    )
    if ask_fn is not None:
        raw = ask_fn(TITAN_SUPERVISOR_PROMPT, user_prompt)
    else:
        raw = ask_ollama(
            TITAN_SUPERVISOR_PROMPT,
            user_prompt,
            model=model,
            temperature=temperature,
        )
    return parse_titan_verdict(str(raw or ""))
