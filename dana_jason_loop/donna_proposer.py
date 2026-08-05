"""Donna — Batch Proposer: capability-gap discovery via local qwen2.5-coder:7b."""

from __future__ import annotations

import os
import re
from typing import Any

from dana_jason_loop.ollama_client import ask_ollama, extract_json_payload

PROPOSAL_REQUIRED_KEYS = (
    "proposed_id",
    "problem_statement",
    "proposed_solution_code_outline",
    "dependencies",
)

DONNA_SYSTEM_PROMPT = """
You are Donna, CAMGRASPER's offline capability scout.
You read recent failure traces / bottleneck logs from an on-device voice agent
(Whisper STT, bilingual EN/FA routing, YOLO SpatialIR, local qwen2.5-coder:7b ReAct,
encrypted vault memory, Piper TTS). Your job is to pitch concrete remedies.

Output rules (non-negotiable):
1. Respond with EXACTLY one JSON array containing EXACTLY 5 objects. No prose,
   no markdown fences, no trailing commentary.
2. Each object MUST use this exact schema:
   {
     "proposed_id": "snake_case_identifier",
     "problem_statement": "one short paragraph naming the gap",
     "proposed_solution_code_outline": "brief Python-oriented outline of the tool/module",
     "dependencies": ["package_or_module_names"]
   }
3. All five proposed_id values MUST be distinct snake_case strings.
4. Prefer remedies that run offline on a Windows laptop with ~8GB VRAM,
   reuse the existing sandbox / vault / tools.json patterns, and avoid new
   foundation models or cloud APIs unless the log explicitly demands them.
5. Base every pitch on evidence in the supplied traces — do not invent
   unrelated product fantasies.
6. STRICT JSON hygiene: use only double-quoted strings. Never use Python
   triple-quotes. Keep proposed_solution_code_outline to ONE line (~40 words)
   with \\n escapes if needed. ASCII preferred; no raw non-ASCII code points inside
   JSON string values.
""".strip()


def _load_trace_blob(source: str | os.PathLike[str] | list[str] | None) -> str:
    if source is None:
        return "(no traces supplied)"
    if isinstance(source, list):
        return "\n".join(str(x) for x in source)
    path = os.fspath(source)
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    return str(source)


def _snake_case(value: str, fallback: str) -> str:
    text = (value or "").strip().lower()
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text or not re.match(r"^[a-z]", text):
        text = fallback
    return text[:64]


def _normalize_proposal(item: Any, index: int) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError(f"Proposal #{index + 1} is not an object: {item!r}")
    missing = [k for k in PROPOSAL_REQUIRED_KEYS if k not in item]
    if missing:
        raise ValueError(f"Proposal #{index + 1} missing keys: {missing}")

    deps = item.get("dependencies") or []
    if isinstance(deps, str):
        deps = [deps]
    if not isinstance(deps, list):
        raise ValueError(f"Proposal #{index + 1} dependencies must be a list")

    return {
        "proposed_id": _snake_case(str(item["proposed_id"]), f"proposal_{index + 1}"),
        "problem_statement": str(item["problem_statement"]).strip(),
        "proposed_solution_code_outline": str(
            item["proposed_solution_code_outline"]
        ).strip(),
        "dependencies": [str(d).strip() for d in deps if str(d).strip()],
    }


def generate_capability_pitches(
    traces: str | os.PathLike[str] | list[str] | None,
    *,
    model: str = "qwen2.5-coder:7b",
    temperature: float = 0.35,
) -> list[dict[str, Any]]:
    """Ask Donna (via Ollama) for exactly five capability pitches.

    ``traces`` may be a path to a log file, a raw string, or a list of
    recent error / bottleneck lines.
    """
    blob = _load_trace_blob(traces)
    user_prompt = (
        "Analyze the following recent system bottleneck / error traces and "
        "propose exactly 5 distinct tool or project pitches as a raw JSON array.\n\n"
        "=== TRACES BEGIN ===\n"
        f"{blob}\n"
        "=== TRACES END ===\n"
    )
    raw = ask_ollama(
        DONNA_SYSTEM_PROMPT,
        user_prompt,
        model=model,
        temperature=temperature,
    )
    try:
        payload = extract_json_payload(raw)
    except ValueError:
        # One repair turn — 8B often emits Python triple-quotes inside "JSON".
        repair = ask_ollama(
            DONNA_SYSTEM_PROMPT,
            (
                "Your previous reply was not valid JSON. Rewrite it as a strict "
                "JSON array of exactly 5 objects with double-quoted one-line "
                "strings only. No triple quotes, no markdown.\n\n"
                f"PREVIOUS_REPLY=\n{raw[:4000]}"
            ),
            model=model,
            temperature=0.05,
        )
        payload = extract_json_payload(repair)

    if not isinstance(payload, list):
        raise ValueError(f"Donna must return a JSON array; got {type(payload).__name__}")
    if len(payload) != 5:
        # Soft truncate / reject — require exactly 5 after normalize when possible.
        if len(payload) > 5:
            payload = payload[:5]
        else:
            raise ValueError(
                f"Donna must return exactly 5 proposals; got {len(payload)}"
            )

    proposals = [_normalize_proposal(item, i) for i, item in enumerate(payload)]
    ids = [p["proposed_id"] for p in proposals]
    if len(set(ids)) != 5:
        # Disambiguate collisions without calling the model again.
        seen: dict[str, int] = {}
        for p in proposals:
            base = p["proposed_id"]
            n = seen.get(base, 0)
            seen[base] = n + 1
            if n:
                p["proposed_id"] = f"{base}_{n + 1}"
    return proposals
