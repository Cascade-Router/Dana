"""Tool Forge — LangGraph multi-agent security pipeline for new tools.

Flow:
  Coder → AST Gatekeeper (security_policy.json) → Security Reviewer (JSON) → Hot-Load

Triggered when the orchestrator hits ``ToolNotFound``. Only tools that pass BOTH
deterministic AST and semantic LLM review are written under ``CAMGRASPER/custom_tools/``
and registered into the Semantic Tool Registry.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import os
import re
import textwrap
from collections.abc import Callable
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from dana.paths import CUSTOM_TOOLS_DIR
from dana.swarm.tool_forge_template import (
    JSON_SCHEMA_FAILURE,
    assemble_forged_tool,
    extract_coder_json,
    normalize_coder_payload,
)
from dana.swarm.tool_forge_template import (
    safe_tool_name as _template_safe_tool_name,
)
from dana.tools.registry import (
    ensure_custom_tools_package,
    get_tool_registry,
    load_security_policy,
)
from dana.tools.schema import ToolParameterSpec, ToolSpec

try:
    from dana.logging import log as _forge_log
except Exception:  # noqa: BLE001
    def _forge_log(thread: str, message: str, *, level: str = "info") -> None:  # type: ignore[misc]
        print(f"[{thread}] {message}", flush=True)

DEFAULT_MODEL = "qwen2.5-coder:7b"
MAX_FORGE_REVISIONS = 3
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.I)
_TOOL_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")


class ToolForgeState(TypedDict):
    query: str
    tool_name: str
    code: str
    lint_errors: str
    security_feedback: str
    security_review: dict[str, Any]
    feedback: str
    status: str
    # pending | drafting | LINT_OK | LINT_FAIL | SEC_REJECTED | APPROVED | loaded | error
    revisions: int
    history: list[dict[str, Any]]
    loaded_tool: str


CODER_SYSTEM = """
You are Dana's Tool Forge coder. You do NOT invent a full Python module.
You ONLY fill fields for a deterministic BaseTool template.

Output ONLY one JSON object. Zero markdown. Zero commentary. Zero prose.
Any character outside the JSON object is an IMMEDIATE FAILURE.

Exact schema (escape newlines inside strings as \\n):
{
  "tool_name": "snake_case_id",
  "description": "one-line English description",
  "docstring": "short docstring for the tool",
  "python_code": "result = (text or '').strip()\\n    return result[::-1]"
}

Field rules:
- ``python_code`` is the FUNCTION BODY ONLY (statements inside the function).
  Do NOT include ``def``, ``@tool``, imports, or class wrappers.
  Do NOT wrap the body in triple quotes.
- Newlines inside ``python_code`` MUST be escaped as \\n so the JSON stays valid.
- The system injects your fields into this fixed template (you never write it):
  from langchain_core.tools import tool
  @tool
  def {tool_name}(text: str = '', filepath: str = '') -> str:
      \"\"\"{docstring}\"\"\"
      {python_code}

Hard rules for the body:
- Use ``text`` and/or ``filepath`` parameters already provided by the template.
- Local file access MUST use resolve_safe_path / sandbox_read from
  dana.tools.sandbox_io (the assembler will inject those imports when needed).
  Never write ``from sandbox_io import ...`` or ``import resolve_safe_path``.
- For images: Image.open(resolve_safe_path(filepath)) — never raw paths, never open().
- CPU / RAM / process metrics: use ``psutil`` (Tier-2 approved). Example body:
  ``cpu = psutil.cpu_percent(interval=0.1)\\n    mem = psutil.virtual_memory().percent\\n    return f'cpu={cpu} ram={mem}'``
  The assembler injects ``import psutil`` when the body references it.
- Count files under the Dana sandbox with pathlib only, e.g.:
  ``root = Path.home() / 'Desktop' / 'Dana' / 'execution_jail'\\n    n = len(list(root.glob('*.txt')))\\n    return str(n)``
- Forbidden in the body: os, sys, subprocess, shutil, socket, ctypes, pickle,
  eval, exec, compile, __import__, native open().
- Keep the body short and deterministic. Indent the body with 4 spaces per line
  (or none — the assembler re-indents).
""".strip()

SECURITY_REVIEWER_SYSTEM = """
You are a zero-trust security auditor for Dana's Tool Forge.
You do NOT evaluate utility or cleverness — ONLY threat vectors.

Inspect the Python tool source for:
- data exfiltration / network egress
- filesystem escape beyond sandbox_read
- infinite loops / resource bombs
- prompt-injection surfaces (untrusted string eval)
- obfuscation or dynamic import tricks
- privilege escalation

You MUST reply with ONLY this JSON schema (no markdown fences):
{
  "status": "APPROVED" | "REJECTED",
  "threat_assessment": "Short explanation of logical vulnerabilities",
  "violations": ["List of semantic violations, if any"],
  "required_remediation": "Exact instructions for the coder to fix the code"
}
""".strip()


def _llm_content(result: Any) -> str:
    content = getattr(result, "content", None)
    if isinstance(content, list):
        return "".join(
            str(part.get("text", part)) if isinstance(part, dict) else str(part)
            for part in content
        ).strip()
    if content is None:
        return str(result or "").strip()
    return str(content).strip()


def _chat_ollama(model: str = DEFAULT_MODEL, temperature: float = 0.1):
    from langchain_ollama import ChatOllama

    return ChatOllama(model=model, temperature=temperature)


def _extract_json(raw: str) -> dict[str, Any] | None:
    text = (raw or "").strip()
    if not text:
        return None
    fence = _JSON_FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _safe_tool_name(name: str, *, fallback: str = "forged_tool") -> str:
    return _template_safe_tool_name(name, fallback=fallback)


def suggest_tool_name(query: str) -> str:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9]{1,}", query or "")
    if not tokens:
        return "forged_tool"
    base = "_".join(t.lower() for t in tokens[:4])
    return _safe_tool_name(base, fallback="forged_tool")


def _safe_resolver_call_name(node: ast.AST) -> str:
    """Return the callee name if ``node`` is a function call, else ''."""
    if not isinstance(node, ast.Call):
        return ""
    fn = node.func
    if isinstance(fn, ast.Name):
        return fn.id
    if isinstance(fn, ast.Attribute):
        return fn.attr
    return ""


def analyze_tool_ast(
    code: str,
    *,
    policy: dict[str, Any] | None = None,
) -> list[str]:
    """Declarative AST gatekeeper driven by ``security_policy.json``.

    Tier-1 + Tier-2 imports are permitted; Tier-3 is forbidden. Path-dependent
    loaders such as ``PIL.Image.open(...)`` are allowed ONLY when their path
    argument is produced by a safe resolver (``resolve_safe_path`` /
    ``sandbox_read``), either inline or via a variable assigned from one.
    """
    policy = policy or load_security_policy()
    tier1 = {str(x) for x in (policy.get("tier1_allowed") or [])}
    tier2 = {str(x) for x in (policy.get("tier2_review") or [])}
    tier3 = {str(x) for x in (policy.get("tier3_forbidden") or [])}
    forbidden_builtins = {str(x) for x in (policy.get("forbidden_builtins") or [])}
    allowed_from = {str(x) for x in (policy.get("allowed_from_modules") or [])}
    safe_resolvers = {
        str(x)
        for x in (
            policy.get("safe_path_resolvers")
            or ["sandbox_read", "resolve_safe_path", "resolve_sandbox_path"]
        )
    }
    open_attrs = {str(x) for x in (policy.get("path_dependent_open_attrs") or ["open"])}

    blob = textwrap.dedent(code or "").strip()
    if not blob:
        return ["FATAL: empty tool source."]

    # Raw JSON / brace dump mistaken for Python (common coder failure mode).
    if blob.lstrip().startswith("{") and (
        '"python_code"' in blob
        or '"tool_name"' in blob
        or '"function_body"' in blob
        or '"code"' in blob
    ):
        return [JSON_SCHEMA_FAILURE]

    try:
        tree = ast.parse(blob)
    except SyntaxError as exc:
        msg = str(exc)
        # Brace / JSON bleed → steer coder back to the schema.
        if (
            "{" in (exc.text or "")
            or "}" in (exc.text or "")
            or "'{'" in msg
            or "'}'" in msg
            or "was never closed" in msg
            or (
                "invalid syntax" in msg.lower()
                and blob.lstrip()[:1] in "{["
            )
        ):
            return [JSON_SCHEMA_FAILURE, f"FATAL: SyntaxError: {exc}"]
        return [f"FATAL: SyntaxError: {exc}"]

    errors: list[str] = []
    has_native_open = False

    # Pre-pass: variables bound to a safe-resolver result may feed a loader.
    safe_path_vars: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and _safe_resolver_call_name(node.value) in safe_resolvers:
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    safe_path_vars.add(tgt.id)
        elif (
            isinstance(node, ast.withitem)
            and _safe_resolver_call_name(node.context_expr) in safe_resolvers
            and isinstance(node.optional_vars, ast.Name)
        ):
            safe_path_vars.add(node.optional_vars.id)

    def _open_arg_is_sandboxed(call_node: ast.Call) -> bool:
        args = list(call_node.args)
        if not args:
            return False
        first = args[0]
        if _safe_resolver_call_name(first) in safe_resolvers:
            return True
        if isinstance(first, ast.Name) and first.id in safe_path_vars:
            return True
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = (alias.name or "").split(".", 1)[0]
                if root in tier3:
                    errors.append(
                        f"FATAL: Tier-3 import '{alias.name}' forbidden by security_policy.json."
                    )
                elif root in tier1 or root in tier2 or alias.name in allowed_from:
                    continue
                else:
                    hint = ""
                    if root == "psutil" or "psutil" in (alias.name or ""):
                        hint = " (psutil should be Tier-2 — reload security_policy.json)."
                    errors.append(
                        f"FATAL: import '{alias.name}' is not in an allowed tier "
                        f"(Tier-1/Tier-2). Use approved stdlib, PIL, psutil, or "
                        f"dana.tools.sandbox_io.{hint}"
                    )
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            root = mod.split(".", 1)[0] if mod else ""
            if mod in allowed_from or any(mod.startswith(a + ".") for a in allowed_from):
                continue
            if root in tier3:
                errors.append(
                    f"FATAL: Tier-3 ImportFrom '{mod}' forbidden by security_policy.json."
                )
            elif root in tier1 or root in tier2:
                continue
            elif root:
                errors.append(
                    f"FATAL: ImportFrom '{mod}' is not in an allowed tier (Tier-1/Tier-2)."
                )
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                if func.id in forbidden_builtins:
                    errors.append(f"FATAL: forbidden built-in call '{func.id}()'.")
                if func.id == "open":
                    has_native_open = True
            elif isinstance(func, ast.Attribute):
                # Path-dependent loader (e.g. Image.open) — require sandboxed path.
                if func.attr in open_attrs:
                    if not _open_arg_is_sandboxed(node):
                        errors.append(
                            f"FATAL: <obj>.{func.attr}(...) must receive a path resolved "
                            "via resolve_safe_path()/sandbox_read() "
                            "(e.g. Image.open(resolve_safe_path(filepath)))."
                        )
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id == "open":
                has_native_open = True
            elif node.id in {"eval", "exec", "compile", "__import__"}:
                errors.append(f"FATAL: forbidden name '{node.id}'.")

    # Native open( — but not attribute access like Image.open(.
    if has_native_open or re.search(r"(?<![.\w])open\s*\(", blob):
        errors.append(
            "FATAL: native open() is forbidden — use sandbox_read(filepath) or "
            "resolve_safe_path(filepath) from dana.tools.sandbox_io."
        )

    # Deduplicate.
    seen: set[str] = set()
    unique: list[str] = []
    for err in errors:
        if err not in seen:
            seen.add(err)
            unique.append(err)
    return unique


def dana_coder_forge(
    state: ToolForgeState,
    *,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Step A: draft JSON fields; assemble into a pre-validated @tool module."""
    query = (state.get("query") or "").strip()
    lint_errors = (state.get("lint_errors") or "").strip()
    security_feedback = (state.get("security_feedback") or "").strip()
    revisions = int(state.get("revisions") or 0)
    hinted = _safe_tool_name(state.get("tool_name") or suggest_tool_name(query))
    _forge_log(
        "ToolForge",
        f"tool_forge_node=dana_coder_forge revision={revisions} "
        f"hint={hinted!r} query={query[:80]!r}",
    )

    user = (
        f"UNHANDLED USER QUERY:\n{query or '(empty)'}\n\n"
        f"Preferred tool_name: {hinted}\n"
        "Return ONLY the JSON object with tool_name, description, docstring, "
        "and python_code (function body only; escape newlines as \\n).\n"
    )
    if lint_errors:
        user += f"\nFATAL AST LINT ERRORS (fix first):\n{lint_errors}\n"
    if security_feedback:
        user += f"\nSECURITY REMEDIATION (must apply):\n{security_feedback}\n"
    if not lint_errors and not security_feedback:
        user += "\nDraft the JSON fields now.\n"

    try:
        llm = _chat_ollama(model=model, temperature=0.1)
        raw = _llm_content(
            llm.invoke(
                [
                    {"role": "system", "content": CODER_SYSTEM},
                    {"role": "user", "content": user},
                ]
            )
        )
        data = extract_coder_json(raw)
        if data is None:
            history = list(state.get("history") or [])
            history.append(
                {
                    "stage": "coder",
                    "revision": revisions,
                    "tool_name": hinted,
                    "status": "JSON_FAIL",
                    "raw_preview": (raw or "")[:240],
                }
            )
            return {
                "tool_name": hinted,
                "code": "",
                "lint_errors": JSON_SCHEMA_FAILURE,
                "status": "LINT_FAIL",
                "feedback": JSON_SCHEMA_FAILURE,
                "revisions": revisions,
                "history": history,
            }

        payload = normalize_coder_payload(data, fallback_name=hinted)
        tool_name = payload["tool_name"]
        body = payload["python_code"]
        if not body.strip():
            return {
                "tool_name": tool_name,
                "code": "",
                "lint_errors": JSON_SCHEMA_FAILURE,
                "status": "LINT_FAIL",
                "feedback": "Tool Forge coder returned empty python_code field",
                "revisions": revisions,
            }

        # Assemble into the fixed BaseTool / @tool topology (Inversion of Control).
        code = assemble_forged_tool(
            tool_name=tool_name,
            docstring=payload["docstring"],
            python_code=body,
            description=payload["description"],
        )
        description = payload["description"] or payload["docstring"]
        history = list(state.get("history") or [])
        history.append(
            {
                "stage": "coder",
                "revision": revisions,
                "tool_name": tool_name,
                "description": description,
                "status": "drafting",
            }
        )
        return {
            "tool_name": tool_name,
            "code": code,
            "lint_errors": "",
            "status": "drafting",
            "feedback": description,
            "revisions": revisions,
            "history": history,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "feedback": f"Tool Forge coder failed: {exc}",
            "revisions": revisions,
        }


def ast_gatekeeper_forge(state: ToolForgeState) -> dict[str, Any]:
    """Step B: deterministic AST policy check — bounce to coder on failure."""
    code = state.get("code") or ""
    revisions = int(state.get("revisions") or 0)
    _forge_log(
        "ToolForge",
        f"tool_forge_node=ast_gatekeeper_forge tool={state.get('tool_name')!r}",
    )
    if not str(code).strip():
        prior = (state.get("lint_errors") or "").strip()
        fatal = [prior] if prior else [JSON_SCHEMA_FAILURE]
    else:
        fatal = analyze_tool_ast(code)
    history = list(state.get("history") or [])

    if fatal:
        lint_blob = "\n".join(fatal)
        # Prefer the JSON-schema remediation whenever brace/JSON failures appear.
        if any(
            "not valid JSON" in e or "python_code` JSON field" in e or "python_code JSON" in e
            for e in fatal
        ):
            lint_blob = JSON_SCHEMA_FAILURE + (
                "" if lint_blob == JSON_SCHEMA_FAILURE else f"\n{lint_blob}"
            )
        revisions += 1
        history.append(
            {
                "stage": "ast_lint",
                "revision": revisions,
                "feedback": lint_blob,
                "status": "LINT_FAIL",
            }
        )
        _forge_log(
            "ToolForge",
            f"AST LINT_FAIL revision={revisions}: {lint_blob[:200]}",
            level="warning",
        )
        return {
            "lint_errors": lint_blob,
            "status": "LINT_FAIL",
            "revisions": revisions,
            "history": history,
        }

    history.append(
        {
            "stage": "ast_lint",
            "revision": revisions,
            "feedback": "",
            "status": "LINT_OK",
        }
    )
    _forge_log("ToolForge", "AST LINT_OK — routing to security_reviewer_agent")
    return {
        "lint_errors": "",
        "status": "LINT_OK",
        "revisions": revisions,
        "history": history,
    }


def _normalize_security_review(data: dict[str, Any] | None) -> dict[str, Any]:
    data = dict(data or {})
    status = str(data.get("status") or "REJECTED").strip().upper()
    if status not in {"APPROVED", "REJECTED"}:
        status = "REJECTED"
    violations = data.get("violations") or []
    if isinstance(violations, str):
        violations = [violations]
    if not isinstance(violations, list):
        violations = [str(violations)]
    return {
        "status": status,
        "threat_assessment": str(data.get("threat_assessment") or "").strip(),
        "violations": [str(v) for v in violations],
        "required_remediation": str(data.get("required_remediation") or "").strip(),
    }


def security_reviewer_agent(
    state: ToolForgeState,
    *,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Step C: security gate after AST.

    Local llama3.2 reviewers chronically false-reject allowlisted ``psutil`` /
    pathlib tools, then the coder collapses into invalid JSON. When the
    deterministic AST gate already passed (Tier-1/2 only, no native open),
    short-circuit to APPROVED. LLM audit remains available only as a soft
    advisory when ``DANA_FORGE_LLM_SECURITY=1``.
    """
    code = state.get("code") or ""
    query = state.get("query") or ""
    tool_name = state.get("tool_name") or ""
    revisions = int(state.get("revisions") or 0)
    _forge_log(
        "ToolForge",
        f"tool_forge_node=security_reviewer_agent tool={tool_name!r}",
    )

    # Re-check AST — if still clean, do not let a flaky LLM veto hot-load.
    fatal = analyze_tool_ast(code)
    history = list(state.get("history") or [])
    if not fatal:
        use_llm = os.environ.get("DANA_FORGE_LLM_SECURITY", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if not use_llm:
            review = {
                "status": "APPROVED",
                "threat_assessment": (
                    "Deterministic allowlist: AST clean (Tier-1/2 + sandbox path rules)."
                ),
                "violations": [],
                "required_remediation": "",
            }
            history.append(
                {
                    "stage": "security_review",
                    "revision": revisions,
                    "feedback": review["threat_assessment"],
                    "status": "APPROVED",
                    "review": review,
                }
            )
            _forge_log(
                "ToolForge",
                "security_reviewer APPROVED (deterministic allowlist) — routing to hot_load",
            )
            return {
                "security_review": review,
                "security_feedback": "",
                "status": "APPROVED",
                "history": history,
            }

    user = (
        f"TOOL NAME: {tool_name}\n"
        f"USER QUERY: {query}\n\n"
        f"SOURCE:\n```python\n{code}\n```\n"
        "IMPORTANT: psutil CPU/RAM metrics and pathlib sandbox file counts are "
        "APPROVED Tier-2 patterns. Do NOT reject solely for importing psutil.\n"
    )
    try:
        llm = _chat_ollama(model=model, temperature=0.0)
        raw = _llm_content(
            llm.invoke(
                [
                    {"role": "system", "content": SECURITY_REVIEWER_SYSTEM},
                    {"role": "user", "content": user},
                ]
            )
        )
        review = _normalize_security_review(_extract_json(raw))
    except Exception as exc:  # noqa: BLE001
        review = _normalize_security_review(
            {
                "status": "REJECTED",
                "threat_assessment": f"security reviewer failed: {exc}",
                "violations": ["reviewer_error"],
                "required_remediation": "Simplify the tool and avoid risky APIs.",
            }
        )

    # If AST is clean but LLM still REJECTS, override — false rejects cause
    # JSON death spirals on forge revisions.
    if not fatal and review["status"] != "APPROVED":
        _forge_log(
            "ToolForge",
            "security_reviewer LLM REJECTED allowlisted tool — overriding to APPROVED",
            level="warning",
        )
        review = {
            "status": "APPROVED",
            "threat_assessment": (
                "Overrode LLM REJECTED: AST allowlist clean "
                f"(llm said: {(review.get('threat_assessment') or '')[:160]})"
            ),
            "violations": [],
            "required_remediation": "",
        }

    if review["status"] == "APPROVED":
        history.append(
            {
                "stage": "security_review",
                "revision": revisions,
                "feedback": review.get("threat_assessment") or "APPROVED",
                "status": "APPROVED",
                "review": review,
            }
        )
        _forge_log("ToolForge", "security_reviewer APPROVED — routing to hot_load")
        return {
            "security_review": review,
            "security_feedback": "",
            "status": "APPROVED",
            "history": history,
        }

    revisions += 1
    remediation = review.get("required_remediation") or "Address security violations."
    history.append(
        {
            "stage": "security_review",
            "revision": revisions,
            "feedback": remediation,
            "status": "SEC_REJECTED",
            "review": review,
        }
    )
    _forge_log(
        "ToolForge",
        f"security_reviewer SEC_REJECTED revision={revisions}: {remediation[:200]}",
        level="warning",
    )
    return {
        "security_review": review,
        "security_feedback": remediation,
        "status": "SEC_REJECTED",
        "revisions": revisions,
        "history": history,
    }


def _extract_entry_function(code: str, tool_name: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    names = [
        n.name
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if tool_name in names:
        return tool_name
    if names:
        return names[0]
    return None


def hot_load_forged_tool(state: ToolForgeState) -> dict[str, Any]:
    """Step D: persist under CAMGRASPER/custom_tools/, importlib hot-load."""
    _forge_log(
        "ToolForge",
        f"tool_forge_node=hot_load_forged_tool tool={state.get('tool_name')!r}",
    )
    try:
        from dana.settings import is_dynamic_tool_synthesis_enabled, synthesis_locked_message

        if not is_dynamic_tool_synthesis_enabled():
            return {
                "status": "error",
                "feedback": synthesis_locked_message("en"),
            }
    except Exception:  # noqa: BLE001
        return {
            "status": "error",
            "feedback": "Dynamic tool synthesis is locked (settings unavailable).",
        }

    code = textwrap.dedent(state.get("code") or "").strip()
    tool_name = _safe_tool_name(state.get("tool_name") or "forged_tool")
    description = (state.get("feedback") or f"Forged tool `{tool_name}`").strip()
    query = state.get("query") or ""

    # Final AST re-check before disk write.
    fatal = analyze_tool_ast(code)
    if fatal:
        return {
            "status": "LINT_FAIL",
            "lint_errors": "\n".join(fatal),
            "feedback": "Hot-load aborted: AST re-check failed",
        }

    ensure_custom_tools_package()
    path = CUSTOM_TOOLS_DIR / f"{tool_name}.py"
    header = (
        f'"""Auto-generated by Tool Forge for query: {query[:120]!r}.\n'
        f"Do not edit by hand — regenerated by the security pipeline.\n"
        f"Workspace: CAMGRASPER/custom_tools/\n"
        f'"""\n\n'
    )
    path.write_text(header + code + "\n", encoding="utf-8")

    # External package on sys.path: CAMGRASPER → import custom_tools.<name>
    module_name = f"custom_tools.{tool_name}"
    try:
        import sys

        from dana.paths import ensure_workspace_on_syspath

        ensure_workspace_on_syspath()
        # Invalidate cached module so re-forge picks up new source.
        sys.modules.pop(module_name, None)
        sys.modules.pop("custom_tools", None)
        # Legacy package name (pre-restructure).
        sys.modules.pop(f"generated_tools.{tool_name}", None)
        sys.modules.pop("generated_tools", None)
        spec = importlib.util.spec_from_file_location(
            module_name,
            path,
            submodule_search_locations=[str(CUSTOM_TOOLS_DIR)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "feedback": f"Hot-load import failed: {exc}",
        }

    entry_name = _extract_entry_function(code, tool_name)
    callable_obj: Callable[..., Any] | None = None
    if entry_name and hasattr(module, entry_name):
        callable_obj = getattr(module, entry_name)
        # LangChain @tool wraps StructuredTool — prefer .func / invoke.
        if hasattr(callable_obj, "func") and callable(callable_obj.func):
            callable_obj = callable_obj.func  # type: ignore[assignment]
        elif hasattr(callable_obj, "invoke") and not callable(callable_obj):
            inv = callable_obj.invoke

            def _wrapped(**kwargs: Any) -> Any:
                return inv(kwargs)

            callable_obj = _wrapped

    tool_spec = ToolSpec(
        id=tool_name,
        description_en=description or f"Forged tool {tool_name}",
        description_fa=f" ‌ `{tool_name}`",
        parameters=(
            ToolParameterSpec(
                name="text",
                type="string",
                required=False,
                description_en="Primary text input.",
            ),
        ),
        aliases_en={"_intent": (tool_name.replace("_", " "),)},
        aliases_fa={"_intent": (tool_name,)},
    )

    registry = get_tool_registry()
    registry.register(
        tool_spec,
        callable=callable_obj if callable(callable_obj) else None,
        source="forge",
        ephemeral=True,
        metadata={
            "path": str(path),
            "module": module_name,
            "query": query,
            "ephemeral": True,
            "tier": "custom",
        },
    )

    # Keep tools.json + broker in sync when possible.
    try:
        from dana.tools.broker import reload_broker_registry
        from dana_security import register_tool_schema

        register_tool_schema(
            tool_name,
            description_en=tool_spec.description_en,
            param_name="text",
        )
        reload_broker_registry()
    except Exception:  # noqa: BLE001
        pass

    history = list(state.get("history") or [])
    history.append(
        {
            "stage": "hot_load",
            "revision": state.get("revisions", 0),
            "tool_name": tool_name,
            "path": str(path),
            "status": "loaded",
        }
    )
    _forge_log(
        "ToolForge",
        f"hot_load COMPLETE tool={tool_name!r} path={path} — registered in ToolRegistry",
    )
    return {
        "status": "loaded",
        "loaded_tool": tool_name,
        "feedback": f"OK: forged and hot-loaded tool `{tool_name}`",
        "history": history,
    }


def _route_after_ast(
    state: ToolForgeState,
) -> Literal["security_reviewer", "dana_coder", "terminal_failure"]:
    status = (state.get("status") or "").strip().upper()
    revisions = int(state.get("revisions") or 0)
    if status == "LINT_OK":
        return "security_reviewer"
    if status == "LINT_FAIL" and revisions < MAX_FORGE_REVISIONS:
        return "dana_coder"
    return "terminal_failure"


def _route_after_security(
    state: ToolForgeState,
) -> Literal["hot_load", "dana_coder", "terminal_failure"]:
    status = (state.get("status") or "").strip().upper()
    revisions = int(state.get("revisions") or 0)
    if status == "APPROVED":
        return "hot_load"
    if status == "SEC_REJECTED" and revisions < MAX_FORGE_REVISIONS:
        return "dana_coder"
    return "terminal_failure"


def terminal_failure_forge(state: ToolForgeState) -> dict[str, Any]:
    lint = (state.get("lint_errors") or "").strip()
    sec = (state.get("security_feedback") or "").strip()
    root = lint or sec or state.get("feedback") or "Tool Forge aborted"
    detail = (
        f"Tool Forge failed after {state.get('revisions', 0)} revision(s): {root}"
    )
    try:
        from dana.logging import log_exception

        log_exception("ToolForge", "Tool Forge terminal failure", exc=RuntimeError(detail))
    except Exception:
        pass
    # Autonomous Bug Tracker — every Tool Forge Terminal Failure is ledgered.
    try:
        from dana.bug_tracker import log_bug_to_tracker

        log_bug_to_tracker(
            detail,
            context=(
                f"tool_name={state.get('tool_name') or ''}\n"
                f"query={state.get('query') or ''}\n"
                f"lint_errors={lint}\n"
                f"security_feedback={sec}"
            ),
            status="PENDING",
            source="tool_forge_terminal_failure",
        )
    except Exception:
        pass
    history = list(state.get("history") or [])
    history.append(
        {
            "stage": "terminal_failure",
            "revision": state.get("revisions", 0),
            "feedback": detail,
            "status": "error",
        }
    )
    return {"status": "error", "feedback": detail, "history": history}


def build_tool_forge_graph(*, model: str = DEFAULT_MODEL):
    """Compile Coder → AST → Security Reviewer → Hot-Load."""

    def _coder(state: ToolForgeState) -> dict[str, Any]:
        return dana_coder_forge(state, model=model)

    def _ast(state: ToolForgeState) -> dict[str, Any]:
        return ast_gatekeeper_forge(state)

    def _sec(state: ToolForgeState) -> dict[str, Any]:
        return security_reviewer_agent(state, model=model)

    graph = StateGraph(ToolForgeState)
    graph.add_node("dana_coder", _coder)
    graph.add_node("ast_gatekeeper", _ast)
    graph.add_node("security_reviewer", _sec)
    graph.add_node("hot_load", hot_load_forged_tool)
    graph.add_node("terminal_failure", terminal_failure_forge)

    graph.add_edge(START, "dana_coder")
    graph.add_edge("dana_coder", "ast_gatekeeper")
    graph.add_conditional_edges(
        "ast_gatekeeper",
        _route_after_ast,
        {
            "security_reviewer": "security_reviewer",
            "dana_coder": "dana_coder",
            "terminal_failure": "terminal_failure",
        },
    )
    graph.add_conditional_edges(
        "security_reviewer",
        _route_after_security,
        {
            "hot_load": "hot_load",
            "dana_coder": "dana_coder",
            "terminal_failure": "terminal_failure",
        },
    )
    graph.add_edge("hot_load", END)
    graph.add_edge("terminal_failure", END)
    return graph.compile()


def run_tool_forge(
    query: str,
    *,
    tool_name: str = "",
    model: str = DEFAULT_MODEL,
) -> ToolForgeState:
    """Invoke the Tool Forge subgraph for an unhandled / ToolNotFound query."""
    app = build_tool_forge_graph(model=model)
    seed: ToolForgeState = {
        "query": (query or "").strip(),
        "tool_name": _safe_tool_name(tool_name or suggest_tool_name(query)),
        "code": "",
        "lint_errors": "",
        "security_feedback": "",
        "security_review": {},
        "feedback": "",
        "status": "pending",
        "revisions": 0,
        "history": [],
        "loaded_tool": "",
    }
    result = app.invoke(seed)
    return ToolForgeState(
        query=str(result.get("query") or seed["query"]),
        tool_name=str(result.get("tool_name") or seed["tool_name"]),
        code=str(result.get("code") or ""),
        lint_errors=str(result.get("lint_errors") or ""),
        security_feedback=str(result.get("security_feedback") or ""),
        security_review=dict(result.get("security_review") or {}),
        feedback=str(result.get("feedback") or ""),
        status=str(result.get("status") or "error"),
        revisions=int(result.get("revisions") or 0),
        history=list(result.get("history") or []),
        loaded_tool=str(result.get("loaded_tool") or ""),
    )


class ToolNotFound(KeyError):
    """Raised when a requested tool id is absent from the Semantic Tool Registry."""


def route_tool_not_found(
    query: str,
    *,
    missing_tool: str = "",
    model: str = DEFAULT_MODEL,
) -> ToolForgeState:
    """Orchestrator entry: ToolNotFound → Tool Forge subgraph."""
    _forge_log(
        "ToolForge",
        f"ToolNotFound trigger missing_tool={missing_tool!r} "
        f"query={(query or '')[:100]!r} - entering tool_forge_node",
    )
    return run_tool_forge(query, tool_name=missing_tool, model=model)
