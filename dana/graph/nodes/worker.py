"""Worker swarm — deterministic extraction workers for ready DAG tasks.

Code-generation hops call a plain LLM (no tools / no ReAct JSON), then Python
extracts a ```python``` fence and writes via the staging file tool.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from typing import Any

from dana.graph.state import (
    DagTask,
    SupervisorState,
    WorkerState,
    empty_worker_state,
)
from dana.system_health import llm_lock  # global RLock — serialize all LLM gens

WORKER_NODE = "workers"

# Direct-write generator — no tool schemas, no JSON tool_calls.
_STDLIB_EPIC_RULE = (
    "When generating Python code for Epics, rely strictly on Python Standard "
    "Library modules (e.g., json, math, os, sys, deque) unless third-party "
    "packages are explicitly requested in the prompt."
)
WORKER_CODE_SYSTEM_PROMPT = (
    "You are a senior Python engineer. Write the code for the requested task. "
    "You MUST output ONLY the raw code wrapped in a single ```python code block. "
    "Do not provide explanations. "
    "CRITICAL: Unless explicitly instructed otherwise, ALL code must be written in "
    "Python. NEVER output HTML, CSS, or JavaScript. "
    + _STDLIB_EPIC_RULE
)
# Backward-compatible alias (llm_client / tests may still import this name).
WORKER_SYSTEM_PROMPT = WORKER_CODE_SYSTEM_PROMPT
WORKER_DOMAIN_CLAMP = (
    "CRITICAL: You are a Python expert. Unless explicitly instructed otherwise, "
    "ALL code must be written in Python. NEVER output HTML, CSS, or JavaScript. "
    + _STDLIB_EPIC_RULE
)

_READ_RE = re.compile(
    r"\b(?:read|open|inspect|load|explore|outline|survey|map)\b",
    re.I,
)
_WRITE_RE = re.compile(r"\b(?:write|create|save|overwrite)\b", re.I)
_EDIT_RE = re.compile(r"\b(?:edit|refactor|patch|update|modify|append)\b", re.I)
_SYMBOL_RE = re.compile(
    r"\b(?:symbol|definition|def(?:inition)?\s+of|class|function|method)\s+"
    r"[`'\"]?([A-Za-z_]\w*(?:[.:][A-Za-z_]\w*)*)",
    re.I,
)
_SYMBOL_BARE_RE = re.compile(
    r"\b(?:get_symbol_definition|find\s+symbol)\b.*?[`'\"]([A-Za-z_]\w*(?:[.:][A-Za-z_]\w*)*)[`'\"]",
    re.I,
)
_FILE_RE = re.compile(
    r"([\w./\\-]+\.(?:py|pyi|c|cc|cpp|cxx|h|hh|hpp|hxx|inl|md|txt|json|toml|yaml|yml|cfg|ini))\b",
    re.I,
)
_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_HTML_MARKUP_RE = re.compile(
    r"<(?:html|!DOCTYPE|script|style|div|body|head)\b",
    re.I,
)
_CODE_SUFFIXES = frozenset(
    {
        ".py",
        ".pyi",
        ".c",
        ".cc",
        ".cpp",
        ".cxx",
        ".h",
        ".hh",
        ".hpp",
        ".hxx",
        ".inl",
    }
)

# Explore hops still use AST tools (no LLM tool-calling).
WORKER_TOOL_REGISTRY: tuple[str, ...] = (
    "get_file_outline",
    "get_symbol_definition",
    "file_editor",
    "read_local_file",
)

ToolFn = Callable[[str, str, str | None], str]
WorkerFactory = Callable[[DagTask, SupervisorState], WorkerState]
AstOutlineFn = Callable[[str], str]
AstSymbolFn = Callable[[str, str], str]


def _log(msg: str) -> None:
    try:
        from dana.logging import log

        log("DagWorker", msg)
    except Exception:  # noqa: BLE001
        print(f"[DagWorker] {msg}", flush=True)


def _default_file_tool(action: str, filepath: str, content: str | None = None) -> str:
    """Legacy direct editor — prefer ``transactional_file_tool`` in run_worker."""
    from dana.tools.file_editor import file_editor

    return file_editor(action, filepath, content)


def _default_outline(file_path: str) -> str:
    from dana.tools.ast_tools import get_file_outline

    return get_file_outline(file_path)


def _default_symbol(file_path: str, symbol_name: str) -> str:
    from dana.tools.ast_tools import get_symbol_definition

    return get_symbol_definition(file_path, symbol_name)


def _first_filepath(text: str) -> str | None:
    m = _FILE_RE.search(text or "")
    if not m:
        return None
    return m.group(1).replace("\\", "/")


def first_filepath_from_text(text: str) -> str | None:
    """Public helper — first ``*.py`` (etc.) path token in ``text``."""
    return _first_filepath(text)


def _coerce_canonical_filepath(path: str, instructions: str) -> str:
    """Prefer exact product filenames from the prompt over planner renames."""
    p = (path or "").replace("\\", "/")
    mentioned = [
        m.group(1).replace("\\", "/") for m in _FILE_RE.finditer(instructions or "")
    ]
    if (
        p.endswith("token_bucket.py")
        and any(x.endswith("rate_limiter.py") for x in mentioned)
    ):
        return next(x for x in mentioned if x.endswith("rate_limiter.py"))
    return p


def _path_suffix(path: str) -> str:
    p = (path or "").replace("\\", "/").rsplit("/", 1)[-1]
    if "." not in p:
        return ""
    return "." + p.rsplit(".", 1)[-1].lower()


def _is_code_path(path: str) -> bool:
    return _path_suffix(path) in _CODE_SUFFIXES


def _extract_symbol(instructions: str) -> str | None:
    for cre in (_SYMBOL_BARE_RE, _SYMBOL_RE):
        m = cre.search(instructions or "")
        if m:
            return m.group(1).strip()
    return None


def _worker_llm_enabled() -> bool:
    raw = (os.environ.get("DONNA_WORKER_LLM") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _content_is_htmlish(content: str | None) -> bool:
    body = content or ""
    if _HTML_MARKUP_RE.search(body):
        return True
    if "```html" in body.lower() or "```css" in body.lower():
        return True
    return False


def with_explicit_path_passthrough(
    action: str,
    tool_name: str = "file_editor",
) -> str:
    """Inject an explicit filepath into worker instructions from the Supervisor."""
    text = (action or "").strip()
    path = _first_filepath(text)
    tool = (tool_name or "file_editor").strip() or "file_editor"
    if not path:
        return text
    if re.search(r"\bTARGET FILEPATH:\b", text, re.I):
        return text
    will_mutate = bool(_WRITE_RE.search(text) or _EDIT_RE.search(text))
    is_explore = bool(_READ_RE.search(text)) and not will_mutate
    if is_explore or tool in {
        "get_file_outline",
        "get_symbol_definition",
        "read_local_file",
    }:
        return f"TARGET FILEPATH: {path}. {text}"
    if tool in {"file_editor", "write_to_file"} or will_mutate:
        return f"TARGET FILEPATH: {path}. Write Python code for this file. {text}"
    return text


def extract_code_from_llm_response(llm_response: str) -> str:
    """Pull the first fenced code block; else return the raw response body."""
    raw = llm_response if isinstance(llm_response, str) else str(llm_response or "")
    m = _CODE_BLOCK_RE.search(raw)
    if m:
        return (m.group(1) or "").strip("\n")
    return raw.strip()


def _expand_compact_python(code: str) -> str:
    """Best-effort newlines when a 3B model emits an entire module as one line."""
    s = (code or "").strip()
    if not s or "\n" in s:
        return code
    if not re.search(r"\b(?:def|class)\s+\w+", s):
        return code
    s = re.sub(r"\b(import\s+[A-Za-z_][\w.]*(?:\s+as\s+\w+)?)\s+(?=class\b|def\b)", r"\1\n", s)
    s = re.sub(r"\b(from\s+[A-Za-z_][\w.]*\s+import\s+[^\n]+?)\s+(?=class\b|def\b)", r"\1\n", s)
    s = re.sub(r"\s+(?=class\s+\w+)", "\n\n", s)
    s = re.sub(r"(?<=[;])\s*", "\n", s)
    # Split compacted method/body keywords after a colon.
    s = re.sub(
        r":\s+(?=def\s|class\s|return\b|if\s|for\s|while\s|with\s|try\b|self\.|root\b|canvas\b|ball\b)",
        ":\n    ",
        s,
    )
    s = re.sub(r"\s+(?=def\s+\w+)", "\n    ", s)
    return s + ("\n" if not s.endswith("\n") else "")


def _align_imports_to_target(path: str, code: str) -> str:
    """Keep test/impl API locked to ``TokenBucket`` on rate_limiter paths."""
    rel = (path or "").replace("\\", "/").lower()
    body = code
    if rel.endswith("test_rate_limiter.py") or "/test_rate_limiter.py" in rel:
        body = re.sub(
            r"\bfrom\s+token_bucket\s+import\b",
            "from rate_limiter import",
            body,
        )
        body = re.sub(r"\bimport\s+token_bucket\b", "import rate_limiter", body)
        body = re.sub(
            r"\bfrom\s+rate_limiter\s+import\s+Product\b",
            "from rate_limiter import TokenBucket",
            body,
        )
        body = re.sub(r"\bProduct\b", "TokenBucket", body)
    elif rel.endswith("rate_limiter.py"):
        body = re.sub(r"\bclass\s+Product\b", "class TokenBucket", body)
        body = re.sub(r"\bProduct\(", "TokenBucket(", body)
    return body


def _sibling_context_for_path(path: str) -> str:
    """Inject a short sibling-file sketch so tests/impl share one API."""
    from pathlib import Path

    from dana.paths import PROJECT_ROOT

    rel = (path or "").replace("\\", "/")
    root = Path(PROJECT_ROOT)
    siblings: list[Path] = []
    if rel.endswith("test_rate_limiter.py"):
        siblings.append(root / "rate_limiter.py")
    elif rel.endswith("rate_limiter.py"):
        siblings.append(root / "tests" / "test_rate_limiter.py")
    chunks: list[str] = []
    for sib in siblings:
        if not sib.is_file():
            continue
        try:
            text = sib.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lines: list[str] = []
        for line in text.splitlines():
            s = line.strip()
            if s.startswith(("import ", "from ", "class ", "def ", "async def ")):
                lines.append(line.rstrip())
            if len(lines) >= 40:
                break
        if lines:
            chunks.append(
                f"SIBLING FILE `{sib.relative_to(root).as_posix()}` "
                f"(match this API exactly):\n```python\n"
                + "\n".join(lines)
                + "\n```"
            )
    return "\n\n".join(chunks)


def _strip_import_time_side_effects(code: str) -> str:
    """Drop trailing demo / ``while True`` loops that hang ``import`` / pytest."""
    lines = (code or "").splitlines()
    cut = len(lines)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r"^#\s*example\b", stripped, re.I):
            cut = i
            break
        if re.match(r"^if\s+__name__\s*==\s*['\"]__main__['\"]\s*:", stripped):
            cut = i
            break
        if re.match(r"^while\s+True\s*:", stripped):
            cut = i
            break
    body = "\n".join(lines[:cut]).rstrip()
    return body + ("\n" if body else "")


def extract_and_save_code(
    llm_response: str,
    target_filepath: str,
    *,
    tool_fn: ToolFn | None = None,
    staging_session_id: str | None = None,
) -> str:
    """Regex-extract code from ``llm_response`` and persist under staging scratch.

    Writes via ``tool_fn`` / ``file_editor`` staging so content lands in
    ``.dana_scratch/<session>/<target_filepath>`` (committed by the supervisor).
    Falls back to a direct ``.dana_scratch/<target_filepath>`` write when no
    staging session is available.
    """
    path = (target_filepath or "").strip().replace("\\", "/")
    if not path:
        return "ERROR: empty target_filepath"

    code = extract_code_from_llm_response(llm_response)
    if not code.strip():
        return "ERROR: empty code after extraction"

    if _is_code_path(path):
        code = _expand_compact_python(code)
        code = _align_imports_to_target(path, code)
        code = _strip_import_time_side_effects(code)

    if _is_code_path(path) and _content_is_htmlish(code):
        return (
            "ERROR: refused HTML/CSS/JS content for Python path "
            f"{path}; expected Python source"
        )

    if tool_fn is not None:
        return tool_fn("write", path, code)

    sid = str(staging_session_id or "").strip()
    if sid:
        from dana.tools.file_editor import file_editor

        return file_editor("write", path, code, staging_session=sid)

    # Direct scratch mirror (no active DAG staging session).
    from pathlib import Path

    from dana.exec.shadow_workspace import default_scratch_base
    from dana.paths import PROJECT_ROOT

    rel = Path(path)
    if rel.is_absolute():
        try:
            rel = rel.resolve().relative_to(Path(PROJECT_ROOT).resolve())
        except ValueError:
            rel = Path(rel.name)
    dest = default_scratch_base() / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(code, encoding="utf-8")
    return f"OK: wrote {len(code)} chars to .dana_scratch/{rel.as_posix()}"


def _worker_should_escalate(repair_attempts: int) -> bool:
    """Senior-dev fallback: cloud after N local repair failures when hybrid is on."""
    if (os.environ.get("DONNA_FORCE_LOCAL") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return False
    try:
        threshold = int(os.environ.get("DONNA_WORKER_ESCALATE_AFTER") or "3")
    except ValueError:
        threshold = 3
    if threshold < 1:
        threshold = 3
    if int(repair_attempts or 0) < threshold:
        return False
    try:
        from dana.settings import is_hybrid_planner_enabled

        if not is_hybrid_planner_enabled():
            return False
    except Exception:  # noqa: BLE001
        return False
    try:
        from dana.graph.cloud_planner import hybrid_cloud_planner_active

        return bool(hybrid_cloud_planner_active())
    except Exception:  # noqa: BLE001
        return False


def _cloud_error_allows_ollama_fallback(exc: BaseException) -> bool:
    """True only for hard client errors (400/401) — not throttling."""
    try:
        import requests

        if isinstance(exc, requests.HTTPError) and exc.response is not None:
            return int(exc.response.status_code) in {400, 401}
    except Exception:  # noqa: BLE001
        pass
    text = str(exc or "")
    if re.search(r"\b401\b|Unauthorized", text, re.I):
        return True
    if re.search(r"\b400\b|Bad Request", text, re.I):
        return True
    return False


def generate_worker_code(
    instructions: str,
    target_filepath: str,
    *,
    model: str | None = None,
    repair_attempts: int = 0,
) -> str:
    """Plain string LLM call — no tools, no JSON schema, no ReAct loop."""
    from dana.core_agent import OLLAMA_MODEL, ask_ollama_messages

    path = (target_filepath or "").strip()
    stem = path.replace("\\", "/").rsplit("/", 1)[-1]
    import_hint = ""
    if stem.startswith("test_") and stem.endswith(".py"):
        mod = stem[len("test_") : -len(".py")]
        if mod:
            import_hint = (
                f"\nIf this is a pytest file, import the product module as "
                f"`from {mod} import ...` (never invent alternate module names)."
            )
    api_lock = ""
    if "rate_limiter" in path.replace("\\", "/").lower():
        api_lock = (
            "\nAPI LOCK (mandatory): The public class name is TokenBucket "
            "(never Product). Tests must use `from rate_limiter import TokenBucket`. "
            "Implementation must define `class TokenBucket`. "
            "Do not add module-level while-True loops, demos, or prints."
        )
    sibling = _sibling_context_for_path(path)
    sibling_block = f"\n\n{sibling}" if sibling else ""
    user = (
        f"TARGET FILEPATH: {path}\n"
        f"TASK:\n{instructions}\n\n"
        "Output ONLY one ```python``` code block with the full file contents. "
        "Use real newlines and indentation — never emit the whole file as one line. "
        f"The file path is exactly {path}; do not invent a different filename."
        f"{import_hint}{api_lock}{sibling_block}"
    )
    messages = [
        {"role": "system", "content": WORKER_CODE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    escalate = _worker_should_escalate(repair_attempts)
    _log(
        f"direct-write generator CALL path={path!r} chars={len(instructions)} "
        f"repair_attempts={int(repair_attempts or 0)} escalate={escalate}"
    )
    # Global LLM lock — never overlap generations (VRAM safety).
    with llm_lock:
        if escalate:
            print(
                "[Worker Escalation] Local model failed 3 times. Escalating to Cloud API.",
                flush=True,
            )
            from dana.graph.cloud_planner import ask_gemini_text

            try:
                raw = ask_gemini_text(
                    messages,
                    temperature=0.1,
                    max_output_tokens=4096,
                    response_mime_type=None,
                )
            except Exception as cloud_exc:  # noqa: BLE001
                # Only hard client errors fall back to Ollama; 429/503 are retried
                # inside ask_gemini_text (exponential backoff).
                if not _cloud_error_allows_ollama_fallback(cloud_exc):
                    err_s = str(cloud_exc)
                    err_s = re.sub(r"([?&]key=)[^&\s]+", r"\1***", err_s, flags=re.I)
                    print(
                        f"[Worker Escalation] Cloud API failed without Ollama fallback "
                        f"({type(cloud_exc).__name__}: {err_s})",
                        flush=True,
                    )
                    raise
                err_s = str(cloud_exc)
                err_s = re.sub(r"([?&]key=)[^&\s]+", r"\1***", err_s, flags=re.I)
                print(
                    f"[Worker Escalation] Cloud API client error "
                    f"({type(cloud_exc).__name__}: {err_s}); "
                    f"falling back to local Ollama.",
                    flush=True,
                )
                model_id = (model or "").strip() or OLLAMA_MODEL
                raw = ask_ollama_messages(
                    messages,
                    model=model_id,
                    num_predict=4096,
                    temperature=0.1,
                )
        else:
            model_id = (model or "").strip() or OLLAMA_MODEL
            raw = ask_ollama_messages(
                messages,
                model=model_id,
                num_predict=4096,
                temperature=0.1,
            )
    text = raw if isinstance(raw, str) else str(raw or "")
    _log(f"direct-write generator RESPONSE chars={len(text)} escalate={escalate}")
    return text


def _dense_summary(task_id: int, instructions: str, outputs: list[dict[str, Any]]) -> str:
    bits: list[str] = [f"task {task_id}: {instructions[:160]}"]
    for row in outputs:
        tool = row.get("tool") or "tool"
        obs = str(row.get("output") or "")
        snippet = obs.replace("\n", " ").strip()
        if len(snippet) > 220:
            snippet = snippet[:220] + "…"
        bits.append(f"{tool}→{snippet}")
    return " | ".join(bits)


def build_isolated_worker(task: DagTask, supervisor: SupervisorState) -> WorkerState:
    """Spawn a worker with a fresh context window (no global chat history)."""
    action = with_explicit_path_passthrough(
        str(task.get("action") or ""),
        str(task.get("tool_name") or "file_editor"),
    )
    # Prefer exact filenames from the supervisor/user prompt over planner renames
    # (e.g. token_bucket.py → rate_limiter.py).
    prompt = str((supervisor or {}).get("user_prompt") or "")
    path = _first_filepath(action)
    if path:
        canon = _coerce_canonical_filepath(path, f"{action}\n{prompt}")
        if canon != path:
            action = action.replace(path, canon)
            if "TARGET FILEPATH:" not in action:
                action = f"TARGET FILEPATH: {canon}. {action}"
            else:
                action = re.sub(
                    r"TARGET FILEPATH:\s*\S+",
                    f"TARGET FILEPATH: {canon}",
                    action,
                    count=1,
                    flags=re.I,
                )
    return empty_worker_state(int(task["task_id"]), action)


def _explore_with_ast(
    path: str,
    instructions: str,
    outputs: list[dict[str, Any]],
    *,
    outline_fn: AstOutlineFn,
    symbol_fn: AstSymbolFn,
) -> None:
    """Prefer structural outline / symbol nav over ``read_local_file`` dumps."""
    symbol = _extract_symbol(instructions)
    if symbol:
        obs = symbol_fn(path, symbol)
        outputs.append(
            {
                "tool": "get_symbol_definition",
                "filepath": path,
                "symbol": symbol,
                "output": obs,
            }
        )
        if str(obs).startswith("ERROR:"):
            outline = outline_fn(path)
            outputs.append(
                {"tool": "get_file_outline", "filepath": path, "output": outline}
            )
        return

    outline = outline_fn(path)
    outputs.append({"tool": "get_file_outline", "filepath": path, "output": outline})


def run_worker(
    worker: WorkerState,
    *,
    tool_fn: ToolFn | None = None,
    edit_content: str | None = None,
    outline_fn: AstOutlineFn | None = None,
    symbol_fn: AstSymbolFn | None = None,
    staging_session_id: str | None = None,
    repair_attempts: int = 0,
) -> WorkerState:
    """Execute one isolated worker hop.

    Mutating code tasks use the Deterministic Extraction Worker (plain LLM →
    regex extract → staging write). Explore hops use AST tools only. Unit-test
    overrides (``tool_fn`` / ``edit_content``) keep the deterministic stub path.
    """
    from dana.tools.file_editor import rollback_workspace, transactional_file_tool

    outline = outline_fn or _default_outline
    symbol = symbol_fn or _default_symbol
    instructions = str(worker.get("instructions") or "")
    path = _first_filepath(instructions)
    if path:
        path = _coerce_canonical_filepath(path, instructions)
    context = list(worker.get("context_window") or [])
    outputs = list(worker.get("tool_outputs") or [])
    tid = int(worker.get("task_id") or 0)
    attempts = int(repair_attempts or 0)
    will_mutate = bool(
        _WRITE_RE.search(instructions) or _EDIT_RE.search(instructions)
    )
    sid = str(staging_session_id or worker.get("staging_session_id") or "").strip()
    if tool_fn is None and will_mutate:
        sid = sid or f"dag-worker-{tid}"
        tools: ToolFn = transactional_file_tool(sid)
    else:
        tools = tool_fn or _default_file_tool
        if tool_fn is not None:
            sid = ""

    context.append(
        {
            "role": "system",
            "content": (
                "Isolated Deterministic Extraction Worker. No ReAct tool loop. "
                f"{WORKER_DOMAIN_CLAMP} "
                + (f"Staging session: {sid}." if sid else "")
            ),
        }
    )
    context.append({"role": "user", "content": instructions})

    if not path:
        err = "no filepath found in task instructions"
        context.append({"role": "assistant", "content": err})
        if sid:
            rollback_workspace(sid)
        return {
            **worker,
            "context_window": context,
            "tool_outputs": outputs,
            "summary": "",
            "status": "failed",
            "staging_session_id": sid,
            "error": err,
        }

    try:
        is_explore = bool(_READ_RE.search(instructions)) and not will_mutate
        use_direct_write = (
            will_mutate
            and edit_content is None
            and tool_fn is None
            and _worker_llm_enabled()
        )
        if is_explore and _is_code_path(path):
            _explore_with_ast(
                path,
                instructions,
                outputs,
                outline_fn=outline,
                symbol_fn=symbol,
            )
        elif use_direct_write:
            # Plain LLM → regex extract → Python saves (no tool_calls).
            import ast

            llm_raw = generate_worker_code(
                instructions, path, repair_attempts=attempts
            )
            context.append({"role": "assistant", "content": llm_raw[:4000]})
            code_probe = extract_code_from_llm_response(llm_raw)
            if _is_code_path(path):
                code_probe = _expand_compact_python(code_probe)
                code_probe = _align_imports_to_target(path, code_probe)
                try:
                    ast.parse(code_probe)
                except SyntaxError as syn_exc:
                    _log(f"syntax retry after {syn_exc}")
                    llm_raw = generate_worker_code(
                        instructions
                        + "\n\nPREVIOUS OUTPUT HAD A SYNTAX ERROR. "
                        "Emit valid Python with real newlines only.",
                        path,
                        repair_attempts=attempts,
                    )
                    context.append({"role": "assistant", "content": llm_raw[:4000]})
            obs = extract_and_save_code(
                llm_raw,
                path,
                tool_fn=tools,
                staging_session_id=sid,
            )
            outputs.append(
                {
                    "tool": "file_editor.write",
                    "filepath": path,
                    "output": obs,
                    "extractor": "deterministic",
                }
            )
            if str(obs).startswith("ERROR:"):
                if sid:
                    rollback_workspace(sid)
                return {
                    **worker,
                    "context_window": context,
                    "tool_outputs": outputs,
                    "summary": "",
                    "status": "failed",
                    "staging_session_id": sid,
                    "error": obs,
                }
        elif will_mutate:
            # Hermetic / injected-content path (tests): no LLM.
            if _is_code_path(path):
                _explore_with_ast(
                    path,
                    instructions,
                    outputs,
                    outline_fn=outline,
                    symbol_fn=symbol,
                )
            prior = tools("read", path, None)
            outputs.append(
                {"tool": "file_editor.read", "filepath": path, "output": prior}
            )
            body = edit_content
            if body is None:
                if prior.startswith("OK:") and "\n" in prior:
                    existing = prior.split("\n", 1)[1]
                else:
                    existing = ""
                marker = (
                    f"\n# DAG-worker task {worker.get('task_id')}: "
                    f"{instructions[:80]}\n"
                )
                if _WRITE_RE.search(instructions) and not existing.strip():
                    body = f"# created by DAG worker\n{marker}"
                else:
                    body = existing + marker
            obs = tools("write", path, body)
            outputs.append(
                {"tool": "file_editor.write", "filepath": path, "output": obs}
            )
        elif _is_code_path(path):
            _explore_with_ast(
                path,
                instructions,
                outputs,
                outline_fn=outline,
                symbol_fn=symbol,
            )
        else:
            obs = tools("read", path, None)
            outputs.append(
                {"tool": "read_local_file", "filepath": path, "output": obs}
            )
    except Exception as exc:  # noqa: BLE001
        err = f"worker tool failure: {exc}"
        context.append({"role": "assistant", "content": err})
        if sid:
            rollback_workspace(sid)
        return {
            **worker,
            "context_window": context,
            "tool_outputs": outputs,
            "summary": "",
            "status": "failed",
            "staging_session_id": sid,
            "error": err,
        }

    failed = any(str(o.get("output") or "").startswith("ERROR:") for o in outputs)
    if failed and any(
        (
            str(o.get("tool") or "").startswith("file_editor.write")
            or str(o.get("tool") or "").startswith("file_editor.append")
        )
        and not str(o.get("output") or "").startswith("ERROR:")
        for o in outputs
    ):
        failed = any(
            (
                str(o.get("tool") or "").startswith("file_editor.write")
                or str(o.get("tool") or "").startswith("file_editor.append")
            )
            and str(o.get("output") or "").startswith("ERROR:")
            for o in outputs
        )
    summary = _dense_summary(tid, instructions, outputs)
    context.append({"role": "assistant", "content": summary})
    return {
        **worker,
        "context_window": context,
        "tool_outputs": outputs,
        "summary": "" if failed else summary,
        "status": "failed" if failed else "completed",
        "staging_session_id": sid,
        "error": "tool returned ERROR" if failed else "",
    }


_REPAIR_ATTEMPTS_RE = re.compile(r"REPAIR_ATTEMPTS:\s*(\d+)", re.I)


def _repair_attempts_from_state(state: SupervisorState) -> int:
    """Read epic.repair_attempts (harness-bumped) for Worker Escalation."""
    fb = state.get("runtime_feedback") or {}
    if isinstance(fb, dict) and fb.get("repair_attempts") is not None:
        try:
            return max(0, int(fb.get("repair_attempts") or 0))
        except (TypeError, ValueError):
            pass
    epics = state.get("epics") or []
    try:
        idx = int(state.get("active_epic_index") or 0)
    except (TypeError, ValueError):
        idx = 0
    if 0 <= idx < len(epics):
        try:
            n = int((epics[idx] or {}).get("repair_attempts") or 0)
            if n:
                return max(0, n)
        except (TypeError, ValueError):
            pass
    # Belt-and-suspenders: broker stamps REPAIR_ATTEMPTS into the repair prompt.
    for blob in (
        str(state.get("user_prompt") or ""),
        str((state.get("runtime_feedback") or {}).get("stderr") or "")
        if isinstance(state.get("runtime_feedback"), dict)
        else "",
    ):
        m = _REPAIR_ATTEMPTS_RE.search(blob)
        if m:
            try:
                return max(0, int(m.group(1)))
            except (TypeError, ValueError):
                pass
    return 0


def workers_node(
    state: SupervisorState,
    *,
    tool_fn: ToolFn | None = None,
    worker_factory: WorkerFactory | None = None,
    outline_fn: AstOutlineFn | None = None,
    symbol_fn: AstSymbolFn | None = None,
) -> dict[str, Any]:
    """Run all ``active_task_ids`` in isolation; return summaries to supervisor."""
    factory = worker_factory or build_isolated_worker
    by_id = {int(t["task_id"]): t for t in (state.get("dag") or [])}
    results: list[dict[str, Any]] = []
    repair_attempts = _repair_attempts_from_state(state)
    if repair_attempts:
        print(
            f"[Worker Escalation] probe repair_attempts={repair_attempts} "
            f"(escalate={_worker_should_escalate(repair_attempts)})",
            flush=True,
        )

    _ = state.get("global_conversation_history")

    for tid in state.get("active_task_ids") or []:
        task = by_id.get(int(tid))
        if task is None:
            results.append(
                {
                    "task_id": int(tid),
                    "status": "failed",
                    "summary": "",
                    "error": "unknown task_id",
                    "context_window": [],
                }
            )
            continue
        worker = factory(task, state)
        if worker.get("context_window"):
            worker = {**worker, "context_window": []}
        finished = run_worker(
            worker,
            tool_fn=tool_fn,
            outline_fn=outline_fn,
            symbol_fn=symbol_fn,
            repair_attempts=repair_attempts,
        )
        _log(
            f"task {tid} → {finished.get('status')} "
            f"(ctx_turns={len(finished.get('context_window') or [])})"
        )
        results.append(
            {
                "task_id": int(tid),
                "status": finished.get("status"),
                "summary": finished.get("summary") or "",
                "error": finished.get("error") or "",
                "staging_session_id": finished.get("staging_session_id") or "",
                "context_window": list(finished.get("context_window") or []),
                "tool_outputs": list(finished.get("tool_outputs") or []),
            }
        )

    open_sessions = [
        str(r.get("staging_session_id") or "").strip()
        for r in results
        if str(r.get("staging_session_id") or "").strip()
    ]
    prev_open = [
        str(s) for s in (state.get("open_staging_sessions") or []) if str(s).strip()
    ]
    return {
        "worker_results": results,
        "status": "evaluating",
        "active_task_ids": [],
        "open_staging_sessions": prev_open + open_sessions,
    }


def make_workers_node(
    tool_fn: ToolFn | None = None,
    worker_factory: WorkerFactory | None = None,
    outline_fn: AstOutlineFn | None = None,
    symbol_fn: AstSymbolFn | None = None,
) -> Callable[[SupervisorState], dict[str, Any]]:
    def _node(state: SupervisorState) -> dict[str, Any]:
        return workers_node(
            state,
            tool_fn=tool_fn,
            worker_factory=worker_factory,
            outline_fn=outline_fn,
            symbol_fn=symbol_fn,
        )

    return _node


__all__ = (
    "WORKER_CODE_SYSTEM_PROMPT",
    "WORKER_DOMAIN_CLAMP",
    "WORKER_NODE",
    "WORKER_SYSTEM_PROMPT",
    "WORKER_TOOL_REGISTRY",
    "build_isolated_worker",
    "extract_and_save_code",
    "extract_code_from_llm_response",
    "first_filepath_from_text",
    "generate_worker_code",
    "make_workers_node",
    "run_worker",
    "with_explicit_path_passthrough",
    "workers_node",
)
