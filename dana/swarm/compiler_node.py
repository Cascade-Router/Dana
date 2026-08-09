"""Meta-Planner compiler_node — Whisper transcript → engineering-grade ReAct prompt.

Uses the low-latency base llama3.2 model as a Systems Engineering Meta-Planner.
"""

from __future__ import annotations

from dana.cascade_router import local_model_name
from dana.paths import DANA_WORKSPACE

_COMPILER_SYSTEM_PROMPT = f"""You are a Systems Engineering Meta-Planner for the Dana assistant.

Your job: rewrite a vague voice transcript into ONE high-level architectural
ticket prompt for the draft_cursor_prompt / self-improvement pipeline.
Output ONLY that prompt string — no preamble, no markdown fences, no explanation.

STRICT OUTPUT TEMPLATE (non-negotiable):
- The output MUST ALWAYS start with the exact phrase:
  "Dana, use the draft_cursor_prompt tool to log a self-improvement ticket to..."
- After that opening phrase, continue with a high-level architectural objective
  and context only (what to change and why), never an implementation dump.
- NEVER output raw code, shell commands, or file names as the primary string.
- Keep semantic routing keywords present so the DeepSeek MoA router is always
  triggered: include "self-improvement" (already in the required opening) and,
  when relevant to the intent, also mention "deepseek".
- Do not invent executable patches; only describe the architectural ticket.

Hard path constraints (when paths are discussed in context, not as primary output):
- Any referenced paths MUST target the active workspace under: {DANA_WORKSPACE}
- Known workspace subdirs include execution_jail/, dana/, dana_security/,
  tracker/, logs/, custom_tools/, cursor_handoffs/, captures/.
- NEVER assume a standard repository root layout (no /usr/src, no ~/projects,
  no generic "repo root" placeholders).
"""


def compile_voice_to_prompt(raw_transcript: str) -> str:
    """Compile a raw Whisper transcript into a constraint-bound ReAct prompt.

    Calls the low-latency base llama3.2 model. Raises on model/transport failure
    so callers can fall back to the raw transcript.
    """
    raw = (raw_transcript or "").strip()
    if not raw:
        return raw

    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage, SystemMessage

    model = local_model_name()
    llm = ChatOllama(model=model, temperature=0.1, num_predict=512, num_ctx=4096)
    response = llm.invoke(
        [
            SystemMessage(content=_COMPILER_SYSTEM_PROMPT),
            HumanMessage(content=f"Voice transcript to compile:\n{raw}"),
        ]
    )
    compiled = getattr(response, "content", None)
    if compiled is None:
        compiled = str(response)
    text = str(compiled or "").strip()
    if not text:
        raise RuntimeError("compiler_node returned empty prompt")
    return text
