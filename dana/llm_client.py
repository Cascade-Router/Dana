"""Local LLM client helpers — ChatOllama + speculative-decoding knobs.

Also provides schema-constrained Ollama calls (``format`` JSON Schema) with
automatic JSON retry-parser middleware for small-model tool pipelines.

Primary runtime is Ollama via LangChain ``ChatOllama``. Speculative decoding
is backend-dependent:

* **Ollama (Modelfile)** — pair a draft model at create time, then tune
  ``draft_num_predict`` in request ``options``::

      # Modelfile
      FROM llama3.2
      DRAFT llama3.2:1b
      PARAMETER draft_num_predict 4

      ollama create dana-spec -f Modelfile

* **llama.cpp server** (GGUF path; not used by default in DANA)::

      ./llama-server -m <main.gguf> -md <draft.gguf> --draft 4
      # or: --model-draft <path_to_draft_model>

Env knobs:
  DANA_SPECULATIVE_DECODING  — ``1``/``true`` enables draft_num_predict injection
  DANA_DRAFT_MODEL           — draft tag/path (default ``llama3.2:1b``)
  DANA_DRAFT_NUM_PREDICT     — draft tokens per step (default ``4``; ``0`` disables)
"""

from __future__ import annotations

import os
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def speculative_decoding_enabled() -> bool:
    raw = (os.environ.get("DANA_SPECULATIVE_DECODING") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def draft_model_name() -> str:
    return (
        (os.environ.get("DANA_DRAFT_MODEL") or "").strip()
        or "llama3.2:1b"
    )


def draft_num_predict() -> int | None:
    """Return draft depth when speculative decoding is on; else None.

    ``DANA_DRAFT_NUM_PREDICT=0`` explicitly disables drafting even when the
    feature flag is set (useful for A/B TPS benchmarks).
    """
    if not speculative_decoding_enabled():
        return None
    raw = (os.environ.get("DANA_DRAFT_NUM_PREDICT") or "4").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 4
    if n < 0:
        n = 4
    return n


def speculative_options() -> dict[str, Any]:
    """Ollama ``options`` fragment for speculative drafting (may be empty)."""
    n = draft_num_predict()
    if n is None:
        return {}
    return {"draft_num_predict": n}


def speculative_launch_notes() -> str:
    """Human-readable server-side flags for operators / benchmarks."""
    draft = draft_model_name()
    n = draft_num_predict()
    depth = 4 if n is None else n
    return (
        "Speculative decoding is primarily a server/model packaging concern.\n"
        f"  Draft model: {draft}\n"
        f"  draft_num_predict: {depth}\n"
        "\n"
        "Ollama Modelfile:\n"
        "  FROM <main_model>\n"
        f"  DRAFT {draft}\n"
        f"  PARAMETER draft_num_predict {depth}\n"
        "  # then: ollama create dana-spec -f Modelfile\n"
        "\n"
        "llama.cpp (when not using Ollama):\n"
        f"  ./llama-server -m <main.gguf> -md <path_to_{draft.replace(':', '_')}.gguf> "
        f"--draft {depth}\n"
        "  # alias: --model-draft <path_to_draft_model>\n"
    )


def merge_ollama_options(base: dict[str, Any] | None = None) -> dict[str, Any]:
    """Merge caller options with speculative ``draft_num_predict`` when enabled."""
    out: dict[str, Any] = dict(base or {})
    out.update(speculative_options())
    return out


class _SpeculativeChatOllama:
    """Composition wrapper that merges ``draft_num_predict`` into Ollama options.

    Wraps rather than subclasses ``ChatOllama``: tests monkeypatch
    ``langchain_ollama.ChatOllama`` to a plain factory function (see
    ``tests/test_support_react.py::patch_scripted_llm``), and ``class Foo(ChatOllama):``
    raises ``TypeError`` when ``ChatOllama`` is a function rather than a class.
    Composition works regardless of whether ``inner`` is a real ``ChatOllama``
    instance or a test double.

    The draft-token option can't be injected via delegated ``invoke``/``stream``
    calls alone: ``bind_tools()`` returns a LangChain ``Runnable`` bound directly
    to ``inner``, bypassing this wrapper entirely on the eventual generate call.
    So the option merge is patched onto ``inner._chat_params`` itself (the hook
    every ChatOllama generate/stream path already calls through) — that patch
    stays live no matter which object callers end up invoking.
    """

    def __init__(self, inner: Any, draft_n: int) -> None:
        self._inner = inner
        self._draft_n = draft_n
        original_chat_params = getattr(inner, "_chat_params", None)
        if callable(original_chat_params):

            def _chat_params_with_draft(
                messages: Any, stop: list[str] | None = None, **kw: Any
            ) -> dict[str, Any]:
                params = original_chat_params(messages, stop=stop, **kw)
                opts = dict(params.get("options") or {})
                opts["draft_num_predict"] = draft_n
                params["options"] = opts
                return params

            inner._chat_params = _chat_params_with_draft

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.invoke(*args, **kwargs)

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        return await self._inner.ainvoke(*args, **kwargs)

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.stream(*args, **kwargs)

    async def astream(self, *args: Any, **kwargs: Any) -> Any:
        async for chunk in self._inner.astream(*args, **kwargs):
            yield chunk

    def bind_tools(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.bind_tools(*args, **kwargs)

    def with_structured_output(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.with_structured_output(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # Full Runnable/BaseChatModel duck-typing fallback (batch, abatch,
        # with_config, ...) for anything not explicitly delegated above.
        return getattr(self._inner, name)


def build_chat_ollama(
    *,
    model: str,
    temperature: float = 0.2,
    num_ctx: int | None = 8192,
    num_predict: int | None = None,
    keep_alive: str | int | None = None,
    **kwargs: Any,
) -> Any:
    """Construct ``ChatOllama`` with optional speculative ``draft_num_predict``.

    LangChain's ``ChatOllama`` has no first-class ``draft_model`` field; the
    draft pairing is done via Modelfile ``DRAFT`` / llama.cpp ``-md``. When
    ``DANA_SPECULATIVE_DECODING=1``, we inject ``draft_num_predict`` into
    request options on every chat call.
    """
    from langchain_ollama import ChatOllama

    draft_n = draft_num_predict()

    init: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        **kwargs,
    }
    if num_ctx is not None:
        init["num_ctx"] = num_ctx
    if num_predict is not None:
        init["num_predict"] = num_predict
    if keep_alive is not None:
        init["keep_alive"] = keep_alive
    else:
        # Zero-latency default — unload after each ChatOllama call.
        init["keep_alive"] = 0

    inner = ChatOllama(**init)
    if draft_n is None:
        return inner
    return _SpeculativeChatOllama(inner, draft_n)


def ollama_format_for_model(model: type[BaseModel] | dict[str, Any] | str) -> Any:
    """Value for Ollama ``format`` — JSON Schema dict, ``\"json\"``, or passthrough."""
    if isinstance(model, str):
        return model
    if isinstance(model, dict):
        return model
    from dana.llm_schemas import schema_for_model

    return schema_for_model(model)


def ask_ollama_structured(
    messages: list[dict[str, str]],
    response_model: type[T],
    *,
    model: str | None = None,
    num_predict: int | None = None,
    max_retries: int = 3,
    temperature: float | None = 0.0,
) -> T:
    """Schema-constrained chat with up to ``max_retries`` JSON parse retries.

    Validates each attempt with ``parse_with_schema_retry``. Previously
    passed the Pydantic JSON Schema via Ollama's native ``format`` kwarg
    through the now-removed ``dana.core.agent_loop``; routed through
    ``ModelProvider`` (the OpenAI-wire bridge) instead, which has no
    equivalent schema-constraint hook — ``parse_with_schema_retry``'s
    retry-on-malformed-JSON loop covers the gap.
    """
    from dana.core.constants import OLLAMA_MODEL
    from dana.core.model_provider import ModelProvider
    from dana.middleware.json_schema_retry import parse_with_schema_retry

    model_id = (model or "").strip() or OLLAMA_MODEL

    def _invoke(msgs: list[dict[str, str]]) -> str:
        return ModelProvider(local_model=model_id).complete(
            msgs,
            num_predict=num_predict or 512,
            temperature=temperature or 0.0,
            allow_cloud=False,
        )

    return parse_with_schema_retry(
        messages,
        response_model,
        invoke=_invoke,
        max_retries=max_retries,
    )


def ask_planner_structured(
    messages: list[dict[str, str]],
    response_model: type[T],
    *,
    model: str | None = None,
    num_predict: int | None = None,
    max_retries: int = 3,
    temperature: float | None = 0.0,
    allow_cloud: bool = True,
) -> T:
    """Route high-level planning through local Ollama.

    The hybrid cloud-structured-planning path this used to try first
    (``dana.graph.cloud_planner.ask_cloud_structured``) was removed with the
    legacy LangGraph stack; ``allow_cloud``/``DANA_FORCE_LOCAL`` are accepted
    for call-site compatibility but no longer change behavior — every call
    now goes straight to local Ollama. Workers must continue calling
    ``ask_ollama_structured`` / local tools only.
    """
    return ask_ollama_structured(
        messages,
        response_model,
        model=model,
        num_predict=num_predict,
        max_retries=max_retries,
        temperature=temperature,
    )


def ask_supervisor_dag_plan(
    prompt: str,
    *,
    model: str | None = None,
    max_retries: int = 3,
) -> Any:
    """Structured supervisor DAGPlan for a complex multi-file prompt."""
    from dana.llm_schemas import DAGPlan

    messages = [
        {
            "role": "system",
            "content": (
                "You are the DAG supervisor. Return ONLY JSON matching DAGPlan: "
                '{"tasks":[{"task_id":int,"action":str,"dependencies":[int,...]},...]}.'
            ),
        },
        {"role": "user", "content": prompt},
    ]
    return ask_ollama_structured(
        messages, DAGPlan, model=model, max_retries=max_retries
    )


def ask_worker_tool_plan(
    instructions: str,
    *,
    model: str | None = None,
    max_retries: int = 3,
) -> Any:
    """Legacy structured WorkerToolPlan helper (DAG workers no longer use this).

    Live DAG code hops use ``generate_worker_code`` + ``extract_and_save_code``
    instead of JSON tool_calls.
    """
    from dana.llm_schemas import WorkerToolPlan

    messages = [
        {
            "role": "system",
            "content": (
                "You are an isolated DAG worker. Return ONLY JSON matching "
                "WorkerToolPlan with tool_calls from: get_file_outline, "
                "get_symbol_definition, file_editor, read_local_file."
            ),
        },
        {"role": "user", "content": instructions},
    ]
    return ask_ollama_structured(
        messages, WorkerToolPlan, model=model, max_retries=max_retries
    )
