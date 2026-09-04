"""Static OpenRouter per-model USD pricing — Cost Tracking.

Prices are USD per 1,000,000 tokens, split prompt/completion, taken from
OpenRouter's own per-model pricing page (https://openrouter.ai/models) at
the time this table was written. This is a hand-maintained snapshot, not a
live lookup against OpenRouter's API — real prices do change — so treat a
number here as "close enough for a live session-cost estimate", never as a
billing-grade source of truth. Update the tuple for a model by hand when
its OpenRouter price changes.
"""

from __future__ import annotations

# model_id -> (usd_per_1m_prompt_tokens, usd_per_1m_completion_tokens).
# Keys are OpenRouter's own model id strings (the exact value sent as the
# request's "model" field), so a lookup here never needs normalization.
_PRICING_PER_1M_USD: dict[str, tuple[float, float]] = {
    "anthropic/claude-3.5-sonnet": (3.00, 15.00),
    "anthropic/claude-3-haiku": (0.25, 1.25),
    "anthropic/claude-3-opus": (15.00, 75.00),
    "openai/gpt-4o": (2.50, 10.00),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "meta-llama/llama-3.3-70b-instruct": (0.12, 0.30),
    "meta-llama/llama-3.3-70b-instruct:free": (0.0, 0.0),
    "meta-llama/llama-3-70b-instruct": (0.35, 0.40),
    "google/gemini-2.5-flash": (0.075, 0.30),
    "qwen/qwen-2.5-coder-32b-instruct": (0.07, 0.16),
}


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    """Returns ``None`` — never a guessed number — when ``model`` isn't in
    the table above (e.g. a local Ollama model id, which OpenRouter never
    priced in the first place). A missing price must surface as "unknown"
    to callers, not silently as a free $0.00.
    """
    pricing = _PRICING_PER_1M_USD.get(model)
    if pricing is None:
        return None
    prompt_rate, completion_rate = pricing
    return (prompt_tokens / 1_000_000) * prompt_rate + (completion_tokens / 1_000_000) * completion_rate


__all__ = ("estimate_cost_usd",)
