"""Centralized runtime constants for Dana's backend.

The single place that reads env-var overrides for values that used to be
hardcoded literals scattered across ``dana.api.server`` and
``dana.core.model_provider`` (``_MAX_REACT_ITERATIONS``, the tool-calling
``num_predict`` default). Import from here rather than redeclaring a literal
at another call site, so each value has exactly one definition to change.
"""

from __future__ import annotations

import os


def _int_env(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# ReAct loop iteration ceiling (dana.api.server._run_react_loop): a turn that
# reaches this many reasoning steps stops with a clear "reached the maximum
# number of reasoning steps" message instead of looping forever. Value is
# UNCHANGED from its prior hardcoded default (30) -- only its location moved
# here; only an explicit DANA_MAX_REACT_ITERATIONS override changes it.
MAX_REACT_ITERATIONS = _int_env("DANA_MAX_REACT_ITERATIONS", 30)

# Max completion tokens for one tool-calling turn (dana.core.model_provider.
# ModelProvider.complete_with_tool_calls) -- previously a hardcoded 1024,
# which a verbose multi-task create_plan call (e.g. a 9-step CAD plan) could
# exhaust mid-generation with no error surfaced anywhere (see the forensic
# RCA on a plan silently truncated from 9 tasks to 8). Raised to 4096.
LLM_MAX_OUTPUT_TOKENS = _int_env("DANA_LLM_MAX_OUTPUT_TOKENS", 4096)

__all__ = ("MAX_REACT_ITERATIONS", "LLM_MAX_OUTPUT_TOKENS")
