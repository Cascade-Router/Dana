"""Donna evals package — headless CI harness for agent routing."""

from __future__ import annotations

__all__ = ["run_harness"]


def run_harness(*args, **kwargs):
    """Lazy import so ``python -m dana.evals.headless_harness`` stays clean."""
    from dana.evals.headless_harness import run_harness as _run

    return _run(*args, **kwargs)
