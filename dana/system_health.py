"""Process-wide LLM concurrency guard."""

from __future__ import annotations

import threading

# Serialize ALL LLM generations (Ollama + Cloud) — never double VRAM load.
llm_lock = threading.RLock()

__all__ = ("llm_lock",)
