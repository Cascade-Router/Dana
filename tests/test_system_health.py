"""dana.system_health.llm_lock: one process-wide lock that local LLM calls
(ModelProvider) and the context distiller share, so two local generations
never run at once."""

from __future__ import annotations

import threading
import time

from dana.core import context_distiller, model_provider
from dana.system_health import llm_lock


def test_model_provider_and_distiller_share_the_one_lock() -> None:
    assert model_provider.llm_lock is llm_lock
    assert context_distiller.llm_lock is llm_lock


def test_lock_is_reentrant_for_nested_calls_on_one_thread() -> None:
    with llm_lock:
        with llm_lock:  # a helper that takes the lock again must not deadlock
            pass


def test_lock_serializes_generations_across_threads() -> None:
    active, peak = 0, 0
    counter = threading.Lock()

    def generate() -> None:
        nonlocal active, peak
        with llm_lock:
            with counter:
                active += 1
                peak = max(peak, active)
            time.sleep(0.02)
            with counter:
                active -= 1

    threads = [threading.Thread(target=generate) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    assert peak == 1
