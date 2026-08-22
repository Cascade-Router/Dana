"""Unit tests for dana.core.context_distiller — Pillar 3's local-GPU rolling
working-memory summarizer. Covers the three safety properties it must hold
under real-world failure modes (Ollama down, the RTX 2080 busy/unreachable,
a misbehaving model output): never raise into the caller, never contend
with a live ReAct turn for the GPU, and never let session["working_memory"]
grow past its cap no matter what the local model actually returns.

No async test plugin needed: each test drives distill_turn's coroutine with
a plain asyncio.run(...) call from an ordinary sync def test_...(), matching
tests/core/test_react_dispatch.py's own convention.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

import dana.core.context_distiller as cd
from dana.system_health import llm_lock


class _FakeProvider:
    def __init__(self, *, text: str = "", raises: Exception | None = None) -> None:
        self._text = text
        self._raises = raises
        self.calls = 0

    def complete(self, messages, **kwargs):  # noqa: ANN001, ANN003 — test double
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return self._text


def _fresh_session(summary: str = "") -> dict:
    return {"working_memory": {"summary": summary, "turn": 0}, "turn_counter": 1}


def test_distill_turn_updates_and_caps_a_normal_response(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeProvider(text="User created a box and asked for a cylinder next.")
    monkeypatch.setattr(cd, "ModelProvider", lambda **_kwargs: fake)
    session = _fresh_session()

    asyncio.run(cd.distill_turn(session, "make a cylinder", "Done — cylinder created."))

    assert fake.calls == 1
    assert session["working_memory"]["summary"] == "User created a box and asked for a cylinder next."


def test_distill_turn_caps_word_count_even_if_the_model_ignores_the_instruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runaway = " ".join(f"word{i}" for i in range(cd._MAX_SUMMARY_WORDS * 3))
    fake = _FakeProvider(text=runaway)
    monkeypatch.setattr(cd, "ModelProvider", lambda **_kwargs: fake)
    session = _fresh_session()

    asyncio.run(cd.distill_turn(session, "hi", "hello"))

    summary = session["working_memory"]["summary"]
    assert len(summary.split()) <= cd._MAX_SUMMARY_WORDS + 1  # +1 for the trailing "…" token
    assert summary.endswith("…")


def test_distill_turn_caps_character_count_for_whitespace_free_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model that answers with one giant unbroken token (dense CJK, a
    stray code blob, ...) has too few "words" for the word-count cap to
    ever trigger — the character ceiling is what actually bounds it."""
    fake = _FakeProvider(text="x" * (cd._MAX_SUMMARY_CHARS * 3))
    monkeypatch.setattr(cd, "ModelProvider", lambda **_kwargs: fake)
    session = _fresh_session()

    asyncio.run(cd.distill_turn(session, "hi", "hello"))

    assert len(session["working_memory"]["summary"]) <= cd._MAX_SUMMARY_CHARS + 1


def test_distill_turn_never_raises_when_local_model_is_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeProvider(raises=ConnectionError("Ollama unreachable"))
    monkeypatch.setattr(cd, "ModelProvider", lambda **_kwargs: fake)
    session = _fresh_session(summary="prior summary")

    asyncio.run(cd.distill_turn(session, "hi", "hello"))  # must not raise

    # Failure degrades silently — the session keeps whatever it already had.
    assert session["working_memory"]["summary"] == "prior summary"


def test_distill_turn_never_raises_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    class _HangingProvider:
        def complete(self, messages, **kwargs):  # noqa: ANN001, ANN003
            raise TimeoutError("stalled")

    monkeypatch.setattr(cd, "ModelProvider", lambda **_kwargs: _HangingProvider())
    session = _fresh_session(summary="prior summary")

    asyncio.run(cd.distill_turn(session, "hi", "hello"))

    assert session["working_memory"]["summary"] == "prior summary"


def test_distill_turn_skips_without_calling_the_model_when_gpu_is_busy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulates a live ReAct turn already holding dana.system_health.llm_lock
    (mid-generation on the local model, on ITS OWN thread — a real live turn
    runs its blocking Ollama call via asyncio.to_thread, a genuinely
    different OS thread than distill_turn's own event-loop thread) —
    distillation must back off entirely rather than queue behind it and
    delay a future live turn.

    llm_lock is a threading.RLock, which is reentrant PER THREAD — acquiring
    it on the SAME thread that later checks it would always look "free"
    regardless of this module's logic, so the holder must run on a real
    separate thread for this test to actually exercise cross-thread
    contention the way production does.
    """
    fake = _FakeProvider(text="should never be reached")
    monkeypatch.setattr(cd, "ModelProvider", lambda **_kwargs: fake)
    session = _fresh_session(summary="prior summary")

    holder_has_lock = threading.Event()
    release_holder = threading.Event()

    def _hold_lock() -> None:
        with llm_lock:
            holder_has_lock.set()
            release_holder.wait(timeout=5)

    holder = threading.Thread(target=_hold_lock, daemon=True)
    holder.start()
    try:
        assert holder_has_lock.wait(timeout=5), "holder thread never acquired llm_lock"
        asyncio.run(cd.distill_turn(session, "hi", "hello"))
    finally:
        release_holder.set()
        holder.join(timeout=5)

    assert fake.calls == 0
    assert session["working_memory"]["summary"] == "prior summary"


def test_distillation_disabled_via_env_flag_is_a_pure_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeProvider(text="should never be reached")
    monkeypatch.setattr(cd, "ModelProvider", lambda **_kwargs: fake)
    monkeypatch.setenv("DANA_CONTEXT_DISTILL", "0")
    session = _fresh_session(summary="prior summary")

    asyncio.run(cd.distill_turn(session, "hi", "hello"))

    assert fake.calls == 0
    assert session["working_memory"]["summary"] == "prior summary"


def test_distill_turn_skips_on_empty_user_text(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeProvider(text="should never be reached")
    monkeypatch.setattr(cd, "ModelProvider", lambda **_kwargs: fake)
    session = _fresh_session()

    asyncio.run(cd.distill_turn(session, "   ", "hello"))

    assert fake.calls == 0


def test_working_memory_never_exceeds_cap_across_many_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulates a long-running (50+ turn) session: since each round's
    "current summary" input is always the ALREADY-capped prior summary, the
    cap is a genuine fixed point, not just a one-shot trim."""
    runaway = " ".join(f"word{i}" for i in range(cd._MAX_SUMMARY_WORDS * 5))
    fake = _FakeProvider(text=runaway)
    monkeypatch.setattr(cd, "ModelProvider", lambda **_kwargs: fake)
    session = _fresh_session()

    for turn in range(60):
        session["turn_counter"] = turn
        asyncio.run(cd.distill_turn(session, f"turn {turn}", f"reply {turn}"))
        summary = session["working_memory"]["summary"]
        assert len(summary) <= cd._MAX_SUMMARY_CHARS + 1
        assert len(summary.split()) <= cd._MAX_SUMMARY_WORDS + 1


def test_schedule_distillation_runs_in_the_background_without_blocking_caller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeProvider(text="background summary")
    monkeypatch.setattr(cd, "ModelProvider", lambda **_kwargs: fake)
    session = _fresh_session()

    async def _main() -> None:
        cd.schedule_distillation(session, "hi", "hello")
        # schedule_distillation must return immediately (no await) — give
        # the background task a few loop iterations to actually run.
        for _ in range(10):
            await asyncio.sleep(0)

    asyncio.run(_main())

    assert fake.calls == 1
    assert session["working_memory"]["summary"] == "background summary"
