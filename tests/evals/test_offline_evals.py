"""The offline agent evaluation suite (tests/evals/eval_harness.py) as part of
the normal test run: one test per canonical task, each on its own /ws/chat
session against the mock CAD engine, no network and no subprocesses.

``python scripts/run_offline_evals.py`` runs the same scenarios and prints
the markdown scorecard.
"""

from __future__ import annotations

import pytest

pytest.importorskip("trimesh")
pytest.importorskip("manifold3d")

from eval_harness import SCENARIOS, Run, install, run_scenario  # noqa: E402


@pytest.fixture
def run_ref(monkeypatch: pytest.MonkeyPatch) -> list[Run]:
    import dana.core.react_dispatch as rd

    ref: list[Run] = []
    install(monkeypatch.setattr, monkeypatch.setenv, ref)
    if rd._token_encoder is rd._TOKEN_ENCODER_UNAVAILABLE:
        pytest.skip("tiktoken's cl100k encoding is unavailable; the token ceilings are in cl100k tokens")
    return ref


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.key for s in SCENARIOS])
def test_canonical_task(scenario, run_ref: list[Run]) -> None:
    run = run_scenario(scenario, run_ref)
    report = "\n".join(f"{'ok ' if c.passed else 'BAD'} {c.name}: {c.detail}" for c in run.checks)
    assert run.error is None, run.error
    assert run.blocked == [], run.blocked
    assert run.passed, f"{scenario.title}\n{report}"
