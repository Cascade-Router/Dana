"""Spatial coordinate TTL + exponential memory decay compaction tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from dana.memory.compaction import CompactionEngine
from dana.memory.store import EpisodicMemoryStore


class _FakeClock:
    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += float(seconds)


@pytest.fixture
def clock() -> _FakeClock:
    return _FakeClock()


@pytest.fixture
def store(tmp_path: Path, clock: _FakeClock) -> EpisodicMemoryStore:
    return EpisodicMemoryStore(tmp_path / "decay_test.db", time_fn=clock)


def test_spatial_coordinate_ttl_expires_after_15_minutes(
    store: EpisodicMemoryStore,
    clock: _FakeClock,
) -> None:
    """Spatial / UI location facts with ttl=900 expire and prune after 15 min."""
    store.add_fact(
        "environment_fact",
        "ui_element_submit_btn_coord",
        {"x": 120, "y": 340},
        ttl_seconds=900,
    )
    assert any(
        f["key"] == "ui_element_submit_btn_coord" for f in store.search_facts("submit")
    )

    clock.advance(899)
    assert any(
        f["key"] == "ui_element_submit_btn_coord" for f in store.search_facts("submit")
    )

    clock.advance(2)  # now past created_at + 900
    hits = store.search_facts("submit")
    assert not any(f["key"] == "ui_element_submit_btn_coord" for f in hits)

    deleted = store.prune_expired_entries()
    assert deleted >= 1
    assert not any(
        f["key"] == "ui_element_submit_btn_coord"
        for f in store.list_facts(include_expired=True)
    )


def test_user_preferences_intact_regardless_of_elapsed_time(
    store: EpisodicMemoryStore,
    clock: _FakeClock,
) -> None:
    """Preferences have no TTL and are exempt from decay compaction."""
    store.add_fact("user_preference", "prefer_dark_mode", True)
    clock.advance(30 * 24 * 3600)  # 30 days

    prefs = store.get_all_preferences()
    assert prefs.get("prefer_dark_mode") is True

    engine = CompactionEngine(time_fn=clock)
    result = engine.compact_memory(store)
    assert result["pruned"] == 0
    assert store.get_all_preferences().get("prefer_dark_mode") is True


def test_compaction_engine_prunes_stale_execution_traces(
    store: EpisodicMemoryStore,
    clock: _FakeClock,
) -> None:
    """W(t)=base*exp(-0.05*delta_h) prunes task_outcome below 0.15."""
    store.add_fact(
        "task_outcome",
        "clicked_login",
        "ok",
        confidence_score=1.0,
    )
    # Need exp(-0.05 * h) < 0.15 → h > -ln(0.15)/0.05 ≈ 37.93 hours
    clock.advance(40 * 3600)

    engine = CompactionEngine(time_fn=clock)
    weight = engine.decay_weight("task_outcome", 1.0, clock.now - 40 * 3600)
    assert weight < 0.15

    result = engine.compact_memory(store)
    assert result["pruned"] == 1
    assert not any(f["key"] == "clicked_login" for f in store.list_facts())
