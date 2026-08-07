"""Async EventBus + TokenBucket burst / rate-limit tests."""

from __future__ import annotations

import asyncio
import time

import pytest

from async_bus import EventBus
from token_bucket import TokenBucket

_TEST_TIMEOUT_S = 5.0


def _run(coro):
    """Run an async body under a hard 5s ceiling (no hanging harness)."""
    return asyncio.run(asyncio.wait_for(coro, timeout=_TEST_TIMEOUT_S))


@pytest.mark.asyncio
def test_token_bucket_consume_immediate_and_refill():
    async def _body() -> None:
        bucket = TokenBucket(capacity=2, refill_rate=20.0)
        assert await bucket.consume(1) is True
        assert await bucket.consume(1) is True
        t0 = time.monotonic()
        assert await bucket.consume(1) is True
        elapsed = time.monotonic() - t0
        # Third token must wait for refill (~0.05s at 20 tok/s).
        assert elapsed >= 0.03

    _run(_body())


@pytest.mark.asyncio
def test_token_bucket_zero_refill_does_not_hang():
    async def _body() -> None:
        bucket = TokenBucket(capacity=1, refill_rate=0.0)
        assert await bucket.consume(1) is True
        assert await bucket.consume(1) is False

    _run(_body())


@pytest.mark.asyncio
def test_burst_publish_is_rate_limited():
    async def _body() -> None:
        # capacity=2, refill 10/s → overflow publishes are delayed.
        bus = EventBus(capacity=2, refill_rate=10.0)
        stamps: list[float] = []

        async def on_msg(_payload: object) -> None:
            stamps.append(time.monotonic())

        bus.subscribe("events", on_msg)

        t0 = time.monotonic()
        for i in range(5):
            await bus.publish("events", {"i": i})
        total = time.monotonic() - t0

        assert len(stamps) == 5
        # First two consume the burst capacity nearly instantly.
        assert stamps[1] - stamps[0] < 0.15
        # Overflow (3 tokens beyond capacity) needs ~0.3s at 10 tok/s.
        assert stamps[-1] - stamps[0] >= 0.2
        assert total >= 0.2

    _run(_body())


@pytest.mark.asyncio
def test_start_full_bucket_allows_burst_then_paces():
    async def _body() -> None:
        bus = EventBus(capacity=3, refill_rate=50.0)
        stamps: list[float] = []

        async def on_msg(_payload: object) -> None:
            stamps.append(time.monotonic())

        bus.subscribe("burst", on_msg)
        for i in range(3):
            await bus.publish("burst", i)

        assert len(stamps) == 3
        assert stamps[-1] - stamps[0] < 0.15

    _run(_body())
