"""Async event bus with per-topic token-bucket publish rate limiting."""
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any
from token_bucket import TokenBucket
Handler = Callable[[Any], Awaitable[Any]]

class EventBus:
    """Pub/sub bus; each topic has its own ``TokenBucket`` for publish pacing."""

    def __init__(self, *, capacity: int=5, refill_rate: float=10.0) -> None:
        self._subscribers: dict[str, list[Handler]] = defaultdict(list)
        self._buckets: dict[str, TokenBucket] = {}
        self._capacity = int(capacity)
        self._refill_rate = float(refill_rate)

    def _bucket_for(self, topic: str) -> TokenBucket:
        bucket = self._buckets.get(topic)
        if bucket is None:
            bucket = TokenBucket(self._capacity, self._refill_rate)
            self._buckets[topic] = bucket
        return bucket

    def subscribe(self, topic: str, handler_coroutine: Handler) -> None:
        self._subscribers[topic].append(handler_coroutine)

    async def publish(self, topic: str, payload: Any) -> None:
        """Rate-limit then deliver ``payload`` to all topic subscribers."""
        await self._bucket_for(topic).consume(1)
        for handler in list(self._subscribers.get(topic, ())):
            await handler(payload)
__all__ = ('EventBus',)
