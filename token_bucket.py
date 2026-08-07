"""Async token-bucket rate limiter (stdlib asyncio only)."""
import asyncio
import time

def _clock() -> float:
    """Prefer the running loop clock; fall back to monotonic wall time."""
    try:
        return asyncio.get_running_loop().time()
    except RuntimeError:
        try:
            return asyncio.get_event_loop().time()
        except RuntimeError:
            return time.monotonic()

class TokenBucket:
    """Capacity-limited bucket that refills at ``refill_rate`` tokens per second."""

    def __init__(self, capacity: int, refill_rate: float) -> None:
        self.capacity = max(0, int(capacity))
        self.refill_rate = float(refill_rate)
        self.tokens = float(self.capacity)
        self._last = _clock()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = _clock()
        elapsed = now - self._last
        self._last = now
        if self.refill_rate > 0 and elapsed > 0:
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)

    async def consume(self, amount: float=1.0) -> bool:
        """Take ``amount`` tokens, sleeping until refill if necessary.

        Returns ``True`` when the tokens were acquired. Returns ``False``
        immediately when ``refill_rate <= 0`` and the bucket cannot satisfy
        the request (avoids infinite ``asyncio.sleep`` loops).
        """
        need = float(amount)
        if need <= 0:
            return True
        while True:
            async with self._lock:
                self._refill()
                if self.tokens >= need:
                    self.tokens -= need
                    return True
                if self.refill_rate <= 0:
                    return False
                deficit = need - self.tokens
                sleep_for = deficit / self.refill_rate
            if sleep_for <= 0:
                async with self._lock:
                    self._refill()
                    if self.tokens >= need:
                        self.tokens -= need
                        return True
                sleep_for = 1.0 / self.refill_rate
                if sleep_for <= 0:
                    return False
            await asyncio.sleep(sleep_for)
__all__ = ('TokenBucket',)
