import asyncio
from types import SimpleNamespace

import numpy as np

from dana.audio.wake_poller import WakePoller


class StubModel:
    def __init__(self, score: float):
        self.score = score

    def predict(self, chunk):
        return {"dana": self.score}


def test_wake_poller_triggers_from_either_stream_and_clears_queues():
    router = SimpleNamespace(
        whisper_queue=asyncio.Queue(),
        standard_queue=asyncio.Queue(),
    )
    events = []

    poller = WakePoller(
        router=router,
        whisper_model=StubModel(0.9),
        standard_model=StubModel(0.1),
        callback=lambda: events.append("wake"),
        threshold=0.5,
        poll_interval=0.0,
    )

    async def _run() -> None:
        await router.whisper_queue.put(np.array([1000, 2000], dtype=np.int16))
        await poller._poll_once()

    asyncio.run(_run())

    assert events == ["wake"]
    assert router.whisper_queue.empty()
    assert router.standard_queue.empty()
