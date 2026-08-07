from __future__ import annotations

import asyncio
import threading
from typing import Any, Callable, Optional

import numpy as np

try:
    from openwakeword.model import Model as OpenWakeWordModel
except Exception:  # pragma: no cover - optional dependency
    OpenWakeWordModel = None  # type: ignore[assignment]


class WakePoller:
    """Poll two audio streams and trigger on either wake-word confidence."""

    def __init__(
        self,
        *,
        router: Any,
        whisper_model: Any = None,
        standard_model: Any = None,
        callback: Optional[Callable[[], None]] = None,
        threshold: float = 0.5,
        poll_interval: float = 0.02,
        wake_token: str = "dana",
    ) -> None:
        self.router = router
        self.whisper_model = whisper_model
        self.standard_model = standard_model
        self.callback = callback
        self.threshold = threshold
        self.poll_interval = poll_interval
        self.wake_token = wake_token
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_sync, name="WakePoller", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _run_sync(self) -> None:
        asyncio.run(self._run())

    async def _run(self) -> None:
        while not self._stop_event.is_set():
            await self._poll_once()
            await asyncio.sleep(self.poll_interval)

    async def _poll_once(self) -> None:
        if self._stop_event.is_set():
            return
        await self._consume_queue(self.router.whisper_queue, self.whisper_model)
        if self._stop_event.is_set():
            return
        await self._consume_queue(self.router.standard_queue, self.standard_model)

    async def _consume_queue(self, queue: asyncio.Queue[np.ndarray], model: Any) -> None:
        if model is None:
            return
        try:
            chunk = queue.get_nowait()
        except Exception:
            return
        prediction = self._predict(chunk, model)
        if self._should_trigger(prediction):
            self._trigger()

    def _predict(self, chunk: np.ndarray, model: Any) -> dict[str, Any]:
        try:
            prediction = model.predict(chunk)
        except TypeError:
            prediction = model.predict(chunk.astype(np.float32))
        except Exception:
            return {}
        if isinstance(prediction, dict):
            return prediction
        return {}

    def _should_trigger(self, prediction: dict[str, Any]) -> bool:
        for key, score in prediction.items():
            try:
                value = float(score)
            except (TypeError, ValueError):
                continue
            if self.wake_token.lower() in str(key).lower() and value >= self.threshold:
                return True
        return False

    def _trigger(self) -> None:
        if self.callback is not None:
            self.callback()
        self.router.whisper_queue = asyncio.Queue()
        self.router.standard_queue = asyncio.Queue()
