"""Centralized, thread-safe TTS speech queue for system-wide voice notifications.

Any component (Meta-Broker, System Health, Router, UI) pushes text via
``TTSManager.enqueue`` / ``enqueue_speech``. A single daemon thread drains
``speech_queue`` sequentially so Piper playback never overlaps.
"""

from __future__ import annotations

import queue
import threading
from typing import Any, Callable, Optional

_EnqueueFn = Callable[..., None]
_WorkerFn = Callable[[], None]


class TTSManager:
    """Process-wide singleton: one speech queue + one consumer thread."""

    _instance: TTSManager | None = None
    _instance_lock = threading.Lock()

    def __init__(self, *, maxsize: int = 64) -> None:
        self.speech_queue: queue.Queue[Any] = queue.Queue(maxsize=max(8, int(maxsize)))
        self._enqueue_lock = threading.Lock()
        self._start_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._worker_target: _WorkerFn | None = None
        self._enqueue_impl: _EnqueueFn | None = None
        self._started = False

    @classmethod
    def instance(cls) -> TTSManager:
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def bind(
        self,
        *,
        speech_queue: queue.Queue[Any] | None = None,
        worker: _WorkerFn | None = None,
        enqueue_impl: _EnqueueFn | None = None,
    ) -> None:
        """Wire the Piper spooler (core_agent) without circular imports at module load."""
        if speech_queue is not None:
            self.speech_queue = speech_queue
        if worker is not None:
            self._worker_target = worker
        if enqueue_impl is not None:
            self._enqueue_impl = enqueue_impl

    def start(self, *, worker: _WorkerFn | None = None) -> threading.Thread | None:
        """Launch the dedicated daemon consumer (idempotent)."""
        if worker is not None:
            self._worker_target = worker
        target = self._worker_target
        if target is None:
            return None
        with self._start_lock:
            if self._thread is not None and self._thread.is_alive():
                return self._thread
            self._thread = threading.Thread(
                target=target,
                name="TTSManager",
                daemon=True,
            )
            self._thread.start()
            self._started = True
            return self._thread

    @property
    def is_running(self) -> bool:
        t = self._thread
        return t is not None and t.is_alive()

    def enqueue(
        self,
        text: str,
        *,
        interruptible: bool | None = None,
        agent_id: str | None = None,
    ) -> None:
        """Push a phrase onto the speech queue (non-blocking for callers).

        Prefers the bound core_agent ``enqueue_speech`` (chunking / barge rules).
        Falls back to a raw queue put when unbound (tests / early boot).
        """
        impl = self._enqueue_impl
        if impl is not None:
            try:
                kwargs: dict[str, Any] = {}
                if interruptible is not None:
                    kwargs["interruptible"] = interruptible
                if agent_id is not None:
                    kwargs["agent_id"] = agent_id
                impl(str(text or ""), **kwargs)
                return
            except Exception:  # noqa: BLE001
                pass
        piece = str(text or "").strip()
        if not piece:
            return
        flag = True if interruptible is None else bool(interruptible)
        aid = str(agent_id or "broker")
        with self._enqueue_lock:
            try:
                self.speech_queue.put_nowait((piece, flag, aid))
            except queue.Full:
                try:
                    _ = self.speech_queue.get_nowait()
                except queue.Empty:
                    return
                try:
                    self.speech_queue.put_nowait((piece, flag, aid))
                except queue.Full:
                    pass

    def notify(self, text: str, *, interruptible: bool = False) -> None:
        """System notification helper (acks / epic / health — uninterruptible by default)."""
        self.enqueue(text, interruptible=interruptible)


def get_tts_manager() -> TTSManager:
    return TTSManager.instance()


def enqueue_speech(
    text: str,
    *,
    interruptible: bool | None = None,
    agent_id: str | None = None,
) -> None:
    """Module-level producer API (alias for ``TTSManager.enqueue``)."""
    get_tts_manager().enqueue(
        text, interruptible=interruptible, agent_id=agent_id
    )


__all__ = (
    "TTSManager",
    "enqueue_speech",
    "get_tts_manager",
)
