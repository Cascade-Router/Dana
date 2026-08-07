from __future__ import annotations

import asyncio
import os
import threading
from typing import Any, Optional

import numpy as np

from dana.paths import resolve_wakeword_onnx

try:
    from openwakeword.model import Model as OpenWakeWordModel
except Exception:  # pragma: no cover - optional dependency
    OpenWakeWordModel = None  # type: ignore[assignment]


class AudioRouter:
    """Split microphone audio into whisper and standard wake-word queues."""

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 1280,
        sample_width: int = 2,
        whisper_gain: float = 2.0,
        *,
        input_device: Optional[int] = None,
        stream_kwargs: Optional[dict[str, Any]] = None,
        wakeword_model_path: Optional[str] = None,
        whisper_model: Optional[Any] = None,
        standard_model: Optional[Any] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.sample_width = sample_width
        self.whisper_gain = whisper_gain
        self.input_device = input_device
        self.stream_kwargs = stream_kwargs or {}
        self.whisper_queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self.standard_queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.whisper_model = whisper_model or self._build_model(wakeword_model_path, vad_threshold=0.15)
        self.standard_model = standard_model or self._build_model(wakeword_model_path, vad_threshold=0.5)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="AudioRouter", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        import sounddevice as sd

        if self._stop_event.is_set():
            return
        try:
            with sd.InputStream(
                device=self.input_device,
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=self.chunk_size,
                **self.stream_kwargs,
            ) as stream:
                while not self._stop_event.is_set():
                    chunk, overflowed = stream.read(self.chunk_size)
                    if overflowed:
                        continue
                    audio = np.asarray(chunk, dtype=np.int16)
                    if audio.size == 0:
                        continue
                    whisper_chunk = self._boost_audio(audio)
                    self._enqueue_chunk_sync(whisper_chunk, self.whisper_queue)
                    self._enqueue_chunk_sync(audio, self.standard_queue)
        except Exception:
            if os.environ.get("DANA_AUDIO_PIPELINE_DEBUG"):
                raise

    @staticmethod
    def _boost_audio(audio: np.ndarray) -> np.ndarray:
        boosted = audio.astype(np.float32) * 2.0
        return np.clip(boosted, -32768.0, 32767.0).astype(np.int16)

    def _enqueue_chunk_sync(self, chunk: np.ndarray, queue: asyncio.Queue[np.ndarray]) -> None:
        queue.put_nowait(chunk)

    def _build_model(self, wakeword_model_path: Optional[str], *, vad_threshold: float) -> Optional[Any]:
        if OpenWakeWordModel is None:
            return None
        try:
            model_path = wakeword_model_path or str(resolve_wakeword_onnx())
            if not model_path:
                return None
            return OpenWakeWordModel(
                wakeword_models=[model_path],
                inference_framework="onnx",
                vad_threshold=vad_threshold,
            )
        except Exception:
            return None

    def flush(self) -> None:
        self.whisper_queue = asyncio.Queue()
        self.standard_queue = asyncio.Queue()
