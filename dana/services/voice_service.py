"""Headless background voice worker: mic capture -> VAD -> Whisper STT.

Wraps the pre-existing ``dana.audio`` wake-word/VAD/Whisper stack (built for
the now-deleted Gradio UI) behind a small idle/listening/processing/speaking
state machine, so ``dana.api.server`` can run this on a daemon thread and
broadcast state transitions as ``voice_state`` websocket events without any
GUI toolkit involved.

This is push-to-talk, not a hot mic: the worker thread parks (mic closed)
until ``request_listen()`` is called — the AssistiveOrb click/hotkey handler
on the frontend, relayed through a ``voice_control`` websocket message (see
``dana.api.server``). One call captures and transcribes exactly one
utterance, then hands the transcript off and stays parked in "processing"
until ``finish_turn()`` is called once the assistant has replied (text +
TTS) — only then does the service re-arm for the next ``request_listen()``.
This keeps this module's own STT-side states from racing the separate,
server-driven "assistant is now speaking a TTS reply" state.

Every dependency this needs — a real input device, ``sounddevice``, the
Whisper model bundle — is optional at runtime: if any of it is missing the
service degrades to sitting idle instead of raising, so a container/CI/dev
box with no microphone still boots the server cleanly.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import numpy as np

VoiceState = Literal["idle", "listening", "processing", "speaking"]
StateCallback = Callable[[VoiceState, str], None]

_LISTEN_CHUNK_S = 0.5
_MAX_UTTERANCE_S = 8.0
_SILENCE_HANGOVER_S = 0.8
_SILENCE_RMS_FLOOR = 150.0


class VoiceService:
    """Mic -> VAD -> Whisper loop, driven from a single daemon thread.

    ``on_state`` fires on every state transition as ``(state, transcript)`` —
    ``transcript`` is only non-empty on the ``"speaking"`` transition, once
    an utterance has been finalized. The caller never has to poll: register
    a callback and read ``.state``/``.hardware_available`` for diagnostics.
    """

    def __init__(self, on_state: StateCallback | None = None) -> None:
        self._on_state: StateCallback = on_state or (lambda *_a: None)
        self._stop_event = threading.Event()
        self._listen_trigger = threading.Event()
        self._cancel_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._state: VoiceState = "idle"
        self._hardware_available = self._probe_hardware()
        if self._hardware_available:
            self._start_whisper_background_load()

    @property
    def state(self) -> VoiceState:
        return self._state

    @property
    def hardware_available(self) -> bool:
        return self._hardware_available

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="VoiceService", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._listen_trigger.set()  # wake a parked _run() so it can observe stop_event promptly
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._set_state("idle", "")

    def request_listen(self) -> bool:
        """Starts one listen -> transcribe cycle. No-op (returns ``False``)
        if a cycle is already in flight — the orb/hotkey handler should
        treat that as "ignored", not an error."""
        if self._state != "idle":
            return False
        self._listen_trigger.set()
        return True

    def cancel(self) -> None:
        """Aborts an in-flight capture — e.g. the orb clicked again while
        ``listening``. No-op once capture has already moved on to
        transcription."""
        if self._state == "listening":
            self._cancel_event.set()

    def finish_turn(self) -> None:
        """Re-arms the service for the next ``request_listen()`` once the
        caller (``dana.api.server``) has finished acting on a handed-off
        transcript — i.e. the assistant's reply has been synthesized and
        played back. No-op if a transcript isn't actually pending."""
        if self._state == "processing":
            self._set_state("idle", "")

    # -- setup -----------------------------------------------------------

    @staticmethod
    def _probe_hardware() -> bool:
        try:
            import sounddevice as sd

            devices = sd.query_devices()
            return any(d.get("max_input_channels", 0) > 0 for d in devices)
        except Exception:  # noqa: BLE001 — no PortAudio backend / no mic is expected on CI
            return False

    @staticmethod
    def _start_whisper_background_load() -> None:
        try:
            from dana.audio.stt import start_whisper_background_load

            start_whisper_background_load(local_files_only=True, device=None)
        except Exception:  # noqa: BLE001 — torch/transformers missing is a graceful no-op here
            pass

    def _set_state(self, state: VoiceState, transcript: str = "") -> None:
        self._state = state
        try:
            self._on_state(state, transcript)
        except Exception:  # noqa: BLE001 — a broken listener must never kill the worker thread
            pass

    # -- worker loop -------------------------------------------------------

    def _run(self) -> None:
        if not self._hardware_available:
            self._set_state("idle", "")
            while not self._stop_event.is_set():
                self._stop_event.wait(1.0)
            return

        self._set_state("idle", "")
        while not self._stop_event.is_set():
            # Parked here (mic closed) until request_listen() sets the
            # trigger — this is the push-to-talk gate.
            triggered = self._listen_trigger.wait(timeout=1.0)
            if not triggered:
                continue
            self._listen_trigger.clear()
            if self._stop_event.is_set():
                break

            self._set_state("listening", "")
            audio = self._capture_utterance()
            if self._stop_event.is_set():
                break
            if self._cancel_event.is_set():
                self._cancel_event.clear()
                self._set_state("idle", "")
                continue
            if audio is None:
                # Nothing captured (silence, transient device error).
                self._set_state("idle", "")
                continue

            self._set_state("processing", "")
            transcript = self._transcribe(audio)
            if transcript:
                # Hand-off signal for dana.api.server: state stays
                # "processing" (now carrying the transcript) rather than
                # auto-idling — finish_turn() re-arms this loop once the
                # assistant has replied. See the class docstring.
                self._set_state("processing", transcript)
            else:
                self._set_state("idle", "")

    def _capture_utterance(self) -> "np.ndarray | None":
        try:
            import numpy as np
            import sounddevice as sd

            from dana.audio.devices import resolve_live_input_device
        except Exception:  # noqa: BLE001
            return None
        try:
            device, rate = resolve_live_input_device()
        except Exception:  # noqa: BLE001
            device, rate = None, 16000

        chunks: list["np.ndarray"] = []
        silence_s = 0.0
        elapsed_s = 0.0
        try:
            with sd.InputStream(device=device, samplerate=rate, channels=1, dtype="int16") as stream:
                while (
                    not self._stop_event.is_set()
                    and not self._cancel_event.is_set()
                    and elapsed_s < _MAX_UTTERANCE_S
                ):
                    frame, _overflowed = stream.read(int(rate * _LISTEN_CHUNK_S))
                    elapsed_s += _LISTEN_CHUNK_S
                    frame = np.asarray(frame, dtype=np.int16).reshape(-1)
                    rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)) + 1e-9)
                    if rms < _SILENCE_RMS_FLOOR:
                        if chunks:
                            silence_s += _LISTEN_CHUNK_S
                    else:
                        silence_s = 0.0
                        chunks.append(frame)
                    if chunks and silence_s >= _SILENCE_HANGOVER_S:
                        break
        except Exception:  # noqa: BLE001 — device unplugged mid-stream, etc.
            return None

        if not chunks:
            return None
        return np.concatenate(chunks)

    @staticmethod
    def _transcribe(audio: "np.ndarray") -> str:
        try:
            from dana.audio.stt import ensure_whisper_bundle, transcribe_audio

            processor, model, device, dtype = ensure_whisper_bundle(timeout=0.5)
            return transcribe_audio(audio, processor, model, device, dtype).strip()
        except Exception:  # noqa: BLE001 — model not ready/loaded yet, or transcription failed
            return ""


__all__ = ("VoiceService", "VoiceState")
