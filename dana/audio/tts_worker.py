"""Thread-safe TTS barge-in controller (flush spool + hard-stop playback)."""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
import wave
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import soundfile as sf
import sounddevice as sd

from dana.core.constants import BARGE_IN_CHUNK_MS, TTS_UTTERANCE_MAX_SECONDS
from dana.logging import log, log_debug, log_exception
from dana.paths import PROJECT_ROOT, TEMP_REPLY_WAV
from dana.paths import TTS_MODELS_DIR as _TTS_DIR

_log = logging.getLogger("dana.audio.tts")

_FlushFn = Callable[[], int]
_StopFn = Callable[..., None]
_UiFn = Callable[[str], None]
_ResetStreamFn = Callable[[], None]

# Default matches core_agent.BARGE_IN_PLAYBACK_GRACE_MS (speaker-onset bleed).
_DEFAULT_PLAYBACK_GRACE_S = 0.4


class TtsWorker:
    """Owns the barge-in flag and coordinates queue flush + hard-stop playback.

    Playback code registers the active ``OutputStream`` so ``interrupt()`` can
    ``abort()`` it without waiting on the writer’s ``playback_lock`` (avoids the
    race where ``sd.stop`` was deferred while a chunk write held the lock).

    Short UI acknowledgments play with ``interruptible=False`` so speaker bleed
    cannot self-barge; LLM synthesis keeps ``interruptible=True`` (zero-latency).
    """

    def __init__(self, *, barge_in_event: threading.Event | None = None) -> None:
        self._barge_in_event = barge_in_event or threading.Event()
        self._stream_lock = threading.Lock()
        self._active_stream: Any | None = None
        self._playback_lock = threading.Lock()
        self._playback_interruptible = True
        self._playback_active = False
        self._playback_started_at = 0.0
        self._flush_fn: _FlushFn | None = None
        self._sd_stop_fn: _StopFn | None = None
        self._set_ui_fn: _UiFn | None = None
        self._reset_stream_fn: _ResetStreamFn | None = None
        self._busy_fn: Callable[[], bool] | None = None

    @property
    def barge_in_event(self) -> threading.Event:
        return self._barge_in_event

    def bind(
        self,
        *,
        flush_fn: _FlushFn | None = None,
        sd_stop_fn: _StopFn | None = None,
        set_ui_fn: _UiFn | None = None,
        reset_stream_fn: _ResetStreamFn | None = None,
        busy_fn: Callable[[], bool] | None = None,
    ) -> None:
        """Inject core_agent callbacks (keeps this module free of PortAudio imports)."""
        if flush_fn is not None:
            self._flush_fn = flush_fn
        if sd_stop_fn is not None:
            self._sd_stop_fn = sd_stop_fn
        if set_ui_fn is not None:
            self._set_ui_fn = set_ui_fn
        if reset_stream_fn is not None:
            self._reset_stream_fn = reset_stream_fn
        if busy_fn is not None:
            self._busy_fn = busy_fn

    def begin_playback(self, *, interruptible: bool = True) -> None:
        """Mark the start of a play_audio turn (state-machine exemption latch)."""
        with self._playback_lock:
            self._playback_active = True
            self._playback_interruptible = bool(interruptible)
            self._playback_started_at = time.perf_counter()

    def end_playback(self) -> None:
        """Clear the playback latch after audio finishes (interruptible or not)."""
        with self._playback_lock:
            self._playback_active = False
            self._playback_interruptible = True
            self._playback_started_at = 0.0

    def is_playback_active(self) -> bool:
        with self._playback_lock:
            return bool(self._playback_active)

    def is_playback_interruptible(self) -> bool:
        with self._playback_lock:
            # Idle / between utterances → allow barge-in arming for the next turn.
            if not self._playback_active:
                return True
            return bool(self._playback_interruptible)

    def in_playback_grace(self, *, grace_s: float | None = None) -> bool:
        """True during the post-onset window where barge-in must stay suppressed."""
        window = (
            float(grace_s)
            if grace_s is not None
            else _DEFAULT_PLAYBACK_GRACE_S
        )
        if window <= 0:
            return False
        with self._playback_lock:
            if not self._playback_active:
                return False
            started = float(self._playback_started_at or 0.0)
        if started <= 0:
            return False
        return (time.perf_counter() - started) < window

    def play_audio(self, *, interruptible: bool = True) -> Any:
        """Context manager: ``with worker.play_audio(interruptible=False): ...``."""
        return _PlaybackSession(self, interruptible=interruptible)

    def register_output_stream(self, stream: Any) -> None:
        with self._stream_lock:
            self._active_stream = stream

    def unregister_output_stream(self, stream: Any | None = None) -> None:
        with self._stream_lock:
            if stream is None or self._active_stream is stream:
                self._active_stream = None

    def is_set(self) -> bool:
        return self._barge_in_event.is_set()

    def clear(self) -> None:
        self._barge_in_event.clear()

    def interrupt(
        self,
        *,
        reason: str = "",
        set_listening: bool = True,
        force: bool = False,
    ) -> int:
        """Hard barge-in: flag → flush spool → abort stream → stop device.

        No-ops instantly when the active utterance is a UI acknowledgment
        (``interruptible=False``), unless ``force=True`` (utterance watchdog).
        """
        if not force and not self.is_playback_interruptible():
            if reason:
                _log.debug(
                    "TTS barge-in ignored (uninterruptible UX ack) reason=%s",
                    reason,
                )
            return 0

        self._barge_in_event.set()

        dropped = 0
        if self._flush_fn is not None:
            try:
                dropped = int(self._flush_fn() or 0)
            except Exception as exc:  # noqa: BLE001
                _log.debug("TTS flush failed during interrupt: %s", exc)

        # Drop any LangGraph stream sentence fragments still coalescing into TTS.
        if self._reset_stream_fn is not None:
            try:
                self._reset_stream_fn()
            except Exception as exc:  # noqa: BLE001
                _log.debug("stream TTS reset failed during interrupt: %s", exc)

        stream = None
        with self._stream_lock:
            stream = self._active_stream
        if stream is not None:
            for meth in ("abort", "stop", "close"):
                fn = getattr(stream, meth, None)
                if not callable(fn):
                    continue
                try:
                    fn()
                    break
                except Exception:  # noqa: BLE001
                    continue

        if self._sd_stop_fn is not None:
            try:
                self._sd_stop_fn(where=f"barge_in:{reason or 'interrupt'}", blocking=False)
            except TypeError:
                try:
                    self._sd_stop_fn()
                except Exception as exc:  # noqa: BLE001
                    _log.debug("sd.stop failed during interrupt: %s", exc)
            except Exception as exc:  # noqa: BLE001
                _log.debug("sd.stop failed during interrupt: %s", exc)

        if set_listening and self._set_ui_fn is not None:
            try:
                self._set_ui_fn("listening")
            except Exception as exc:  # noqa: BLE001
                _log.debug("UI listening transition failed: %s", exc)

        if reason:
            _log.info("TTS barge-in (%s); flushed=%s", reason, dropped)
        return dropped

    def consume_if_set(self) -> bool:
        """If barge-in latched, clear it and return True (skip next spool item)."""
        if not self._barge_in_event.is_set():
            return False
        self._barge_in_event.clear()
        return True

    def is_tts_busy(self) -> bool:
        if self._busy_fn is None:
            return False
        try:
            return bool(self._busy_fn())
        except Exception:  # noqa: BLE001
            return False


class _PlaybackSession:
    def __init__(self, worker: TtsWorker, *, interruptible: bool) -> None:
        self._worker = worker
        self._interruptible = interruptible

    def __enter__(self) -> TtsWorker:
        self._worker.begin_playback(interruptible=self._interruptible)
        return self._worker

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._worker.end_playback()


_CONTROLLER: TtsWorker | None = None
_CONTROLLER_LOCK = threading.Lock()


def get_tts_worker(*, barge_in_event: threading.Event | None = None) -> TtsWorker:
    """Process-wide TTS barge-in controller (lazy singleton)."""
    global _CONTROLLER
    with _CONTROLLER_LOCK:
        if _CONTROLLER is None:
            _CONTROLLER = TtsWorker(barge_in_event=barge_in_event)
        elif barge_in_event is not None and _CONTROLLER.barge_in_event is not barge_in_event:
            # Keep a single shared Event object with core_agent.
            _CONTROLLER._barge_in_event = barge_in_event
        return _CONTROLLER


# ---------------------------------------------------------------------------
# Piper voice management (model paths, download, cached PiperVoice instances)
# ---------------------------------------------------------------------------

TTS_MODELS_DIR = str(_TTS_DIR)
PIPER_TEMP_WAV = str(TEMP_REPLY_WAV)
# Pre-rendered canned UX acknowledgments (skip live Piper during LLM load).
AUDIO_CACHE_DIR = Path(PROJECT_ROOT) / "dana" / "assets" / "audio_cache"
# Canonical UX phrases → WAV filenames. Lookup uses fuzzy keys (lower + no punct).
_CANNED_UX_WAV_FILES: dict[str, str] = {
    "The ticket is on the board.": "the_ticket_is_on_the_board.wav",
    "Yes?": "yes.wav",
    "Standing by.": "standing_by.wav",
    "I didn't catch that.": "i_didnt_catch_that.wav",
    "Dana is ready.": "dana_is_ready.wav",
    "Developer mode active.": "developer_mode_active.wav",
    "Chat mode active.": "chat_mode_active.wav",
    "Vision mode active.": "vision_mode_active.wav",
    "Research mode active.": "research_mode_active.wav",
    "Memory cleared.": "memory_cleared.wav",
}
# Default voice: en_US-hfc_female-medium (CC BY-NC-SA 4.0 — see docs/LEGAL_AND_IP.md).
# Override via DANA_PIPER_VOICE (e.g. en_US-ljspeech-high for public-domain weights).
PIPER_VOICE_ID = (
    os.environ.get("DANA_PIPER_VOICE", "en_US-hfc_female-medium").strip()
    or "en_US-hfc_female-medium"
)
PIPER_EN_ONNX = os.path.join(TTS_MODELS_DIR, f"{PIPER_VOICE_ID}.onnx")
PIPER_EN_JSON = os.path.join(TTS_MODELS_DIR, f"{PIPER_VOICE_ID}.onnx.json")
DEFAULT_PIPER_ONNX = PIPER_EN_ONNX
# Offline migration fallback if preferred download fails.
_PIPER_LEGACY_VOICE_ID = "en_US-ljspeech-high"
_PIPER_LEGACY_ONNX = os.path.join(TTS_MODELS_DIR, f"{_PIPER_LEGACY_VOICE_ID}.onnx")
_PIPER_LEGACY_JSON = os.path.join(TTS_MODELS_DIR, f"{_PIPER_LEGACY_VOICE_ID}.onnx.json")
# length_scale < 1.0 speeds speech (VITS). Default 0.75 for snappier replies.
try:
    PIPER_LENGTH_SCALE = float(os.environ.get("DANA_PIPER_LENGTH_SCALE", "0.75"))
except ValueError:
    PIPER_LENGTH_SCALE = 0.75
PIPER_LENGTH_SCALE = max(0.5, min(2.0, PIPER_LENGTH_SCALE))
# Incomplete localization voices are disabled for the public release.
# Related local Piper assets remain gitignored under tts_models/.
_PIPER_VOICE_RELPATHS: dict[str, str] = {
    "en_US-ljspeech-high": "ljspeech/high/en_US-ljspeech-high",
    "en_US-ljspeech-medium": "ljspeech/medium/en_US-ljspeech-medium",
    "en_US-lessac-medium": "lessac/medium/en_US-lessac-medium",
    "en_US-hfc_female-medium": "hfc_female/medium/en_US-hfc_female-medium",
}
_PIPER_HF_BASE = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US"
)


def _piper_hf_urls(voice_id: str) -> tuple[tuple[str, str], tuple[str, str]]:
    rel = _PIPER_VOICE_RELPATHS.get(voice_id, f"ljspeech/high/{voice_id}")
    onnx = os.path.join(TTS_MODELS_DIR, f"{voice_id}.onnx")
    js = os.path.join(TTS_MODELS_DIR, f"{voice_id}.onnx.json")
    return (
        (onnx, f"{_PIPER_HF_BASE}/{rel}.onnx"),
        (js, f"{_PIPER_HF_BASE}/{rel}.onnx.json"),
    )


PIPER_MODEL_URLS: tuple[tuple[str, str], ...] = _piper_hf_urls(PIPER_VOICE_ID)
_piper_voice_cache: dict[str, Any] = {}


def _download_file(url: str, dest: str) -> None:
    """Stream-download url to dest (atomic replace)."""
    import requests

    print(f"[TTS] Downloading Piper model -> {os.path.basename(dest)} ...", flush=True)
    with requests.get(url, stream=True, timeout=180) as resp:
        resp.raise_for_status()
        tmp_path = dest + ".partial"
        with open(tmp_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1024 * 256):
                if chunk:
                    fh.write(chunk)
        os.replace(tmp_path, dest)
    print(
        f"[TTS] Saved {os.path.basename(dest)} "
        f"({os.path.getsize(dest) / (1024 * 1024):.1f} MB)",
        flush=True,
    )


def _piper_file_ready(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > 0


def download_piper_models() -> None:
    """Download the English Piper voice into tts_models/ if missing.

    Prefers the configured default (hfc_female-medium). If download fails and a
    legacy ljspeech voice is already on disk, keep that path so offline /
    mocked test environments do not hard-fail on network.
    """
    global PIPER_EN_ONNX, PIPER_EN_JSON, DEFAULT_PIPER_ONNX
    os.makedirs(TTS_MODELS_DIR, exist_ok=True)
    urls = _piper_hf_urls(PIPER_VOICE_ID)
    try:
        for dest, url in urls:
            if _piper_file_ready(dest):
                continue
            try:
                _download_file(url, dest)
            except Exception as exc:  # noqa: BLE001
                try:
                    if os.path.isfile(dest + ".partial"):
                        os.remove(dest + ".partial")
                except OSError:
                    pass
                raise RuntimeError(
                    f"Failed to download Piper model from {url}: {exc}"
                ) from exc
    except RuntimeError:
        if (
            PIPER_VOICE_ID != _PIPER_LEGACY_VOICE_ID
            and _piper_file_ready(_PIPER_LEGACY_ONNX)
            and _piper_file_ready(_PIPER_LEGACY_JSON)
        ):
            PIPER_EN_ONNX = _PIPER_LEGACY_ONNX
            PIPER_EN_JSON = _PIPER_LEGACY_JSON
            DEFAULT_PIPER_ONNX = PIPER_EN_ONNX
            print(
                "[TTS] WARNING: preferred Piper voice download failed; "
                f"falling back to legacy {_PIPER_LEGACY_VOICE_ID}.",
                flush=True,
            )
            return
        raise


def piper_model_path_for_text(text: str) -> str:
    """Always route Piper to the English voice (public release)."""
    _ = text
    return PIPER_EN_ONNX


def get_piper_voice(model_path: str) -> Any:
    """Load (and cache) a PiperVoice for the given .onnx path (lazy onnx import)."""
    voice = _piper_voice_cache.get(model_path)
    if voice is not None:
        return voice
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Piper model missing: {model_path}")
    from piper import PiperVoice

    log("Audio", f"Loading Piper voice: {os.path.basename(model_path)}")
    t_load = time.perf_counter()
    voice = PiperVoice.load(model_path)
    _piper_voice_cache[model_path] = voice
    try:
        from dana.perf import log_perf

        log_perf(
            "piper_voice_load",
            (time.perf_counter() - t_load) * 1000.0,
            model=os.path.basename(model_path),
        )
    except Exception:  # noqa: BLE001
        pass
    return voice


def synthesize_to_file(voice: Any, text: str, path: str) -> bool:
    """Write Piper speech to a WAV path.

    Collects audio from ``voice.synthesize`` first so empty/failed TTS never
    opens a half-initialized ``wave`` writer (``# channels not specified``).

    Returns:
        True when a valid WAV was written; False when TTS produced no audio
        (caller should skip playback).
    """
    from piper.config import SynthesisConfig

    from dana.audio.tts_manager import sanitize_text_for_tts

    # Defaults used when the voice omits format metadata.
    channels = 1
    sampwidth = 2
    framerate = 22050
    try:
        cfg_rate = int(getattr(getattr(voice, "config", None), "sample_rate", 0) or 0)
        if cfg_rate > 0:
            framerate = cfg_rate
    except Exception:  # noqa: BLE001
        pass

    utterance = sanitize_text_for_tts(text or "")
    if not utterance:
        # Empty / markdown-only input — skip without warning spam.
        return False

    chunks: list[Any] = []
    piper_bytes = 0
    t_piper0 = time.perf_counter()
    ttfb_logged = False
    syn_config = SynthesisConfig(length_scale=float(PIPER_LENGTH_SCALE))
    try:
        for chunk in voice.synthesize(utterance, syn_config=syn_config):
            if chunk is None:
                continue
            try:
                raw = chunk.audio_int16_bytes
            except Exception:  # noqa: BLE001
                raw = b""
            if not raw:
                continue
            if not ttfb_logged:
                ttfb_logged = True
                try:
                    from dana.perf import log_perf

                    log_perf(
                        "piper_ttfb",
                        (time.perf_counter() - t_piper0) * 1000.0,
                        chars=len(utterance),
                    )
                except Exception:  # noqa: BLE001
                    pass
            piper_bytes += len(raw)
            chunks.append(chunk)
    except Exception as exc:  # noqa: BLE001
        log(
            "Audio",
            f"WARNING: TTS returned empty audio data, skipping synthesis ({exc})",
        )
        return False

    if not chunks:
        log("Audio", "WARNING: TTS returned empty audio data, skipping synthesis")
        return False

    log_debug(
        "Audio",
        f"Piper synthesize chunks={len(chunks)} bytes={piper_bytes} "
        f"chars={len(utterance)} length_scale={PIPER_LENGTH_SCALE} "
        f"dt_ms={(time.perf_counter() - t_piper0) * 1000.0:.1f}",
    )

    first = chunks[0]
    try:
        channels = int(getattr(first, "sample_channels", None) or channels)
        sampwidth = int(getattr(first, "sample_width", None) or sampwidth)
        framerate = int(getattr(first, "sample_rate", None) or framerate)
    except Exception:  # noqa: BLE001
        pass
    if channels < 1:
        channels = 1
    if sampwidth < 1:
        sampwidth = 2
    if framerate < 1:
        framerate = 22050

    try:
        with wave.open(path, "wb") as wav_file:
            # Explicit format BEFORE any frames (required by wave module).
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sampwidth)
            wav_file.setframerate(framerate)
            for chunk in chunks:
                try:
                    frame_bytes = chunk.audio_int16_bytes
                except Exception:  # noqa: BLE001
                    frame_bytes = b""
                if frame_bytes:
                    wav_file.writeframes(frame_bytes)
    except Exception as exc:  # noqa: BLE001
        log(
            "Audio",
            f"WARNING: TTS returned empty audio data, skipping synthesis ({exc})",
        )
        return False

    try:
        if not os.path.isfile(path) or os.path.getsize(path) < 44:
            log("Audio", "WARNING: TTS returned empty audio data, skipping synthesis")
            return False
    except OSError:
        log("Audio", "WARNING: TTS returned empty audio data, skipping synthesis")
        return False
    return True


# ---------------------------------------------------------------------------
# Barge-in coordination / hardware fault recovery
# ---------------------------------------------------------------------------


def _safe_sd_stop(*, where: str = "", blocking: bool = True) -> None:
    """Stop PortAudio playback under ``playback_lock`` (best-effort)."""
    from dana.core import shared_state as state

    acquired = False
    try:
        acquired = (
            state.playback_lock.acquire(blocking=blocking)
            if blocking
            else state.playback_lock.acquire(blocking=False)
        )
        if not acquired:
            log_debug("Audio", f"sd.stop deferred (lock busy) where={where or '-'}")
            return
        t0 = time.perf_counter()
        try:
            sd.stop()
        except Exception as exc:  # noqa: BLE001
            log_debug("Audio", f"sd.stop ignored ({where or '-'}): {exc}")
        else:
            log_debug(
                "Audio",
                f"sd.stop ok where={where or '-'} dt_ms={(time.perf_counter() - t0) * 1000.0:.1f}",
            )
    finally:
        if acquired:
            state.playback_lock.release()


def _bind_tts_barge_controller() -> None:
    """Wire callbacks into the shared ``TtsWorker`` (idempotent)."""

    from dana.core import shared_state as state

    def _reset_stream() -> None:
        try:
            from dana.agentic import reset_stream_sentence_tts

            reset_stream_sentence_tts()
        except Exception:  # noqa: BLE001
            pass

    state._tts_barge.bind(
        flush_fn=_flush_tts_queue_lazy,
        sd_stop_fn=_safe_sd_stop,
        set_ui_fn=state.set_ui_state,
        reset_stream_fn=_reset_stream,
        busy_fn=state.tts_busy.is_set,
    )


def _flush_tts_queue_lazy() -> int:
    from dana.audio.tts_manager import flush_tts_queue

    return flush_tts_queue()


def interrupt_tts(*, reason: str = "barge_in", force: bool = False) -> int:
    """Hard-stop TTS: latch barge-in, flush spool, abort PortAudio stream."""
    from dana.core import shared_state as state

    _bind_tts_barge_controller()
    return int(
        state._tts_barge.interrupt(reason=reason, set_listening=True, force=force)
    )


def _wait_tts_clear_of_user_speech(text: str) -> bool:
    """Hold a dequeued phrase while VAD capture is active.

    Returns False if the phrase should be discarded (interrupt / shutdown /
    hold timeout) instead of spoken.
    """
    from dana.core import shared_state as state

    if not state.vad_capture_active.is_set():
        return True
    log_debug(
        "TTS",
        f"holding spool item while user speaks chars={len(text)} "
        f"(max={state._TTS_HOLD_FOR_VAD_MAX_S:.0f}s)",
    )
    deadline = time.perf_counter() + state._TTS_HOLD_FOR_VAD_MAX_S
    while state.vad_capture_active.is_set() and not state.stop_event.is_set():
        if state.tts_interrupt_event.is_set():
            log_debug("TTS", "discard held spool item (barge-in during VAD hold)")
            return False
        if time.perf_counter() >= deadline:
            log(
                "TTS",
                "WARNING: discard held spool item — user speech exceeded hold window",
            )
            return False
        time.sleep(0.05)
    if state.stop_event.is_set() or state.tts_interrupt_event.is_set():
        return False
    return True


def reset_tts_audio_state(
    reason: str = "",
    *,
    ui_state: str = "idle",
    flush_queue: bool = True,
) -> int:
    """Force-release TTS / PortAudio locks after timeout or hung Piper playback.

    Without this, ``speech_idle`` stays cleared and ``tts_busy`` may remain set,
    which permanently blocks the wake-word listener.
    """
    from dana.core import shared_state as state

    _bind_tts_barge_controller()
    if flush_queue:
        dropped = interrupt_tts(reason=f"reset:{reason or 'unspecified'}")
    else:
        state.tts_interrupt_event.set()
        _safe_sd_stop(where=f"reset_tts:{reason or 'unspecified'}")
        dropped = 0
    state.tts_busy.clear()
    state.speech_idle.set()
    # Do not clear vad_capture_active here — record_utterance may still own the mic.
    try:
        state.set_ui_state(ui_state)
    except Exception:  # noqa: BLE001
        pass
    if reason:
        log(
            "Audio",
            f"TTS state reset ({reason}); flushed {dropped} queued item(s); "
            f"-> {ui_state}/listening",
        )
    return dropped


def wait_for_speech_idle(timeout: float = 20.0) -> None:
    """Block until queued TTS has finished playing (or timeout + hard recovery)."""
    from dana.core import shared_state as state

    if state.speech_idle.wait(timeout=timeout):
        return
    try:
        reset_tts_audio_state(f"timed out waiting for TTS after {timeout:.1f}s")
    finally:
        # Guarantee wake-word gates even if interrupt/flush races a worker.
        state.tts_busy.clear()
        state.speech_idle.set()


def report_audio_hardware_fault(exc: BaseException, *, where: str = "audio") -> None:
    """Signal Main that PortAudio/hardware failed so it can soft-recover before freeze."""
    from dana.core import shared_state as state

    detail = f"{where}: {type(exc).__name__}: {exc}"
    with state._audio_hardware_fault_lock:
        state._audio_hardware_fault_detail = detail
    state.audio_hardware_fault.set()
    log_exception("Audio", f"TTS Engine Failure / PortAudio fault ({where})", exc=exc)


def consume_audio_hardware_fault() -> str:
    """Return and clear the pending hardware-fault detail (empty if none)."""
    from dana.core import shared_state as state

    if not state.audio_hardware_fault.is_set():
        return ""
    with state._audio_hardware_fault_lock:
        detail = state._audio_hardware_fault_detail
        state._audio_hardware_fault_detail = ""
    state.audio_hardware_fault.clear()
    return detail


def soft_recover_audio_hardware(detail: str = "") -> None:
    """Main-loop soft restart after PaErrorCode: release locks and log device state."""
    from dana.core import shared_state as state

    from dana.audio.mic_input import (
        _device_rate,
        ensure_mic_ingest_thread,
        flush_audio_buffer_queue,
        list_input_devices,
        list_output_devices,
        request_mic_ingest_restart,
    )

    reason = detail or "PortAudio hardware fault"
    log("Main", f"Audio hardware fault — soft restart ({reason})")
    reset_tts_audio_state(f"hardware fault soft-restart: {reason}", ui_state="idle")
    flush_audio_buffer_queue()
    try:
        list_input_devices()
        list_output_devices()
    except Exception as exc:  # noqa: BLE001
        log_exception("Main", "Failed listing audio devices during soft restart", exc=exc)
    # Autonomous audio: always rebind System Default (device=None).
    state.AUDIO_INPUT_DEVICE = None
    state.AUDIO_OUTPUT_DEVICE = None
    state.AUDIO_INPUT_RATE = _device_rate(None)
    try:
        from dana.audio.devices import get_default_audio_devices

        din, dout = get_default_audio_devices()
        log("Main", f"Soft restart OS defaults: in={din} out={dout}")
    except Exception as exc:  # noqa: BLE001
        log_exception("Main", "Failed re-querying default audio devices", exc=exc)
    try:
        # Nudge PortAudio to drop stale streams.
        sd.stop()
    except Exception:  # noqa: BLE001
        pass
    request_mic_ingest_restart()
    ensure_mic_ingest_thread()
    log("Main", "Audio soft restart complete — returning to idle/listening")


def _is_portaudio_error(exc: BaseException) -> bool:
    """True for sounddevice PortAudioError or messages carrying PaErrorCode."""
    name = type(exc).__name__
    if name == "PortAudioError":
        return True
    msg = str(exc)
    return "PaErrorCode" in msg or "PortAudio" in msg


# ---------------------------------------------------------------------------
# PCM playback
# ---------------------------------------------------------------------------


def _device_output_samplerate(output_device: Optional[int]) -> Optional[int]:
    """Host default output sample rate for ``output_device`` (if queryable)."""
    try:
        info = sd.query_devices(output_device if output_device is not None else None)
        rate = int(float(info.get("default_samplerate") or 0))
        return rate if rate > 0 else None
    except Exception:  # noqa: BLE001
        return None


def _resample_pcm(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear-resample mono float32 PCM when host rate != Piper rate.

    Playing 22050 Hz buffers on a 44100 Hz WASAPI path without resampling
    can make speech sound ~2x too fast (common with virtual mixers like Sonar).
    """
    src = int(src_rate)
    dst = int(dst_rate)
    if src <= 0 or dst <= 0 or src == dst:
        return np.asarray(audio, dtype=np.float32).reshape(-1)
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size < 2:
        return samples
    n_out = max(1, int(round(samples.size * float(dst) / float(src))))
    x_old = np.linspace(0.0, 1.0, num=samples.size, endpoint=False, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False, dtype=np.float64)
    return np.interp(x_new, x_old, samples).astype(np.float32)


def _play_pcm_interruptible(
    audio_data: np.ndarray,
    samplerate: int,
    output_device: Optional[int],
    *,
    interruptible: bool = True,
) -> bool:
    """Stream PCM to the speaker in short chunks; return True if barge-in aborted.

    Holds ``playback_lock`` for the OutputStream lifecycle so a second utterance
    cannot open the device while chunks are still draining. Barge-in registers
    the live stream so ``interrupt_tts()`` can ``abort()`` without waiting on
    this lock (avoids deferred ``sd.stop`` races).

    ``interruptible=False`` plays UI acknowledgments without arming barge-in.
    """
    from dana.core import shared_state as state

    audio = np.asarray(audio_data, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = audio.reshape(-1)
    if audio.size == 0:
        return False

    from dana.audio.devices import get_default_audio_devices, stream_device_kwargs

    # Autonomous audio: always System Default (device=None).
    try:
        _din, dout = get_default_audio_devices()
    except Exception:  # noqa: BLE001
        dout = None
    play_device: Optional[int] = None
    state.AUDIO_OUTPUT_DEVICE = None
    if dout is not None:
        log_debug("Audio", f"System Default speaker current=[{dout}]")

    play_rate = int(samplerate)
    host_rate = _device_output_samplerate(None)
    if host_rate is not None and host_rate != play_rate:
        log_debug(
            "Audio",
            f"resample PCM {play_rate} Hz -> {host_rate} Hz "
            f"(device=None) to prevent sped-up playback",
        )
        audio = _resample_pcm(audio, play_rate, host_rate)
        play_rate = host_rate

    _bind_tts_barge_controller()
    chunk = max(1, int(round(play_rate * (BARGE_IN_CHUNK_MS / 1000.0))))
    stream_kwargs: dict[str, Any] = {
        "samplerate": int(play_rate),
        "channels": 1,
        "dtype": "float32",
        "blocksize": chunk,
    }
    stream_kwargs.update(stream_device_kwargs(None))

    interrupted = False
    used_fallback = False
    t_start = time.perf_counter()
    n_chunks = 0
    bytes_written = 0
    log_debug(
        "Audio",
        f"playback alloc samples={audio.size} sr={play_rate} "
        f"chunk={chunk} ({BARGE_IN_CHUNK_MS:.0f}ms) device={play_device} "
        f"interruptible={interruptible}",
    )
    # Honor the turn-level latch when the spooler already called begin_playback;
    # otherwise (unit tests / direct play) open a local session.
    owns_session = not state._tts_barge.is_playback_active()
    if owns_session:
        state._tts_barge.begin_playback(interruptible=interruptible)

    def _write_chunks(stream: Any) -> None:
        from dana.core import shared_state as state

        nonlocal interrupted, n_chunks, bytes_written
        if interruptible:
            state._tts_barge.register_output_stream(stream)
        try:
            for start in range(0, audio.size, chunk):
                if state.stop_event.is_set():
                    interrupted = True
                    break
                if interruptible and state._tts_barge.is_set():
                    interrupted = True
                    log_debug(
                        "Audio",
                        f"playback interrupt at chunk={n_chunks} "
                        f"offset={start}/{audio.size}",
                    )
                    try:
                        stream.abort()
                    except Exception:  # noqa: BLE001
                        pass
                    break
                piece = audio[start : start + chunk]
                if piece.size < chunk:
                    pad = np.zeros(chunk, dtype=np.float32)
                    pad[: piece.size] = piece
                    piece = pad
                stream.write(piece.reshape(-1, 1))
                n_chunks += 1
                bytes_written += int(piece.size) * 4
        finally:
            if interruptible:
                state._tts_barge.unregister_output_stream(stream)

    try:
        with state.playback_lock:
            log_debug(
                "Audio",
                f"playback start t={t_start:.3f} samples={audio.size}",
            )
            try:
                with sd.OutputStream(**stream_kwargs) as stream:
                    _write_chunks(stream)
            except Exception as exc:  # noqa: BLE001
                if interrupted or (interruptible and state._tts_barge.is_set()):
                    interrupted = True
                elif _is_portaudio_error(exc):
                    # Re-query OS defaults and retry System Default without crashing.
                    log(
                        "Audio",
                        f"PortAudioError on System Default speaker ({exc}); "
                        "re-querying defaults and retrying",
                    )
                    state.AUDIO_OUTPUT_DEVICE = None
                    play_device = None
                    try:
                        get_default_audio_devices()
                    except Exception:  # noqa: BLE001
                        pass
                    fallback_kwargs = {
                        "samplerate": int(play_rate),
                        "channels": 1,
                        "dtype": "float32",
                        "blocksize": chunk,
                    }
                    fallback_kwargs.update(stream_device_kwargs(None))
                    try:
                        with sd.OutputStream(**fallback_kwargs) as stream:
                            _write_chunks(stream)
                    except Exception as exc2:  # noqa: BLE001
                        if _is_portaudio_error(exc2):
                            report_audio_hardware_fault(exc2, where="OutputStream")
                            # Soft recover on worker thread; do not crash main.
                            try:
                                soft_recover_audio_hardware(str(exc2))
                            except Exception:  # noqa: BLE001
                                pass
                        else:
                            raise
                else:
                    # Fallback: start play under lock, then poll outside so barge-in can stop.
                    log(
                        "Audio",
                        f"WARNING: interruptible OutputStream failed ({exc}); using sd.play",
                    )
                    try:
                        kwargs: dict[str, Any] = {
                            "samplerate": int(play_rate),
                            "blocking": False,
                        }
                        kwargs.update(stream_device_kwargs(None))
                        sd.play(audio, **kwargs)
                        used_fallback = True
                    except Exception as exc2:  # noqa: BLE001
                        if _is_portaudio_error(exc2):
                            log(
                                "Audio",
                                "PortAudioError on sd.play System Default; "
                                "retrying once after re-query",
                            )
                            try:
                                get_default_audio_devices()
                            except Exception:  # noqa: BLE001
                                pass
                            try:
                                sd.play(audio, samplerate=int(play_rate), blocking=False)
                                used_fallback = True
                                state.AUDIO_OUTPUT_DEVICE = None
                            except Exception as exc3:  # noqa: BLE001
                                if _is_portaudio_error(exc3):
                                    report_audio_hardware_fault(exc3, where="sd.play")
                                    try:
                                        soft_recover_audio_hardware(str(exc3))
                                    except Exception:  # noqa: BLE001
                                        pass
                                else:
                                    log_exception("Audio", "TTS Engine Failure", exc=exc3)
                        else:
                            log_exception("Audio", "TTS Engine Failure", exc=exc2)

        if used_fallback and not interrupted:
            duration = audio.size / float(max(1, play_rate))
            deadline = time.perf_counter() + duration + 0.5
            log_debug("Audio", f"playback fallback sd.play duration_s={duration:.2f}")
            while time.perf_counter() < deadline:
                if state.stop_event.is_set():
                    interrupted = True
                    break
                if interruptible and state._tts_barge.is_set():
                    interrupted = True
                    _safe_sd_stop(where="playback_fallback_interrupt", blocking=False)
                    break
                time.sleep(0.03)
            else:
                try:
                    with state.playback_lock:
                        sd.wait()
                except Exception:
                    pass
    finally:
        if owns_session:
            state._tts_barge.end_playback()

    log_debug(
        "Audio",
        f"playback end interrupted={interrupted} chunks={n_chunks} "
        f"bytes≈{bytes_written} fallback={used_fallback} "
        f"dt_ms={(time.perf_counter() - t_start) * 1000.0:.1f}",
    )
    return interrupted


def half_duplex_mic_drop(stop_flag: threading.Event) -> None:
    """Half-duplex: discard mic frames while TTS plays (no barge-in evaluation).

    Speaker echo / room bleed must never call ``TtsWorker.interrupt()``. The
    mic queue is flushed continuously until playback ends; wake-word and VAD
    stand down on the same shared stream for the duration.
    """
    from dana.core import shared_state as state

    from dana.audio.mic_input import flush_audio_buffer_queue, get_mic_frame

    if state.vad_capture_active.is_set():
        return
    flush_audio_buffer_queue()
    try:
        while (
            not stop_flag.is_set()
            and not state.stop_event.is_set()
            and state.tts_busy.is_set()
            and not state.vad_capture_active.is_set()
        ):
            # Drain then discard — do not run Silero / wake scoring.
            frame = get_mic_frame(timeout=0.05)
            if frame is None:
                flush_audio_buffer_queue()
                time.sleep(0.01)
                continue
            # Drop frame (half-duplex deaf period).
    except Exception as exc:  # noqa: BLE001
        log_debug("BargeIn", f"half-duplex mic drop unavailable ({exc})")
    finally:
        flush_audio_buffer_queue()


# Back-compat alias — former stream-barge watcher is now half-duplex only.
barge_in_watch = half_duplex_mic_drop


def _boost_audio_thread_priority() -> None:
    """Raise OS priority so local LLM inference does not starve the sound buffer."""
    try:
        if os.name == "nt":
            import ctypes

            # THREAD_PRIORITY_ABOVE_NORMAL = 1
            handle = ctypes.windll.kernel32.GetCurrentThread()
            ctypes.windll.kernel32.SetThreadPriority(handle, 1)
        else:
            # Best-effort: nicer values are lower; negative raises priority when permitted.
            os.nice(-5)
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Canned UX audio cache
# ---------------------------------------------------------------------------


def _normalize_canned_ux_key(text: str) -> str:
    """Lowercase + strip common punctuation so cache hits ignore trailing marks."""
    import re

    from dana.audio.tts_manager import sanitize_text_for_tts

    key = sanitize_text_for_tts(text or "")
    key = key.lower()
    key = re.sub(r"[.,?!;:\"'`…]+", "", key)
    key = re.sub(r"\s+", " ", key).strip()
    return key


# Fuzzy lookup: normalized phrase → WAV filename (built once from canon map).
_CANNED_UX_FUZZY_WAV: dict[str, str] = {
    _normalize_canned_ux_key(phrase): filename
    for phrase, filename in _CANNED_UX_WAV_FILES.items()
}


def canned_ux_cache_path(text: str) -> Optional[Path]:
    """Return cache WAV path when ``text`` fuzzy-matches a canned UX acknowledgment."""
    key = _normalize_canned_ux_key(text or "")
    if not key:
        return None
    filename = _CANNED_UX_FUZZY_WAV.get(key)
    if not filename:
        return None
    return AUDIO_CACHE_DIR / filename


def ensure_canned_ux_audio_cache() -> None:
    """Pre-synthesize standard UX WAV files under ``dana/assets/audio_cache/``."""
    AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for phrase, filename in _CANNED_UX_WAV_FILES.items():
        path = AUDIO_CACHE_DIR / filename
        if path.is_file() and path.stat().st_size > 44:
            continue
        try:
            model_path = piper_model_path_for_text(phrase)
            voice = get_piper_voice(model_path)
            if synthesize_to_file(voice, phrase, str(path)):
                log("TTS", f"Cached UX audio: {filename}")
            else:
                log("TTS", f"WARNING: failed to cache UX audio for {phrase!r}")
        except Exception as exc:  # noqa: BLE001
            log("TTS", f"WARNING: UX audio cache skip ({filename}): {exc}")


def _play_ready_chime(output_device: Optional[int]) -> bool:
    """Short mechanical ready tone — no Piper during peak startup CPU."""
    sr = 22050
    duration_s = 0.16
    n = max(1, int(sr * duration_s))
    t = np.linspace(0.0, duration_s, n, endpoint=False, dtype=np.float32)
    tone = (0.22 * np.sin(2.0 * np.pi * 880.0 * t)).astype(np.float32)
    # Soft attack/release so the chime does not click.
    fade = min(n // 8, int(0.02 * sr))
    if fade > 0:
        tone[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
        tone[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return _play_pcm_interruptible(tone, sr, output_device)


def _play_cached_wav(
    path: Path,
    output_device: Optional[int],
    *,
    interruptible: bool = True,
) -> bool:
    """Play a pre-rendered WAV via the interruptible PCM path (no Piper)."""
    audio_data, samplerate = sf.read(str(path), dtype="float32")
    if getattr(audio_data, "size", 0) == 0:
        log("TTS", f"WARNING: empty cached WAV, falling back to Piper: {path.name}")
        raise ValueError("empty cached wav")
    frames = int(np.asarray(audio_data).reshape(-1).shape[0])
    log_debug(
        "TTS",
        f"cache hit {path.name} frames={frames} sr={samplerate} "
        f"interruptible={interruptible}",
    )
    interrupted = _play_pcm_interruptible(
        np.asarray(audio_data, dtype=np.float32),
        int(samplerate),
        output_device,
        interruptible=interruptible,
    )
    if interrupted:
        _safe_sd_stop(where="tts_cache_interrupted", blocking=False)
        log("TTS", "playback interrupted (barge-in)")
    else:
        time.sleep(0.08)
        log_debug(
            "TTS",
            f"Playback finished ({frames / float(samplerate):.2f}s) [cache]",
        )
    return interrupted


def _synthesize_and_play(
    text: str,
    output_device: Optional[int],
    *,
    interruptible: bool = True,
    agent_id: str | None = None,
) -> bool:
    """Worker-only: synth + interruptible playback under ``playback_lock``.

    Returns True if barge-in aborted playback. Producers must use ``enqueue_speech``.
    Canned UX strings play from ``dana/assets/audio_cache/`` when available.
    Stage 8.8 — specialized ``agent_id`` voices bypass the receptionist Piper path.
    """
    from dana.core import shared_state as state

    from dana.audio.tts_manager import sanitize_text_for_tts

    text = sanitize_text_for_tts(text or "")
    if not text:
        return False

    try:
        from dana.audio.multi_voice_tts import (
            set_active_tts_agent,
            synthesize_speech,
            uses_receptionist_piper,
        )

        set_active_tts_agent(agent_id or "broker")
        use_piper = uses_receptionist_piper(agent_id)
    except Exception:  # noqa: BLE001
        use_piper = True

    # Specialized personas: multi-voice WAV → same interruptible PCM path.
    if not use_piper:
        tmp_path: str | None = None
        try:
            wav = synthesize_speech(text, agent_id=agent_id)
            tmp_path = str(wav)
            return _play_cached_wav(
                Path(tmp_path), output_device, interruptible=interruptible
            )
        except Exception as exc:  # noqa: BLE001
            log_debug("TTS", f"multi-voice play failed ({exc}); Piper fallback")
        finally:
            if tmp_path:
                try:
                    if os.path.isfile(tmp_path) and "dana_" in os.path.basename(tmp_path):
                        os.remove(tmp_path)
                except OSError:
                    pass

    cache_path = canned_ux_cache_path(text)
    if cache_path is not None and cache_path.is_file() and cache_path.stat().st_size > 44:
        try:
            return _play_cached_wav(
                cache_path, output_device, interruptible=interruptible
            )
        except Exception as exc:  # noqa: BLE001
            log_debug("TTS", f"cache play failed ({exc}); live Piper fallback")

    model_path = piper_model_path_for_text(text)
    lang = "en"
    t_synth0 = time.perf_counter()
    log_debug(
        "TTS",
        f"Piper route -> {lang} ({os.path.basename(model_path)}) chars={len(text)} "
        f"interruptible={interruptible} agent={agent_id or 'broker'}",
    )

    voice = get_piper_voice(model_path)
    out_path = PIPER_TEMP_WAV
    interrupted = False
    try:
        try:
            if not synthesize_to_file(voice, text, out_path):
                log("TTS", "WARNING: skipping playback — Piper produced no audio")
                return False
            # Persist newly rendered canned UX into the cache for next time.
            if cache_path is not None:
                try:
                    AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    import shutil

                    shutil.copyfile(out_path, str(cache_path))
                except Exception as exc:  # noqa: BLE001
                    log_debug("TTS", f"cache write skipped: {exc}")
            audio_data, samplerate = sf.read(out_path, dtype="float32")
            if getattr(audio_data, "size", 0) == 0:
                log("TTS", "WARNING: TTS returned empty audio data, skipping synthesis")
                return False
            frames = int(np.asarray(audio_data).reshape(-1).shape[0])
            log_debug(
                "TTS",
                f"Piper buffer ready frames={frames} sr={samplerate} "
                f"synth_ms={(time.perf_counter() - t_synth0) * 1000.0:.1f}",
            )

            interrupted = _play_pcm_interruptible(
                np.asarray(audio_data, dtype=np.float32),
                int(samplerate),
                output_device,
                interruptible=interruptible,
            )
            if interrupted:
                _safe_sd_stop(where="tts_play_interrupted", blocking=False)
                log("TTS", "playback interrupted (barge-in)")
            else:
                time.sleep(0.08)
                log_debug(
                    "TTS",
                    f"Playback finished ({frames / float(samplerate):.2f}s)",
                )
        except Exception as exc:  # noqa: BLE001
            if _is_portaudio_error(exc):
                if not state.audio_hardware_fault.is_set():
                    report_audio_hardware_fault(exc, where="tts_worker/Piper")
            else:
                log_exception("TTS", "TTS Engine Failure", exc=exc)
            raise
    finally:
        try:
            if os.path.isfile(out_path):
                os.remove(out_path)
        except OSError:
            pass
    return interrupted


def speak_text(text: str, output_device: Optional[int] = None) -> bool:
    """Legacy name — producers should prefer ``enqueue_speech`` (non-blocking).

    When called off the TTS worker, this enqueues and returns False (not interrupted).
    The TTS worker calls ``_synthesize_and_play`` directly.
    """
    from dana.core import shared_state as state

    from dana.audio.tts_manager import enqueue_speech_impl

    if threading.current_thread() is state._tts_worker_thread:
        return _synthesize_and_play(
            text, output_device if output_device is not None else state.AUDIO_OUTPUT_DEVICE
        )
    enqueue_speech_impl(text)
    return False


def _speak_with_timeout(
    text: str,
    output_device: Optional[int],
    *,
    max_seconds: float = TTS_UTTERANCE_MAX_SECONDS,
    interruptible: bool = True,
    agent_id: str | None = None,
) -> bool:
    """Play on a watchdog-guarded helper thread; abort if it exceeds ``max_seconds``."""
    result: list[bool] = [False]
    error: list[BaseException | None] = [None]

    def _run() -> None:
        _boost_audio_thread_priority()
        try:
            result[0] = bool(
                _synthesize_and_play(
                    text,
                    output_device,
                    interruptible=interruptible,
                    agent_id=agent_id,
                )
            )
        except BaseException as exc:  # noqa: BLE001
            error[0] = exc

    worker = threading.Thread(target=_run, name="TTSUtterance", daemon=True)
    worker.start()
    worker.join(timeout=max_seconds)
    if worker.is_alive():
        log(
            "TTS",
            f"WARNING: utterance exceeded {max_seconds:.0f}s — "
            "aborting playback and releasing audio device",
        )
        # Hard timeout always wins — even UX acks must not hang forever.
        interrupt_tts(reason="utterance_timeout", force=True)
        worker.join(timeout=2.0)
        if worker.is_alive():
            log(
                "TTS",
                "WARNING: utterance thread still alive after abort — forcing state reset",
            )
            reset_tts_audio_state(
                "hung TTSUtterance thread",
                ui_state="listening",
                flush_queue=False,
            )
        return True
    if error[0] is not None:
        raise error[0]
    return result[0]


def maybe_play_boot_ready_audio() -> None:
    """Play ``Dana is ready.`` only once after Ollama + Piper + wake-word arm.

    Safe to call from multiple boot threads; fires at most once per process.
    Never replay on quiet-mic re-arm or mid-session mic energy.
    """
    from dana.core import shared_state as state

    from dana.audio.tts_manager import enqueue_speech_impl

    if state._boot_ready_audio_played:
        return
    if (os.environ.get("DANA_SKIP_BOOT_READY") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        state._boot_ready_audio_played = True
        return
    if not (
        state.ollama_ready.is_set()
        and state.piper_voices_ready.is_set()
        and state.wakeword_armed.is_set()
    ):
        return
    with state._boot_ready_audio_lock:
        if state._boot_ready_audio_played:
            return
        state._boot_ready_audio_played = True
    log(
        "TTS",
        "Boot complete (ollama_ready + Piper + wake armed) — playing ready signal",
    )
    try:
        enqueue_speech_impl("Dana is ready.")
    except Exception as exc:  # noqa: BLE001
        log("TTS", f"WARNING: boot ready enqueue failed ({exc})")


def tts_worker() -> None:
    """TTS consumer: block on ``tts_queue``, honor VAD hold, then play under lock.

    Piper onnx is loaded on first speak only — never at import or spooler boot.
    """
    from dana.core import shared_state as state

    from dana.audio.tts_manager import _parse_tts_spool_item, flush_tts_queue
    # Not yet migrated (CLI/process-lifecycle bucket, Phase 7) — trivial,
    # side-effect-only, no shared state; safe as a temporary bridge import.
    from dana.core_agent import _nt_hide_console_if_mp_child

    state._tts_worker_thread = threading.current_thread()
    _nt_hide_console_if_mp_child()
    _boost_audio_thread_priority()
    log("TTS", "Initializing offline Piper TTS spooler...")
    try:
        download_piper_models()
    except Exception as exc:  # noqa: BLE001
        log("TTS", f"ERROR downloading Piper models: {exc}")
        state.stop_event.set()
        return

    if state.AUDIO_OUTPUT_DEVICE is not None:
        try:
            out_name = sd.query_devices()[state.AUDIO_OUTPUT_DEVICE].get("name", "?")
        except Exception:
            out_name = "?"
        log("TTS", f"playback device [{state.AUDIO_OUTPUT_DEVICE}] {out_name}")
    else:
        log("TTS", "playback device: system default")

    # Model files on disk are enough to arm the spooler; onnx loads on first speak.
    if not _piper_file_ready(PIPER_EN_ONNX):
        log("TTS", f"ERROR: Piper model missing after download: {PIPER_EN_ONNX}")
        state.stop_event.set()
        return
    log(
        "TTS",
        f"Piper model present ({os.path.basename(PIPER_EN_ONNX)}); "
        "onnx load deferred until first speak.",
    )

    # Defer ready audio until Ollama warm-up + wake-word arming also complete.
    state.piper_voices_ready.set()
    if state.tts_queue.empty() and not state.tts_busy.is_set():
        state.speech_idle.set()
    maybe_play_boot_ready_audio()

    log("TTS", "spooler ready; waiting for messages (barge-in armed).")
    _bind_tts_barge_controller()
    while not state.stop_event.is_set():
        try:
            try:
                raw_item = state.tts_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if raw_item is None:
                state.speech_idle.set()
                break

            text, item_interruptible, item_agent = _parse_tts_spool_item(raw_item)

            # Drop orphaned spool items after a barge-in latch.
            if state._tts_barge.is_set():
                flush_tts_queue()
                state._tts_barge.clear()
                if state.tts_queue.empty() and not state.tts_busy.is_set():
                    state.speech_idle.set()
                continue

            # Hold while the user is speaking — do not overlap mic capture.
            if not _wait_tts_clear_of_user_speech(text):
                if state.tts_queue.empty() and not state.tts_busy.is_set():
                    state.speech_idle.set()
                continue

            state._tts_barge.clear()
            state.tts_busy.set()
            state.speech_idle.clear()
            # Latch exemption BEFORE any mic/VAD path can see tts_busy (self-barge race).
            state._tts_barge.begin_playback(interruptible=item_interruptible)
            prev_ui = state.get_ui_state()
            state.set_ui_state("speaking")
            watcher_stop = threading.Event()
            # Half-duplex: always discard mic frames during TTS (acks + LLM speech).
            watcher = threading.Thread(
                target=half_duplex_mic_drop,
                args=(watcher_stop,),
                name="HalfDuplexMicDrop",
                daemon=True,
            )
            watcher.start()
            interrupted = False
            turn_t0 = time.perf_counter()
            time.sleep(0.05)
            try:
                log_debug(
                    "TTS",
                    f'play start t={turn_t0:.3f} chars={len(text)} '
                    f'interruptible={item_interruptible} agent={item_agent} '
                    f'pending={state.tts_queue.qsize()} preview="{text[:80]}"',
                )
                interrupted = bool(
                    _speak_with_timeout(
                        text,
                        state.AUDIO_OUTPUT_DEVICE,
                        interruptible=item_interruptible,
                        agent_id=item_agent,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                if _is_portaudio_error(exc):
                    if not state.audio_hardware_fault.is_set():
                        report_audio_hardware_fault(exc, where="tts_worker")
                else:
                    log_exception("TTS", "TTS Engine Failure", exc=exc)
                interrupted = True
            finally:
                watcher_stop.set()
                if watcher is not None:
                    try:
                        watcher.join(timeout=1.0)
                    except Exception:
                        pass
                _safe_sd_stop(where="tts_worker_turn_end", blocking=False)
                barged = bool(
                    item_interruptible
                    and (interrupted or state.tts_interrupt_event.is_set())
                )
                state.tts_busy.clear()
                state._tts_barge.end_playback()
                if barged:
                    # Instantly dump any pending system messages after cut-off.
                    dropped = flush_tts_queue()
                    from dana.audio.mic_input import flush_audio_buffer_queue

                    flush_audio_buffer_queue()
                    state.tts_interrupt_event.clear()
                    state.set_ui_state("listening")
                    state.speech_idle.set()
                    log(
                        "BargeIn",
                        f"Flushed {dropped} TTS spool item(s); state -> listening "
                        f"dt_ms={(time.perf_counter() - turn_t0) * 1000.0:.1f}",
                    )
                else:
                    if prev_ui in ("thinking", "transcribing", "followup", "listening"):
                        state.set_ui_state(prev_ui)
                    elif prev_ui == "speaking":
                        state.set_ui_state("listening")
                    else:
                        # Boot ready / standby TTS starts from idle — must return
                        # to idle or WakeWord never consumes mic frames.
                        state.set_ui_state("idle")
                    if state.tts_queue.empty():
                        state.speech_idle.set()
                    else:
                        state.speech_idle.clear()
                    log_debug(
                        "TTS",
                        f"play end ok dt_ms="
                        f"{(time.perf_counter() - turn_t0) * 1000.0:.1f} "
                        f"pending={state.tts_queue.qsize()} ui={state.get_ui_state()}",
                    )
        except Exception as exc:  # noqa: BLE001
            log_exception("TTS", "TTS Engine Failure (spooler)", exc=exc)
            if _is_portaudio_error(exc):
                report_audio_hardware_fault(exc, where="TTS spooler")
            reset_tts_audio_state(
                f"TTS spooler error: {exc}",
                ui_state="listening",
            )

    log("TTS", "spooler stopped.")


def audio_worker() -> None:
    """Backward-compatible entrypoint — runs the TTS output spooler consumer."""
    tts_worker()


__all__ = (
    "TtsWorker",
    "canned_ux_cache_path",
    "download_piper_models",
    "ensure_canned_ux_audio_cache",
    "get_piper_voice",
    "get_tts_worker",
    "half_duplex_mic_drop",
    "interrupt_tts",
    "maybe_play_boot_ready_audio",
    "piper_model_path_for_text",
    "reset_tts_audio_state",
    "soft_recover_audio_hardware",
    "speak_text",
    "synthesize_to_file",
    "tts_worker",
    "wait_for_speech_idle",
)
