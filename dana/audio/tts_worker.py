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









from dana.logging import log, log_debug

from dana.paths import PROJECT_ROOT

from dana.paths import TTS_MODELS_DIR as _TTS_DIR



_log = logging.getLogger("dana.audio.tts")



_FlushFn = Callable[[], int]

_StopFn = Callable[..., None]

_UiFn = Callable[[str], None]

_ResetStreamFn = Callable[[], None]



# Default matches core_agent.BARGE_IN_PLAYBACK_GRACE_MS (speaker-onset bleed).


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

        self._flush_fn: _FlushFn | None = None

        self._sd_stop_fn: _StopFn | None = None

        self._set_ui_fn: _UiFn | None = None

        self._reset_stream_fn: _ResetStreamFn | None = None



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



    def is_playback_interruptible(self) -> bool:

        with self._playback_lock:

            # Idle / between utterances → allow barge-in arming for the next turn.

            if not self._playback_active:

                return True

            return bool(self._playback_interruptible)



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

