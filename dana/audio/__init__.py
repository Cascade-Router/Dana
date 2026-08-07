"""Audio helpers (Silero VAD + TTS barge-in / playback interrupt)."""

from dana.audio.devices import (
    SYSTEM_DEFAULT_LABEL,
    get_default_audio_devices,
    resolve_live_input_device,
    stream_device_kwargs,
)
from dana.audio.noise_floor import (
    ABSOLUTE_MIN_SPEECH_FLOOR,
    NOISE_FLOOR_MULTIPLIER,
    compute_ambient_baseline,
    compute_dynamic_speech_floor,
)
from dana.audio.tts_manager import TTSManager, enqueue_speech, get_tts_manager
from dana.audio.tts_worker import TtsWorker, get_tts_worker
from dana.audio.vad_consumer import (
    SILERO_SPEECH_THRESHOLD,
    SILERO_WINDOW_SAMPLES,
    get_silero_vad,
    is_speech_frame,
    trigger_tts_barge_in,
)
from dana.audio.audio_pipeline import AudioRouter
from dana.audio.wake_poller import WakePoller

__all__ = [
    "ABSOLUTE_MIN_SPEECH_FLOOR",
    "AudioRouter",
    "NOISE_FLOOR_MULTIPLIER",
    "SILERO_SPEECH_THRESHOLD",
    "SILERO_WINDOW_SAMPLES",
    "SYSTEM_DEFAULT_LABEL",
    "TTSManager",
    "TtsWorker",
    "compute_ambient_baseline",
    "compute_dynamic_speech_floor",
    "enqueue_speech",
    "get_default_audio_devices",
    "get_silero_vad",
    "get_tts_manager",
    "get_tts_worker",
    "is_speech_frame",
    "resolve_live_input_device",
    "stream_device_kwargs",
    "trigger_tts_barge_in",
    "WakePoller",
]
