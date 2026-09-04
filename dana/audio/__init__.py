"""Audio helpers (Silero VAD + TTS barge-in / playback interrupt)."""

from dana.audio.devices import (
    SYSTEM_DEFAULT_LABEL,
    get_default_audio_devices,
    resolve_live_input_device,
    stream_device_kwargs,
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

__all__ = [
    "SILERO_SPEECH_THRESHOLD",
    "SILERO_WINDOW_SAMPLES",
    "SYSTEM_DEFAULT_LABEL",
    "TTSManager",
    "TtsWorker",
    "enqueue_speech",
    "get_default_audio_devices",
    "get_silero_vad",
    "get_tts_manager",
    "get_tts_worker",
    "is_speech_frame",
    "resolve_live_input_device",
    "stream_device_kwargs",
    "trigger_tts_barge_in",
]
