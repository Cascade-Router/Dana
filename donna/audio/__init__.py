"""Audio helpers (Silero VAD + TTS barge-in / playback interrupt)."""

from donna.audio.tts_worker import TtsWorker, get_tts_worker
from donna.audio.vad_consumer import (
    SILERO_SPEECH_THRESHOLD,
    SILERO_WINDOW_SAMPLES,
    get_silero_vad,
    is_speech_frame,
    trigger_tts_barge_in,
)

__all__ = [
    "SILERO_SPEECH_THRESHOLD",
    "SILERO_WINDOW_SAMPLES",
    "TtsWorker",
    "get_silero_vad",
    "get_tts_worker",
    "is_speech_frame",
    "trigger_tts_barge_in",
]
