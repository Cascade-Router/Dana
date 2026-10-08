"""stt.transcribe_audio's guards against Whisper hallucinating on silence:
silent audio never reaches the model, and known non-speech transcripts come
back empty, while real (including one-word) commands pass through.

Whisper itself is faked: the processor/model return a fixed transcript, so
these run without the model weights.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from dana.audio import stt

torch = pytest.importorskip("torch")  # transcribe_audio builds torch inputs


class FakeWhisper:
    """Stands in for the Whisper processor + model pair."""

    def __init__(self, transcript: str):
        self.transcript = transcript
        self.generate_calls = 0

    # processor
    def __call__(self, audio: Any, *, sampling_rate: int, return_tensors: str) -> dict[str, Any]:
        assert sampling_rate == stt.SAMPLE_RATE
        return {"input_features": torch.zeros((1, 80, 10))}

    def batch_decode(self, _ids: Any, **_kwargs: Any) -> list[str]:
        return [self.transcript]

    # model
    def generate(self, **_kwargs: Any) -> Any:
        self.generate_calls += 1
        return torch.zeros((1, 1), dtype=torch.long)


def _speech(seconds: float = 1.5) -> np.ndarray:
    t = np.arange(int(stt.SAMPLE_RATE * seconds)) / stt.SAMPLE_RATE
    return (3000 * np.sin(2 * np.pi * 220 * t)).astype(np.int16)


def _transcribe(audio: np.ndarray, whisper: FakeWhisper) -> str:
    return stt.transcribe_audio(audio, whisper, whisper, "cpu", torch.float32)


def test_silent_audio_never_reaches_whisper() -> None:
    whisper = FakeWhisper("you know.")
    assert _transcribe(np.zeros(16000, dtype=np.int16), whisper) == ""
    assert whisper.generate_calls == 0


@pytest.mark.parametrize(
    "phantom",
    ["you know.", "You know", "Thank you for watching.", "Thanks for watching!", "[BLANK_AUDIO]", "(breathing)", "...",
     "Thank you."],
)
def test_known_non_speech_transcripts_are_dropped(phantom: str) -> None:
    assert _transcribe(_speech(), FakeWhisper(phantom)) == ""


@pytest.mark.parametrize(
    "command",
    ["Make the box 20 millimetres wider.", "stop", "Yes.", "Do you know the bed size?", "export everything"],
)
def test_real_commands_pass_through(command: str) -> None:
    # One-word replies matter here ("stop" while a print is running), so the
    # filter has no "fewer than two words" rule; and "you know" only counts as
    # noise when it is the whole transcript.
    assert _transcribe(_speech(), FakeWhisper(command)) == command
