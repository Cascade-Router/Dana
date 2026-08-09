"""OS default mic/speaker resolution (sounddevice / PortAudio)."""

from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock, patch

from dana.audio.devices import (
    SYSTEM_DEFAULT_LABEL,
    get_default_audio_devices,
    resolve_live_input_device,
    stream_device_kwargs,
)


def test_get_default_audio_devices_returns_os_pair() -> None:
    """``get_default_audio_devices()`` returns the current OS in/out pair."""
    din, dout = get_default_audio_devices()
    assert isinstance(din, (int, type(None)))
    assert isinstance(dout, (int, type(None)))
    if din is not None:
        assert din >= 0
    if dout is not None:
        assert dout >= 0


def test_get_default_audio_devices_matches_sounddevice_default() -> None:
    import sounddevice as sd

    din, dout = get_default_audio_devices()
    raw = sd.default.device
    try:
        expect_in, expect_out = raw[0], raw[1]
    except Exception:  # noqa: BLE001
        expect_in, expect_out = raw, raw

    def _norm(value: object) -> Optional[int]:
        if value is None:
            return None
        try:
            idx = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return idx if idx >= 0 else None

    assert din == _norm(expect_in)
    assert dout == _norm(expect_out)


def test_get_default_audio_devices_tolerates_query_failure() -> None:
    mock_default = MagicMock()
    type(mock_default).device = property(
        lambda self: (_ for _ in ()).throw(RuntimeError("no audio"))
    )
    with patch("sounddevice.default", mock_default):
        din, dout = get_default_audio_devices()
    assert din is None and dout is None


def test_stream_device_kwargs_omits_none() -> None:
    assert stream_device_kwargs(None) == {}
    assert stream_device_kwargs(3) == {"device": 3}


def test_resolve_live_input_device_keeps_live_default() -> None:
    with (
        patch("dana.audio.devices._input_samplerate", return_value=44100),
        patch("dana.audio.devices._probe_input_rms", return_value=0.01),
    ):
        idx, rate, reason = resolve_live_input_device(None, floor=1e-4)
    assert idx is None
    assert rate == 44100
    assert reason == "default_live"


def test_resolve_live_input_device_fallback_first_live() -> None:
    devices = [
        {"max_input_channels": 2, "name": "Silent Default", "hostapi": 0, "default_samplerate": 44100},
        {"max_input_channels": 2, "name": "Mapper", "hostapi": 0, "default_samplerate": 44100},
        {"max_input_channels": 2, "name": "HD Audio Mic", "hostapi": 0, "default_samplerate": 48000},
    ]
    hostapis = [{"name": "MME"}, {"name": "Windows WDM-KS"}]

    def _rms(device, rate, seconds=0.35):  # noqa: ANN001
        if device is None or device == 0:
            return 1e-6
        if device == 2:
            return 0.002
        return 0.0

    with (
        patch("dana.audio.devices._input_samplerate", return_value=44100),
        patch("dana.audio.devices._probe_input_rms", side_effect=_rms),
        patch("dana.audio.devices.get_default_audio_devices", return_value=(0, 5)),
        patch("sounddevice.query_devices", return_value=devices),
        patch("sounddevice.query_hostapis", return_value=hostapis),
    ):
        idx, rate, reason = resolve_live_input_device(None, floor=1e-4)
    assert idx == 2
    assert rate == 48000
    assert reason == "fallback_live"


def test_resolve_live_input_device_quiet_when_none_live() -> None:
    devices = [
        {"max_input_channels": 2, "name": "VB-Audio Cable", "hostapi": 0, "default_samplerate": 44100},
        {"max_input_channels": 2, "name": "Dead Mic", "hostapi": 1, "default_samplerate": 44100},
    ]
    hostapis = [{"name": "MME"}, {"name": "Windows WDM-KS"}]

    with (
        patch("dana.audio.devices._input_samplerate", return_value=44100),
        patch("dana.audio.devices._probe_input_rms", return_value=0.0),
        patch("dana.audio.devices.get_default_audio_devices", return_value=(None, None)),
        patch("sounddevice.query_devices", return_value=devices),
        patch("sounddevice.query_hostapis", return_value=hostapis),
    ):
        idx, rate, reason = resolve_live_input_device(None, floor=1e-4)
    assert idx is None
    assert rate == 44100
    assert reason == "quiet_mic"


def test_system_default_label_constant() -> None:
    assert SYSTEM_DEFAULT_LABEL == "System Default (Auto)"


def test_gui_audio_is_autonomous_system_default() -> None:
    """Mic/Speaker menus removed; streams always use System Default."""
    from dana.core_agent import DanaGUI, load_audio_settings, set_engine_engaged

    mic_id, speaker_id, _rate = load_audio_settings()
    assert mic_id is None
    assert speaker_id is None
    assert SYSTEM_DEFAULT_LABEL  # helpers kept

    set_engine_engaged(False)
    try:
        app = DanaGUI()
    except Exception as exc:  # noqa: BLE001
        import pytest

        pytest.skip(f"Tk unavailable: {exc}")
    try:
        assert app.mic_menu is None
        assert app.speaker_menu is None
        assert app.apply_note is not None
        note = str(app.apply_note.cget("text"))
        assert "System Default" in note
    finally:
        set_engine_engaged(False)
        try:
            app.destroy()
        except Exception:  # noqa: BLE001
            pass
