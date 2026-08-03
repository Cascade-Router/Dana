"""OS default mic/speaker resolution (sounddevice / PortAudio)."""

from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock, patch

from dana.audio.devices import (
    SYSTEM_DEFAULT_LABEL,
    get_default_audio_devices,
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


def test_system_default_label_constant() -> None:
    assert SYSTEM_DEFAULT_LABEL == "System Default (Auto)"


def test_gui_audio_menus_prepend_system_default() -> None:
    """Null mic/speaker settings map to System Default (Auto) in the GUI."""
    from dana.core_agent import DonnaGUI, set_engine_engaged

    set_engine_engaged(False)
    try:
        app = DonnaGUI()
    except Exception as exc:  # noqa: BLE001
        import pytest

        pytest.skip(f"Tk unavailable: {exc}")
    try:
        assert SYSTEM_DEFAULT_LABEL in list(app.mic_menu.cget("values"))
        assert SYSTEM_DEFAULT_LABEL in list(app.speaker_menu.cget("values"))
        assert list(app.mic_menu.cget("values"))[0] == SYSTEM_DEFAULT_LABEL
        assert list(app.speaker_menu.cget("values"))[0] == SYSTEM_DEFAULT_LABEL
        assert app._mic_by_label.get(SYSTEM_DEFAULT_LABEL) is None
        assert app._speaker_by_label.get(SYSTEM_DEFAULT_LABEL) is None
    finally:
        set_engine_engaged(False)
        try:
            app.destroy()
        except Exception:  # noqa: BLE001
            pass
