"""Windows / PortAudio default mic+speaker resolution for sounddevice streams."""

from __future__ import annotations

from typing import Any, Optional

SYSTEM_DEFAULT_LABEL = "System Default (Auto)"


def get_default_audio_devices() -> tuple[Optional[int], Optional[int]]:
    """Return the current OS default (input_idx, output_idx).

    Values come from ``sounddevice.default.device``. Either index may be ``None``
    when PortAudio has not bound a default; callers should then omit ``device``
    so sounddevice targets the system default at stream-open time.
    """
    import sounddevice as sd

    try:
        raw = sd.default.device
    except Exception:  # noqa: BLE001
        return None, None

    # sounddevice returns `_InputOutputPair` (supports [0]/[1] but not len()).
    try:
        din_raw, dout_raw = raw[0], raw[1]  # type: ignore[index]
    except Exception:  # noqa: BLE001
        if isinstance(raw, (list, tuple)) and len(raw) >= 2:
            din_raw, dout_raw = raw[0], raw[1]
        else:
            din_raw, dout_raw = raw, raw

    def _as_idx(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            idx = int(value)
        except (TypeError, ValueError):
            return None
        return idx if idx >= 0 else None

    return _as_idx(din_raw), _as_idx(dout_raw)


def default_device_samplerate(kind: str = "input") -> Optional[int]:
    """Host default sample rate for the current OS default input or output."""
    import sounddevice as sd

    din, dout = get_default_audio_devices()
    idx = din if kind == "input" else dout
    try:
        info = sd.query_devices(idx if idx is not None else None)
        if isinstance(info, dict):
            return int(round(float(info.get("default_samplerate", 0)))) or None
    except Exception:  # noqa: BLE001
        return None
    return None


def stream_device_kwargs(device: Optional[int]) -> dict[str, Any]:
    """Kwargs for InputStream/OutputStream: omit ``device`` when using OS default."""
    if device is None:
        return {}
    return {"device": int(device)}
