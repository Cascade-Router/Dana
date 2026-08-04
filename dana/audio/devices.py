"""Windows / PortAudio default mic+speaker resolution for sounddevice streams."""

from __future__ import annotations

from typing import Any, Optional

SYSTEM_DEFAULT_LABEL = "System Default (Auto)"
_FALLBACK_RATE = 16000
_SKIP_NAME_TOKENS = ("mapper", "primary sound", "cable", "vb-audio")


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


def _input_samplerate(device: Optional[int]) -> int:
    """Native sample rate for ``device`` (``None`` → OS default input)."""
    import sounddevice as sd

    if device is None:
        rate = default_device_samplerate("input")
        return int(rate) if rate is not None and rate > 0 else _FALLBACK_RATE
    try:
        info = sd.query_devices(int(device))
        return int(round(float(info.get("default_samplerate", _FALLBACK_RATE)))) or _FALLBACK_RATE
    except Exception:  # noqa: BLE001
        return _FALLBACK_RATE


def _probe_input_rms(device: Optional[int], rate: int, seconds: float = 0.35) -> float:
    """Short blocking capture RMS; 0.0 on failure / empty."""
    import numpy as np
    import sounddevice as sd

    frames = max(1, int(round(float(rate) * float(seconds))))
    for channels in (1, 2):
        kwargs: dict[str, Any] = {
            "samplerate": int(rate),
            "channels": channels,
            "dtype": "float32",
            "blocking": True,
        }
        if device is not None:
            kwargs["device"] = int(device)
        try:
            audio = sd.rec(frames, **kwargs)
            samples = np.asarray(audio, dtype=np.float32)
            if samples.ndim > 1:
                samples = samples[:, 0]
            samples = samples.reshape(-1)
            if samples.size == 0:
                return 0.0
            return float(np.sqrt(np.mean(np.square(samples))))
        except Exception:  # noqa: BLE001
            continue
    return 0.0


def resolve_live_input_device(
    default_idx: Optional[int] = None,
    floor: float = 1e-4,
) -> tuple[Optional[int], int, str]:
    """Acoustic-aware input bind: keep OS default when live, else first live mic.

    Returns ``(device_idx, rate, reason)`` where:
    - ``device_idx is None`` means true OS default (omit ``device`` at open),
      or quiet-mic fallback when ``reason == "quiet_mic"``;
    - ``reason`` is ``"default_live"``, ``"fallback_live"``, or ``"quiet_mic"``.
    """
    import sounddevice as sd

    rate = _input_samplerate(default_idx)
    try:
        rms = _probe_input_rms(default_idx, rate, seconds=0.35)
    except Exception:  # noqa: BLE001
        rms = 0.0
    if rms >= float(floor):
        return default_idx, rate, "default_live"

    skip: set[int] = set()
    if default_idx is not None:
        skip.add(int(default_idx))
    else:
        din, _dout = get_default_audio_devices()
        if din is not None:
            skip.add(int(din))

    try:
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
    except Exception:  # noqa: BLE001
        return None, rate, "quiet_mic"

    for idx, dev in enumerate(devices):
        if idx in skip:
            continue
        if int(dev.get("max_input_channels", 0)) < 1:
            continue
        name_l = str(dev.get("name", "")).lower()
        try:
            api = str(hostapis[int(dev["hostapi"])]["name"]).lower()
        except Exception:  # noqa: BLE001
            api = ""
        if "wdm-ks" in api:
            continue
        if any(tok in name_l for tok in _SKIP_NAME_TOKENS):
            continue
        cand_rate = int(round(float(dev.get("default_samplerate", rate)))) or rate
        try:
            cand_rms = _probe_input_rms(idx, cand_rate, seconds=0.35)
        except Exception:  # noqa: BLE001
            continue
        if cand_rms >= float(floor):
            return idx, cand_rate, "fallback_live"

    return None, rate, "quiet_mic"
