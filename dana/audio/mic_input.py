"""Microphone device selection and the continuous mic-ingestion producer/consumer.

Moved out of ``dana.core_agent`` (Phase 3 of the decomposition). One
InputStream producer thread (``mic_ingest_worker``) pushes 16 kHz VAD frames
into a shared queue; ``record_utterance`` is the consumer used by the
conversation loop to capture one utterance via Silero VAD.

Excludes ``input_txt_ingest_worker`` — despite living alongside these
functions in the original file, it polls a text/task file, not the
microphone, and is reserved for a dedicated ingestion module in a later
phase.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from typing import Any, Optional

import numpy as np
import sounddevice as sd

from dana.audio import noise_floor
from dana.audio.dc_blocker import DcBlocker
from dana.audio.stt import prepare_audio_for_whisper, resample_to_16k
from dana.audio.tts_worker import _is_portaudio_error
from dana.core import shared_state as state
from dana.core.constants import (
    BARGE_IN_AMBIENT_MULT,
    BARGE_IN_RMS,
    DEAD_MIC_RMS_FLOOR,
    POST_ACK_FLUSH_SEC,
    SAMPLE_RATE,
    VAD_FRAME_MS,
    VAD_FRAME_SAMPLES,
    VAD_MAX_SECONDS,
    VAD_MIN_SPEECH_MS,
    VAD_PRE_ROLL_FRAMES,
    VAD_SILENCE_MS,
)
from dana.logging import log, log_debug
from dana.paths import _nt_hide_console_if_mp_child

# ---------------------------------------------------------------------------
# Dynamic audio configuration (settings.json)
# ---------------------------------------------------------------------------


def _device_rate(index: Optional[int]) -> int:
    """Native sample rate for a device index, or the OS default input when None."""
    if index is None:
        try:
            from dana.audio.devices import default_device_samplerate

            rate = default_device_samplerate("input")
            if rate is not None and rate > 0:
                return int(rate)
        except Exception:  # noqa: BLE001
            pass
        return SAMPLE_RATE
    try:
        return int(round(float(sd.query_devices()[index]["default_samplerate"])))
    except Exception:
        return SAMPLE_RATE


def _validate_mic_id(mic_id: Optional[int]) -> bool:
    """True for System Default (None) or a live INPUT device index."""
    if mic_id is None:
        return True
    devices = sd.query_devices()
    if mic_id < 0 or mic_id >= len(devices):
        return False
    return int(devices[mic_id].get("max_input_channels", 0)) >= 1


def _validate_speaker_id(speaker_id: Optional[int]) -> bool:
    """True for System Default (None) or a live OUTPUT device index."""
    if speaker_id is None:
        return True
    devices = sd.query_devices()
    if speaker_id < 0 or speaker_id >= len(devices):
        return False
    return int(devices[speaker_id].get("max_output_channels", 0)) >= 1


def _parse_settings_device_id(raw: Any) -> Optional[int]:
    """Parse settings.json mic/speaker id; null / missing → System Default."""
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip().lower() in {"", "none", "null", "default", "auto"}:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def save_audio_settings(mic_id: Optional[int], speaker_id: Optional[int]) -> None:
    # Preserve non-audio flags (e.g. enable_dynamic_tool_synthesis) across audio saves.
    payload: dict[str, Any] = {}
    if os.path.isfile(state.SETTINGS_FILE):
        try:
            with open(state.SETTINGS_FILE, "r", encoding="utf-8") as fh:
                existing = json.load(fh)
            if isinstance(existing, dict):
                payload.update(existing)
        except Exception:  # noqa: BLE001
            pass
    payload["mic_id"] = None if mic_id is None else int(mic_id)
    payload["speaker_id"] = None if speaker_id is None else int(speaker_id)
    if "enable_dynamic_tool_synthesis" not in payload:
        payload["enable_dynamic_tool_synthesis"] = True
    if "assistant_language" not in payload:
        payload["assistant_language"] = "en"
    if "whisper_language" not in payload:
        payload["whisper_language"] = "english"
    with open(state.SETTINGS_FILE, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    log("Audio", f"Saved audio settings -> {os.path.basename(state.SETTINGS_FILE)}")


def interactive_audio_setup() -> tuple[Optional[int], Optional[int]]:
    """First-run terminal wizard: pick mic + speaker, persist to settings.json.

    Press Enter with no ID to keep Windows System Default (Auto).
    """
    print("\n=== Dana first-run audio setup ===", flush=True)
    print(
        "No settings.json found. Let's configure your microphone and speakers.\n"
        "Press Enter with no ID to use System Default (Auto).\n",
        flush=True,
    )

    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    print("Available INPUT devices:", flush=True)
    print(f"{'Index':<7} {'Rate':<8} {'Ch':<4} {'HostAPI':<18} Name", flush=True)
    print("-" * 72, flush=True)
    for idx, dev in enumerate(devices):
        if int(dev.get("max_input_channels", 0)) < 1:
            continue
        try:
            api = str(hostapis[int(dev["hostapi"])]["name"])
        except Exception:
            api = "?"
        rate = int(round(float(dev.get("default_samplerate", 0))))
        print(
            f"{idx:<7} {rate:<8} {int(dev['max_input_channels']):<4} "
            f"{api:<18} {dev.get('name', '')}",
            flush=True,
        )

    print("\nAvailable OUTPUT devices:", flush=True)
    print(f"{'Index':<7} {'Rate':<8} {'Ch':<4} {'HostAPI':<18} Name", flush=True)
    print("-" * 72, flush=True)
    for idx, dev in enumerate(devices):
        if int(dev.get("max_output_channels", 0)) < 1:
            continue
        try:
            api = str(hostapis[int(dev["hostapi"])]["name"])
        except Exception:
            api = "?"
        rate = int(round(float(dev.get("default_samplerate", 0))))
        print(
            f"{idx:<7} {rate:<8} {int(dev['max_output_channels']):<4} "
            f"{api:<18} {dev.get('name', '')}",
            flush=True,
        )

    while True:
        raw = input(
            "\nPlease enter the ID of your preferred Microphone "
            "(Enter = System Default): "
        ).strip()
        if raw == "":
            mic_id: Optional[int] = None
            break
        try:
            mic_id = int(raw)
        except ValueError:
            print("Please enter a numeric device index (or Enter for default).", flush=True)
            continue
        if not _validate_mic_id(mic_id):
            print(f"Invalid microphone ID: {mic_id}", flush=True)
            continue
        break

    while True:
        raw = input(
            "Please enter the ID of your preferred Speaker/Headphones "
            "(Enter = System Default): "
        ).strip()
        if raw == "":
            speaker_id: Optional[int] = None
            break
        try:
            speaker_id = int(raw)
        except ValueError:
            print("Please enter a numeric device index (or Enter for default).", flush=True)
            continue
        if not _validate_speaker_id(speaker_id):
            print(f"Invalid speaker ID: {speaker_id}", flush=True)
            continue
        break

    save_audio_settings(mic_id, speaker_id)
    print(
        f"Saved settings.json (mic={mic_id}, speaker={speaker_id}). "
        "Delete settings.json to re-run this wizard.\n",
        flush=True,
    )
    return mic_id, speaker_id


def load_audio_settings() -> tuple[Optional[int], Optional[int], int]:
    """Always bind System Default (``device=None``) for mic + speaker.

    Sticky ``mic_id`` / ``speaker_id`` values in settings.json are ignored so
    streams follow the live OS default. Helpers in ``dana.audio.devices`` remain
    available for logging / re-query after PortAudio faults.
    """
    if not os.path.isfile(state.SETTINGS_FILE):
        try:
            save_audio_settings(None, None)
        except Exception:  # noqa: BLE001
            pass

    try:
        from dana.audio.devices import get_default_audio_devices

        din, dout = get_default_audio_devices()
    except Exception:  # noqa: BLE001
        din, dout = None, None
    rate = _device_rate(None)
    log(
        "Audio",
        f"Autonomous audio -> System Default "
        f"(current in={din} out={dout}) @ {rate} Hz",
    )
    return None, None, rate


# ---------------------------------------------------------------------------
# Device enumeration / probing
# ---------------------------------------------------------------------------


def list_input_devices() -> None:
    """Print a clean table of sounddevice input devices."""
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    n_in = sum(1 for d in devices if int(d.get("max_input_channels", 0)) >= 1)
    log("Audio", f"INPUT devices: {n_in} available (DANA_DEBUG=1 for full table)")
    log_debug("Audio", "Available INPUT devices:")
    log_debug("Audio", f"{'Index':<7} {'Rate':<8} {'Ch':<4} {'HostAPI':<18} Name")
    log_debug("Audio", "-" * 72)
    for idx, dev in enumerate(devices):
        channels = int(dev.get("max_input_channels", 0))
        if channels < 1:
            continue
        try:
            api = str(hostapis[int(dev["hostapi"])]["name"])
        except Exception:
            api = "?"
        rate = int(round(float(dev.get("default_samplerate", 0))))
        name = str(dev.get("name", ""))
        log_debug("Audio", f"{idx:<7} {rate:<8} {channels:<4} {api:<18} {name}")


def list_output_devices() -> None:
    """Print a clean table of sounddevice output devices (speaker routing check)."""
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    n_out = sum(1 for d in devices if int(d.get("max_output_channels", 0)) >= 1)
    log("Audio", f"OUTPUT devices: {n_out} available (DANA_DEBUG=1 for full table)")
    log_debug("Audio", "Available OUTPUT devices (Windows speaker / monitor routing):")
    log_debug("Audio", f"{'Index':<7} {'Rate':<8} {'Ch':<4} {'HostAPI':<18} Name")
    log_debug("Audio", "-" * 72)
    for idx, dev in enumerate(devices):
        channels = int(dev.get("max_output_channels", 0))
        if channels < 1:
            continue
        try:
            api = str(hostapis[int(dev["hostapi"])]["name"])
        except Exception:
            api = "?"
        rate = int(round(float(dev.get("default_samplerate", 0))))
        name = str(dev.get("name", ""))
        log_debug("Audio", f"{idx:<7} {rate:<8} {channels:<4} {api:<18} {name}")
    try:
        default_out = sd.default.device[1]
        if default_out is not None and 0 <= int(default_out) < len(devices):
            out_name = devices[int(default_out)].get("name", "?")
            log("Audio", f"Default OUTPUT device: [{default_out}] {out_name}")
        else:
            log_debug("Audio", f"Default OUTPUT device: {default_out}")
    except Exception as exc:  # noqa: BLE001
        log("Audio", f"WARNING: could not resolve default OUTPUT device: {exc}")


def find_steelseries_speaker() -> Optional[tuple[int, str]]:
    """
    Prefer a SteelSeries Sonar playback endpoint that reaches the headset.
    Chat is best for voice; then Media; then Gaming.
    """
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    preferred_apis = ("mme", "wasapi", "directsound")
    channel_rank = ("chat", "media", "gaming")
    matches: list[tuple[int, int, int, int, str]] = []
    # api_rank, channel_rank_i, -channels, idx, name

    for idx, dev in enumerate(devices):
        if int(dev.get("max_output_channels", 0)) < 1:
            continue
        name = str(dev.get("name", ""))
        name_l = name.lower()
        if "steelseries" not in name_l:
            continue
        # Skip mic-monitor / capture-looking outputs.
        if "microphone" in name_l and "chat" not in name_l:
            continue
        try:
            api = str(hostapis[int(dev["hostapi"])]["name"]).lower()
        except Exception:
            api = ""
        if "wdm-ks" in api:
            continue
        api_rank = next(
            (i for i, token in enumerate(preferred_apis) if token in api),
            len(preferred_apis),
        )
        ch_rank = next(
            (i for i, token in enumerate(channel_rank) if token in name_l),
            len(channel_rank),
        )
        channels = int(dev.get("max_output_channels", 0))
        matches.append((api_rank, ch_rank, -channels, idx, name))

    if not matches:
        return None
    matches.sort()
    _a, _c, _ch, idx, name = matches[0]
    return idx, name


def pick_output_device(preferred: Optional[int] = None) -> Optional[int]:
    """Resolve TTS playback device (honors --speaker when provided).

    ``None`` / System Default keeps PortAudio on the live Windows default so
    mid-session Bluetooth switches are picked up on the next OutputStream open.
    """
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()

    if preferred is not None:
        if preferred < 0 or preferred >= len(devices):
            log("Audio", f"ERROR: --speaker {preferred} is out of range.")
            return None
        dev = devices[preferred]
        if int(dev.get("max_output_channels", 0)) < 1:
            log("Audio", f"ERROR: --speaker {preferred} is not an OUTPUT device.")
            return None
        try:
            api = str(hostapis[int(dev["hostapi"])]["name"])
        except Exception:
            api = "?"
        log(
            "Main",
            f"Selected speaker [{preferred}] {dev.get('name')} ({api}) via --speaker",
        )
        return preferred

    try:
        from dana.audio.devices import get_default_audio_devices

        _din, default_out = get_default_audio_devices()
        if default_out is not None and 0 <= int(default_out) < len(devices):
            name = devices[int(default_out)].get("name", "?")
            log(
                "Audio",
                f"Using System Default speaker (current=[{default_out}] {name})",
            )
        else:
            log("Audio", "Using System Default speaker (sounddevice device=None)")
    except Exception:
        log("Audio", "Using System Default speaker (sounddevice device=None)")
    return None


def find_steelseries_mic() -> Optional[tuple[int, int, str]]:
    """
    Prefer a SteelSeries virtual microphone input.
    Returns (index, sample_rate, name) or None.

    Never selects WDM-KS — that host API spams PaErrorCode -9999 / capture-pin
    failures on Sonar devices (especially after TTS playback).
    """
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    # MME is the most reliable for repeated sd.rec / InputStream on Windows Sonar.
    preferred_apis = ("mme", "wasapi", "directsound")
    matches: list[tuple[int, int, int, str, int]] = []
    # rank_api, prefer_mic_name (0 better), idx, name, rate

    for idx, dev in enumerate(devices):
        if int(dev.get("max_input_channels", 0)) < 1:
            continue
        name = str(dev.get("name", ""))
        name_l = name.lower()
        if "steelseries" not in name_l:
            continue
        if "stream" in name_l and "microphone" not in name_l:
            continue
        try:
            api = str(hostapis[int(dev["hostapi"])]["name"]).lower()
        except Exception:
            api = ""
        if "wdm-ks" in api:
            continue
        api_rank = next(
            (i for i, token in enumerate(preferred_apis) if token in api),
            len(preferred_apis),
        )
        # Prefer plain "Microphone" endpoints over chat-capture aliases.
        mic_rank = 0 if name_l.strip().startswith("steelseries sonar - microphone") else 1
        if "microphone" not in name_l:
            mic_rank = 2
        rate = int(round(float(dev.get("default_samplerate", SAMPLE_RATE))))
        matches.append((api_rank, mic_rank, idx, name, rate))

    if not matches:
        return None

    # Among remaining SteelSeries inputs, prefer the endpoint with real signal.
    scored: list[tuple[float, int, int, int, str, int]] = []
    for api_rank, mic_rank, idx, name, rate in matches:
        try:
            rms = probe_mic_rms(idx, rate, seconds=0.25)
        except Exception:
            rms = -1.0
        log("Audio", f"SteelSeries candidate [{idx}] RMS={rms:.6f} {name}")
        scored.append((-rms, api_rank, mic_rank, idx, name, rate))

    scored.sort()
    _neg_rms, _api_rank, _mic_rank, idx, name, rate = scored[0]
    return idx, rate, name


def pick_input_device(preferred: Optional[int] = None) -> tuple[Optional[int], int]:
    """Resolve mic index + native sample rate (honors --mic when provided).

    When ``preferred`` is None, return System Default (device=None) so the next
    InputStream open follows the live Windows recording endpoint.
    """
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()

    def hostapi_name(dev: dict) -> str:
        try:
            return str(hostapis[int(dev["hostapi"])]["name"]).lower()
        except Exception:
            return ""

    # Explicit --mic always wins.
    if preferred is not None:
        if preferred < 0 or preferred >= len(devices):
            log("Audio", f"ERROR: --mic {preferred} is out of range.")
            return None, SAMPLE_RATE
        dev = devices[preferred]
        if int(dev.get("max_input_channels", 0)) < 1:
            log("Audio", f"ERROR: --mic {preferred} is not an INPUT device.")
            return None, SAMPLE_RATE
        rate = int(round(float(dev.get("default_samplerate", SAMPLE_RATE))))
        api = hostapi_name(dev)
        log(
            "Audio",
            f"Selected mic device [{preferred}] {dev.get('name')} ({api}, {rate} Hz) via --mic",
        )
        return preferred, rate

    rate = _device_rate(None)
    try:
        from dana.audio.devices import get_default_audio_devices

        default_in, _dout = get_default_audio_devices()
        if default_in is not None and 0 <= int(default_in) < len(devices):
            name = devices[int(default_in)].get("name", "?")
            log(
                "Audio",
                f"Using System Default mic (current=[{default_in}] {name}, {rate} Hz)",
            )
        else:
            log("Audio", f"Using System Default mic (sounddevice device=None, {rate} Hz)")
    except Exception:
        log("Audio", f"Using System Default mic (sounddevice device=None, {rate} Hz)")
    return None, rate


def probe_mic_rms(device_idx: Optional[int], rate: int, seconds: float = 0.4) -> float:
    """Capture a short clip and return RMS level (0.0 ~= muted / dead)."""
    frames = max(1, int(round(rate * seconds)))
    last_exc: Optional[Exception] = None
    for channels in (1, 2):
        kwargs: dict[str, Any] = {
            "samplerate": rate,
            "channels": channels,
            "dtype": "float32",
            "blocking": True,
        }
        if device_idx is not None:
            kwargs["device"] = device_idx
        try:
            audio = sd.rec(frames, **kwargs)
            samples = np.asarray(audio, dtype=np.float32)
            if samples.ndim > 1:
                samples = samples[:, 0]
            samples = samples.reshape(-1)
            if samples.size == 0:
                return 0.0
            return float(np.sqrt(np.mean(np.square(samples))))
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            continue
    if last_exc is not None:
        raise last_exc
    return 0.0


def ensure_live_mic(
    device_idx: Optional[int],
    rate: int,
    *,
    min_rms: float = 1e-4,
    allow_fallback: bool = True,
) -> tuple[Optional[int], int]:
    """Warn / optionally auto-fallback when the selected mic looks muted or silent."""
    # Keep intentional SteelSeries / --mic choices even if ambient RMS is low.
    keep_name = ""
    keep_api = ""
    try:
        if device_idx is not None:
            devices = sd.query_devices()
            hostapis = sd.query_hostapis()
            keep_name = str(devices[device_idx].get("name", ""))
            keep_api = str(hostapis[int(devices[device_idx]["hostapi"])]["name"]).lower()
    except Exception:
        keep_name = ""
    keep_locked = bool(
        device_idx is not None
        and (("steelseries" in keep_name.lower()) or ("wdm-ks" in keep_api))
    )

    # Acoustic-aware scan when fallback is allowed and device is not locked.
    if allow_fallback and not keep_locked:
        from dana.audio.devices import resolve_live_input_device

        live_idx, live_rate, reason = resolve_live_input_device(
            device_idx, floor=float(min_rms)
        )
        try:
            rms = probe_mic_rms(live_idx, live_rate, seconds=0.5)
        except Exception as exc:  # noqa: BLE001
            log("Audio", f"WARNING: mic RMS probe failed on [{live_idx}]: {exc}")
            rms = 0.0
        noise_floor._mic_ambient_rms = float(rms)
        log(
            "Audio",
            f"Mic RMS probe [{live_idx}]: {rms:.6f} (resolve={reason})",
        )
        if reason == "quiet_mic":
            state.quiet_mic_mode.set()
            state.wakeword_armed.clear()
            msg = (
                "[Audio] No active physical mic detected — falling back to "
                "Text-Only / Quiet Mic mode"
            )
            print(msg, flush=True)
            log("Audio", msg)
            log(
                "Audio",
                f"WARNING: mic [{device_idx}] ambient RMS is abnormally low; "
                "enabling quiet-mic adaptive VAD floors + Whisper gain. "
                "Speak into the headset to verify the endpoint is live.",
            )
        else:
            state.quiet_mic_mode.clear()
            if reason == "fallback_live":
                log(
                    "Audio",
                    f"Auto-fallback to live mic [{live_idx}] @ {live_rate} Hz "
                    f"(RMS={rms:.6f})",
                )
        return live_idx, live_rate

    try:
        # ~500 ms probe distinguishes dead/virtual endpoints from live ambient.
        rms = probe_mic_rms(device_idx, rate, seconds=0.5)
    except Exception as exc:  # noqa: BLE001
        log("Audio", f"WARNING: mic RMS probe failed on [{device_idx}]: {exc}")
        rms = 0.0

    noise_floor._mic_ambient_rms = float(rms)
    log("Audio", f"Mic RMS probe [{device_idx}]: {rms:.6f}")
    if rms < DEAD_MIC_RMS_FLOOR:
        state.quiet_mic_mode.set()
        state.wakeword_armed.clear()
        msg = (
            "[Audio] No active physical mic detected — falling back to "
            "Text-Only / Quiet Mic mode"
        )
        print(msg, flush=True)
        log("Audio", msg)
        # Soft normalization hint for Whisper / VAD on near-silent probes (e.g. 0.000015).
        log(
            "Audio",
            f"WARNING: mic [{device_idx}] ambient RMS is abnormally low ({rms:.6f}); "
            "enabling quiet-mic adaptive VAD floors + Whisper gain. "
            "Speak into the headset to verify the endpoint is live.",
        )
    else:
        state.quiet_mic_mode.clear()
    if rms >= min_rms:
        return device_idx, rate

    if (not allow_fallback) or keep_locked:
        # If we somehow landed on WDM-KS, force a SteelSeries MME/WASAPI rematch.
        if "wdm-ks" in keep_api:
            steel = find_steelseries_mic()
            if steel is not None:
                idx, new_rate, name = steel
                log(
                    "Audio",
                    f"Replacing unstable WDM-KS mic [{device_idx}] with [{idx}] {name}",
                )
                return idx, new_rate
        log(
            "Audio",
            f"WARNING: mic [{device_idx}] ambient RMS is low ({rms:.6f}); "
            "keeping selected device (speak into the headset to verify).",
        )
        return device_idx, rate

    log("Audio", "WARNING: no live mic found; keeping original selection.")
    return device_idx, rate


# ---------------------------------------------------------------------------
# Mic ingestion producer / consumer
# ---------------------------------------------------------------------------


def flush_input_buffer(seconds: float = POST_ACK_FLUSH_SEC) -> None:
    """Discard pending mic frames so TTS echo / buffer tail does not enter VAD."""
    # Producer keeps running; just drop queued frames (~seconds worth).
    target = max(1, int(round(float(seconds) * 1000.0 / float(VAD_FRAME_MS))))
    dropped = 0
    for _ in range(target):
        try:
            _ = state.audio_buffer_queue.get_nowait()
            dropped += 1
        except queue.Empty:
            break
    log_debug(
        "Conversation",
        f"Flushed {dropped} mic frame(s) (~{dropped * VAD_FRAME_MS:.0f} ms) after ack.",
    )


def _run_with_timeout(
    fn: Any,
    *,
    timeout_s: float,
    label: str,
) -> tuple[bool, Any, BaseException | None]:
    """Run ``fn`` on a daemon thread; return (ok, result, error).

    If the call hangs past ``timeout_s``, returns ok=False without blocking the
    caller forever (the worker may still be stuck in PortAudio).
    """
    box: list[Any] = []
    err: list[BaseException] = []

    def _target() -> None:
        try:
            box.append(fn())
        except BaseException as exc:  # noqa: BLE001
            err.append(exc)

    worker = threading.Thread(target=_target, name=f"MicTimed:{label}", daemon=True)
    t0 = time.perf_counter()
    worker.start()
    worker.join(timeout=max(0.05, float(timeout_s)))
    if worker.is_alive():
        log(
            "Audio",
            f"ERROR: {label} hung after {timeout_s:.1f}s "
            f"(elapsed_ms={(time.perf_counter() - t0) * 1000.0:.0f}) — "
            "PortAudio device acquisition/read blocked",
        )
        return False, None, TimeoutError(f"{label} timed out after {timeout_s:.1f}s")
    if err:
        return False, None, err[0]
    return True, (box[0] if box else None), None


def _open_input_stream_with_timeout(
    stream_kwargs: dict[str, Any],
    *,
    timeout_s: float = state.MIC_STREAM_OPEN_TIMEOUT_S,
    label: str = "InputStream.open",
) -> Any | None:
    """Open+start an InputStream with a hang timeout. Returns stream or None."""

    def _open() -> Any:
        stream = sd.InputStream(**stream_kwargs)
        stream.start()
        return stream

    ok, stream, err = _run_with_timeout(_open, timeout_s=timeout_s, label=label)
    if not ok:
        if err is not None and not isinstance(err, TimeoutError):
            log("Audio", f"ERROR: {label} failed: {type(err).__name__}: {err}")
        return None
    return stream


def _read_input_stream_with_timeout(
    stream: Any,
    frames: int,
    *,
    timeout_s: float = state.MIC_STREAM_READ_TIMEOUT_S,
    label: str = "InputStream.read",
) -> tuple[np.ndarray | None, bool]:
    """Read frames with a hang timeout. Returns (chunk, overflowed) or (None, False)."""

    def _read() -> tuple[Any, bool]:
        data, overflowed = stream.read(frames)
        return data, bool(overflowed)

    ok, result, err = _run_with_timeout(_read, timeout_s=timeout_s, label=label)
    if not ok or result is None:
        if err is not None and not isinstance(err, TimeoutError):
            log_debug("Audio", f"{label} error: {err}")
        return None, False
    data, overflowed = result
    return np.asarray(data, dtype=np.float32), overflowed


def _close_input_stream(stream: Any, *, label: str = "InputStream") -> None:
    """Best-effort stop+close; never raises."""
    if stream is None:
        return
    try:
        stream.stop()
    except Exception:
        pass
    try:
        stream.close()
    except Exception:
        pass
    log_debug("Audio", f"{label} closed")


def flush_audio_buffer_queue() -> int:
    """Drop all pending mic frames (state transitions / barge-in / standby)."""
    n = 0
    while True:
        try:
            _ = state.audio_buffer_queue.get_nowait()
            n += 1
        except queue.Empty:
            break
    if n:
        log_debug("MicIngest", f"Flushed {n} stale audio frame(s)")
    return n


def get_mic_frame(*, timeout: float = 0.25) -> Optional[np.ndarray]:
    """Pull one 16 kHz mono float32 VAD frame from the ingest queue."""
    # Stage 7.3 — drop frames while Ghost Typist is typing (no Whisper path).
    try:
        from dana.memory.blackboard import is_typing

        if is_typing():
            flush_audio_buffer_queue()
            return None
    except Exception:  # noqa: BLE001
        pass
    try:
        frame = state.audio_buffer_queue.get(timeout=max(0.01, float(timeout)))
    except queue.Empty:
        return None
    arr = np.asarray(frame, dtype=np.float32).reshape(-1)
    if arr.size < VAD_FRAME_SAMPLES:
        pad = np.zeros(VAD_FRAME_SAMPLES, dtype=np.float32)
        pad[: arr.size] = arr
        return pad
    if arr.size > VAD_FRAME_SAMPLES:
        return arr[:VAD_FRAME_SAMPLES].copy()
    return arr


def request_mic_ingest_restart() -> None:
    """Ask the producer to reopen InputStream (device change / soft recovery)."""
    state.mic_ingest_restart.set()
    flush_audio_buffer_queue()


def ensure_mic_ingest_thread() -> None:
    """Start the continuous mic producer once (idempotent)."""
    t = state._mic_ingest_thread
    if t is not None and t.is_alive():
        return
    t = threading.Thread(target=mic_ingest_worker, name="MicIngest", daemon=True)
    state._mic_ingest_thread = t
    t.start()
    log("Main", "Started thread: MicIngest")


def mic_ingest_worker() -> None:
    """Continuous producer: open InputStream once, push 16 kHz VAD frames to queue."""
    _nt_hide_console_if_mp_child()
    stream: Any = None
    stream_channels = 1
    next_err_log = 0.0
    log("MicIngest", "Producer starting (single shared InputStream)...")

    def _close() -> None:
        nonlocal stream
        held = state.mic_lock.acquire(timeout=state.MIC_STREAM_OPEN_TIMEOUT_S)
        try:
            _close_input_stream(stream, label="MicIngest.InputStream")
            stream = None
        finally:
            if held:
                state.mic_lock.release()
        state.mic_ingest_ready.clear()
        state.wake_mic_released.set()

    def _open() -> bool:
        nonlocal stream, stream_channels
        _close()
        from dana.audio.devices import resolve_live_input_device, stream_device_kwargs

        # Acoustic-aware: OS default when live, else first live physical index.
        live_idx, rate, reason = resolve_live_input_device(
            None, floor=float(DEAD_MIC_RMS_FLOOR)
        )
        state.AUDIO_INPUT_DEVICE = live_idx
        state.AUDIO_INPUT_RATE = rate
        if reason == "quiet_mic":
            state.quiet_mic_mode.set()
        elif reason == "fallback_live":
            state.quiet_mic_mode.clear()
            log(
                "MicIngest",
                f"Binding fallback live mic [{live_idx}] @ {rate} Hz",
            )
        else:
            state.quiet_mic_mode.clear()
            log(
                "MicIngest",
                f"Binding System Default input (device={live_idx}) @ {rate} Hz",
            )

        native_frame = max(
            1, int(round(VAD_FRAME_SAMPLES * rate / float(SAMPLE_RATE)))
        )
        last_exc: Optional[BaseException] = None
        for channels in (1, 2):
            kwargs: dict[str, Any] = {
                "samplerate": rate,
                "channels": channels,
                "dtype": "float32",
                "blocksize": native_frame,
            }
            kwargs.update(stream_device_kwargs(live_idx))
            held = state.mic_lock.acquire(timeout=state.MIC_STREAM_OPEN_TIMEOUT_S)
            if not held:
                last_exc = TimeoutError("mic_lock timeout")
                continue
            try:
                state.wake_mic_released.clear()
                candidate = _open_input_stream_with_timeout(
                    kwargs,
                    timeout_s=state.MIC_STREAM_OPEN_TIMEOUT_S,
                    label="MicIngest.InputStream.open",
                )
                if candidate is None:
                    state.wake_mic_released.set()
                    last_exc = TimeoutError("InputStream open timed out")
                    continue
                stream = candidate
                stream_channels = channels
                state.AUDIO_INPUT_DEVICE = live_idx
                state.AUDIO_INPUT_RATE = int(kwargs["samplerate"])
                state.mic_ingest_ready.set()
                state.wake_mic_released.set()
                flush_audio_buffer_queue()
                log(
                    "MicIngest",
                    f"InputStream open device={live_idx} "
                    f"rate={state.AUDIO_INPUT_RATE} ch={channels} "
                    f"block={native_frame} (-> {VAD_FRAME_MS}ms @16k) "
                    f"reason={reason}",
                )
                print(
                    f"[Debug] MicIngest InputStream open device={live_idx} "
                    f"rate={state.AUDIO_INPUT_RATE} ch={channels} block={native_frame}",
                    flush=True,
                )
                return True
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                state.wake_mic_released.set()
                if _is_portaudio_error(exc):
                    # Re-resolve live endpoint and retry next channel / outer loop.
                    try:
                        live_idx, rate, reason = resolve_live_input_device(
                            None, floor=float(DEAD_MIC_RMS_FLOOR)
                        )
                        state.AUDIO_INPUT_DEVICE = live_idx
                    except Exception:  # noqa: BLE001
                        rate = _device_rate(None)
                    state.AUDIO_INPUT_RATE = rate
                    native_frame = max(
                        1,
                        int(round(VAD_FRAME_SAMPLES * rate / float(SAMPLE_RATE))),
                    )
                    log(
                        "MicIngest",
                        f"PortAudioError on mic device={live_idx} — "
                        f"re-resolved, will retry ({exc})",
                    )
            finally:
                state.mic_lock.release()
        log("MicIngest", f"ERROR: could not open mic stream ({last_exc})")
        return False

    if not _open():
        # Keep retrying so a late device bind can recover.
        while not state.stop_event.is_set():
            time.sleep(1.0)
            if _open():
                break

    while not state.stop_event.is_set():
        if state.mic_ingest_restart.is_set():
            state.mic_ingest_restart.clear()
            log("MicIngest", "Restart requested — reopening InputStream")
            if not _open():
                time.sleep(0.5)
            continue
        if stream is None:
            if not _open():
                time.sleep(0.5)
            continue
        native_frame = max(
            1, int(round(VAD_FRAME_SAMPLES * state.AUDIO_INPUT_RATE / float(SAMPLE_RATE)))
        )
        try:
            chunk, overflowed = _read_input_stream_with_timeout(
                stream,
                native_frame,
                timeout_s=state.MIC_STREAM_READ_TIMEOUT_S,
                label="MicIngest.InputStream.read",
            )
            if chunk is None:
                raise TimeoutError("MicIngest read timed out")
            if overflowed:
                log_debug("MicIngest", "input overflow")
            arr = np.asarray(chunk, dtype=np.float32)
            if arr.ndim > 1:
                arr = arr[:, 0]
            frame_16k = resample_to_16k(arr.reshape(-1), state.AUDIO_INPUT_RATE)
            if frame_16k.size < VAD_FRAME_SAMPLES:
                pad = np.zeros(VAD_FRAME_SAMPLES, dtype=np.float32)
                pad[: frame_16k.size] = frame_16k
                frame_16k = pad
            elif frame_16k.size > VAD_FRAME_SAMPLES:
                frame_16k = frame_16k[:VAD_FRAME_SAMPLES]
            # Stage 7.3 — do not enqueue mic frames while Ghost Typist types.
            try:
                from dana.memory.blackboard import is_typing

                if is_typing():
                    continue
            except Exception:  # noqa: BLE001
                pass
            try:
                state.audio_buffer_queue.put_nowait(frame_16k.copy())
            except queue.Full:
                try:
                    _ = state.audio_buffer_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    state.audio_buffer_queue.put_nowait(frame_16k.copy())
                except queue.Full:
                    pass
        except Exception as exc:  # noqa: BLE001
            now = time.monotonic()
            if now >= next_err_log:
                log("MicIngest", f"WARNING: read/reopen cycle ({exc})")
                next_err_log = now + 5.0
            if _is_portaudio_error(exc):
                try:
                    from dana.audio.devices import get_default_audio_devices

                    din, _dout = get_default_audio_devices()
                    log(
                        "MicIngest",
                        f"PortAudioError on read — re-queried defaults in={din}",
                    )
                except Exception:  # noqa: BLE001
                    pass
                state.AUDIO_INPUT_DEVICE = None
            _close()
            time.sleep(0.35)

    _close()
    flush_audio_buffer_queue()
    log("MicIngest", "Producer stopped.")


def adaptive_barge_in_rms() -> float:
    """Dynamic barge-in gate: absolute floor + multiple of ambient (filters TTS bleed)."""
    ambient = float(noise_floor._mic_ambient_rms or 0.0)
    return max(BARGE_IN_RMS, ambient * BARGE_IN_AMBIENT_MULT, 0.08)


def record_utterance(
    max_seconds: Optional[float] = None,
    *,
    ignore_onset_ms: float = 0.0,
) -> tuple[np.ndarray, float, str, bool]:
    """
    Capture speech with Silero VAD; stop shortly after the user finishes talking.

    ``ignore_onset_ms`` skips early VAD hits (TTS echo / buffer tail after ack).

    Returns (audio_16k, rms_raw, stop_reason, speech_started).
    """
    from dana.audio.vad_consumer import (
        SILERO_SPEECH_THRESHOLD,
        is_speech_frame,
        prepare_frame_for_silero,
        reset_silero_states,
    )

    silence_needed = max(1, int(round(VAD_SILENCE_MS / VAD_FRAME_MS)))
    min_speech_frames = max(1, int(round(VAD_MIN_SPEECH_MS / VAD_FRAME_MS)))
    limit_s = float(VAD_MAX_SECONDS if max_seconds is None else max_seconds)
    max_frames = max(1, int(limit_s * 1000 / VAD_FRAME_MS))
    ignore_onset_frames = max(0, int(round(float(ignore_onset_ms) / VAD_FRAME_MS)))

    try:
        reset_silero_states()
    except Exception as exc:  # noqa: BLE001
        log("Conversation", f"WARNING: Silero VAD reset failed ({exc})")

    log(
        "Conversation",
        f"VAD recording (Silero queue consumer @16 kHz, frame={VAD_FRAME_MS}ms, "
        f"silence_cut={VAD_SILENCE_MS}ms, max={limit_s:.0f}s, "
        f"speech_prob>{SILERO_SPEECH_THRESHOLD})...",
    )

    collected: list[np.ndarray] = []
    pre_roll: list[np.ndarray] = []
    speech_started = False
    silence_frames = 0
    speech_frames = 0
    stop_reason = "max_timeout"
    floor_listening_emitted = False
    t0 = time.perf_counter()
    # Streaming DC / rumble kill — state persists across frames for this utterance.
    dc_blocker = DcBlocker()

    def consume_frame(frame_idx: int, samples_16k: np.ndarray) -> bool:
        """Return True when recording should stop."""
        nonlocal speech_started, silence_frames, speech_frames, stop_reason
        nonlocal floor_listening_emitted

        # DC-block before Silero so offset cannot inflate probabilities.
        samples_16k = dc_blocker.apply(samples_16k)

        if samples_16k.size < VAD_FRAME_SAMPLES:
            padded = np.zeros(VAD_FRAME_SAMPLES, dtype=np.float32)
            padded[: samples_16k.size] = samples_16k
            samples_16k = padded
        elif samples_16k.size > VAD_FRAME_SAMPLES:
            samples_16k = samples_16k[:VAD_FRAME_SAMPLES]

        # Quiet-mic: boost a copy for Silero only; keep raw frames for Whisper.
        speech_floor = noise_floor.get_dynamic_speech_floor()
        vad_frame, frame_rms_raw, vad_gain = prepare_frame_for_silero(
            samples_16k,
            noise_floor=speech_floor,
        )
        try:
            is_speech = bool(is_speech_frame(vad_frame, sample_rate=SAMPLE_RATE))
        except Exception as exc:  # noqa: BLE001
            log("Conversation", f"WARNING: Silero VAD frame error: {exc}")
            is_speech = False

        # STATE_CHANGE listening when dynamic noise floor is breached.
        if not floor_listening_emitted and not (
            state.tts_busy.is_set() or frame_idx < ignore_onset_frames
        ):
            try:
                if frame_rms_raw >= speech_floor:
                    floor_listening_emitted = True
                    from dana.ui.status_bus import emit_state_change

                    emit_state_change("listening")
            except Exception:  # noqa: BLE001
                pass

        if not speech_started:
            pre_roll.append(samples_16k.copy())
            if len(pre_roll) > VAD_PRE_ROLL_FRAMES:
                pre_roll.pop(0)

            # Half-duplex: never barge-in from VAD while TTS is playing — speaker
            # echo must not cut Dana. Mic onset is ignored until playback ends.
            if state.tts_busy.is_set() or frame_idx < ignore_onset_frames:
                return False
            if is_speech:
                speech_started = True
                collected.extend(pre_roll)
                speech_frames = 1
                silence_frames = 0
                gain_note = (
                    f", vad_gain=x{vad_gain:.1f}" if vad_gain > 1.01 else ""
                )
                log_debug(
                    "Conversation",
                    f"Speech onset at {(frame_idx + 1) * VAD_FRAME_MS} ms "
                    f"(rms_raw={frame_rms_raw:.5f}{gain_note})",
                )
            return False

        collected.append(samples_16k.copy())
        if is_speech:
            speech_frames += 1
            silence_frames = 0
        else:
            silence_frames += 1
            if speech_frames >= min_speech_frames and silence_frames >= silence_needed:
                stop_reason = "silence_cutoff"
                return True
        return False

    # Producer keeps the InputStream; this consumer takes over queue draining.
    if not state.mic_ingest_ready.wait(timeout=2.0):
        log(
            "Conversation",
            "ERROR: MicIngest not ready — cannot start VAD consumer",
        )
        audio = np.zeros(VAD_FRAME_SAMPLES, dtype=np.float32)
        return audio, 0.0, "mic_ingest_not_ready", False

    state.vad_abort_event.clear()
    state.vad_capture_active.set()
    log_debug("Conversation", "VAD consumer attached to audio_buffer_queue")
    try:
        for frame_idx in range(max_frames):
            if state.stop_event.is_set():
                stop_reason = "shutdown"
                break
            if state.vad_abort_event.is_set():
                stop_reason = "text_override"
                log(
                    "Conversation",
                    "VAD aborted for text/chat override — returning to standby",
                )
                break
            samples_16k = get_mic_frame(timeout=0.35)
            if samples_16k is None:
                if not state.mic_ingest_ready.is_set():
                    log(
                        "Conversation",
                        "ERROR: MicIngest stalled during VAD — aborting capture",
                    )
                    stop_reason = "ingest_stalled"
                    break
                continue
            if consume_frame(frame_idx, samples_16k):
                break
        else:
            stop_reason = "max_timeout"
    finally:
        state.vad_capture_active.clear()
        # Drop residual frames so wake-word standby does not see stale speech.
        flush_audio_buffer_queue()

    # STATE_CHANGE idle when VAD times out without speech onset.
    if (not speech_started) and stop_reason in (
        "max_timeout",
        "silence_cutoff",
    ):
        try:
            from dana.ui.status_bus import emit_state_change

            emit_state_change(
                "idle",
                tool="vad_timeout",
                message="No speech detected — disarmed",
            )
        except Exception:  # noqa: BLE001
            pass

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if not collected:
        audio = (
            np.concatenate(pre_roll).astype(np.float32)
            if pre_roll
            else np.zeros(VAD_FRAME_SAMPLES, dtype=np.float32)
        )
        log(
            "Conversation",
            f"VAD captured no speech onset (elapsed={elapsed_ms:.0f} ms, reason={stop_reason})",
        )
    else:
        audio = np.concatenate(collected).astype(np.float32)
        log(
            "Conversation",
            f"VAD stop reason={stop_reason}; elapsed={elapsed_ms:.0f} ms; "
            f"speech_frames={speech_frames}; silence_tail={silence_frames * VAD_FRAME_MS} ms; "
            f"samples={audio.size} ({audio.size / SAMPLE_RATE:.2f}s)",
        )

    peak = float(np.max(np.abs(audio)) + 1e-9)
    rms_raw = float(np.sqrt(np.mean(np.square(audio))) + 1e-9)
    audio = prepare_audio_for_whisper(audio, rms_raw=rms_raw)
    log(
        "Conversation",
        f"Captured audio peak={peak:.4f}, rms_raw={rms_raw:.4f}",
    )
    return audio, rms_raw, stop_reason, speech_started


__all__ = (
    "adaptive_barge_in_rms",
    "ensure_live_mic",
    "ensure_mic_ingest_thread",
    "find_steelseries_mic",
    "find_steelseries_speaker",
    "flush_audio_buffer_queue",
    "flush_input_buffer",
    "get_mic_frame",
    "interactive_audio_setup",
    "list_input_devices",
    "list_output_devices",
    "load_audio_settings",
    "mic_ingest_worker",
    "pick_input_device",
    "pick_output_device",
    "probe_mic_rms",
    "record_utterance",
    "request_mic_ingest_restart",
    "save_audio_settings",
)
