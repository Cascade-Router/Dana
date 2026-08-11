"""OpenWakeWord wake-word detection thread + scoring/verification helpers.

Extracted verbatim from ``dana.core_agent`` (Phase 7 of the core_agent.py
decomposition; see docs/architecture/phase7_core_agent_decomposition.md).
``wakeword_worker`` is one of the four daemon threads ``dana.core.app_runtime
.agent_loop()`` spawns; the rest of this module is its scoring (openWakeWord
onset gate) and phrase-verification (Whisper second gate) helpers.
"""

from __future__ import annotations

import os
import re
import time
from collections import deque
from typing import Any, Optional

import numpy as np
import sounddevice as sd
from openwakeword.model import Model as OpenWakeWordModel

import dana.core.shared_state as state
from dana.audio.mic_input import get_mic_frame
from dana.audio.noise_floor import audio_buffer_rms, should_skip_wake_predict
from dana.audio.stt import (
    _sanitize_whisper_generation_config,
    _whisper_generate_kwargs,
    prepare_audio_for_whisper,
)
from dana.audio.tts_worker import maybe_play_boot_ready_audio
from dana.core.constants import (
    SAMPLE_RATE,
    WAKE_CHUNK,
    WAKE_COOLDOWN_SEC,
    WAKE_MIN_CONSECUTIVE,
    WAKE_ONSET_BELOW,
    WAKE_ONSET_LOOKBACK,
    WAKE_PHRASE_ALIASES,
    WAKE_PHRASE_REJECT,
    WAKE_PHRASE_TOKENS,
    WAKE_PHRASE_VERIFY,
    WAKE_PHRASE_WINDOW_CHUNKS,
    WAKE_THRESHOLD,
)
from dana.core.shared_state import (
    get_ui_state,
    is_engine_engaged,
    is_recording,
    mic_ingest_ready,
    ollama_ready,
    quiet_mic_mode,
    stop_event,
    tts_busy,
    vad_capture_active,
    wakeword_armed,
    whisper_bundle_lock,
)
from dana.logging import log, log_debug
from dana.paths import _nt_hide_console_if_mp_child, resolve_wakeword_onnx

# Nothing outside this module reads the active wake token (only
# state._shared_wakeword_model is shared cross-module, for stream-barge
# during TTS) -- private here rather than a bare import from shared_state.
_shared_wakeword_token: str = "dana"

def wake_score_hit(
    prediction: dict[str, Any],
    *,
    require_token: str = "dana",
    threshold: float = WAKE_THRESHOLD,
) -> Optional[str]:
    """Return matched wake-word key if score crosses threshold for require_token."""
    token = (require_token or "dana").lower()
    for key, score in prediction.items():
        try:
            value = float(score)
        except (TypeError, ValueError):
            continue
        key_l = str(key).lower()
        if value >= threshold and token in key_l:
            return f"{key}={value:.2f}"
    return None
def _normalize_wake_text(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()
def _wake_text_matches_dana(normalized: str) -> bool:
    """True if Whisper text is Dana or a known Dana mishearing."""
    if not normalized:
        return False
    if any(token in normalized for token in WAKE_PHRASE_TOKENS):
        return True
    # Exact / near-exact alias match (avoid accepting long unrelated sentences).
    if normalized in WAKE_PHRASE_ALIASES:
        return True
    for alias in WAKE_PHRASE_ALIASES:
        if normalized == alias or normalized.startswith(alias + " ") or normalized.endswith(" " + alias):
            return True
        # Short wake buffers are often just the misheard phrase.
        if len(normalized) <= len(alias) + 4 and alias in normalized:
            return True
    return False
def wake_phrase_confirmed(audio_16k: np.ndarray) -> bool:
    """Second gate: Whisper must hear Dana / Hey Dana in the wake buffer.

    When WAKE_PHRASE_VERIFY is False, openWakeWord score+onset alone starts the session.
    """
    if not WAKE_PHRASE_VERIFY:
        return True
    if audio_16k.size < SAMPLE_RATE // 4:
        return False

    with whisper_bundle_lock:
        bundle = state.whisper_bundle
    if bundle is None:
        # Whisper not loaded yet — keep energy/score gates only.
        return True

    processor, model, device, dtype = bundle
    try:
        import torch

        audio_prep = prepare_audio_for_whisper(audio_16k.astype(np.float32))
        inputs = processor(
            audio_prep,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )
        moved = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                if value.is_floating_point():
                    moved[key] = value.to(device=device, dtype=dtype)
                else:
                    moved[key] = value.to(device=device)
            else:
                moved[key] = value
        _sanitize_whisper_generation_config(model)
        gen_kwargs = _whisper_generate_kwargs(
            max_new_tokens=32,
            language="english",
            task="transcribe",
            model=model,
        )
        with torch.no_grad():
            generated_ids = model.generate(
                **moved,
                **gen_kwargs,
            )
        text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]
    except Exception as exc:  # noqa: BLE001
        log("WakeWord", f"WARNING: phrase verify failed ({exc}); allowing score gate only")
        return True

    normalized = _normalize_wake_text(text)
    if any(rej == normalized or rej in normalized for rej in WAKE_PHRASE_REJECT):
        log("WakeWord", f"Phrase verify REJECT (noise alias) -> \"{text.strip()}\"")
        print(f"[Debug] Wake phrase verify: \"{text.strip()}\" -> REJECT", flush=True)
        return False
    if _wake_text_matches_dana(normalized):
        log("WakeWord", f"Phrase verify PASS -> \"{text.strip()}\"")
        print(f"[Debug] Wake phrase verify: \"{text.strip()}\" -> PASS", flush=True)
        return True
    # Anything else (incl. short Whisper mishears like "Oh no.") is inconclusive:
    # OpenWakeWord score+onset already fired; only the explicit noise aliases above
    # are hard-rejected ("don't know" on hush).
    log(
        "WakeWord",
        f"Phrase verify inconclusive -> \"{text.strip()}\"; allowing score+energy gate",
    )
    print(
        f"[Debug] Wake phrase verify: \"{text.strip()}\" -> INCONCLUSIVE (allow)",
        flush=True,
    )
    return True
def wakeword_worker() -> None:
    _nt_hide_console_if_mp_child()

    # Prefer custom Dana wake model (dana.onnx), with legacy dana.onnx fallback.
    wake_token = "dana"
    model_paths: list[str]
    onnx_path = str(resolve_wakeword_onnx())
    if os.path.isfile(onnx_path):
        model_paths = [onnx_path]
        log("WakeWord", f"Loading OpenWakeWord model: {onnx_path}")
        try:
            from openwakeword.utils import download_models

            # Feature extractors (melspec/embedding) live in the package resources.
            download_models()
        except Exception as exc:  # noqa: BLE001
            log("WakeWord", f"WARNING: could not refresh OWW feature models ({exc})")
    else:
        print(
            "[Warning] dana.onnx / dana.onnx not found! Temporary Alexa wake-word "
            "enabled for mic debugging. Say 'Alexa' (not Dana). Place dana.onnx "
            "(or legacy dana.onnx) in assets/models/ to switch back.",
            flush=True,
        )
        log(
            "WakeWord",
            "WARNING: wake-word ONNX missing — temporary Alexa debug wake-word active.",
        )
        try:
            import openwakeword
            from openwakeword.utils import download_models

            models_dir = os.path.join(
                os.path.dirname(openwakeword.__file__), "resources", "models"
            )
            alexa_path = os.path.join(models_dir, "alexa_v0.1.onnx")
            if not os.path.isfile(alexa_path):
                log("WakeWord", "Downloading OpenWakeWord ONNX models for Alexa debug...")
                download_models()
            if not os.path.isfile(alexa_path):
                print(
                    "[Warning] Alexa debug model also missing. Voice wake-word disabled. "
                    "Use manual triggers (.trigger_ask).",
                    flush=True,
                )
                while not stop_event.is_set():
                    time.sleep(1)
                return
            model_paths = [alexa_path]
            wake_token = "alexa"
        except Exception as exc:  # noqa: BLE001
            print(
                f"[Warning] Could not load debug wake model ({exc}). "
                "Voice wake-word disabled. Use manual triggers.",
                flush=True,
            )
            log("WakeWord", f"WARNING: debug wake load failed ({exc})")
            while not stop_event.is_set():
                time.sleep(1)
            return

    try:
        oww = OpenWakeWordModel(
            wakeword_models=model_paths,
            inference_framework="onnx",
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"[Warning] Failed to load wake model ({exc}). "
            "Voice wake-word disabled. Use manual triggers.",
            flush=True,
        )
        log("WakeWord", f"WARNING: wake model load failed ({exc})")
        while not stop_event.is_set():
            time.sleep(1)
        return

    log("WakeWord", f"Models ready: {list(getattr(oww, 'models', {}).keys())}")
    global _shared_wakeword_token
    state._shared_wakeword_model = oww
    _shared_wakeword_token = wake_token
    if wake_token == "dana":
        print("Say 'Dana' to wake.", flush=True)
        listen_msg = "Dana"
    else:
        print("DEBUG: Say 'Alexa' to wake (temporary until dana.onnx is added).", flush=True)
        listen_msg = "Alexa (debug)"
    log(
        "WakeWord",
        f"Listening for {listen_msg} on mic [{state.AUDIO_INPUT_DEVICE}] @ {state.AUDIO_INPUT_RATE} Hz "
        "(or .trigger_ask)...",
    )
    print(
        f"[Debug] WakeWord using device={state.AUDIO_INPUT_DEVICE} "
        f"rate={state.AUDIO_INPUT_RATE} threshold={WAKE_THRESHOLD} "
        f"consec={WAKE_MIN_CONSECUTIVE} onset_below={WAKE_ONSET_BELOW} token={wake_token}",
        flush=True,
    )

    if not mic_ingest_ready.wait(timeout=8.0):
        log(
            "WakeWord",
            "WARNING: MicIngest not ready after 8s — continuing (will wait on queue)",
        )

    # Do not arm wake triggers until Ollama warm-up finishes (avoids CPU/TTS fights).
    log("WakeWord", "Waiting for Ollama warm-up before arming listener...")
    if not ollama_ready.wait(timeout=180.0):
        log(
            "WakeWord",
            "WARNING: Ollama warm-up not signaled after 180s — arming wake-word anyway",
        )
        ollama_ready.set()
    if quiet_mic_mode.is_set():
        log(
            "WakeWord",
            "Quiet Mic / Text-Only mode — wake-word polling disarmed "
            "(awaiting physical mic energy or text trigger)",
        )
        wakeword_armed.clear()
    else:
        log("WakeWord", "Ollama ready — wake-word listener armed")
        wakeword_armed.set()
    maybe_play_boot_ready_audio()

    cooldown_until = 0.0
    next_rms_log = 0.0
    consecutive_hits = 0
    score_history: deque[float] = deque(maxlen=WAKE_ONSET_LOOKBACK)
    audio_ring: deque[np.ndarray] = deque(maxlen=WAKE_PHRASE_WINDOW_CHUNKS)
    next_sticky_reset = 0.0
    # Assemble WAKE_CHUNK (80ms) from shared VAD frames (30ms).
    wake_accum: list[np.ndarray] = []
    wake_accum_samples = 0

    def _reset_wake_accum() -> None:
        nonlocal wake_accum_samples
        wake_accum.clear()
        wake_accum_samples = 0

    def _pull_wake_audio() -> Optional[np.ndarray]:
        """Consumer: build one WAKE_CHUNK from audio_buffer_queue frames."""
        nonlocal wake_accum_samples
        while wake_accum_samples < WAKE_CHUNK:
            if (
                tts_busy.is_set()
                or is_recording.is_set()
                or vad_capture_active.is_set()
                or get_ui_state() != "idle"
            ):
                return None
            frame = get_mic_frame(timeout=0.2)
            if frame is None:
                return None
            wake_accum.append(frame)
            wake_accum_samples += int(frame.size)
        merged = np.concatenate(wake_accum).astype(np.float32, copy=False)
        audio = merged[:WAKE_CHUNK].copy()
        remainder = merged[WAKE_CHUNK:]
        wake_accum.clear()
        wake_accum_samples = 0
        if remainder.size:
            wake_accum.append(remainder)
            wake_accum_samples = int(remainder.size)
        return audio

    while not stop_event.is_set():
        # Stay disarmed until warm-up (and after soft recoveries that clear the flag).
        if not ollama_ready.is_set():
            _reset_wake_accum()
            time.sleep(0.1)
            continue

        # Yield while TTS / VAD / turn owns the audio queue (half-duplex:
        # mic frames during TTS are discarded by half_duplex_mic_drop).
        if (
            tts_busy.is_set()
            or is_recording.is_set()
            or vad_capture_active.is_set()
            or get_ui_state() != "idle"
        ):
            _reset_wake_accum()
            time.sleep(0.05)
            continue

        if time.monotonic() < cooldown_until:
            time.sleep(0.05)
            continue

        audio = _pull_wake_audio()
        if audio is None:
            continue

        audio_ring.append(audio.copy())
        chunk_rms = audio_buffer_rms(audio)
        skip_wake = should_skip_wake_predict(chunk_rms)

        now = time.monotonic()
        if now >= next_rms_log:
            log_debug("Debug", f"Live Mic RMS: {chunk_rms:.6f}")
            # TEMP WIRETAP: wake-path diagnosis
            try:
                _dev_idx = state.AUDIO_INPUT_DEVICE
                if _dev_idx is not None:
                    _dev_name = str(sd.query_devices(int(_dev_idx)).get("name", "?"))
                else:
                    _dev_name = str(sd.query_devices(None).get("name", "?"))
            except Exception:
                _dev_idx = state.AUDIO_INPUT_DEVICE
                _dev_name = "?"
            print(
                f"!!! [WIRETAP] Mic: [{_dev_idx}] {_dev_name} "
                f"| Stream Rate: {state.AUDIO_INPUT_RATE} "
                f"| Array Shape: {audio.shape} "
                f"| RMS: {chunk_rms:.6f} "
                f"| Skipping: {skip_wake}",
                flush=True,
            )
            next_rms_log = now + 2.0

        # Dead / virtual mic silence: skip OpenWakeWord to prevent phantom wakes.
        if skip_wake:
            consecutive_hits = 0
            continue

        # Quiet-mic fallback: stay disarmed until physical energy (or text) arrives.
        if quiet_mic_mode.is_set():
            quiet_mic_mode.clear()
            wakeword_armed.set()
            log(
                "WakeWord",
                f"Physical mic energy detected (rms={chunk_rms:.6f}) — "
                "re-arming wake-word polling",
            )
            # Do NOT replay boot-ready TTS here — that caused random
            # "Dana is ready" mid-session when quiet-mic mode cleared.

        pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
        try:
            prediction = oww.predict(pcm)
        except Exception:
            try:
                prediction = oww.predict(audio)
            except Exception as exc:  # noqa: BLE001
                log("WakeWord", f"WARNING: predict failed: {exc}")
                consecutive_hits = 0
                continue

        pred = prediction if isinstance(prediction, dict) else {}
        best_score = 0.0
        for key, score in pred.items():
            try:
                value = float(score)
            except (TypeError, ValueError):
                continue
            key_l = str(key).lower()
            if wake_token in key_l:
                best_score = max(best_score, value)
            if value >= 0.50:
                # Near-miss / hit visibility for live threshold tuning.
                log(
                    "WakeWord",
                    f"score={value:.3f} key={key} "
                    f"(threshold={WAKE_THRESHOLD:.2f})",
                )
            elif value > 0.20:
                log_debug("Debug", f"Wake word score: {value:.4f} ({key})")

        score_history.append(best_score)
        hit = wake_score_hit(pred, require_token=wake_token)
        # Sticky high scores on hush never dip; real "Dana" rises from a low baseline.
        recently_low = any(s < WAKE_ONSET_BELOW for s in score_history)
        if hit and recently_low:
            consecutive_hits += 1
        else:
            if hit and not recently_low and now >= next_sticky_reset:
                log(
                    "WakeWord",
                    f"Rejected sticky false wake ({hit}); resetting detector "
                    f"(score never dipped below {WAKE_ONSET_BELOW:.2f})",
                )
                try:
                    oww.reset()
                except Exception:
                    pass
                next_sticky_reset = now + 2.0
            consecutive_hits = 0

        if consecutive_hits < WAKE_MIN_CONSECUTIVE:
            continue

        wake_audio = np.concatenate(list(audio_ring)) if audio_ring else audio
        # Diagnostic only — do not hard-reject on RMS (SteelSeries Sonar chat
        # mics often sit ~0.002–0.003 even on a real "Dana").
        wake_rms = (
            float(np.sqrt(np.mean(np.square(wake_audio)))) if wake_audio.size else 0.0
        )
        log_debug("WakeWord", f"Wake candidate buffer_rms={wake_rms:.5f} hit={hit}")

        if wake_token == "dana" and not wake_phrase_confirmed(wake_audio):
            consecutive_hits = 0
            cooldown_until = time.monotonic() + 1.5
            try:
                oww.reset()
            except Exception:
                pass
            continue

        if not is_engine_engaged():
            # Soft STANDBY — ignore wake hits until Dashboard ENGAGE.
            consecutive_hits = 0
            cooldown_until = time.monotonic() + 1.0
            try:
                oww.reset()
            except Exception:
                pass
            continue

        log("WakeWord", f"Wake word detected ({hit}) -> yield to VAD consumer")
        print(f"[Debug] Wake word HIT ({hit}) on device={state.AUDIO_INPUT_DEVICE}", flush=True)
        consecutive_hits = 0
        audio_ring.clear()
        score_history.clear()
        _reset_wake_accum()
        # Do NOT flush here — VAD takes the next frames from audio_buffer_queue.
        log_debug("WakeWord", "Consumer yielded; VAD will pull next mic frames")
        is_recording.set()
        cooldown_until = time.monotonic() + WAKE_COOLDOWN_SEC
        try:
            oww.reset()
        except Exception:
            pass

    log("WakeWord", "Stopped.")
