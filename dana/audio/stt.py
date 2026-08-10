"""Whisper STT: model loading, hallucination filtering, and transcription.

Moved out of ``dana.core_agent`` (Phase 3 of the decomposition). Loading
happens on a background daemon thread (``start_whisper_background_load``) so
wake-word arming is never blocked on the HF/torch import tax; callers block
on ``ensure_whisper_bundle`` only when they actually need a transcript.
"""

from __future__ import annotations

import re
from typing import Any, Optional

import numpy as np

from dana.audio.dc_blocker import remove_dc_offset
from dana.core import shared_state as state
from dana.core.constants import (
    SAMPLE_RATE,
    WAKE_PHRASE_VERIFY,
    WHISPER_GAIN_RMS_CEIL,
    WHISPER_ID,
    WHISPER_INITIAL_PROMPT,
    WHISPER_MAX_GAIN,
    WHISPER_MAX_WORDS_PER_SEC,
    WHISPER_MIN_RMS_FOR_GAIN,
    WHISPER_TARGET_RMS,
    WHISPER_TASK,
    _STT_NAME_FIXES,
)
from dana.logging import log, log_debug

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_whisper(local_files_only: bool, device):
    # Latency path: Distil-Whisper on cuda:0 FP16 (3B LLM leaves enough VRAM headroom).
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    _ = local_files_only  # Prefer cache first; download only on OSError miss.
    _ = device
    if torch.cuda.is_available():
        whisper_device = torch.device("cuda:0")
        whisper_dtype = torch.float16
    else:
        whisper_device = torch.device("cpu")
        whisper_dtype = torch.float32
    log(
        "Conversation",
        f"Loading {WHISPER_ID} (local_files_only=True first, "
        f"device={whisper_device}, dtype={whisper_dtype})...",
    )
    import time

    t0 = time.perf_counter()

    def _load_pair(*, files_only: bool):
        proc = AutoProcessor.from_pretrained(
            WHISPER_ID,
            local_files_only=files_only,
        )
        mdl = AutoModelForSpeechSeq2Seq.from_pretrained(
            WHISPER_ID,
            dtype=whisper_dtype,
            local_files_only=files_only,
        ).to(whisper_device)
        return proc, mdl

    try:
        processor, model = _load_pair(files_only=True)
    except OSError as exc:
        log(
            "Conversation",
            f"{WHISPER_ID} not in local cache ({exc}); "
            "falling back to download (local_files_only=False)...",
        )
        processor, model = _load_pair(files_only=False)

    model.eval()
    # Prefer max_new_tokens-only length control. Whisper configs ship with
    # max_length=448; leaving both set triggers transformers warnings.
    _sanitize_whisper_generation_config(model)
    log(
        "Conversation",
        f"Whisper ready in {time.perf_counter() - t0:.1f}s on {whisper_device} "
        f"(dtype={whisper_dtype}).",
    )
    return processor, model, whisper_dtype, whisper_device


def _sanitize_whisper_generation_config(model: Any) -> None:
    """Drop conflicting length / processor fields from Whisper generation_config.

    Transformers warns when both ``max_length`` (often a tiny leftover like 20
    or the stock 448) and ``max_new_tokens`` are set. Callers pass
    ``max_new_tokens`` per generate(); strip every length field from config.
    """
    for obj_name in ("generation_config", "config"):
        gc = getattr(model, obj_name, None)
        if gc is None:
            continue
        for attr in ("max_length", "max_new_tokens", "min_length"):
            if not hasattr(gc, attr):
                continue
            try:
                setattr(gc, attr, None)
            except Exception:  # noqa: BLE001
                try:
                    delattr(gc, attr)
                except Exception:  # noqa: BLE001
                    pass


def _whisper_is_english_only(model: Any | None = None) -> bool:
    """True for *.en Distil/Whisper checkpoints that reject language/task kwargs."""
    if model is not None:
        gc = getattr(model, "generation_config", None)
        if gc is not None and getattr(gc, "is_multilingual", None) is False:
            return True
        if getattr(model, "is_multilingual", None) is False:
            return True
    mid = str(WHISPER_ID or "").lower()
    return mid.endswith(".en")


def _whisper_generate_kwargs(
    *,
    max_new_tokens: int,
    language: str,
    task: str,
    model: Any | None = None,
) -> dict[str, Any]:
    """Generation kwargs for Whisper STT — max_new_tokens only, no logits_processor."""
    kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "condition_on_prev_tokens": False,
        # Intentionally omitted: max_length (including any leftover max_length=20),
        # logits_processor, suppress_tokens, begin_suppress_tokens — transformers
        # builds SuppressTokens* processors from generation_config; passing them
        # again duplicates and warns. Length is controlled only by max_new_tokens.
    }
    # English-only Distil-Whisper raises if language/task are set.
    if not _whisper_is_english_only(model):
        kwargs["language"] = language
        kwargs["task"] = task
    return kwargs


def ensure_whisper_bundle(timeout: float = 180.0):
    """Block until background Whisper load finishes; return (proc, model, device, dtype)."""
    if not state.whisper_ready.wait(timeout=timeout):
        raise TimeoutError("Whisper background load timed out")
    with state.whisper_bundle_lock:
        bundle = state.whisper_bundle
    if bundle is None:
        raise RuntimeError(
            state._whisper_load_error or "Whisper failed to load (no bundle)"
        )
    return bundle


def start_whisper_background_load(local_files_only: bool, device) -> None:
    """Kick off Whisper HF load on a daemon thread; wake-word stays unblocked."""
    import threading

    state.whisper_ready.clear()
    state._whisper_load_error = None

    def _load() -> None:
        try:
            processor, model, whisper_dtype, whisper_device = load_whisper(
                local_files_only, device
            )
            with state.whisper_bundle_lock:
                state.whisper_bundle = (
                    processor,
                    model,
                    whisper_device,
                    whisper_dtype,
                )
            if WAKE_PHRASE_VERIFY:
                log("WakeWord", "Whisper phrase-verify gate armed (must hear 'Dana').")
            else:
                log(
                    "WakeWord",
                    "Whisper phrase-verify DISABLED — openWakeWord score onset starts session.",
                )
        except OSError as exc:
            state._whisper_load_error = (
                f"Whisper weights unavailable for {WHISPER_ID} "
                f"(cache miss and download failed): {exc}"
            )
            log("Conversation", f"ERROR: {state._whisper_load_error}")
            state.stop_event.set()
        except Exception as exc:  # noqa: BLE001
            state._whisper_load_error = f"ERROR loading Whisper: {exc}"
            log("Conversation", state._whisper_load_error)
            state.stop_event.set()
        finally:
            state.whisper_ready.set()

    threading.Thread(target=_load, name="WhisperLoad", daemon=True).start()
    log(
        "Conversation",
        "Whisper load started in background; wake-word / VAD remain available.",
    )


def resample_to_16k(audio: np.ndarray, src_rate: int) -> np.ndarray:
    """Lightweight linear resample to 16 kHz for VAD / Whisper / OpenWakeWord."""
    return resample_audio(audio, src_rate, SAMPLE_RATE)


def resample_audio(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear-resample a 1-D float audio buffer between sample rates."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if src_rate == dst_rate or audio.size == 0:
        return audio
    dst_len = max(1, int(round(audio.size * dst_rate / float(src_rate))))
    x_old = np.linspace(0.0, 1.0, num=audio.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=dst_len, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


# ---------------------------------------------------------------------------
# Hallucination / echo filtering
# ---------------------------------------------------------------------------


def is_whisper_prompt_echo(text: str) -> bool:
    """True when STT regurgitated the Whisper initial_prompt / fixture bias.

    Quiet-mic turns used to invent ``read the file project_omega_status.txt``
    (and similar bias phrases), which the broker then treated as a real file
    read and spoke the confidential Omega fixture aloud.
    """
    cleaned = re.sub(r"\s+", " ", (text or "").strip().lower())
    if not cleaned:
        return False
    bias = re.sub(r"\s+", " ", WHISPER_INITIAL_PROMPT.lower())
    if cleaned == bias or cleaned in bias or bias in cleaned:
        return True
    # Legacy bias tokens that must never become a live transcript on their own.
    legacy_markers = (
        "project_omega_status",
        "project omega",
        "file_jail_enforcer",
        "confidential status report",
        "draft_cursor_prompt",
        "patch_ledger",
    )
    hits = sum(1 for m in legacy_markers if m in cleaned)
    if hits >= 2:
        return True
    # Isolated Omega filename "read" with no other user intent → treat as echo.
    if "project_omega" in cleaned and re.search(
        r"\bread\s+(?:the\s+)?file\b", cleaned
    ):
        # Allow only when the user also states a clear non-bias intent verb+object.
        if not re.search(
            r"\b(summarize|explain|what\s+does|tell\s+me\s+about|status\s+of)\b",
            cleaned,
        ):
            return True
    return False


def is_punctuation_or_whitespace_only(text: str) -> bool:
    """True when transcript is only punctuation / whitespace (no letters or digits)."""
    raw = (text or "").strip()
    if not raw:
        return True
    if state.ARABIC_SCRIPT_RE.search(raw) or re.search(r"[A-Za-z0-9]", raw):
        return False
    return bool(state._PUNCT_OR_SPACE_ONLY_RE.fullmatch(raw))


def is_whisper_rate_hallucination(text: str, duration_s: float) -> bool:
    """True when word density exceeds a realistic human speaking rate.

    Word count is ``len(transcript.split())`` — never ``len(transcript)``
    (character length). Rate is strictly ``word_count / audio_duration_s``;
    values ``> WHISPER_MAX_WORDS_PER_SEC`` (5.0) return True.
    """
    transcript = (text or "").strip()
    if not transcript:
        return False
    # Unit-correct: words, not characters.
    word_count = len(transcript.split())
    if word_count <= 0:
        return False
    dur = float(duration_s)
    if dur <= 0.0:
        # No usable duration — only reject dense dumps on near-zero audio.
        return word_count >= 8
    rate = word_count / dur
    return rate > float(WHISPER_MAX_WORDS_PER_SEC)


def is_whisper_hallucination(
    text: str,
    *,
    audio_duration_s: Optional[float] = None,
) -> bool:
    """Reject empty, ultra-short, bracketed non-speech, or known silence hallucinations.

    Short non-Latin script utterances are allowed when they contain letters.
    Do not hardcode language-specific spam tokens here — diagnose low-SNR
    captures via STT debug logs instead.
    When ``audio_duration_s`` is provided, also apply the duration-to-word
    sanity check (words/sec above ``WHISPER_MAX_WORDS_PER_SEC``).
    """
    raw = (text or "").strip()
    if not raw:
        return True

    if is_punctuation_or_whitespace_only(raw):
        return True

    if is_whisper_prompt_echo(raw):
        return True

    if audio_duration_s is not None and is_whisper_rate_hallucination(
        raw, float(audio_duration_s)
    ):
        return True

    # Breathing / non-speech often lands as [sigh], (breathing), [BLANK_AUDIO], etc.
    if re.fullmatch(r"(?:\s*[\(\[][^\)\]]*[\)\]]\s*)+", raw):
        return True
    paren_stripped = re.sub(r"[\(\[][^\)\]]*[\)\]]", "", raw).strip(" .,!?;:\"'`-")
    if not paren_stripped:
        return True

    # Non-Latin script: keep short real phrases; only drop pure noise/punct.
    if state.ARABIC_SCRIPT_RE.search(raw):
        letters = state.ARABIC_SCRIPT_RE.findall(raw)
        return len(letters) < 1

    cleaned = raw.lower().strip(" .,!?;:\"'`")
    if cleaned in state.WHISPER_HALLUCINATIONS or cleaned in state.WHISPER_AMBIENT_SILENT:
        return True

    # Explicit non-speech tags Whisper emits even without brackets.
    non_speech = {
        "sigh",
        "breathing",
        "breath",
        "inhale",
        "exhale",
        "cough",
        "laughter",
        "blank_audio",
        "silence",
        "music",
        "applause",
    }
    if cleaned in non_speech:
        return True

    words = [w for w in cleaned.replace("-", " ").split() if w]
    if len(words) < 2:
        return True

    # Extra phrase-level traps Whisper-tiny loves on noise.
    bad_phrases = (
        "thank you for watching",
        "thanks for watching",
        "please subscribe",
        "like and subscribe",
        "see you next time",
        "don't forget to subscribe",
        "i'm going to be playing with you",
        "i am going to be playing with you",
        "i'm going to play with you",
        "i am going to play with you",
        "thanks for listening",
        "thank you for listening",
        "subtitles by",
        "transcript by",
        "draft_cursor_prompt",
        "patch ledger",
        "the ticket is on the board",
    )
    return any(p in cleaned for p in bad_phrases)


def is_silent_non_speech_transcript(text: str) -> bool:
    """STT artifacts that must return to listening with no LLM and no apology TTS.

    Covers empty / punctuation-only transcripts, bracketed non-speech, and known
    Whisper ambient-noise hallucinations (e.g. "Thank you", "Bye").
    """
    raw = (text or "").strip()
    if not raw:
        return True
    if is_punctuation_or_whitespace_only(raw):
        return True
    if re.fullmatch(r"(?:\s*[\(\[][^\)\]]*[\)\]]\s*)+", raw):
        return True
    stripped = re.sub(r"[\(\[][^\)\]]*[\)\]]", "", raw).strip(" .,!?;:\"'`-")
    if not stripped:
        return True
    cleaned = raw.lower().strip(" .,!?;:\"'`")
    if cleaned in state.WHISPER_AMBIENT_SILENT:
        return True
    ambient_phrases = (
        "thank you for watching",
        "thanks for watching",
        "please subscribe",
        "like and subscribe",
        "thanks for listening",
        "thank you for listening",
        "see you next time",
        "subtitles by",
        "transcript by",
    )
    return any(p == cleaned or p in cleaned for p in ambient_phrases)


def correct_known_stt_names(text: str) -> str:
    """Repair Whisper mangling of known household names / common phrases."""
    from dana.tools.stt_corrector import correct_stt

    out = correct_stt(text or "")
    if not out:
        return out
    for pattern, repl in _STT_NAME_FIXES:
        out = pattern.sub(repl, out)
    # Collapse "Amirhosein, Amirhosein" / "Narges, and Narges" after repair.
    out = re.sub(r"\b(Amirhosein)(?:\s*,\s*(?:and\s+)?|\s+and\s+|\s+)\1\b", r"\1", out, flags=re.I)
    out = re.sub(r"\b(Narges)(?:\s*,\s*(?:and\s+)?|\s+and\s+|\s+)\1\b", r"\1", out, flags=re.I)
    return out


# ---------------------------------------------------------------------------
# Audio conditioning + transcription
# ---------------------------------------------------------------------------


def prepare_audio_for_whisper(
    audio: np.ndarray,
    *,
    rms_raw: float | None = None,
) -> np.ndarray:
    """DC-block, gain-normalize quiet mic buffers, then peak-limit for Whisper."""
    audio = remove_dc_offset(np.asarray(audio, dtype=np.float32))
    if audio.size == 0:
        return audio
    # Always measure RMS after DC removal (offset-inflated energy mis-scales gain).
    rms_raw = float(np.sqrt(np.mean(np.square(audio))) + 1e-9)
    # Soft headphone path: normalize toward target whenever below soft ceil.
    # Absolute min (0.00005) replaces the stricter calibrated speech floor so
    # conversational quiet speech still reaches Whisper crisply.
    if WHISPER_MIN_RMS_FOR_GAIN < rms_raw < WHISPER_GAIN_RMS_CEIL:
        gain = min(WHISPER_TARGET_RMS / rms_raw, WHISPER_MAX_GAIN)
        audio = audio * float(gain)
        log_debug(
            "Conversation",
            f"Whisper gain x{gain:.1f} (rms_raw={rms_raw:.5f})",
        )
    elif rms_raw <= WHISPER_MIN_RMS_FOR_GAIN:
        log(
            "Conversation",
            f"Whisper gain skipped (rms_raw={rms_raw:.5f} below "
            f"{WHISPER_MIN_RMS_FOR_GAIN:.5f})",
        )
    peak = float(np.max(np.abs(audio)) + 1e-9)
    if peak > 1e-4:
        audio = np.clip(audio / peak * 0.9, -1.0, 1.0)
    return audio.astype(np.float32, copy=False)


def _whisper_initial_prompt_text() -> str:
    """Deprecated: live STT no longer feeds initial_prompt into the decoder."""
    return ""


def _whisper_prompt_ids(processor, device) -> Optional[Any]:
    """Always ``None`` — do not condition Whisper on prior text / ticket logs."""
    return None


def transcribe_audio(
    audio: np.ndarray,
    whisper_processor,
    whisper_model,
    device,
    whisper_dtype,
) -> str:
    import torch

    raw = np.asarray(audio, dtype=np.float32)
    # Duration from captured samples (before gain) — used for rate hallucination.
    audio_duration_s = float(raw.size) / float(SAMPLE_RATE) if raw.size else 0.0
    audio = prepare_audio_for_whisper(raw)
    inputs = whisper_processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    )
    moved = {}
    for key, value in inputs.items():
        if hasattr(value, "to"):
            if value.is_floating_point():
                moved[key] = value.to(device=device, dtype=whisper_dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value

    # Language from settings.json (default english). task=transcribe keeps
    # speech in the chosen language rather than translating.
    from dana.settings import get_whisper_language

    whisper_lang = get_whisper_language()
    # Fresh VAD capture: never condition on previous text / ticket logs.
    _sanitize_whisper_generation_config(whisper_model)
    gen_kwargs = _whisper_generate_kwargs(
        max_new_tokens=128,
        language=whisper_lang,
        task=WHISPER_TASK,
        model=whisper_model,
    )
    with torch.no_grad():
        generated_ids = whisper_model.generate(
            **moved,
            **gen_kwargs,
        )
    text = whisper_processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    text = correct_known_stt_names(text)
    # Hard discard: physically impossible speaking rate → empty transcript.
    if is_whisper_rate_hallucination(text, audio_duration_s):
        log(
            "Conversation",
            "Dropped physically impossible transcript (rate limit exceeded).",
        )
        return ""
    return text


__all__ = (
    "correct_known_stt_names",
    "ensure_whisper_bundle",
    "is_punctuation_or_whitespace_only",
    "is_silent_non_speech_transcript",
    "is_whisper_hallucination",
    "is_whisper_prompt_echo",
    "is_whisper_rate_hallucination",
    "load_whisper",
    "prepare_audio_for_whisper",
    "resample_audio",
    "resample_to_16k",
    "start_whisper_background_load",
    "transcribe_audio",
)
