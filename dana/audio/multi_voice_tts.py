"""Stage 8.1 / 8.8 — Multi-voice TTS routed by ``agent_id``.

``dana`` / broker / llama prefer Piper (receptionist).
Specialized agents (jason, moa, vision, typist) use offline ``pyttsx3``
profiles with distinct rate / gender so they never steal the default
receptionist Piper voice.
"""

from __future__ import annotations

import os
import tempfile
import threading
import wave
from pathlib import Path
from typing import Any, Literal

VoiceId = Literal["dana", "jason", "moa", "vision", "typist"]

JASON_ANDON_LINE = (
    "Operator failure detected. Taking control of the visual state."
)

# Stage 8.8 — agent_id → voice profile (pitch/rate stand-ins for distinct personas).
AGENT_VOICE_MAP: dict[str, str] = {
    "dana": "dana",
    "broker": "dana",
    "llama": "dana",
    "receptionist": "dana",
    "chat": "dana",
    "chat_node": "dana",
    "react_agent": "dana",
    "jason": "jason",
    "jason_supervisor": "jason",
    "cto": "jason",
    "moa": "moa",
    "moa_reasoner": "moa",
    "deepseek": "moa",
    "vision": "vision",
    "yolo": "vision",
    "florence": "vision",
    "vision_agent": "vision",
    "typist": "typist",
    "ghost_typist": "typist",
    "ghosttypist": "typist",
}

# Human-readable profile labels (for logs / orb tooltips).
VOICE_PROFILE_LABELS: dict[str, str] = {
    "dana": "voice_1_receptionist",
    "jason": "voice_jason_male",
    "moa": "voice_2_deep",
    "vision": "voice_3_robotic",
    "typist": "voice_typist",
}

# pyttsx3 rate (words/min-ish). Piper path ignores rate.
VOICE_RATES: dict[str, int] = {
    "dana": 165,
    "jason": 175,
    "moa": 128,  # deeper / slower
    "vision": 205,  # clipped / robotic
    "typist": 188,
}

# GUI persona colors (Stage 8.5.2 transcript tags) — shared with Assistive Orb.
PERSONA_COLORS: dict[str, str] = {
    "jason": "#9c27b0",
    "llama": "#0288d1",
    "broker": "#0288d1",
    "dana": "#0288d1",
    "deepseek": "#d32f2f",
    "moa": "#d32f2f",
    "vision": "#388e3c",
    "yolo": "#388e3c",
    "florence": "#388e3c",
    "typist": "#f57c00",
    "ghost_typist": "#f57c00",
    "default": "#00E676",
}

# Patch 8.3.1 — Piper is optional; ImportError → silent pyttsx3 fallback.
PIPER_AVAILABLE = True
try:
    from dana.core_agent import DEFAULT_PIPER_ONNX as _DEFAULT_PIPER_ONNX
except ImportError:
    PIPER_AVAILABLE = False
    _DEFAULT_PIPER_ONNX = None  # type: ignore[assignment]

_ACTIVE_AGENT_LOCK = threading.Lock()
_ACTIVE_TTS_AGENT = "broker"


def _log(msg: str) -> None:
    try:
        from dana.logging import log

        log("MultiVoiceTTS", msg)
    except Exception:  # noqa: BLE001
        print(f"[MultiVoiceTTS] {msg}", flush=True)


def normalize_agent_id(agent_id: str | None) -> str:
    raw = (agent_id or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not raw:
        return "broker"
    # Strip common suffixes: MoA_Reasoner → moa_reasoner
    return raw


def resolve_voice_id(agent_id: str | None) -> str:
    """Map ``agent_id`` / aliases → canonical voice profile id."""
    key = normalize_agent_id(agent_id)
    if key in AGENT_VOICE_MAP:
        return AGENT_VOICE_MAP[key]
    # Fuzzy: moa_reasoner, vision_agent, …
    for prefix, voice in (
        ("moa", "moa"),
        ("deepseek", "moa"),
        ("jason", "jason"),
        ("vision", "vision"),
        ("yolo", "vision"),
        ("florence", "vision"),
        ("typist", "typist"),
        ("broker", "dana"),
        ("llama", "dana"),
        ("dana", "dana"),
    ):
        if key.startswith(prefix) or prefix in key:
            return voice
    return "dana"


def persona_color_for_agent(agent_id: str | None) -> str:
    """Return GUI persona hex color for ``agent_id``."""
    key = normalize_agent_id(agent_id)
    if key in PERSONA_COLORS:
        return PERSONA_COLORS[key]
    voice = resolve_voice_id(key)
    # Map voice → closest persona color.
    voice_to_persona = {
        "dana": "broker",
        "jason": "jason",
        "moa": "moa",
        "vision": "vision",
        "typist": "typist",
    }
    return PERSONA_COLORS.get(voice_to_persona.get(voice, "default"), PERSONA_COLORS["default"])


def set_active_tts_agent(agent_id: str | None) -> str:
    """Publish the agent currently speaking (orb pulse + routing)."""
    global _ACTIVE_TTS_AGENT
    resolved = normalize_agent_id(agent_id) or "broker"
    with _ACTIVE_AGENT_LOCK:
        _ACTIVE_TTS_AGENT = resolved
    return resolved


def get_active_tts_agent() -> str:
    with _ACTIVE_AGENT_LOCK:
        return _ACTIVE_TTS_AGENT


def uses_receptionist_piper(agent_id: str | None) -> bool:
    """True when playback should stay on the default Piper receptionist path."""
    return resolve_voice_id(agent_id) == "dana"


def synthesize_speech(
    text: str,
    *,
    voice_id: VoiceId | str = "dana",
    agent_id: str | None = None,
    out_path: str | Path | None = None,
) -> Path:
    """Synthesize ``text`` for ``voice_id`` / ``agent_id`` and return a WAV path.

    Jason / MoA / Vision / Typist never load CUDA models (VRAM hard-cap friendly).
    """
    body = (text or "").strip() or "..."
    if agent_id:
        vid = resolve_voice_id(agent_id)
    else:
        vid = resolve_voice_id(voice_id)
    if vid not in VOICE_PROFILE_LABELS:
        vid = "dana"

    if out_path is None:
        fd, name = tempfile.mkstemp(prefix=f"donna_{vid}_", suffix=".wav")
        os.close(fd)
        dest = Path(name)
    else:
        dest = Path(out_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

    label = VOICE_PROFILE_LABELS.get(vid, vid)
    _log(f"synth voice={vid} profile={label} chars={len(body)}")

    if vid == "dana":
        # Receptionist — Piper first, then pyttsx3 female fallback.
        if _synthesize_piper(body, dest):
            return dest
        if _synthesize_pyttsx3(body, dest, prefer_male=False, rate=VOICE_RATES["dana"]):
            return dest
        _write_silence_wav(dest, duration_s=1.0)
        _log("donna TTS fallback — wrote silence placeholder")
        return dest

    prefer_male = vid in {"jason", "moa", "typist"}
    rate = int(VOICE_RATES.get(vid, 165))
    ok = _synthesize_pyttsx3(body, dest, prefer_male=prefer_male, rate=rate)
    if not ok:
        _write_silence_wav(dest, duration_s=1.0)
        _log(f"{vid} pyttsx3 unavailable — wrote silence placeholder")
    return dest


def synthesize_jason_andon_line(*, out_path: str | Path | None = None) -> Path:
    """CTO Andon recovery utterance."""
    return synthesize_speech(
        JASON_ANDON_LINE,
        voice_id="jason",
        out_path=out_path,
    )


def _synthesize_piper(text: str, dest: Path) -> bool:
    """Piper path; returns False quietly when unavailable (pyttsx3 fallback)."""
    if not PIPER_AVAILABLE or _DEFAULT_PIPER_ONNX is None:
        return False
    try:
        from dana.core_agent import get_piper_voice, synthesize_to_file
    except ImportError:
        return False
    try:
        model = str(_DEFAULT_PIPER_ONNX)
        if not Path(model).is_file():
            return False
        voice = get_piper_voice(model)
        return bool(synthesize_to_file(voice, text, str(dest)))
    except ImportError:
        return False
    except Exception:  # noqa: BLE001
        # Soft-fail without terminal stack traces — caller falls back to pyttsx3.
        return False


def _synthesize_pyttsx3(
    text: str,
    dest: Path,
    *,
    prefer_male: bool,
    rate: int = 165,
) -> bool:
    """Offline OS TTS (SAPI5 on Windows) — no VRAM."""
    try:
        import pyttsx3
    except Exception as exc:  # noqa: BLE001
        _log(f"pyttsx3 import failed: {exc}")
        return False
    try:
        engine = pyttsx3.init()
        voices = list(engine.getProperty("voices") or [])
        chosen = None
        for v in voices:
            name = f"{getattr(v, 'name', '')} {getattr(v, 'id', '')}".lower()
            if prefer_male and any(
                k in name for k in ("david", "male", "mark", "james", "richard")
            ):
                chosen = v
                break
            if not prefer_male and any(
                k in name for k in ("zira", "female", "hazel", "susan")
            ):
                chosen = v
                break
        if chosen is None and voices:
            # Prefer last voice for male when unmarked; first for female.
            chosen = voices[-1] if prefer_male else voices[0]
        if chosen is not None:
            engine.setProperty("voice", chosen.id)
        engine.setProperty("rate", int(rate))
        # Optional pitch (ignored by many SAPI drivers — rate is the real cue).
        try:
            engine.setProperty("pitch", 40 if prefer_male else 60)
        except Exception:  # noqa: BLE001
            pass
        # save_to_file is async until runAndWait.
        engine.save_to_file(text, str(dest))
        engine.runAndWait()
        try:
            engine.stop()
        except Exception:  # noqa: BLE001
            pass
        return dest.is_file() and dest.stat().st_size > 44
    except Exception as exc:  # noqa: BLE001
        _log(f"pyttsx3 synth failed: {exc}")
        return False


def _write_silence_wav(path: Path, *, duration_s: float = 1.0, rate: int = 22050) -> None:
    n_frames = max(1, int(rate * float(duration_s)))
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"\x00\x00" * n_frames)


def write_tone_wav(
    path: str | Path,
    *,
    duration_s: float = 1.0,
    freq_hz: float = 440.0,
    rate: int = 22050,
    amplitude: float = 0.2,
) -> Path:
    """Utility for dry ducking demos (no TTS dependency)."""
    import math
    import struct

    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    n = max(1, int(rate * float(duration_s)))
    frames = bytearray()
    for i in range(n):
        sample = amplitude * math.sin(2.0 * math.pi * freq_hz * (i / rate))
        frames += struct.pack("<h", int(max(-1.0, min(1.0, sample)) * 32767))
    with wave.open(str(dest), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(bytes(frames))
    return dest


def svg_logo_loader_notes() -> dict[str, Any]:
    """Optional Stage 8.8 outline: how to load a static SVG onto the orb Canvas.

    Tk Canvas has no native SVG. Compatible approaches (not wired by default):

    1. ``svglib`` + ``reportlab`` → render SVG to a ``.png``, then
       ``tk.PhotoImage`` / ``PIL.ImageTk`` onto the Canvas.
    2. Pre-bake a circular logo PNG under ``dana/assets/`` and
       ``canvas.create_image(...)`` inside ``AssistiveTouchOrb._draw_orb``.
    3. Stage 8.9.9 default: LANCZOS PNG via ``dana.ui.logo`` + PhotoImage /
       CTkImage; smooth Canvas polygons if the asset is missing.

    Returns a small dict describing the preferred asset path convention.
    """
    return {
        "preferred_asset": "dana/ui/assets/dana_logo_highres.png",
        "libraries": ("Pillow", "customtkinter"),
        "canvas_api": "create_image(cx, cy, image=photo) / CTkImage",
        "fallback": "smooth create_polygon mark (AssistiveTouchOrb._draw_smooth_mark)",
    }
