"""Stage 8.1 — Multi-voice audio mixer (Donna Channel 0, Jason Channel 1).

Jason's channel ducks the Receptionist to 0.2 while he speaks, then restores
Channel 0 to full volume.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

CHANNEL_DONNA = 0
CHANNEL_JASON = 1
DONNA_FULL_VOLUME = 1.0
DONNA_DUCK_VOLUME = 0.2

_LOCK = threading.Lock()
_INIT_DONE = False
_MIXER_OK = False
_DONNA_CHANNEL: Any = None
_JASON_CHANNEL: Any = None

# Dry-run bookkeeping (tests / headless CI without a sound device).
_DRY_VOLUME: float = DONNA_FULL_VOLUME
_DRY_EVENTS: list[dict[str, Any]] = []
_DRY_JASON_PLAYING = threading.Event()


def _dry_run() -> bool:
    return os.environ.get("DONNA_AUDIO_DRY_RUN", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _log(msg: str) -> None:
    try:
        from dana.logging import log

        log("AudioMixer", msg)
    except Exception:  # noqa: BLE001
        print(f"[AudioMixer] {msg}", flush=True)


def ensure_mixer(*, frequency: int = 22050, size: int = -16) -> bool:
    """Initialize pygame.mixer with at least 2 channels. Idempotent."""
    global _INIT_DONE, _MIXER_OK, _DONNA_CHANNEL, _JASON_CHANNEL
    with _LOCK:
        if _INIT_DONE:
            return _MIXER_OK
        _INIT_DONE = True
        if _dry_run():
            _MIXER_OK = True
            _log("dry-run mixer armed (no pygame device)")
            return True
        try:
            import pygame

            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=frequency, size=size, channels=2, buffer=512)
            pygame.mixer.set_num_channels(max(2, pygame.mixer.get_num_channels()))
            _DONNA_CHANNEL = pygame.mixer.Channel(CHANNEL_DONNA)
            _JASON_CHANNEL = pygame.mixer.Channel(CHANNEL_JASON)
            _DONNA_CHANNEL.set_volume(DONNA_FULL_VOLUME)
            _JASON_CHANNEL.set_volume(DONNA_FULL_VOLUME)
            _MIXER_OK = True
            _log("pygame.mixer ready channels=2 (Donna=0, Jason=1)")
            return True
        except Exception as exc:  # noqa: BLE001
            _MIXER_OK = False
            _log(f"pygame.mixer init failed ({exc}); dry bookkeeping only")
            return False


def get_donna_volume() -> float:
    """Current Channel 0 volume (dry or live)."""
    if _dry_run() or not _MIXER_OK or _DONNA_CHANNEL is None:
        return float(_DRY_VOLUME)
    try:
        return float(_DONNA_CHANNEL.get_volume())
    except Exception:  # noqa: BLE001
        return float(_DRY_VOLUME)


def _set_donna_volume(vol: float) -> None:
    global _DRY_VOLUME
    v = max(0.0, min(1.0, float(vol)))
    _DRY_VOLUME = v
    if _DONNA_CHANNEL is not None:
        try:
            _DONNA_CHANNEL.set_volume(v)
        except Exception:  # noqa: BLE001
            pass


def play_donna(audio_file: str | Path, *, block: bool = False) -> bool:
    """Play Receptionist audio on Channel 0."""
    ensure_mixer()
    path = Path(audio_file)
    _DRY_EVENTS.append({"event": "play_donna", "path": str(path), "t": time.time()})
    if _dry_run() or not _MIXER_OK or _DONNA_CHANNEL is None:
        # Approximate duration from file size when possible; else short stub.
        dur = _estimate_wav_duration_s(path) or 1.0

        def _stub() -> None:
            _set_donna_volume(DONNA_FULL_VOLUME)
            time.sleep(dur)

        if block:
            _stub()
        else:
            threading.Thread(target=_stub, name="DonnaPlayDry", daemon=True).start()
        return True
    try:
        import pygame

        sound = pygame.mixer.Sound(str(path))
        _set_donna_volume(DONNA_FULL_VOLUME)
        _DONNA_CHANNEL.play(sound)
        if block:
            while _DONNA_CHANNEL.get_busy():
                time.sleep(0.05)
        return True
    except Exception as exc:  # noqa: BLE001
        _log(f"play_donna failed: {exc}")
        return False


def play_jason(audio_file: str | Path, *, block: bool = True) -> bool:
    """Duck Donna to 0.2, play Jason on Channel 1, then restore Donna to 1.0."""
    ensure_mixer()
    path = Path(audio_file)
    _DRY_EVENTS.append({"event": "play_jason_start", "path": str(path), "t": time.time()})
    _set_donna_volume(DONNA_DUCK_VOLUME)
    _DRY_EVENTS.append(
        {"event": "duck", "volume": DONNA_DUCK_VOLUME, "t": time.time()}
    )
    _DRY_JASON_PLAYING.set()

    def _restore() -> None:
        _DRY_JASON_PLAYING.clear()
        _set_donna_volume(DONNA_FULL_VOLUME)
        _DRY_EVENTS.append(
            {"event": "restore", "volume": DONNA_FULL_VOLUME, "t": time.time()}
        )
        _DRY_EVENTS.append({"event": "play_jason_end", "t": time.time()})

    if _dry_run() or not _MIXER_OK or _JASON_CHANNEL is None:
        dur = _estimate_wav_duration_s(path) or 1.5

        def _stub() -> None:
            try:
                time.sleep(dur)
            finally:
                _restore()

        if block:
            _stub()
        else:
            threading.Thread(target=_stub, name="JasonPlayDry", daemon=True).start()
        return True

    try:
        import pygame

        sound = pygame.mixer.Sound(str(path))
        _JASON_CHANNEL.set_volume(DONNA_FULL_VOLUME)
        _JASON_CHANNEL.play(sound)

        def _wait_and_restore() -> None:
            try:
                while _JASON_CHANNEL.get_busy():
                    time.sleep(0.05)
            finally:
                _restore()

        if block:
            _wait_and_restore()
        else:
            threading.Thread(
                target=_wait_and_restore, name="JasonDuckRestore", daemon=True
            ).start()
        return True
    except Exception as exc:  # noqa: BLE001
        _log(f"play_jason failed: {exc}")
        _restore()
        return False


def _estimate_wav_duration_s(path: Path) -> float | None:
    try:
        import wave

        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate() or 1
            return max(0.05, float(frames) / float(rate))
    except Exception:  # noqa: BLE001
        return None


def dry_events() -> list[dict[str, Any]]:
    """Test helper: copy of dry-run mixer event log."""
    return list(_DRY_EVENTS)


def clear_dry_events() -> None:
    _DRY_EVENTS.clear()
