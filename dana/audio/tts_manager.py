"""Centralized, thread-safe TTS speech queue for system-wide voice notifications.

Any component (Meta-Broker, System Health, Router, UI) pushes text via
``TTSManager.enqueue`` / ``enqueue_speech``. A single daemon thread drains
``speech_queue`` sequentially so Piper playback never overlaps.
"""

from __future__ import annotations

import queue
import re
import threading
from typing import Any, Callable

from dana.core.constants import TTS_CHUNK_MAX_CHARS
from dana.logging import log_debug

_EnqueueFn = Callable[..., None]
_WorkerFn = Callable[[], None]


class TTSManager:
    """Process-wide singleton: one speech queue + one consumer thread."""

    _instance: TTSManager | None = None
    _instance_lock = threading.Lock()

    def __init__(self, *, maxsize: int = 64) -> None:
        self.speech_queue: queue.Queue[Any] = queue.Queue(maxsize=max(8, int(maxsize)))
        self._enqueue_lock = threading.Lock()
        self._start_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._worker_target: _WorkerFn | None = None
        self._enqueue_impl: _EnqueueFn | None = None
        self._started = False

    @classmethod
    def instance(cls) -> TTSManager:
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def bind(
        self,
        *,
        speech_queue: queue.Queue[Any] | None = None,
        worker: _WorkerFn | None = None,
        enqueue_impl: _EnqueueFn | None = None,
    ) -> None:
        """Wire the Piper spooler (core_agent) without circular imports at module load."""
        if speech_queue is not None:
            self.speech_queue = speech_queue
        if worker is not None:
            self._worker_target = worker
        if enqueue_impl is not None:
            self._enqueue_impl = enqueue_impl

    def start(self, *, worker: _WorkerFn | None = None) -> threading.Thread | None:
        """Launch the dedicated daemon consumer (idempotent)."""
        if worker is not None:
            self._worker_target = worker
        target = self._worker_target
        if target is None:
            return None
        with self._start_lock:
            if self._thread is not None and self._thread.is_alive():
                return self._thread
            self._thread = threading.Thread(
                target=target,
                name="TTSManager",
                daemon=True,
            )
            self._thread.start()
            self._started = True
            return self._thread

    @property
    def is_running(self) -> bool:
        t = self._thread
        return t is not None and t.is_alive()

    def enqueue(
        self,
        text: str,
        *,
        interruptible: bool | None = None,
        agent_id: str | None = None,
    ) -> None:
        """Push a phrase onto the speech queue (non-blocking for callers).

        Prefers the bound core_agent ``enqueue_speech`` (chunking / barge rules).
        Falls back to a raw queue put when unbound (tests / early boot).
        """
        impl = self._enqueue_impl
        if impl is not None:
            try:
                kwargs: dict[str, Any] = {}
                if interruptible is not None:
                    kwargs["interruptible"] = interruptible
                if agent_id is not None:
                    kwargs["agent_id"] = agent_id
                impl(str(text or ""), **kwargs)
                return
            except Exception:  # noqa: BLE001
                pass
        piece = str(text or "").strip()
        if not piece:
            return
        flag = True if interruptible is None else bool(interruptible)
        aid = str(agent_id or "broker")
        with self._enqueue_lock:
            try:
                self.speech_queue.put_nowait((piece, flag, aid))
            except queue.Full:
                try:
                    _ = self.speech_queue.get_nowait()
                except queue.Empty:
                    return
                try:
                    self.speech_queue.put_nowait((piece, flag, aid))
                except queue.Full:
                    pass

    def notify(self, text: str, *, interruptible: bool = False) -> None:
        """System notification helper (acks / epic / health — uninterruptible by default)."""
        self.enqueue(text, interruptible=interruptible)


def get_tts_manager() -> TTSManager:
    return TTSManager.instance()


def enqueue_speech(
    text: str,
    *,
    interruptible: bool | None = None,
    agent_id: str | None = None,
) -> None:
    """Module-level producer API (alias for ``TTSManager.enqueue``)."""
    get_tts_manager().enqueue(
        text, interruptible=interruptible, agent_id=agent_id
    )


def strip_code_blocks_for_tts(text: str) -> str:
    """Replace markdown fenced code with a short spoken placeholder.

    Prevents Piper from reading raw Python/JSON aloud (long TTS ceiling crash).
    """
    from dana.core import shared_state as state

    raw = text or ""
    if "```" not in raw:
        return raw
    out = state._CODE_FENCE_TTS_RE.sub("[Code block generated]", raw)
    out = state._CODE_FENCE_TTS_UNCLOSED_RE.sub("[Code block generated]", out)
    out = re.sub(r"(?:\s*\[Code block generated\]\s*){2,}", " [Code block generated] ", out)
    return out.strip()


def sanitize_text_for_tts(text: str) -> str:
    """Strip markdown emphasis/code markers before Piper synthesis.

    Returns empty string when nothing speakable remains (caller must skip TTS).
    Newlines become pause punctuation so OCR/vision dumps are not rushed.
    """
    from dana.core import shared_state as state

    out = strip_code_blocks_for_tts(text or "")
    out = state._TTS_MD_MARKERS_RE.sub("", out)
    # Florence / vision dumps often arrive as newline-joined tokens with no stops.
    out = re.sub(r"[\r\n]+", ". ", out)
    out = re.sub(r"\s+", " ", out).strip()
    # Soft-break very long comma-less runs so Piper inserts breath pauses.
    if len(out) > 160 and not re.search(r"[.!?]", out):
        out = re.sub(r"(.{40,80}?)\s+", r"\1. ", out)
    if not out or state._PUNCT_OR_SPACE_ONLY_RE.match(out):
        return ""
    return out


def chunk_text_for_tts(text: str, *, max_chars: int = TTS_CHUNK_MAX_CHARS) -> list[str]:
    """Split long speakable text into sentence-sized chunks under ``max_chars``.

    Keeps short UX phrases intact. Prevents the 90s watchdog from aborting a
    single monolithic Piper render of OCR / multi-paragraph answers.
    """
    raw = sanitize_text_for_tts(text or "")
    if not raw:
        return []
    limit = max(80, int(max_chars))
    if len(raw) <= limit:
        return [raw]
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw) if s.strip()]
    if not sentences:
        sentences = [raw]
    chunks: list[str] = []
    buf = ""
    for sentence in sentences:
        if not buf:
            candidate = sentence
        else:
            candidate = f"{buf} {sentence}"
        if len(candidate) <= limit:
            buf = candidate
            continue
        if buf:
            chunks.append(buf)
        if len(sentence) <= limit:
            buf = sentence
        else:
            # Hard-wrap oversized sentence on spaces.
            words = sentence.split()
            buf = ""
            for word in words:
                piece = word if not buf else f"{buf} {word}"
                if len(piece) <= limit:
                    buf = piece
                else:
                    if buf:
                        chunks.append(buf)
                    buf = word
    if buf:
        chunks.append(buf)
    return chunks or [raw]


def enqueue_speech_impl(
    text: str,
    *,
    interruptible: bool | None = None,
    agent_id: str | None = None,
) -> None:
    """Producer API: push text into the TTS spooler and return immediately.

    Never opens PortAudio or blocks on Piper — the ``tts_worker`` owns playback.
    Caps pending phrases while busy so stream handlers cannot hammer the device.
    Long utterances are chunked into sequential spool items (sentence-level) so
    playback can start before LangGraph finishes the full turn.

    ``interruptible=False`` marks UI acknowledgments (wake "Yes?", mode acks) so
    VAD/wake barge-in cannot cut them (Apple-style self-barge exemption).
    When omitted, canned UX cache hits default to uninterruptible.

    Stage 8.8 — ``agent_id`` selects a multi-voice profile (broker/moa/jason/…).
    """
    from dana.core import shared_state as state

    pieces = chunk_text_for_tts(text or "")
    if not pieces:
        with state._tts_enqueue_lock:
            if state.tts_queue.empty() and not state.tts_busy.is_set():
                state.speech_idle.set()
        return

    try:
        from dana.audio.multi_voice_tts import (
            normalize_agent_id,
            set_active_tts_agent,
        )

        aid = normalize_agent_id(agent_id) if agent_id else "broker"
        if agent_id:
            set_active_tts_agent(aid)
    except Exception:  # noqa: BLE001
        aid = (agent_id or "broker").strip().lower() or "broker"

    # Lazy import — dana.audio.tts_worker imports several functions back from
    # this module, so this side of the cycle must resolve at call time.
    from dana.audio.tts_worker import canned_ux_cache_path

    with state._tts_enqueue_lock:
        for piece in pieces:
            if interruptible is None:
                # Canned UX WAVs (Yes?, mode active, …) are uninterruptible by default.
                try:
                    piece_interruptible = canned_ux_cache_path(piece) is None
                except Exception:  # noqa: BLE001
                    piece_interruptible = True
            else:
                piece_interruptible = bool(interruptible)
            state.speech_idle.clear()
            pending = state.tts_queue.qsize()
            if state.tts_busy.is_set() and pending >= state._SPEECH_MAX_PENDING_WHILE_BUSY:
                log_debug(
                    "TTS",
                    f"spool busy — drop overflow chars={len(piece)} pending={pending}",
                )
                break
            try:
                state.tts_queue.put_nowait((piece, piece_interruptible, aid))
                log_debug(
                    "TTS",
                    f"spooled chars={len(piece)} interruptible={piece_interruptible} "
                    f"agent={aid} pending={state.tts_queue.qsize()} "
                    f"busy={state.tts_busy.is_set()} vad={state.vad_capture_active.is_set()}",
                )
            except queue.Full:
                log_debug("TTS", f"spool full — drop chars={len(piece)}")
                break


def _parse_tts_spool_item(item: Any) -> tuple[str, bool, str]:
    """Normalize queue items to ``(text, interruptible, agent_id)``."""
    if isinstance(item, tuple) and item:
        text = str(item[0] or "")
        flag = bool(item[1]) if len(item) > 1 else True
        agent = str(item[2] or "broker") if len(item) > 2 else "broker"
        return text, flag, agent
    return str(item or ""), True, "broker"


def flush_tts_queue() -> int:
    """Instantly dump pending system messages (barge-in / interrupt).

    Uses the internal deque clear under the queue mutex so the agent does not
    keep talking after being cut off.
    """
    from dana.core import shared_state as state

    with state._tts_enqueue_lock:
        with state.tts_queue.mutex:
            n = len(state.tts_queue.queue)
            state.tts_queue.queue.clear()
            state.tts_queue.not_full.notify_all()
        if n:
            log_debug("TTS", f"Flushed {n} pending spool item(s)")
        return n


def flush_speech_queue() -> int:
    """Alias for ``flush_tts_queue`` (legacy call sites)."""
    return flush_tts_queue()


# Bind the real chunking/canned-UX/multi-voice implementation as this
# process's TTSManager enqueue_impl (moved here from core_agent.py's
# module-level wiring -- same effect, now co-located with both pieces).
try:
    get_tts_manager().bind(enqueue_impl=enqueue_speech_impl)
except Exception:  # noqa: BLE001
    pass


__all__ = (
    "TTSManager",
    "chunk_text_for_tts",
    "enqueue_speech",
    "enqueue_speech_impl",
    "flush_speech_queue",
    "flush_tts_queue",
    "get_tts_manager",
    "sanitize_text_for_tts",
    "strip_code_blocks_for_tts",
)
