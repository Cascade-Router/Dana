"""DC blocker / high-pass smoke tests for VAD rumble kill."""

from __future__ import annotations

import numpy as np

from dana.core.constants import DC_BLOCKER_R
from dana.audio.dc_blocker import DcBlocker, remove_dc_offset


def test_dc_blocker_kills_constant_offset() -> None:
    blocker = DcBlocker(r=DC_BLOCKER_R)
    # Pure DC at mic-plausible level — should collapse toward ~0 after settle.
    dc = np.full(1600, 0.05, dtype=np.float32)
    out = blocker.apply(dc)
    tail = out[800:]
    assert float(np.mean(np.abs(tail))) < 0.005, float(np.mean(np.abs(tail)))
    print("[PASS] DC blocker kills constant offset")


def test_dc_blocker_streaming_chunks_match_whole() -> None:
    rng = np.random.default_rng(0)
    signal = (0.02 + 0.1 * rng.standard_normal(4800)).astype(np.float32)
    whole = DcBlocker(r=DC_BLOCKER_R).apply(signal)
    stream = DcBlocker(r=DC_BLOCKER_R)
    parts = [stream.apply(signal[i : i + 480]) for i in range(0, signal.size, 480)]
    streamed = np.concatenate(parts)
    assert streamed.shape == whole.shape
    assert float(np.max(np.abs(streamed - whole))) < 1e-5
    print("[PASS] streaming DC blocker matches whole-buffer apply")


def test_remove_dc_offset_preserves_ac_energy() -> None:
    t = np.arange(1600, dtype=np.float32) / 16000.0
    tone = (0.2 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    biased = tone + 0.08
    cleaned = remove_dc_offset(biased)
    # AC energy should remain; DC component should be tiny.
    assert float(np.abs(np.mean(cleaned))) < 0.01
    assert float(np.sqrt(np.mean(np.square(cleaned)))) > 0.1
    print("[PASS] remove_dc_offset preserves speech-band energy")


if __name__ == "__main__":
    test_dc_blocker_kills_constant_offset()
    test_dc_blocker_streaming_chunks_match_whole()
    test_remove_dc_offset_preserves_ac_energy()
    print("OK")
