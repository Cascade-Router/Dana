"""Live wake-word diagnostic: tail dana_runtime.log for MicIngest / WakeWord signals.

Usage:
  .venv\\Scripts\\python.exe scripts\\diagnostics\\wake_word_live_diag.py --seconds 45
"""

from __future__ import annotations

import argparse
import re
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "logs" / "dana_runtime.log"

RMS_RE = re.compile(r"RMS:\s*([0-9.]+)")
SCORE_RE = re.compile(r"dana[=:]?\s*([0-9.]+)", re.I)
WAKE_HIT_RE = re.compile(r"Wake word detected\s*\(([^)]+)\)", re.I)


def _tail_follow(path: Path, *, seconds: float) -> list[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    matched: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        fh.seek(0, 2)
        deadline = time.time() + max(5.0, float(seconds))
        buf = ""
        print(f"[diag] Tailing {path} for {seconds:.0f}s — say 'Dana' now.", flush=True)
        while time.time() < deadline:
            chunk = fh.read()
            if not chunk:
                time.sleep(0.15)
                continue
            buf += chunk
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                low = line.lower()
                keep = (
                    "wiretap" in low
                    or "[wakeword]" in low
                    or "[micingest]" in low
                    or "[audio]" in low
                    or "ollama_ready" in low
                    or "engage" in low
                    or "standby" in low
                    or "quiet mic" in low
                    or "engine" in low and ("engage" in low or "standby" in low)
                )
                if keep:
                    print(line, flush=True)
                    matched.append(line)
    return matched


def summarize(lines: list[str]) -> None:
    rms_vals: list[float] = []
    skipping_false = 0
    skipping_true = 0
    scores: list[float] = []
    hits: list[str] = []
    quiet = False
    ollama_ready = None
    engage = None

    for line in lines:
        m = RMS_RE.search(line)
        if m:
            try:
                rms_vals.append(float(m.group(1)))
            except ValueError:
                pass
        if "Skipping: False" in line:
            skipping_false += 1
        if "Skipping: True" in line:
            skipping_true += 1
        if "Quiet Mic" in line or "quiet_mic" in line or "Text-Only" in line:
            quiet = True
        if "ollama_ready=True" in line or "Wake-word arming allowed" in line:
            ollama_ready = True
        if "ollama_ready=False" in line:
            ollama_ready = False
        if "ENGAGE engine" in line or "Engine ENGAGED" in line:
            engage = True
        if "STANDBY engine" in line or "Engine STANDBY" in line:
            engage = False
        hm = WAKE_HIT_RE.search(line)
        if hm:
            hits.append(hm.group(1))
            sm = SCORE_RE.search(hm.group(1))
            if sm:
                scores.append(float(sm.group(1)))
        # debug score lines
        if "Wake candidate" in line or "score=" in line.lower():
            sm = SCORE_RE.search(line)
            if sm:
                scores.append(float(sm.group(1)))

    print("\n======== WAKE DIAG SUMMARY ========", flush=True)
    if rms_vals:
        print(
            f"RMS: n={len(rms_vals)} min={min(rms_vals):.6f} "
            f"max={max(rms_vals):.6f} mean={sum(rms_vals)/len(rms_vals):.6f}",
            flush=True,
        )
    else:
        print("RMS: (no wiretap samples in window)", flush=True)
    print(f"Skipping False/True counts: {skipping_false}/{skipping_true}", flush=True)
    print(f"quiet_mic_mode signals in window: {quiet}", flush=True)
    print(f"ollama_ready (from logs): {ollama_ready}", flush=True)
    print(f"engine_engaged (ENGAGE/STANDBY from logs): {engage}", flush=True)
    if scores:
        print(
            f"OpenWakeWord scores: {sorted(scores)} "
            f"(threshold typically 0.80)",
            flush=True,
        )
    else:
        print("OpenWakeWord scores: (none logged in window)", flush=True)
    print(f"Wake hits: {hits or '(none)'}", flush=True)
    print("===================================", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=40.0)
    args = ap.parse_args()
    lines = _tail_follow(LOG, seconds=args.seconds)
    summarize(lines)


if __name__ == "__main__":
    main()
