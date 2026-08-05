"""TPS / TTFT benchmark — raw Ollama chat, bypasses the ReAct loop.

Usage:
  python tests/test_inference_speed.py
  python tests/test_inference_speed.py --compare   # baseline vs speculative

Env:
  DONNA_LOCAL_MODEL / OLLAMA_MODEL  — main model (default llama3.2)
  DONNA_SPECULATIVE_DECODING        — set by --compare / --speculative
  DONNA_DRAFT_MODEL                 — draft tag (default llama3.2:1b)
  DONNA_DRAFT_NUM_PREDICT           — draft depth (default 4; 0 = off)
  OLLAMA_URL                        — default http://127.0.0.1:11434
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Repo root on sys.path when run as a script.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

BENCHMARK_PROMPT = (
    "Write a Python script for a multi-threaded TCP server that accepts "
    "client connections, echoes messages, handles graceful shutdown with "
    "a signal handler, and includes brief comments explaining the design."
)


@dataclass
class SpeedResult:
    label: str
    model: str
    speculative: bool
    draft_num_predict: int | None
    ttft_sec: float
    total_sec: float
    completion_tokens: int
    prompt_tokens: int
    tps: float
    chars: int
    text_preview: str


def _ollama_base() -> str:
    return (os.environ.get("OLLAMA_URL") or "http://127.0.0.1:11434").rstrip("/")


def _main_model() -> str:
    return (
        (os.environ.get("DONNA_LOCAL_MODEL") or "").strip()
        or (os.environ.get("OLLAMA_MODEL") or "").strip()
        or "llama3.2"
    )


def stream_raw_chat(
    *,
    model: str,
    prompt: str,
    num_predict: int = 256,
    speculative: bool,
    draft_num_predict: int = 4,
) -> SpeedResult:
    """Bypass ReAct: POST /api/chat with a single user message and stream."""
    from dana.llm_client import merge_ollama_options

    # Isolate A/B: force env for merge_ollama_options for this call only.
    prev_spec = os.environ.get("DONNA_SPECULATIVE_DECODING")
    prev_draft_n = os.environ.get("DONNA_DRAFT_NUM_PREDICT")
    try:
        if speculative:
            os.environ["DONNA_SPECULATIVE_DECODING"] = "1"
            os.environ["DONNA_DRAFT_NUM_PREDICT"] = str(draft_num_predict)
        else:
            os.environ["DONNA_SPECULATIVE_DECODING"] = "0"
            os.environ.pop("DONNA_DRAFT_NUM_PREDICT", None)

        options = merge_ollama_options(
            {
                "num_ctx": 4096,
                "num_predict": int(num_predict),
                "temperature": 0.2,
            }
        )
        draft_n = options.get("draft_num_predict")

        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "keep_alive": "5m",
            "options": options,
        }

        url = f"{_ollama_base()}/api/chat"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        t0 = time.perf_counter()
        ttft: float | None = None
        parts: list[str] = []
        completion_tokens = 0
        prompt_tokens = 0
        eval_duration_ns = 0

        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    msg = data.get("message") or {}
                    piece = str(msg.get("content") or "")
                    if piece:
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        parts.append(piece)
                    if data.get("done"):
                        completion_tokens = int(data.get("eval_count") or 0)
                        prompt_tokens = int(data.get("prompt_eval_count") or 0)
                        eval_duration_ns = int(data.get("eval_duration") or 0)
                        break
        except urllib.error.URLError as exc:
            raise SystemExit(
                f"Ollama unreachable at {url}: {exc}\n"
                "Start Ollama and pull the model before benchmarking."
            ) from exc

        total = time.perf_counter() - t0
        text = "".join(parts)
        if ttft is None:
            ttft = total

        # Prefer Ollama's eval_duration for generation TPS; fall back to wall clock.
        if completion_tokens > 0 and eval_duration_ns > 0:
            tps = completion_tokens / (eval_duration_ns / 1e9)
        elif completion_tokens > 0 and (total - ttft) > 0:
            tps = completion_tokens / (total - ttft)
        else:
            # Rough char heuristic if server omitted counts.
            approx_tokens = max(1, len(text) // 4)
            gen = max(1e-6, total - ttft)
            tps = approx_tokens / gen
            completion_tokens = approx_tokens

        label = "speculative" if speculative else "baseline"
        return SpeedResult(
            label=label,
            model=model,
            speculative=speculative,
            draft_num_predict=int(draft_n) if draft_n is not None else None,
            ttft_sec=ttft,
            total_sec=total,
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            tps=tps,
            chars=len(text),
            text_preview=text[:160].replace("\n", " "),
        )
    finally:
        if prev_spec is None:
            os.environ.pop("DONNA_SPECULATIVE_DECODING", None)
        else:
            os.environ["DONNA_SPECULATIVE_DECODING"] = prev_spec
        if prev_draft_n is None:
            os.environ.pop("DONNA_DRAFT_NUM_PREDICT", None)
        else:
            os.environ["DONNA_DRAFT_NUM_PREDICT"] = prev_draft_n


def _print_result(r: SpeedResult) -> None:
    print(f"\n=== {r.label} ===")
    print(f"model:              {r.model}")
    print(f"speculative:        {r.speculative} (draft_num_predict={r.draft_num_predict})")
    print(f"TTFT:               {r.ttft_sec * 1000.0:.1f} ms")
    print(f"total:              {r.total_sec:.2f} s")
    print(f"prompt tokens:      {r.prompt_tokens}")
    print(f"completion tokens:  {r.completion_tokens}")
    print(f"TPS:                {r.tps:.2f} tok/s")
    print(f"chars:              {r.chars}")
    print(f"preview:            {r.text_preview!r}")


def _print_delta(base: SpeedResult, spec: SpeedResult) -> None:
    if base.tps <= 0:
        print("\nTPS delta: n/a (baseline TPS was 0)")
        return
    delta = spec.tps - base.tps
    pct = (delta / base.tps) * 100.0
    print("\n=== TPS delta (speculative − baseline) ===")
    print(f"Δ TPS:  {delta:+.2f} tok/s ({pct:+.1f}%)")
    print(f"Δ TTFT: {(spec.ttft_sec - base.ttft_sec) * 1000.0:+.1f} ms")
    if delta > 0:
        print("Verdict: speculative path yielded a net speedup on this host.")
    elif delta < 0:
        print(
            "Verdict: no net speedup — draft acceptance may be low, or the "
            "active Ollama model has no DRAFT/MTP pairing (GGUF path often "
            "ignores draft_num_predict). See dana.llm_client.speculative_launch_notes()."
        )
    else:
        print("Verdict: TPS unchanged within measurement noise.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Raw Ollama TTFT/TPS benchmark")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run baseline then speculative and print TPS delta",
    )
    parser.add_argument(
        "--speculative",
        action="store_true",
        help="Single run with DONNA_SPECULATIVE_DECODING=1",
    )
    parser.add_argument("--num-predict", type=int, default=256)
    parser.add_argument(
        "--draft-num-predict",
        type=int,
        default=int(os.environ.get("DONNA_DRAFT_NUM_PREDICT") or "4"),
    )
    parser.add_argument("--model", default=_main_model())
    parser.add_argument(
        "--prompt",
        default=BENCHMARK_PROMPT,
        help="Complex raw prompt (ReAct bypassed)",
    )
    args = parser.parse_args(argv)

    from dana.llm_client import draft_model_name, speculative_launch_notes

    print("Inference speed benchmark (ReAct bypassed)")
    print(f"endpoint: {_ollama_base()}/api/chat")
    print(f"model:    {args.model}")
    print(f"draft:    {draft_model_name()} (packaging note)")
    print(f"prompt:   {args.prompt[:72]}...")
    print()
    print(speculative_launch_notes())

    if args.compare:
        base = stream_raw_chat(
            model=args.model,
            prompt=args.prompt,
            num_predict=args.num_predict,
            speculative=False,
            draft_num_predict=args.draft_num_predict,
        )
        _print_result(base)
        spec = stream_raw_chat(
            model=args.model,
            prompt=args.prompt,
            num_predict=args.num_predict,
            speculative=True,
            draft_num_predict=args.draft_num_predict,
        )
        _print_result(spec)
        _print_delta(base, spec)
        return 0

    speculative = bool(args.speculative) or (
        (os.environ.get("DONNA_SPECULATIVE_DECODING") or "").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    result = stream_raw_chat(
        model=args.model,
        prompt=args.prompt,
        num_predict=args.num_predict,
        speculative=speculative,
        draft_num_predict=args.draft_num_predict,
    )
    _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
