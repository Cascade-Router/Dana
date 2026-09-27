"""Pre-flight: point OLLAMA_URL/OLLAMA_BASE_URL in .env at the running RunPod GPU pod.

RunPod pod IDs change whenever a pod is recreated, and the proxy URL is
easy to mistype (``1ugryg...`` vs ``lugryg...`` cost a debugging round
already), so this reads the live pod list instead of trusting a pasted URL.

Picks the pod that is RUNNING, has at least one GPU and exposes Ollama's
port (11434) over HTTP. If that doesn't narrow it to exactly one pod it
refuses to guess — pass ``--pod-id`` or ``--name`` to choose.

Needs the ``runpod`` SDK, which is deliberately NOT in requirements.txt
(that file is the HF Space image): ``pip install "runpod<1.12"`` — 1.12+
requires tomlkit>=0.15.1, which conflicts with gradio's own pin.

Usage:
    python scripts/update_runpod_url.py            # update .env
    python scripts/update_runpod_url.py --dry-run  # show what would change
    python scripts/update_runpod_url.py --no-verify
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import requests
from dotenv import dotenv_values, set_key

OLLAMA_PORT = 11434
ENV_KEYS = ("OLLAMA_URL", "OLLAMA_BASE_URL")
DEFAULT_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def proxy_url(pod_id: str) -> str:
    return f"https://{pod_id}-{OLLAMA_PORT}.proxy.runpod.net"


def exposes_ollama(pod: dict[str, Any]) -> bool:
    """``ports`` is RunPod's configured-ports string, e.g. "11434/http,22/tcp"."""
    exposed = {p.strip().lower() for p in str(pod.get("ports") or "").split(",")}
    return f"{OLLAMA_PORT}/http" in exposed


def select_pod(pods: list[dict[str, Any]], *, pod_id: str | None = None, name: str | None = None) -> dict[str, Any]:
    """Returns the single target pod, or raises SystemExit explaining why not."""
    if pod_id:
        matches = [p for p in pods if p.get("id") == pod_id]
    elif name:
        matches = [p for p in pods if p.get("name") == name]
    else:
        matches = [
            p for p in pods
            if p.get("desiredStatus") == "RUNNING" and int(p.get("gpuCount") or 0) > 0 and exposes_ollama(p)
        ]
    if len(matches) == 1:
        pod = matches[0]
        if pod.get("desiredStatus") != "RUNNING":
            raise SystemExit(f"Pod {pod.get('id')} ({pod.get('name')}) is {pod.get('desiredStatus')}, not RUNNING.")
        return pod

    listing = "\n".join(
        f"  {p.get('id')}  {p.get('name')!r:24} {p.get('desiredStatus'):10} gpus={p.get('gpuCount')} ports={p.get('ports')}"
        for p in pods
    ) or "  (no pods on this account)"
    reason = "No pod matched" if not matches else f"{len(matches)} pods matched"
    criteria = f"id={pod_id!r}" if pod_id else f"name={name!r}" if name else f"RUNNING + GPU + {OLLAMA_PORT}/http exposed"
    raise SystemExit(f"{reason} ({criteria}). Pods on this account:\n{listing}\nUse --pod-id or --name to choose.")


def verify_ollama(url: str, timeout: float = 20.0) -> tuple[bool, str]:
    """Ollama answers GET / with "Ollama is running"; RunPod's proxy returns an
    empty 404 when it can't reach the service (wrong ID, port not exposed,
    Ollama bound to 127.0.0.1 only)."""
    try:
        resp = requests.get(url + "/", timeout=timeout)
    except requests.RequestException as exc:
        return False, f"request failed: {exc}"
    if resp.status_code == 200 and "ollama is running" in resp.text.lower():
        return True, "Ollama is running"
    return False, f"HTTP {resp.status_code} {resp.text[:120]!r}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument("--pod-id", help="Target this pod ID instead of auto-selecting.")
    parser.add_argument("--name", help="Target the pod with this exact name instead of auto-selecting.")
    parser.add_argument("--dry-run", action="store_true", help="Print the change without writing .env.")
    parser.add_argument("--no-verify", action="store_true", help="Skip the 'Ollama is running' health check.")
    args = parser.parse_args(argv)

    if not args.env_file.is_file():
        raise SystemExit(f"{args.env_file} not found.")
    env = dotenv_values(args.env_file)
    api_key = (env.get("RUNPOD_API_KEY") or os.environ.get("RUNPOD_API_KEY") or "").strip()
    if not api_key:
        raise SystemExit(f"RUNPOD_API_KEY is not set in {args.env_file} or the environment.")

    import runpod  # deferred: only this script needs it, see module docstring

    runpod.api_key = api_key
    pods = runpod.get_pods() or []
    pod = select_pod(pods, pod_id=args.pod_id, name=args.name)
    url = proxy_url(pod["id"])
    gpu = (pod.get("machine") or {}).get("gpuDisplayName") or "?"
    print(f"Pod: {pod['id']} ({pod.get('name')}, {pod.get('gpuCount')}x {gpu})")
    print(f"URL: {url}")

    if not args.no_verify:
        ok, detail = verify_ollama(url)
        print(f"Health check: {detail}")
        if not ok:
            print("Not updating .env — the endpoint isn't serving Ollama. Use --no-verify to write it anyway.", file=sys.stderr)
            return 1

    for key in ENV_KEYS:
        old = env.get(key)
        if old == url:
            print(f"{key}: unchanged")
            continue
        print(f"{key}: {old or '(unset)'} -> {url}")
        if not args.dry_run:
            set_key(args.env_file, key, url, quote_mode="never")
    if args.dry_run:
        print("Dry run: .env not modified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
