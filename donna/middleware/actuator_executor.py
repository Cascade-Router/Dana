"""Stage 4.2 — Multi-threaded actuator middleware (action_queue consumer).

Polls Blackboard ``action_queue`` for ``pending`` rows, runs tools on a
``ThreadPoolExecutor``, and writes ``completed`` / ``failed`` results back.

Run:
    python -m donna.middleware.actuator_executor
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import psutil

from donna.middleware.resource_cap import apply_cpu_half_affinity

# Stage 7.1 — pin this daemon to the first 50% of logical cores.
try:
    apply_cpu_half_affinity()
except Exception:  # noqa: BLE001
    pass

# Stage 4.4 QoS — yield CPU to foreground Chat/Audio and user apps.
try:
    psutil.Process(os.getpid()).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception:  # noqa: BLE001
    pass

from donna.memory.blackboard import (
    claim_next_pending,
    resolve_action,
)
from donna.telemetry import log_actuator_done, log_actuator_start

DEFAULT_POLL_INTERVAL_S = 0.5
DEFAULT_WORKERS = 2

# Stage 6.5 — physical operators that pull the Andon Cord on failure.
_OPERATOR_ANDON_TOOLS = frozenset({"navigate_and_click", "type_stealth_text"})


def execute_tool_payload(
    tool_name: str,
    arguments: dict[str, Any] | None,
    *,
    db_path: Any = None,
) -> str:
    """Dispatch a queued tool by name (standalone, no LangGraph)."""
    name = (tool_name or "").strip()
    args = dict(arguments or {})
    if not name:
        return "ERROR: empty tool_name"

    if name == "draft_cursor_prompt":
        from donna.tools.general.draft_cursor_prompt import (
            draft_cursor_prompt as _draft,
        )

        return str(
            _draft(
                objective=str(args.get("objective") or ""),
                context=str(args.get("context") or ""),
            )
        )

    if name == "analyze_visual_context":
        from donna.vision_tools import analyze_visual_context as _analyze

        src = str(args.get("source") or "screen").strip().lower() or "screen"
        if src == "camera":
            src = "webcam"
        return str(_analyze(source=src))

    if name == "ocr_with_region":
        from donna.tools.visual_tools import ocr_with_region as _ocr

        return str(_ocr(query=str(args.get("query") or "").strip()))

    if name == "type_stealth_text":
        from donna.operators.ghost_typist import type_stealth_text as _type_stealth

        wait_raw = args.get("wait_hotkey", True)
        if isinstance(wait_raw, str):
            wait_hotkey = wait_raw.strip().lower() not in {"0", "false", "no", "off"}
        else:
            wait_hotkey = bool(wait_raw)
        return str(
            _type_stealth(
                str(args.get("text") or ""),
                wait_hotkey=wait_hotkey,
                hotkey=str(args.get("hotkey") or "f9"),
            )
        )

    if name == "navigate_and_click":
        from donna.operators.nav_and_click import navigate_and_click as _nav_click

        return str(
            _nav_click(
                str(args.get("query") or args.get("target") or "Target"),
                visual_context=(
                    str(args["visual_context"])
                    if args.get("visual_context") is not None
                    else None
                ),
            )
        )

    if name == "press_key":
        from donna.operators.keystroke import press_key as _press_key

        return str(
            _press_key(
                str(args.get("key_name") or args.get("key") or args.get("vk") or "")
            )
        )

    if name == "wake_cto":
        from donna.management.jason_supervisor import handle_wake_cto

        return str(handle_wake_cto(args, db_path=db_path))

    if name == "click_close_button":
        # Logical recovery step — dismiss blocking UI (Esc when live).
        if os.environ.get("DONNA_OS_DRY_RUN", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return "OK: click_close_button dry_run dismissed"
        try:
            from donna.tools import os_control as _osc

            # Prefer allowlisted hotkey path if present.
            if hasattr(_osc, "execute_os_hotkey"):
                _osc.execute_os_hotkey("esc")  # type: ignore[attr-defined]
            elif hasattr(_osc, "_tap_vk"):
                _osc._tap_vk(0x1B)  # type: ignore[attr-defined]
            return "OK: click_close_button sent Escape"
        except Exception as exc:  # noqa: BLE001
            return f"OK: click_close_button soft-fail ({exc})"

    # Prefer the core IR dispatcher for the rest of the registry.
    try:
        from donna.core_agent import execute_tool_call
        from donna.tools.schema import ToolCall

        return str(
            execute_tool_call(
                ToolCall(tool_id=name, arguments=args, raw_text="")
            )
        )
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: tool {name} failed: {exc}"


def _maybe_pull_andon(
    *,
    tool_name: str,
    status: str,
    result: str,
    action_id: int,
    arguments: dict[str, Any],
    session_id: str,
    db_path: Any = None,
) -> dict[str, Any] | None:
    """Stage 6.5 — wake Jason when a physical operator fails."""
    if status != "failed":
        return None
    if (tool_name or "").strip() not in _OPERATOR_ANDON_TOOLS:
        return None
    # Retries already at max depth: record failure but do not re-wake.
    depth = int(arguments.get("_andon_depth") or 0)
    if depth >= 2:
        return None
    error_context = (result or "").strip() or "operator failed"
    if error_context.upper().startswith("ERROR:"):
        error_context = error_context[6:].strip() or error_context
    try:
        from donna.management.jason_supervisor import trigger_andon_cord

        return trigger_andon_cord(
            task_name=tool_name,
            error_context=error_context,
            failed_action_id=int(action_id),
            failed_arguments=arguments,
            session_id=session_id,
            db_path=db_path,
            run_recovery=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[actuator] ANDON trigger failed: {exc}", flush=True)
        return None


def process_action(action: dict[str, Any], *, db_path: Any = None) -> dict[str, Any]:
    """Run one claimed action and resolve the Blackboard row."""
    # Stage 7.4 — arm human-yield hooks for --once / embedded actuator paths.
    try:
        from donna.middleware.human_yield import start_human_yield_listener

        start_human_yield_listener()
    except Exception:  # noqa: BLE001
        pass
    aid = int(action.get("action_id") or 0)
    tool_name = str(action.get("tool_name") or "")
    session_id = str(action.get("session_id") or "")
    arguments = action.get("arguments") if isinstance(action.get("arguments"), dict) else {}
    t0 = time.perf_counter()
    try:
        log_actuator_start(
            tool_name,
            action_id=aid,
            session_id=session_id,
        )
    except Exception:  # noqa: BLE001
        pass

    status = "completed"
    error_context = ""
    try:
        from donna.middleware.kill_switch import is_halted

        if is_halted():
            status = "cancelled"
            result = "HALTED: cancelled by GLOBAL_HALT_EVENT"
            error_context = "halted by GLOBAL_HALT_EVENT"
        else:
            result = execute_tool_payload(tool_name, arguments, db_path=db_path)
            result_s = str(result)
            if result_s.startswith("HALTED:"):
                status = "cancelled"
                error_context = result_s
            elif result_s.startswith("ERROR:"):
                status = "failed"
                error_context = result_s
    except Exception as exc:  # noqa: BLE001
        status = "failed"
        result = f"ERROR: tool {tool_name} failed: {exc}"
        error_context = str(exc)

    latency_ms = (time.perf_counter() - t0) * 1000.0
    resolve_action(
        aid,
        status=status,
        result=str(result),
        error_context=error_context,
        db_path=db_path,
    )
    andon: dict[str, Any] | None = None
    if status == "failed" and (tool_name or "").strip() in _OPERATOR_ANDON_TOOLS:
        andon = _maybe_pull_andon(
            tool_name=tool_name,
            status=status,
            result=str(result),
            action_id=aid,
            arguments=dict(arguments or {}),
            session_id=session_id,
            db_path=db_path,
        )
    try:
        log_actuator_done(
            tool_name,
            action_id=aid,
            session_id=session_id,
            ok=status == "completed",
            latency_ms=latency_ms,
            payload={
                "status": status,
                "result_chars": len(str(result)),
                "andon": bool(andon),
            },
        )
    except Exception:  # noqa: BLE001
        pass
    # Stage 4.3/4.4 — silent toast after DB resolve; never block the worker.
    try:
        from donna.middleware.toast_notify import (
            format_actuator_toast,
            show_silent_toast_async,
        )
        from donna.telemetry import log_notification_toast

        title, body = format_actuator_toast(tool_name, status)
        log_notification_toast(
            body,
            action_id=aid,
            session_id=session_id,
            tool_name=tool_name,
            payload={"status": status, "shown": None},
        )
        show_silent_toast_async(title, body)
    except Exception:  # noqa: BLE001
        pass
    out = {
        "action_id": aid,
        "tool_name": tool_name,
        "status": status,
        "result": str(result),
        "error_context": error_context,
        "latency_ms": latency_ms,
    }
    if andon is not None:
        out["andon"] = andon
    return out


def poll_once(
    executor: ThreadPoolExecutor,
    inflight: set[Future[Any]],
    *,
    db_path: Any = None,
    max_claim: int = 1,
) -> int:
    """Claim up to ``max_claim`` pending actions and submit them to the pool."""
    claimed = 0
    for _ in range(max(1, int(max_claim))):
        action = claim_next_pending(db_path=db_path)
        if action is None:
            break
        fut = executor.submit(process_action, action, db_path=db_path)
        inflight.add(fut)
        claimed += 1
    # Reap finished futures so the set does not grow forever.
    done = {f for f in inflight if f.done()}
    for f in done:
        try:
            f.result()
        except Exception as exc:  # noqa: BLE001
            print(f"[actuator] worker error: {exc}", flush=True)
        inflight.discard(f)
    return claimed


def run_forever(
    *,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    workers: int = DEFAULT_WORKERS,
    db_path: Any = None,
) -> None:
    """Robust polling loop with a bounded thread pool."""
    # Stage 7.2 — ensure panic hotkey is armed even when actuator is standalone.
    try:
        from donna.middleware.kill_switch import start_kill_switch_listener

        start_kill_switch_listener()
    except Exception:  # noqa: BLE001
        pass
    # Stage 7.4 — physical human-yield LL hooks.
    try:
        from donna.middleware.human_yield import start_human_yield_listener

        start_human_yield_listener()
    except Exception:  # noqa: BLE001
        pass
    interval = max(0.05, float(poll_interval_s))
    n_workers = max(1, int(workers))
    print(
        f"[actuator] starting workers={n_workers} poll={interval}s",
        flush=True,
    )
    inflight: set[Future[Any]] = set()
    with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="donna-act") as pool:
        while True:
            try:
                n = poll_once(
                    pool,
                    inflight,
                    db_path=db_path,
                    max_claim=n_workers,
                )
                if n:
                    print(f"[actuator] claimed {n} action(s)", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[actuator] ERROR: {exc}\n{traceback.format_exc()}",
                    flush=True,
                )
            time.sleep(interval)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Donna Stage 4.2 actuator daemon (action_queue worker)."
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL_S,
        help=f"Poll interval seconds (default {DEFAULT_POLL_INTERVAL_S})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"ThreadPoolExecutor size (default {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Claim and process at most one pending action, then exit",
    )
    args = parser.parse_args(argv)
    if args.once:
        action = claim_next_pending()
        if action is None:
            print("[actuator] no pending actions", flush=True)
            return 0
        stats = process_action(action)
        print(
            f"[actuator] action_id={stats['action_id']} "
            f"status={stats['status']} latency_ms={stats['latency_ms']:.1f}",
            flush=True,
        )
        print(stats.get("result", ""), flush=True)
        return 0 if stats.get("status") == "completed" else 1
    run_forever(poll_interval_s=float(args.interval), workers=int(args.workers))
    return 0


if __name__ == "__main__":
    sys.exit(main())
