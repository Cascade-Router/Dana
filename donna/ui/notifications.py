"""Windows toast + planner handoff for the Shell Watchdog.

Imports are resilient: ``win11toast`` / planner deps are optional so Linux/CI
can import this module without GUI or LangGraph installed.
"""

from __future__ import annotations

import threading
from typing import Any, Callable

WATCHDOG_TOAST_TITLE = "Dānā Shell Watchdog"

# Optional sinks for unit tests (override without real toasts / planner).
NotifyFn = Callable[[str, str], bool]
PlannerFn = Callable[[str, str], Any]
ErrorHook = Callable[[str, str], None]


def show_watchdog_toast(title: str, message: str, *, app_id: str = "Donna") -> bool:
    """Best-effort native toast; never raises. No-op off Windows / when disabled."""
    try:
        from donna.middleware.toast_notify import show_silent_toast

        return bool(
            show_silent_toast(
                title or WATCHDOG_TOAST_TITLE,
                message or "Shell error detected",
                app_id=app_id,
            )
        )
    except Exception:  # noqa: BLE001
        return False


def show_watchdog_toast_async(title: str, message: str, *, app_id: str = "Donna") -> None:
    """Fire-and-forget toast so feed paths never block on WinRT."""

    def _run() -> None:
        try:
            show_watchdog_toast(title, message, app_id=app_id)
        except Exception:  # noqa: BLE001
            pass

    threading.Thread(target=_run, name="donna-watchdog-toast", daemon=True).start()


def submit_to_planner(trace: str, summary: str = "") -> dict[str, Any] | None:
    """Queue a best-effort plan request from an error trace.

    Calls ``build_structured_plan`` with the trace as intent. Failures (missing
    deps, import errors) return None and never break tray startup.
    """
    intent_bits = [
        "Shell Watchdog detected a terminal/subprocess error. Propose a fix candidate.",
    ]
    if summary:
        intent_bits.append(f"Summary: {summary}")
    if trace:
        intent_bits.append("Trace:\n" + trace[:4000])
    intent = "\n\n".join(intent_bits)
    try:
        from donna.agentic_planning import build_structured_plan

        plan = build_structured_plan(intent)
        return plan if isinstance(plan, dict) else None
    except Exception:  # noqa: BLE001
        return None


def notify_shell_error(
    trace: str,
    summary: str,
    *,
    notify: NotifyFn | None = None,
    submit: PlannerFn | None = None,
    on_error: ErrorHook | None = None,
) -> dict[str, Any]:
    """Notification + planner integration layer (fully injectable for tests)."""
    title = WATCHDOG_TOAST_TITLE
    message = (summary or "Shell error detected").strip()
    payload: dict[str, Any] = {
        "title": title,
        "message": message,
        "trace": trace or "",
        "summary": summary or "",
        "notified": False,
        "plan": None,
    }

    if on_error is not None:
        try:
            on_error(trace, summary)
        except Exception:  # noqa: BLE001
            pass

    notify_fn = notify or show_watchdog_toast
    try:
        payload["notified"] = bool(notify_fn(title, message))
    except Exception:  # noqa: BLE001
        payload["notified"] = False

    planner = submit or submit_to_planner
    try:
        payload["plan"] = planner(trace, summary)
    except Exception:  # noqa: BLE001
        payload["plan"] = None

    return payload


def make_watchdog_error_handler(
    *,
    notify: NotifyFn | None = None,
    submit: PlannerFn | None = None,
    on_error: ErrorHook | None = None,
) -> ErrorHook:
    """Return an ``on_error(trace, summary)`` sink wired to toast + planner."""

    def _handler(trace: str, summary: str) -> None:
        notify_shell_error(
            trace,
            summary,
            notify=notify,
            submit=submit,
            on_error=on_error,
        )

    return _handler
