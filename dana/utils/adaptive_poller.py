from __future__ import annotations

import threading
from typing import Callable


class AdaptivePoller:
    """Adaptive polling with exponential backoff for telemetry refresh loops.

    ``callback`` may optionally return a bool: ``True`` (there was activity)
    resets the interval to ``t_min`` so the poller stays responsive while
    busy; ``False``/``None`` (the original contract) lets it keep backing
    off toward ``t_max`` while idle.

    **Tk callers: do not call ``start()``.** ``start()`` runs ``callback`` on
    a real background thread, and Tkinter (CPython 3.12+) raises
    ``RuntimeError: main thread is not in main loop`` — or, observed in
    practice, simply stalls the polling thread indefinitely — the moment
    that thread touches *any* Tk API, including ``widget.after(0, ...)``
    itself (registering the callback is itself a Tcl/Tk call, not merely
    "running" one). There is no safe way to hand off to the Tk main thread
    from inside this poller's own thread. Tk callers should instead call
    ``note_activity()`` synchronously from their own main-thread
    ``self.after()`` chain to get the next adaptive delay — see
    ``DanaGUI._master_telemetry_tick`` for the pattern. ``start()``/``stop()``
    remain here for non-Tk callback use (e.g. pure I/O polling).
    """

    def __init__(
        self,
        callback: Callable[[], bool | None],
        *,
        t_min: float = 0.05,
        t_max: float = 0.5,
        gamma: float = 1.5,
    ) -> None:
        self.callback = callback
        self.t_min = t_min
        self.t_max = t_max
        self.gamma = gamma
        self._interval = t_min
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="AdaptivePoller", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def note_activity(self, had_activity: bool | None) -> float:
        """Update the interval from one poll's result and return it.

        Safe to call from any single thread that owns its own scheduling
        (e.g. a Tk main-thread ``self.after()`` chain) without ever starting
        this poller's background thread.
        """
        if had_activity:
            self._interval = self.t_min
        else:
            self._interval = min(self.t_max, self._interval * self.gamma)
        return self._interval

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                had_activity = self.callback()
            except Exception:  # noqa: BLE001
                # A misbehaving callback must never silently kill the poller
                # thread — fall back to backoff-as-idle for this tick.
                had_activity = None
            # Event.wait (not time.sleep) so stop() interrupts immediately
            # instead of blocking the caller for up to t_max seconds.
            if self._stop_event.wait(self._interval):
                break
            self.note_activity(had_activity)
