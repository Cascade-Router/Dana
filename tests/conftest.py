"""Pytest bootstrap: keep CAMGRASPER repo root on ``sys.path``.

Tests live under ``tests/``; packages ``dana`` / ``dana_security`` stay at repo root.
"""

from __future__ import annotations

import sys
import tkinter
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_root_s = str(_ROOT)
if _root_s not in sys.path:
    sys.path.insert(0, _root_s)
for _sub in ("scripts", "scripts/diagnostics"):
    _p = str(_ROOT / _sub)
    if Path(_p).is_dir() and _p not in sys.path:
        sys.path.insert(0, _p)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "asyncio: async test body (executed via asyncio.wait_for)",
    )


@pytest.fixture(autouse=True)
def _stub_premium_logo_unless_stage899(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """Avoid cross-test Tk PhotoImage / CTkImage ``pyimage`` collisions.

    Stage 8.9.9 explicitly asserts CTkImage wiring and keeps real loaders.
    """
    nodeid = getattr(request.node, "nodeid", "") or ""
    if "stage899" in nodeid or "premium_logo" in nodeid:
        return

    monkeypatch.setattr("dana.ui.logo.load_premium_logo", lambda *_a, **_k: None)
    monkeypatch.setattr("dana.ui.logo.apply_window_icon", lambda *_a, **_k: False)
    monkeypatch.setattr("dana.ui.logo.force_apply_window_icon", lambda *_a, **_k: False)
    monkeypatch.setattr("dana.ui.logo.schedule_window_icon", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "dana.ui.logo.load_premium_logo_photoimage",
        lambda *_a, **_k: None,
    )


def _cancel_pending_after_events(root: tkinter.Misc) -> None:
    """Cancel every ``after()`` callback still queued on ``root``.

    ``Tk.destroy``/``CTk.destroy`` tear down the widget tree but never call
    ``after_cancel`` on their own pending timers first. Each scheduled
    callback is a dynamically-named Tcl command (e.g.
    ``"<id>_windows_set_titlebar_icon"``, customtkinter's Windows title-bar
    icon setter); once its window is gone that command still fires at its
    scheduled time against a dead interpreter, printing
    ``invalid command name "..." ("after" script)`` to stderr (Tcl's
    after-error path bypasses Python exceptions entirely, so this is
    silent to the test itself). This happens on *every* create/destroy
    cycle, not just ones a test fails before reaching its own ``destroy()``
    — reproduced with 150 back-to-back bare ``customtkinter.CTk()`` cycles.
    Across a ~450-file suite the accumulated dangling commands eventually
    corrupt the shared Tcl interpreter state, surfacing as unrelated
    ``_tkinter.TclError`` ("can't find a usable init.tcl", or "no such file
    or directory" for a ``.tcl`` file that verifiably exists on disk) on
    whichever GUI test happens to run next.
    """
    try:
        pending = root.tk.call("after", "info")
    except Exception:  # noqa: BLE001
        return
    for after_id in pending:
        try:
            root.after_cancel(after_id)
        except Exception:  # noqa: BLE001
            pass


_original_tk_destroy = tkinter.Tk.destroy


def _patched_tk_destroy(self: tkinter.Tk) -> None:
    """``Tk.destroy`` wrapper: sweep pending ``after()`` events afterward.

    Order matters: canceling *before* the widget-tree teardown races each
    child widget's own Tcl-command cleanup (e.g. ``CTkTextbox.destroy``
    deletes its own tracked commands; if ``after_cancel`` already deleted
    one first via its shared Tcl command table, that widget's own delete
    call then raises ``can't delete Tcl command`` — reproduced empirically).
    Running the original destroy first lets every widget finish its own
    bookkeeping normally; ``self.tk`` stays usable afterward (confirmed:
    Tk.destroy tears down the widget tree, not the interpreter object), so
    only genuinely orphaned root-level timers (customtkinter's Windows
    title-bar icon setter, scheduled once in ``CTk.__init__`` and never
    canceled by anything) are left to sweep up here.
    """
    _original_tk_destroy(self)
    _cancel_pending_after_events(self)


tkinter.Tk.destroy = _patched_tk_destroy


@pytest.fixture(autouse=True)
def _teardown_lingering_tk_root():
    """Destroy any leftover Tk/CTk root after every test (GUI or not).

    GUI tests that ``assert`` before reaching their own trailing
    ``app.destroy()`` leave that Tcl interpreter alive for the rest of the
    process. customtkinter widgets only unregister from its global
    ``AppearanceModeTracker`` / ``ScalingTracker`` callback lists inside
    ``destroy()`` (see ``CTkAppearanceModeBaseClass.destroy`` /
    ``CTkScalingBaseClass.destroy`` — both cascade from ``CTk.destroy``), so
    a leaked root also leaks those registrations, each holding a strong
    reference back to the dead widget/interpreter.

    Calling ``destroy()`` here is the sanctioned customtkinter cleanup path
    (not reaching into its private tracker dicts directly): it cascades
    through the whole widget tree and unregisters everything in one call —
    and, via the patch above, also cancels that root's pending after()
    events first.
    """
    yield
    root = getattr(tkinter, "_default_root", None)
    if root is None:
        return
    try:
        root.destroy()
    except tkinter.TclError:
        pass
    except Exception:  # noqa: BLE001 — teardown must never fail a passing test
        pass
    tkinter._default_root = None
