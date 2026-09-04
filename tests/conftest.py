"""Pytest bootstrap: keep CAMGRASPER repo root on ``sys.path``.

Tests live under ``tests/``; the ``dana`` package stays at repo root.
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

import dana.api.sessions as _sessions_module  # noqa: E402 — needs the sys.path bootstrap above first
import dana.core.react_dispatch as _react_dispatch_module  # noqa: E402
import dana.plugins.os.file_system as _file_system_module  # noqa: E402
import dana.plugins.planning.task_board as _task_board_module  # noqa: E402


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "asyncio: async test body (executed via asyncio.wait_for)",
    )


@pytest.fixture(autouse=True)
def _isolate_chat_sessions_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Global safety net: redirects Local Chat Session Persistence's
    on-disk store (dana.api.sessions.SESSIONS_DIR) to a throwaway per-test
    directory for EVERY test in this suite, not just the ones that
    explicitly test it. dana.api.server's _finish_turn auto-saves after
    every completed ReAct turn (dana.api.server._persist_turn), so any
    test anywhere that drives a real /ws/chat turn would otherwise write a
    real session file into the actual AGENT_WORKSPACE_DIR/data/sessions/
    on disk. A test file that specifically exercises this feature (see
    tests/api/test_sessions_api.py) may still redirect it again to its own
    tmp_path via its own fixture — same effective value, harmless.
    """
    monkeypatch.setattr(_sessions_module, "SESSIONS_DIR", tmp_path / "sessions")


@pytest.fixture(autouse=True)
def _isolate_os_tools_sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Global safety net, same rationale as ``_isolate_chat_sessions_dir``
    above: redirects the "os_tools" capability domain's real sandbox root
    (dana.plugins.os.file_system._SANDBOX_ROOT) to a throwaway per-test
    directory for EVERY test, not just tests/plugins/os/test_file_system.py
    (which already does this itself, redundantly-but-harmlessly, via its
    own fixture). Any test anywhere that drives a real /ws/chat turn
    through a HITL-approved write_file call would otherwise write a real
    file into the actual AGENT_WORKSPACE_DIR on disk.
    """
    root = tmp_path / "agent_workspace"
    root.mkdir(exist_ok=True)
    monkeypatch.setattr(_file_system_module, "_SANDBOX_ROOT", root)


@pytest.fixture(autouse=True)
def _reset_user_skills_registry():
    """Global safety net: Autonomous Skill Acquisition's registry
    (dana.core.react_dispatch's TOOL_HANDLERS / _USER_SKILL_TOOL_IDS /
    _CAPABILITY_TOOL_IDS["user_skills"]) is process-wide, mutable, global
    state — a skill saved/loaded by ANY test (test_skill_loader.py,
    test_skills_api.py, or any future one) would otherwise leak into every
    later test in the WHOLE suite via these shared module-level dicts.
    Teardown-only: the registry is empty at process start and after any
    earlier test's own cleanup here, so there's nothing to reset going in.
    """
    yield
    rd = _react_dispatch_module
    for tool_id in list(rd._USER_SKILL_TOOL_IDS):
        rd.TOOL_HANDLERS.pop(tool_id, None)
        rd._USER_SKILL_SCHEMAS.pop(tool_id, None)
    rd._USER_SKILL_TOOL_IDS.clear()
    rd._CAPABILITY_TOOL_IDS["user_skills"] = frozenset()
    rd._LLM_TOOL_IDS = rd._CORE_TOOL_IDS.union(*rd._CAPABILITY_TOOL_IDS.values())
    rd._tool_ids_for_plugins.cache_clear()
    rd._llm_tools_schema_cached.cache_clear()


@pytest.fixture(autouse=True)
def _reset_task_board_plan():
    """Global safety net, same rationale as ``_reset_user_skills_registry``
    above: Task Planner / Executive Function's ``_ACTIVE_PLAN``
    (dana.plugins.planning.task_board) is process-wide, mutable, global
    state — a plan created by ANY test (e.g. one driving a real /ws/chat
    turn that calls ``create_plan``) would otherwise leak into every later
    test in the WHOLE suite via this shared module-level dict.
    Teardown-only: the plan is already empty at process start and after
    any earlier test's own cleanup here, so there's nothing to reset going
    in.
    """
    yield
    plan = _task_board_module._ACTIVE_PLAN
    plan["objective"] = ""
    plan["tasks"] = []
    plan["current_task_id"] = None


@pytest.fixture(autouse=True)
def _reset_plan_gate_state():
    """Global safety net, same rationale as ``_reset_task_board_plan``
    above — and for a near-identical reason: the Plan-and-Execute
    Gatekeeper's ``_PLAN_STATE_REGISTRY`` (dana.core.react_dispatch) is
    process-wide, mutable, module-level state, session-scoped but keyed by
    whatever session_id happens to be ambient at the time. A plan opened by
    ANY test (a fixture that pre-opens the gate for its own module's
    dispatch tests, or a real /ws/chat turn that calls ``create_plan``)
    would otherwise leak into every later test in the WHOLE suite that
    reuses the same (often just the ambient default) session_id —
    including one that specifically asserts ``build_system_prompt()``
    renders NO active-plan anchor, or one that expects a geometry tool to
    still be gated. Teardown-only: the registry is empty at process start
    and after any earlier test's own cleanup here, so there's nothing to
    reset going in.
    """
    yield
    _react_dispatch_module._PLAN_STATE_REGISTRY.clear()


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
