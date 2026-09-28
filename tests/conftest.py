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
import dana.audio.multi_voice_tts as _multi_voice_tts_module  # noqa: E402
import dana.core.react_dispatch as _react_dispatch_module  # noqa: E402
import dana.plugins.os.file_system as _file_system_module  # noqa: E402
import dana.plugins.planning.task_board as _task_board_module  # noqa: E402


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "asyncio: async test body (executed via asyncio.wait_for)",
    )
    config.addinivalue_line(
        "markers",
        "requires_audio_output: needs a real audio output device; skipped when sounddevice finds none (e.g. CI)",
    )


def _has_audio_output_device() -> bool:
    try:
        import sounddevice as sd

        sd.query_devices(kind="output")
    except Exception:  # noqa: BLE001 — no PortAudio, no devices, or no default output
        return False
    return True


def pytest_runtest_setup(item: pytest.Item) -> None:
    # Without an output device, sd.OutputStream fails immediately and playback
    # returns before any simulated barge-in can fire, so these tests can't
    # measure what they're meant to.
    if item.get_closest_marker("requires_audio_output") and not _has_audio_output_device():
        pytest.skip("no audio output device available")


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
def _mock_tts_hardware_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """Global safety net: dana.audio.multi_voice_tts._synthesize_pyttsx3 is a
    real, synchronous call into Windows SAPI via the pyttsx3/COM bridge —
    hardware/OS-integration code with no business running in a test suite
    regardless of whether it happens to be fast or slow on any given
    machine. EVERY test that drives a real /ws/chat turn ending in a
    plain-text assistant reply hits this path (the "dana" receptionist
    voice falls through to it whenever Piper isn't available in this
    environment) — not just tests/audio's own dedicated TTS tests — so
    this is a global, autouse fixture, not a per-file one.

    Correction on an earlier live-debugging session's conclusion: this
    call was originally suspected to be THE cause of a hang in
    tests/api/test_chat_attachments.py. Isolated properly afterward (by
    replacing dana.api.server._speak_reply — the caller of this whole TTS
    pipeline — with a no-op and confirming the hang PERSISTED): the actual
    cause was unrelated (see _disable_context_distillation below). This
    mock is kept anyway on its own merits — a hardware/COM call is exactly
    the kind of thing a test suite should never depend on being fast, or
    even present, on every machine that runs it — but it is a defensive
    good practice here, not the fix for that specific incident.

    Writes the SAME silence-placeholder WAV synthesize_speech's own
    fallback path already writes on a genuine pyttsx3 failure
    (_write_silence_wav) — a real, valid (if silent) WAV file, never empty/
    malformed bytes — so any downstream code that reads the returned Path
    back still sees a well-formed file. Returns True (matching
    _synthesize_pyttsx3's own real return type), so synthesize_speech's
    success branch runs, not its own "wrote silence placeholder" failure-
    fallback branch — the TTS *logic* actually fires and is exercised, only
    the hardware call itself is short-circuited.
    """

    def _fake_synthesize_pyttsx3(text: str, dest: Path, *, prefer_male: bool, rate: int = 165) -> bool:
        _multi_voice_tts_module._write_silence_wav(dest, duration_s=0.05)
        return True

    monkeypatch.setattr(_multi_voice_tts_module, "_synthesize_pyttsx3", _fake_synthesize_pyttsx3)

    # Piper is tried FIRST for the "dana" receptionist voice (synthesize_
    # speech's own branch order) — confirmed live this environment has a
    # real ONNX model on disk (tts_models/en_US-hfc_female-medium.onnx), so
    # every "dana" voice turn was running REAL neural TTS inference before
    # ever reaching the pyttsx3 mock above, real CPU cost multiplying across
    # every plain-text assistant turn in a test run. Short-circuited the
    # same way: pretend unavailable so synthesize_speech falls through to
    # the (now mocked) pyttsx3 path immediately, same as a real environment
    # with no Piper model installed.
    monkeypatch.setattr(_multi_voice_tts_module, "_synthesize_piper", lambda text, dest: False)


@pytest.fixture(autouse=True)
def _disable_context_distillation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Global safety net — THE actual fix for the tests/api/test_chat_
    attachments.py hang the TTS mock above was originally (incorrectly)
    suspected to be.

    dana.api.server._finish_turn calls dana.core.context_distiller.
    schedule_distillation after EVERY completed turn, which does
    ``asyncio.create_task(distill_turn(...))`` — genuinely fire-and-forget,
    never awaited by its caller. distill_turn itself is well-behaved in
    isolation (asyncio.to_thread + asyncio.wait_for(timeout=20s)), but
    that 20s bound is on the ASYNCIO SIDE only: the underlying blocking
    HTTP call to a local Ollama endpoint (ModelProvider(...).complete, via
    to_thread) keeps occupying its OS thread for however long THAT call
    actually takes to fail against a local Ollama daemon that doesn't
    exist in a test/CI environment — asyncio.wait_for gives up waiting,
    but cannot forcibly kill a thread already blocked in a synchronous
    call. Confirmed live: isolated the hang by replacing dana.api.server.
    _speak_reply (the TTS entry point) with a no-op — the hang PERSISTED,
    ruling TTS out; setting DANA_CONTEXT_DISTILL=0 for the same run made
    the whole file pass. Each completed turn silently leaked one
    permanently-blocked worker out of asyncio.to_thread's shared, FIXED-
    SIZE default executor; the 2nd or 3rd turn in a test file was enough
    to exhaust it, so a LATER, entirely unrelated to_thread call (FreeCAD
    subprocess execution, this file's own TTS mock, anything) queued
    forever waiting for a worker that would never free up.

    monkeypatch.setenv (not a manual os.environ write) — reverted
    automatically after each test, and distillation_enabled() re-reads
    the env var fresh on every call, so no import-order dependency here.
    """
    monkeypatch.setenv("DANA_CONTEXT_DISTILL", "0")


@pytest.fixture(autouse=True)
def _force_auto_approve_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Global safety net: a developer's local ``.env`` setting
    ``DANA_AUTO_APPROVE=true`` (dana.api.server._default_auto_approve) would
    otherwise seed every test's /ws/chat session as auto-approved, so any
    test expecting a ``hitl_approval_required`` prompt fails locally while
    passing in CI (no ``.env`` there).

    Patches the seed function itself, not just the env var:
    dana.core.model_provider.ensure_dotenv_loaded re-reads ``.env`` with
    ``override=True`` on many calls mid-test, which would silently put a
    plain ``monkeypatch.setenv`` back to ``true``. Only patched when a test
    module already imported dana.api.server, so tests that never touch the
    server don't pay for importing it.
    """
    monkeypatch.setenv("DANA_AUTO_APPROVE", "0")
    server = sys.modules.get("dana.api.server")
    if server is not None:
        monkeypatch.setattr(server, "_default_auto_approve", lambda: False)


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
    above: Task Planner / Executive Function's ``_PLANS_BY_SESSION``
    (dana.plugins.planning.task_board) is process-wide, mutable, module-
    level state — a plan created by ANY test (e.g. one driving a real
    /ws/chat turn that calls ``create_plan``) would otherwise leak into
    every later test in the WHOLE suite via this shared dict, regardless
    of which session_id that test happened to use (task_board is now
    session-scoped, not a single global plan, but this fixture's own job —
    leave every test a clean slate — doesn't change: it just needs to
    clear every session's entry, not one fixed set of fields).
    Teardown-only: the dict is already empty at process start and after
    any earlier test's own cleanup here, so there's nothing to reset going
    in.
    """
    yield
    _task_board_module._PLANS_BY_SESSION.clear()


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


@pytest.fixture(autouse=True)
def _reset_measurement_gate_state():
    """Global safety net, same rationale as ``_reset_plan_gate_state``
    above: the Position-Before-Measurement Gate's
    ``_BOUNDING_BOX_MEASURED_BY_SESSION`` (dana.core.react_dispatch) is
    process-wide, mutable, module-level state, session-scoped but keyed by
    whatever session_id happens to be ambient at the time — a measurement
    recorded by ANY test would otherwise leak into every later test in the
    WHOLE suite that reuses the same session_id. Teardown-only: the dict is
    empty at process start and after any earlier test's own cleanup here.
    """
    yield
    _react_dispatch_module._BOUNDING_BOX_MEASURED_BY_SESSION.clear()


@pytest.fixture(autouse=True)
def _reset_truncation_nudge_state():
    """Global safety net, same rationale as ``_reset_measurement_gate_state``
    above: the Truncation Recovery Nudge's ``_OUTPUT_TRUNCATED_BY_SESSION``
    (dana.core.react_dispatch) is process-wide, mutable, module-level state,
    session-scoped but keyed by whatever session_id happens to be ambient at
    the time. Teardown-only: the dict is empty at process start and after
    any earlier test's own cleanup here.
    """
    yield
    _react_dispatch_module._OUTPUT_TRUNCATED_BY_SESSION.clear()


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
