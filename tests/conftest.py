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

import dana.audio.multi_voice_tts as _multi_voice_tts_module  # noqa: E402 — needs the sys.path bootstrap above first
import dana.core.react_dispatch as _react_dispatch_module  # noqa: E402
import dana.plugins.os.file_system as _file_system_module  # noqa: E402
import dana.plugins.planning.task_board as _task_board_module  # noqa: E402
from workspace_isolation import (  # noqa: E402 — tests/ is on sys.path (rootdir conftest)
    snapshot_real_workspace,
    workspace_changes,
    workspace_redirects,
)

# Every module that copies a workspace directory into a module-level Path,
# imported up front so _isolate_workspace_dirs redirects it before a test
# runs rather than missing one a test imports lazily halfway through.
for _module_name in (
    "dana.api.artifacts_registry",
    "dana.api.cad",
    "dana.api.sessions",
    "dana.api.workspace",
    "dana.platform.mock",
    "dana.plugins.freecad.engine",
    "dana.plugins.freecad.printability",
    "dana.plugins.freecad.py_export",
    "dana.plugins.manufacturing.bom_exporter",
    "dana.plugins.memory.core_memory",
    "dana.tools.image_to_3d",
    "dana.tools.sim_wrapper_generator",
    "dana.tools.urdf_builder",
):
    __import__(_module_name)


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


_REAL_WORKSPACE_BEFORE: dict[str, tuple[int, int]] = {}


def pytest_sessionstart(session: pytest.Session) -> None:
    _REAL_WORKSPACE_BEFORE.update(snapshot_real_workspace())


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Tripwire behind _isolate_workspace_dirs: the run fails if any test
    added, removed or modified a file in the real freecad_output/, exports/
    or agent_workspace/ (all gitignored, so git status would never show it).
    Catches what the redirect can't, e.g. a plugin hot-reload re-executing a
    module mid-test and restoring its real paths."""
    changes = workspace_changes(_REAL_WORKSPACE_BEFORE, snapshot_real_workspace())
    if not changes:
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    lines = [f"Tests modified the real workspace ({len(changes)} change(s)):", *changes[:40]]
    if reporter is not None:
        reporter.write_sep("!", "real workspace modified", red=True)
        for line in lines:
            reporter.write_line(line)
    else:
        print("\n".join(lines), file=sys.stderr)
    session.exitstatus = pytest.ExitCode.TESTS_FAILED


@pytest.fixture(autouse=True)
def _isolate_workspace_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Global safety net: every loaded ``dana`` module global that points
    into the real ``freecad_output/``, ``exports/`` or ``agent_workspace/``
    (engine._OUTPUT_DIR, py_export._EXPORT_DIR, sessions.SESSIONS_DIR,
    file_system._SANDBOX_ROOT, ...) is redirected to the same layout under
    this test's ``tmp_path/workspace`` — found by sweeping the modules (see
    tests/workspace_isolation.py), so a new module-level path is covered
    without editing this file. A test that sets its own temp directory
    still wins: its value is no longer under a real root, so it's left alone.
    pytest_sessionfinish above fails the run if anything still got through.
    """
    root = tmp_path / "workspace"
    for module, attr, replacement in list(workspace_redirects(root)):
        monkeypatch.setattr(module, attr, replacement)
    # file_system's sandbox root must exist before its first resolve.
    Path(_file_system_module._SANDBOX_ROOT).mkdir(parents=True, exist_ok=True)


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
def _no_dotenv_reload_mid_test(monkeypatch: pytest.MonkeyPatch) -> None:
    """Global safety net: dana.core.model_provider.ensure_dotenv_loaded
    re-reads the developer's real ``.env`` with ``override=True`` on many
    calls (by design, so hand-edits apply without a restart). Mid-test that
    silently undoes a test's own ``monkeypatch.setenv``/``delenv`` — e.g.
    test_cad_mutation_auto_injects_screenshot_and_visual_verification sets
    DANA_HEADLESS=false and a later reload put the local ``true`` back. Local
    runs failed while CI (no ``.env``) passed.

    Values ``.env`` put in os.environ at import time are left alone, so a
    local ``DANA_HEADLESS=true`` still keeps the FreeCAD GUI from launching
    during a test run. ``DANA_CLOUD_PRIMARY`` is cleared because it changes
    which provider and tool schema a turn resolves, which tests assume is
    the default local path; a test that needs it sets it itself.
    image_analysis imports ensure_dotenv_loaded by name, so it's patched too.
    """
    import dana.core.model_provider as model_provider

    monkeypatch.setattr(model_provider, "ensure_dotenv_loaded", lambda: None)
    image_analysis = sys.modules.get("dana.plugins.vision.image_analysis")
    if image_analysis is not None:
        monkeypatch.setattr(image_analysis, "ensure_dotenv_loaded", lambda: None)
    monkeypatch.delenv("DANA_CLOUD_PRIMARY", raising=False)


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


def _clear_session_state() -> None:
    rd = _react_dispatch_module
    for registry in rd._PERSISTED_SESSION_REGISTRIES.values():
        registry.clear()
    rd._OUTPUT_TRUNCATED_BY_SESSION.clear()
    _task_board_module._PLANS_BY_SESSION.clear()
    mock = sys.modules.get("dana.platform.mock")
    if mock is not None:
        for registry in mock._PERSISTED_MOCK_REGISTRIES.values():
            registry.clear()
    artifacts = sys.modules.get("dana.api.artifacts_registry")
    if artifacts is not None:
        artifacts._artifacts.clear()
    server = sys.modules.get("dana.api.server")
    if server is not None:
        # Per-connection session dicts (CadCallLog, turn queue, HITL state)
        # and the mesh/audio download tokens.
        server._active_sessions.clear()
        server._MESH_REGISTRY.clear()
        server._AUDIO_REGISTRY.clear()


@pytest.fixture(autouse=True)
def _reset_session_state():
    """Global safety net: every per-session, process-wide registry starts
    and ends each test empty — the plan/FSM gates, object registry,
    topology DAG, assembly state, measurement/collision gates
    (react_dispatch._PERSISTED_SESSION_REGISTRIES, the same list session
    persistence saves), task boards, the mock driver's geometry, the
    artifact registry, and the server's live sessions (each holding its
    CadCallLog and turn state). They're keyed by session_id, and most tests
    share the ambient default one, so anything left behind showed up in an
    unrelated later test (2f05ca2). Cleared before as well as after, so a
    test that fails mid-way can't seed the next one either.
    """
    _clear_session_state()
    yield
    _clear_session_state()


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


@pytest.fixture
def interactive_tool_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    """Turn off react_dispatch's headless tool pruning (and the screenshot
    short-circuit) for a test that checks the full interactive tool set.
    Patches is_headless only — DANA_HEADLESS itself stays set, so the
    FreeCAD GUI-launch lock in engine.py is untouched."""
    import dana.core.react_dispatch as react_dispatch

    monkeypatch.setattr(react_dispatch, "is_headless", lambda: False)
