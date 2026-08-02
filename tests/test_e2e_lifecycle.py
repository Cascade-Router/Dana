"""End-to-end lifecycle stress test for Donna agentic upgrade.

Simulates: Vault daemon → wake → English STT → ReAct → write_vault_memory →
English SpatialIR synthesis → English TTS, with RAM/VRAM profiling.
Also covers memory conflict compaction and inject_keystrokes dry-run routing.
"""

from __future__ import annotations

import gc
import json
import os
import secrets
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Isolate E2E vault daemon from any production instance on 47475.
os.environ.setdefault("DONNA_VAULT_PORT", "47476")
# Never touch the real OS input stack during E2E.
os.environ["DONNA_OS_DRY_RUN"] = "1"

import psutil  # noqa: E402

from test_support_react import patch_scripted_llm


class _MiniMonkey:
    """Lightweight stand-in for pytest monkeypatch in this script-style E2E."""

    def setattr(self, target, name=None, value=None):  # noqa: ANN001
        import langchain_ollama as _lo

        if isinstance(target, str) and value is not None and name is None:
            if target == "langchain_ollama.ChatOllama":
                setattr(_lo, "ChatOllama", value)
                return
        if isinstance(target, str) and callable(name) and value is None:
            if target == "langchain_ollama.ChatOllama":
                setattr(_lo, "ChatOllama", name)
                return
        raise AssertionError(f"unsupported setattr {target!r}")


def _install_script(script) -> None:  # noqa: ANN001
    patch_scripted_llm(_MiniMonkey(), script)
from dana.agentic import REACT_MAX_ITERS, run_react_loop  # noqa: E402
from dana.os_automation import inject_keystrokes, sanitize_keystroke_text  # noqa: E402
from dana.sanitize import sanitize_log_message, sanitize_tool_trace  # noqa: E402
from dana.tools.broker import IntentBroker  # noqa: E402
from dana.tools.ipc import VaultRequest, VaultResponse  # noqa: E402
from dana.tools.schema import ToolCall  # noqa: E402
from dana.vault_service import (  # noqa: E402
    VaultClient,
    VaultKeyDaemon,
    _META_KEY,
    _vault_port,
    consolidate_vault_memory,
    memory_value)
from dana.prompts.spatial_synthesis import build_agent_system_prompt  # noqa: E402
from spatial_context import SPATIAL_AGGREGATOR  # noqa: E402


@dataclass
class ResourceSample:
    label: str
    rss_mb: float
    vram_mb: float | None = None


@dataclass
class E2EReport:
    ok: bool
    steps: list[str] = field(default_factory=list)
    samples: list[ResourceSample] = field(default_factory=list)
    peak_rss_mb: float = 0.0
    peak_vram_mb: float | None = None
    tts_wav: str | None = None
    final_fa: str = ""
    errors: list[str] = field(default_factory=list)

    def add(self, step: str) -> None:
        self.steps.append(step)
        print(f"[E2E] {step}", flush=True)


def _rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def _vram_mb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    except Exception:
        return None


def _sample(report: E2EReport, label: str) -> None:
    sample = ResourceSample(label=label, rss_mb=_rss_mb(), vram_mb=_vram_mb())
    report.samples.append(sample)
    report.peak_rss_mb = max(report.peak_rss_mb, sample.rss_mb)
    if sample.vram_mb is not None:
        report.peak_vram_mb = max(report.peak_vram_mb or 0.0, sample.vram_mb)
    vram = f"{sample.vram_mb:.1f} MB" if sample.vram_mb is not None else "n/a"
    print(f"[E2E][MEM] {label}: RSS={sample.rss_mb:.1f} MB VRAM={vram}", flush=True)


def _start_daemon(vault_path: str) -> tuple[VaultKeyDaemon, threading.Thread]:
    daemon = VaultKeyDaemon(vault_path=vault_path)
    thread = threading.Thread(target=daemon.serve_forever, name="e2e-vault", daemon=True)
    thread.start()
    deadline = time.time() + 5.0
    while time.time() < deadline:
        try:
            from dana.vault_service import _rpc

            if _rpc({"op": "ping"}, timeout=0.3).get("ok"):
                return daemon, thread
        except OSError:
            time.sleep(0.05)
    raise RuntimeError(f"E2E vault daemon failed to bind port {_vault_port()}")


def _inject_english_spatial_ir() -> str:
    """Synthesize English SpatialIR as if YOLO saw laptop + cup."""
    SPATIAL_AGGREGATOR.set_vision_source("screen")
    SPATIAL_AGGREGATOR.set_ui_state("thinking")
    w, h = 640, 480
    dets = [
        ([120, 100, 420, 340], "laptop (center)", 0.91),
        ([480, 40, 560, 120], "cup (top-right)", 0.77),
    ]
    SPATIAL_AGGREGATOR.update_from_dets(dets, frame_shape=(h, w, 3))
    block = SPATIAL_AGGREGATOR.synthesize_prompt_block()
    assert "laptop" in block
    return block


def _run_react_with_vault(client: VaultClient, spatial: str, report: E2EReport) -> str:
    """STT → ReAct → write_vault_memory → English FINAL."""
    user_text = "Remember this IP on screen: 10.0.0.42 and tell me what you see."
    SPATIAL_AGGREGATOR.update_transcript(user=user_text)
    system = build_agent_system_prompt(
        spatial_block=spatial,
        labels_csv="laptop, cup",
        profile_summary="{}",
        reply_lang="en")
    scripted = [
        "TOOL: write_vault_memory(key=remembered_ip, value=10.0.0.42)",
        "TOOL: describe_spatial_scene(focus=dominant)",
        (
            "FINAL: Saved IP 10.0.0.42. I see a laptop in the center "
            "and a cup at the top-right."
        ),
    ]
    idx = {"n": 0}

    def ask_fn(_messages: list[dict[str, str]]) -> str:
        n = idx["n"]
        idx["n"] = n + 1
        return scripted[min(n, len(scripted) - 1)]

    def execute_fn(tc: ToolCall) -> str:
        if tc.tool_id == "write_vault_memory":
            key = str(tc.arguments.get("key") or "")
            value = tc.arguments.get("value")
            client.write_memory(key, value)
            return f"OK: saved {key}={value!r}"
        if tc.tool_id == "describe_spatial_scene":
            return f"SpatialIR={spatial} | focus=dominant"
        if tc.tool_id == "read_vault_memory":
            key = str(tc.arguments.get("key") or "")
            try:
                val = client.read_memory(key)
                return f"OK: {key}={val!r}"
            except KeyError:
                return f"OK: key '{key}' not found"
        return f"OK: {tc.tool_id}"

    _install_script(ask_fn)
    result = run_react_loop(
        user_text=user_text,
        system_prompt=system,
        execute_fn=execute_fn,
        max_iters=REACT_MAX_ITERS,
        broker=IntentBroker())
    report.add(
        f"ReAct done iters={result.iterations} "
        f"trace={json.dumps(sanitize_tool_trace(result.tool_trace), ensure_ascii=True)}"
    )
    assert result.iterations <= REACT_MAX_ITERS
    assert client.read_memory("remembered_ip") == "10.0.0.42"
    assert "laptop" in result.final_text or "cup" in result.final_text
    return result.final_text


def _run_memory_conflict_compaction(client: VaultClient, report: E2EReport) -> None:
    """Write conflicting project-directory facts; vault must resolve without bloat."""
    client.write_memory("project_directory", "C:/Users/Example/OldProject")
    assert client.read_memory("project_directory") == "C:/Users/Example/OldProject"
    report.add("Conflict seed: project_directory=C:/Users/Example/OldProject")

    client.write_memory("current_project_directory", "C:/Users/Example/Project")
    report.add(
        "Conflict update consolidation="
        + json.dumps(client.last_consolidation, ensure_ascii=True)
    )

    profile = client.get_profile()
    assert "project_directory" not in profile
    assert memory_value(profile, "current_project_directory") == (
        "C:/Users/Example/Project"
    )
    entry = profile["current_project_directory"]
    assert isinstance(entry, dict)
    assert "last_updated" in entry
    assert entry["status"] == "active"
    assert _META_KEY in profile
    assert "current_project_directory" in profile[_META_KEY]

    before_keys = set(k for k in profile.keys() if k != _META_KEY)
    client.write_memory("session_tmp_id", "sess_deadbeefcafebabe0123456789abcdef")
    after = client.get_profile()
    assert "session_tmp_id" not in after
    assert client.last_consolidation.get("pruned_transient") is True
    after_user_keys = set(k for k in after.keys() if k != _META_KEY)
    assert after_user_keys == before_keys
    report.add("Memory conflict + transient prune resolved cleanly")


def _run_inject_keystrokes_pipeline(report: E2EReport) -> None:
    """Simulate typing an extracted visual element through the tool IR loop."""
    broker = IntentBroker()
    assert "inject_keystrokes" in broker.registry
    assert "read_clipboard_context" in broker.registry

    blocked = sanitize_keystroke_text("Ctrl+Alt+Del")
    assert blocked.blocked
    report.add(f"Keystroke guard blocked chord: {blocked.reason}")

    visual_text = "10.0.0.42"
    scripted = [
        f"TOOL: inject_keystrokes(text={visual_text})",
        "FINAL: Typed the IP from the screen.",
    ]
    idx = {"n": 0}
    typed: dict[str, Any] = {}

    def ask_fn(_m: list[dict[str, str]]) -> str:
        n = idx["n"]
        idx["n"] = n + 1
        return scripted[min(n, len(scripted) - 1)]

    def execute_fn(tc: ToolCall) -> str:
        if tc.tool_id == "inject_keystrokes":
            result = inject_keystrokes(str(tc.arguments.get("text") or ""))
            typed.update(result)
            if not result.get("ok"):
                return f"ERROR: {result.get('error')}"
            return (
                f"OK: inject_keystrokes dry_run chars={result.get('chars_typed')} "
                f"preview={result.get('preview')!r}"
            )
        return f"OK: {tc.tool_id}"

    _install_script(ask_fn)
    result = run_react_loop(
        user_text="Type out the IP you see on screen",
        system_prompt=build_agent_system_prompt(
            spatial_block="vis=screen|ui=idle|dom=none@center|scene=[]|intent=type",
            labels_csv="",
            profile_summary="{}",
            reply_lang="en"),
        execute_fn=execute_fn,
        max_iters=REACT_MAX_ITERS,
        broker=broker)
    assert typed.get("ok") is True
    assert typed.get("dry_run") is True
    assert typed.get("chars_typed") == len(visual_text)
    assert any(t.get("tool") == "inject_keystrokes" for t in result.tool_trace)
    report.add(
        "inject_keystrokes pipeline OK "
        + json.dumps(sanitize_tool_trace(result.tool_trace), ensure_ascii=True)
    )


def _force_max_iter_and_gc(report: E2EReport) -> None:
    """Hit the ReAct cap, then GC and confirm RSS does not explode."""
    before = _rss_mb()

    def ask_fn(_m: list[dict[str, str]]) -> str:
        return "TOOL: describe_spatial_scene(focus=all)"

    def execute_fn(tc: ToolCall) -> str:
        blob = bytearray(2 * 1024 * 1024)
        blob[0] = 1
        return f"OK: {tc.tool_id} bytes={len(blob)}"

    _install_script(ask_fn)
    result = run_react_loop(
        user_text="what do you see",
        system_prompt="You are Donna. Reply FINAL only after tools.",
        execute_fn=execute_fn,
        max_iters=REACT_MAX_ITERS,
        broker=IntentBroker())
    assert result.iterations == REACT_MAX_ITERS
    del result
    gc.collect()
    after = _rss_mb()
    report.add(f"Max-iter GC: RSS before={before:.1f} after={after:.1f} MB")
    if after - before > 80.0:
        raise AssertionError(f"Possible leak: RSS grew {after - before:.1f} MB")


def _synthesize_language_tts(text: str, out_wav: Path, report: E2EReport) -> None:
    """Offline Piper English TTS (skip gracefully if voice files missing)."""
    project = Path(__file__).resolve().parent.parent
    en_onnx = project / "tts_models" / "en_US-hfc_female-medium.onnx"
    if not en_onnx.is_file():
        # Graceful skip when preferred voice absent (no network in CI).
        legacy = project / "tts_models" / "en_US-ljspeech-high.onnx"
        if legacy.is_file():
            en_onnx = legacy
            report.add(f"TTS using legacy fallback {legacy.name}")
        else:
            report.add(f"TTS skipped (missing {en_onnx.name})")
            return
    from piper import PiperVoice

    voice = PiperVoice.load(str(en_onnx))
    import wave

    with wave.open(str(out_wav), "wb") as wf:
        voice.synthesize_wav(text, wf)
    size = out_wav.stat().st_size
    assert size > 1000
    report.tts_wav = str(out_wav)
    report.add(f"English TTS wrote {out_wav.name} ({size} bytes)")


def _assert_log_sanitization() -> None:
    dirty = (
        "password=SuperSecret123 session_token=abcDEF1234567890 "
        "OK: saved remembered_ip='10.0.0.42' "
        "data_key_b64=YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY3ODkwYWJjZA=="
    )
    clean = sanitize_log_message(dirty)
    assert "SuperSecret123" not in clean
    assert "10.0.0.42" not in clean
    assert "abcDEF1234567890" not in clean or "***" in clean
    assert "YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY3ODkwYWJjZA==" not in clean


def _assert_local_compaction_unit() -> None:
    """Offline unit check of consolidate_vault_memory without daemon."""
    profile: dict = {}
    profile, r1 = consolidate_vault_memory(profile, "saved_ip", "1.1.1.1")
    assert r1["action"] == "write"
    profile, r2 = consolidate_vault_memory(profile, "remembered_ip", "9.9.9.9")
    assert "saved_ip" in r2["deprecated"]
    assert "saved_ip" not in profile
    assert memory_value(profile, "remembered_ip") == "9.9.9.9"


def _run_reflection_self_correction(client: VaultClient, report: E2EReport) -> None:
    """Force a tool failure → Reflector lesson → next prompt injects the lesson."""
    from dana.reflector import load_lessons

    broker = IntentBroker()

    def _provider():
        return load_lessons(client)

    broker.set_lessons_provider(_provider)

    # Turn 1: valid IR, but execute_fn returns a typed ERROR (simulates bad arg payload).
    scripted = [
        "TOOL: write_vault_memory(key=remembered_ip, value=not-an-ip)",
        "FINAL: I could not save that value.",
    ]
    idx = {"n": 0}

    def ask_fn(_m: list[dict[str, str]]) -> str:
        n = idx["n"]
        idx["n"] = n + 1
        return scripted[min(n, len(scripted) - 1)]

    def execute_fn(tc: ToolCall) -> str:
        if tc.tool_id == "write_vault_memory":
            return (
                "ERROR: invalid argument type for value='not-an-ip'; "
                "expected dotted IPv4 before write_vault_memory"
            )
        return f"OK: {tc.tool_id}"

    _install_script(ask_fn)
    result = run_react_loop(
        user_text="Remember this IP address on my screen: not-an-ip",
        system_prompt=build_agent_system_prompt(
            spatial_block="vis=screen|ui=idle|dom=none@center|scene=[]|intent=",
            labels_csv="",
            profile_summary="{}",
            reply_lang="en"),
        execute_fn=execute_fn,
        max_iters=REACT_MAX_ITERS,
        broker=broker,
        vault_client=client,
        enable_reflection=True,
        reflect_fn=None,  # offline deterministic critic
    )
    assert result.had_errors
    assert result.reflection is not None
    assert result.reflection.get("triggered")
    assert result.reflection.get("rule")
    assert result.reflection.get("persisted") is True
    report.add(
        f"Reflector distilled rule in {result.reflection_ms:.1f} ms: "
        f"{result.reflection.get('rule')}"
    )
    report.add(f"Reflector latency_ms={result.reflection_ms:.1f}")

    lessons = load_lessons(client)
    assert lessons, "lessons_learned missing from vault"
    assert any(
        "write_vault_memory" in (L.tool_id or "") or "write_vault_memory" in L.rule
        for L in lessons
    )

    # Turn 2: broker must inject the lesson into the system prompt.
    base = build_agent_system_prompt(
        spatial_block="vis=screen|ui=idle|dom=none@center|scene=[]|intent=",
        labels_csv="",
        profile_summary="{}",
        reply_lang="en")
    augmented = broker.augment_system_prompt(
        base, "Remember this IP address on my screen: 10.0.0.9"
    )
    assert "Lessons learned" in augmented
    assert "Rule:" in augmented
    report.add("Next ReAct prompt injected lessons_learned OK")


def _run_tool_synthesis(report: E2EReport) -> None:
    """Architect a reverse_string tool via sandbox → register → hot-reload broker."""
    from dana_security import (
        GENERATED_TOOLS_PATH,
        TOOLS_JSON_PATH,
        architect_new_tool,
        execute_dynamic_tool,
        validate_ast)
    from dana.settings import load_donna_settings
    from dana.tools.broker import IntentBroker, reload_broker_registry
    from dana.tools.schema import ToolCall

    # Production lock must block synthesis when flag is false.
    load_donna_settings(force_reload=True)
    locked = architect_new_tool(
        "should_not_exist",
        "def should_not_exist(text):\n    return text\n")
    assert locked.get("ok") is False and locked.get("locked") is True
    broker = IntentBroker()
    obs = broker.dispatch(
        ToolCall(tool_id="architect_new_tool", arguments={"tool_name": "x", "python_code": "y"}),
        {"architect_new_tool": lambda _c: "SHOULD_NOT_RUN"})
    assert "LOCKED" in str(obs)
    report.add("Production synthesis lock active (architect_new_tool blocked)")

    # Temporarily enable for sandbox verification, then restore.
    import dana.settings as ds

    prev = dict(ds.load_donna_settings())
    ds._CACHE = {**prev, "enable_dynamic_tool_synthesis": True}

    try:
        validate_ast("import os\ndef bad(text): return os.getcwd()")
        raise AssertionError("AST should have blocked os import")
    except Exception as exc:
        assert "Blocked" in str(exc) or "os" in str(exc).lower()
        report.add(f"Sandbox AST blocked os import: {exc}")

    code = "def reverse_string(text):\n    return text[::-1]\n"
    result = architect_new_tool("reverse_string", code, test_input="abc")
    assert result.get("ok") is True, result
    assert result.get("test_result") == "cba"
    report.add(
        f"architect_new_tool OK test_result={result.get('test_result')!r} "
        f"elapsed_ms={result.get('elapsed_ms'):.1f}"
    )

    tools = json.loads(TOOLS_JSON_PATH.read_text(encoding="utf-8"))
    ids = [t["id"] for t in tools.get("tools", [])]
    assert "reverse_string" in ids
    assert "reverse_string" in GENERATED_TOOLS_PATH.read_text(encoding="utf-8")

    broker = reload_broker_registry()
    assert "reverse_string" in broker.registry
    report.add("Broker hot-reloaded tools.json — reverse_string registered")

    sand = execute_dynamic_tool("reverse_string", "dana")
    assert sand.ok and sand.result == "annod"
    report.add("Dynamic reverse_string sandbox execution OK")

    # Cleanup dynamic registration so the repo stays clean after E2E.
    tools["tools"] = [t for t in tools["tools"] if t.get("id") != "reverse_string"]
    TOOLS_JSON_PATH.write_text(
        json.dumps(tools, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    text = GENERATED_TOOLS_PATH.read_text(encoding="utf-8")
    marker_start = "# --- begin dynamic tool: reverse_string ---"
    marker_end = "# --- end dynamic tool: reverse_string ---"
    if marker_start in text and marker_end in text:
        pre, rest = text.split(marker_start, 1)
        _, post = rest.split(marker_end, 1)
        GENERATED_TOOLS_PATH.write_text(pre.rstrip() + "\n" + post.lstrip(), encoding="utf-8")
    reload_broker_registry()
    report.add("Cleaned reverse_string registration after verification")

    # Restore production lock.
    ds._CACHE = prev
    load_donna_settings(force_reload=True)


def run_e2e() -> E2EReport:
    report = E2EReport(ok=False)
    daemon: VaultKeyDaemon | None = None
    tmpdir = tempfile.mkdtemp(prefix="donna_e2e_")
    vault_path = os.path.join(tmpdir, "e2e_memory.enc")
    wav_path = Path(tmpdir) / "e2e_en.wav"

    try:
        _sample(report, "baseline")
        _assert_log_sanitization()
        report.add("Log sanitization OK")
        _assert_local_compaction_unit()
        report.add("Local consolidate_vault_memory unit OK")

        req = VaultRequest(op="ping")
        assert req.redacted_dict()["op"] == "ping"
        resp = VaultResponse(ok=True, session_token="tok_secret", data_key_b64="x" * 64)
        assert resp.redacted_dict()["session_token"] == "***"
        report.add("Typed VaultRequest/VaultResponse OK")

        report.add(f"Starting vault daemon on port {_vault_port()} vault={vault_path}")
        daemon, _thread = _start_daemon(vault_path)
        _sample(report, "after_daemon")

        client = VaultClient()
        password = "e2e-test-" + secrets.token_hex(4)
        recovery = secrets.token_urlsafe(16)
        client.unlock(password, create=True, recovery_key=recovery)
        report.add("Vault unlocked (create) — credentials not logged")
        _sample(report, "after_unlock")

        wake_event = threading.Event()
        wake_event.set()
        assert wake_event.is_set()
        report.add("Wake word simulated (event set)")

        spatial = _inject_english_spatial_ir()
        report.add(f"English SpatialIR injected: {spatial[:120]}...")
        _sample(report, "after_spatial")

        final_text = _run_react_with_vault(client, spatial, report)
        report.final_fa = final_text
        report.add(
            "English FINAL: "
            + final_text[:100].encode("ascii", "backslashreplace").decode("ascii")
        )
        _sample(report, "after_react")

        _run_memory_conflict_compaction(client, report)
        _sample(report, "after_compaction")

        _run_inject_keystrokes_pipeline(report)
        _sample(report, "after_keystrokes")

        _run_reflection_self_correction(client, report)
        _sample(report, "after_reflection")

        _run_tool_synthesis(report)
        _sample(report, "after_synthesis")

        _synthesize_language_tts(final_text, wav_path, report)
        _sample(report, "after_tts")

        _force_max_iter_and_gc(report)
        _sample(report, "after_gc")

        report.ok = True
        report.add("E2E lifecycle PASSED")
    except Exception as exc:  # noqa: BLE001
        report.errors.append(f"{exc}\n{traceback.format_exc()}")
        report.add(f"E2E FAILED: {exc}")
        print(traceback.format_exc(), flush=True)
    finally:
        if daemon is not None:
            daemon.stop()
            time.sleep(0.2)
        try:
            if os.path.isfile(vault_path):
                os.remove(vault_path)
        except OSError:
            pass

    return report


def main() -> int:
    print(f"[E2E] DONNA_VAULT_PORT={_vault_port()}", flush=True)
    report = run_e2e()
    summary = {
        "ok": report.ok,
        "peak_rss_mb": round(report.peak_rss_mb, 1),
        "peak_vram_mb": (
            round(report.peak_vram_mb, 1) if report.peak_vram_mb is not None else None
        ),
        "steps": report.steps,
        "tts_wav": report.tts_wav,
        "errors": report.errors,
    }
    print("[E2E] SUMMARY " + json.dumps(summary, ensure_ascii=True), flush=True)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
