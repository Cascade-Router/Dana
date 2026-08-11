"""Dana process lifecycle: CLI parsing, singleton lock, vault unlock/reset,
daemon-thread orchestration, and graceful shutdown.

Extracted verbatim from ``dana.core_agent`` (Phase 6 of the core_agent.py
decomposition; see the approved refactor plan). This module owns ``main()``
and ``agent_loop()`` -- the two entry points that bootstrap either the
CustomTkinter dashboard (dana.ui.app_gui.DanaGUI) or the headless background
loop, spawn the Tracker/WakeWord/Conversation/InputIngest daemon threads,
and tear them down again on shutdown.

A handful of business-logic functions that ``agent_loop()`` spawns as thread
targets (``tracker_worker``, ``wakeword_worker``, ``input_txt_ingest_worker``)
stay in ``dana.core_agent`` because they are also relied on elsewhere
(directly or via ``dana.core.agent_loop``'s existing lazy-import pattern for
the vision/wake-word helpers they share). Importing them here happens inside
``agent_loop()`` itself, mirroring the same lazy-import precedent
``dana.core.agent_loop`` already uses for its own reverse dependency on
``dana.core_agent`` -- not a new workaround, just the existing one applied
consistently to avoid a real module-level cycle between this file and
``dana.core_agent``.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import queue
import secrets
import signal
import socket
import sys
import threading
import time
from typing import Any, Optional

import dana.core.shared_state as state
from dana.agentic import get_dana_mode, set_dana_mode
from dana.audio.audio_pipeline import AudioRouter
from dana.audio.wake_poller import WakePoller
from dana.audio.mic_input import (
    ensure_live_mic,
    ensure_mic_ingest_thread,
    list_input_devices,
    list_output_devices,
    load_audio_settings,
    _validate_mic_id,
    _validate_speaker_id,
)
from dana.audio.noise_floor import calibrate_noise_floor
from dana.audio.tts_manager import enqueue_speech_impl as enqueue_speech, flush_tts_queue
from dana.audio.tts_worker import (
    consume_audio_hardware_fault,
    soft_recover_audio_hardware,
    tts_worker,
    wait_for_speech_idle,
    _synthesize_and_play,
)
from dana.core.agent_loop import conversation_worker
from dana.core.constants import SENDGRID_MAIL_URL, WAKE_COOLDOWN_SEC
from dana.core.shared_state import (
    MEMORY_FILE,
    SETTINGS_FILE,
    TRIGGER_FILE,
    _tts_manager,
    audio_hardware_fault,
    camera_tool,
    emit_trace,
    get_ui_state,
    has_vault_prompt_listener,
    is_recording,
    mic_ingest_ready,
    notify_vault_unlocked,
    request_vault_unlock,
    screen_tool,
    speech_idle,
    speech_queue,
    stop_event,
    tts_interrupt_event,
    unregister_dictation_sessions_listener,
    unregister_spec_approval_listener,
    unregister_transcript_listener,
    unregister_vault_prompt_listener,
    wakeword_armed,
    _dual_wake_poller,
    _dual_wake_router,
)
from dana.logging import CONVERSATION_LOG_PATH, enable_runtime_file_logging, log
from dana.paths import PROJECT_ROOT, chdir_project_root, resolve_wakeword_onnx
from dana.secure_memory import SecureMemory
from dana.vault_service import VaultClient
from dana.ui.app_gui import DanaGUI
from dana.ui.tray_icon import run_system_tray

import requests

# ---------------------------------------------------------------------------
# Singleton lock (keep socket open for process lifetime)
# ---------------------------------------------------------------------------

_SINGLETON_PORT = 47474
_singleton_socket: Optional[socket.socket] = None

# ``dana_vault`` mirrors the exact pre-existing pattern from core_agent.py:
# a bare-imported-looking module global reassigned via ``global dana_vault``
# by both unlock_dana_memory() and execute_lockdown_shutdown() below. It is
# intentionally NOT synced with ``state.dana_vault`` (the shared_state
# canonical copy) -- that mismatch predates this move; relocating both
# reader and writer together here preserves the exact existing behavior
# rather than silently changing it.
dana_vault: Optional[SecureMemory] = None


def enforce_singleton() -> None:
    """Bind a local TCP port so only one Dana process can run."""
    global _singleton_socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # Do not set SO_REUSEADDR — that would defeat the singleton check.
        sock.bind(("127.0.0.1", _SINGLETON_PORT))
        sock.listen(1)
    except OSError:
        print("[System] Dana is already running.", flush=True)
        try:
            sock.close()
        except OSError:
            pass
        sys.exit(0)
    _singleton_socket = sock
def populate_vault_hot_cache(client: Optional["VaultClient"] = None) -> None:
    """Prefetch core identity keys into VAULT_HOT_CACHE after a successful unlock."""
    client = client if client is not None else state.vault_client
    user_name = "Amirhosein"
    family_partner = "Narges"
    try:
        raw = client.read_memory("user_name")
        if raw is not None and str(raw).strip():
            user_name = str(raw).strip()
    except Exception:  # noqa: BLE001
        pass
    try:
        raw = client.read_memory("family_partner")
        if raw is not None and str(raw).strip():
            family_partner = str(raw).strip()
    except Exception:  # noqa: BLE001
        pass
    state.VAULT_HOT_CACHE = {
        "user_name": user_name,
        "family_partner": family_partner,
    }
    log(
        "Memory",
        f"Vault hot-cache ready "
        f"(user_name={user_name!r}, family_partner={family_partner!r}).",
    )


def execute_lockdown_shutdown() -> None:
    """Speak lockdown ack, purge vault RAM key, hard-kill the process."""
    global dana_vault

    log("Security", "Lockdown fast-path triggered — purging vault and exiting.")
    try:
        # Prefer spooler; fall back to direct play if the worker is already dead.
        flush_tts_queue()
        enqueue_speech("Initiating lockdown. Vault secured. Goodbye.")
        wait_for_speech_idle(timeout=4.0)
    except Exception as exc:  # noqa: BLE001
        print(f"[Security] Lockdown TTS failed: {exc}", flush=True)
        log("Security", f"WARNING: lockdown TTS failed ({exc})")
        try:
            _synthesize_and_play(
                "Initiating lockdown. Vault secured. Goodbye.",
                state.AUDIO_OUTPUT_DEVICE,
            )
        except Exception:
            pass

    try:
        state.vault_client.lock_vault()
        print(
            "[Security] Vault session purged. Password will be required next boot.",
            flush=True,
        )
        log("Security", "Vault daemon RAM key + sessions purged.")
    except Exception as exc:  # noqa: BLE001
        print(f"[Security] Failed to contact daemon for purge: {exc}", flush=True)
        log("Security", f"ERROR: lockdown purge failed ({exc})")

    try:
        if dana_vault is not None:
            dana_vault.lock()
    except Exception:  # noqa: BLE001
        pass
    dana_vault = None
    state.dana_profile = {}
    state.VAULT_HOT_CACHE = {}

    # Force immediate termination of all threads and GUI.
    os._exit(0)
def select_device():
    import torch

    # Intel/x86 AVX CPU path for PyTorch ops (no-op / ignored on unsupported builds).
    try:
        torch.backends.mkldnn.enabled = True
    except Exception:
        pass

    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        log(
            "Main",
            f"Accelerator: CUDA ({name}) - YOLO + Whisper on cuda:0; brain=Ollama",
        )
        return device
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        log("Main", "Accelerator: Apple MPS (CUDA unavailable)")
        return torch.device("mps")
    log(
        "Main",
        "Accelerator: CPU fallback (MKLDNN enabled). "
        "Install CUDA wheels: pip install torch torchvision "
        "--index-url https://download.pytorch.org/whl/cu126",
    )
    return torch.device("cpu")


def select_dtype(device):
    import torch

    if device.type == "cuda":
        major, _minor = torch.cuda.get_device_capability(0)
        if major >= 8 and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def email_recovery_key(recovery_key: str) -> None:
    """Optionally email the backup recovery key via SendGrid v3 API (.env credentials)."""
    choice = input(
        "Would you like to email your Backup Recovery Key to yourself? (y/n): "
    ).strip().lower()
    if choice not in ("y", "yes"):
        print(
            f"\n[Memory] Backup Recovery Key (save this somewhere safe):\n{recovery_key}\n",
            flush=True,
        )
        return

    sendgrid_api_key = (os.getenv("SENDGRID_API_KEY") or "").strip()
    sendgrid_from_email = (os.getenv("SENDGRID_FROM_EMAIL") or "").strip()
    if not sendgrid_api_key or not sendgrid_from_email:
        print(
            "[System] SENDGRID_API_KEY or SENDGRID_FROM_EMAIL missing from .env. "
            "Skipping email recovery setup.",
            flush=True,
        )
        print(
            f"\n[Memory] Backup Recovery Key (save this somewhere safe):\n{recovery_key}\n",
            flush=True,
        )
        log(
            "Memory",
            "WARNING: SendGrid env vars missing; printed recovery key to terminal.",
        )
        return

    destination = input(
        "Enter Destination Email Address (where the key should be sent): "
    ).strip()
    if not destination:
        print(
            "[Memory] No destination email provided. Skipping send.",
            flush=True,
        )
        print(
            f"\n[Memory] Backup Recovery Key (save this somewhere safe):\n{recovery_key}\n",
            flush=True,
        )
        return

    payload = {
        "personalizations": [
            {
                "to": [{"email": destination}],
            }
        ],
        "from": {"email": sendgrid_from_email},
        "subject": "Dānā: Secure Memory Recovery Key",
        "content": [
            {
                "type": "text/plain",
                "value": (
                    "Your Dana Backup Recovery Key is below.\n"
                    "Store it offline. Anyone with this key can unlock Dana's memory vault.\n\n"
                    f"{recovery_key}\n"
                ),
            }
        ],
    }
    headers = {
        "Authorization": f"Bearer {sendgrid_api_key}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(
            SENDGRID_MAIL_URL,
            headers=headers,
            json=payload,
            timeout=30,
        )
        if resp.status_code == 202:
            print("[Memory] Recovery key emailed successfully via SendGrid.", flush=True)
            log("Memory", "Recovery key emailed successfully via SendGrid.")
            return
        print(
            f"[Memory] SendGrid rejected the request "
            f"(HTTP {resp.status_code}): {resp.text[:300]}",
            flush=True,
        )
        print(
            f"\n[Memory] Backup Recovery Key (save this somewhere safe):\n{recovery_key}\n",
            flush=True,
        )
        log(
            "Memory",
            f"WARNING: SendGrid HTTP {resp.status_code}; printed recovery key.",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[Memory] Email failed: {exc}", flush=True)
        print(
            f"\n[Memory] Backup Recovery Key (save this somewhere safe):\n{recovery_key}\n",
            flush=True,
        )
        log("Memory", f"WARNING: recovery email failed ({exc})")


def unlock_dana_memory() -> SecureMemory:
    """Unlock via in-RAM vault daemon (Option B). Password only if resume fails.

    Daemon handshake (try_resume_session) ALWAYS runs before any password prompt.
    """
    global dana_vault
    state.vault_client = VaultClient()
    try:
        state.vault_client.ensure_ready()
    except Exception as exc:  # noqa: BLE001
        print(f"[Memory Error] Vault daemon unavailable: {exc}", flush=True)
        log("Memory", f"ERROR vault daemon: {exc}")
        raise SystemExit(1) from exc

    # --- HARD GATE: resume first; password ONLY in the else branch ---
    resumed = False
    try:
        resumed = bool(state.vault_client.try_resume_session())
    except Exception as exc:  # noqa: BLE001
        log("Memory", f"try_resume_session failed ({exc})")
        resumed = False

    if resumed:
        state.dana_profile = dict(state.vault_client.profile)
        vault = SecureMemory(path=MEMORY_FILE)
        try:
            from dana.vault_service import _rpc
            import base64 as _b64

            resp = _rpc(
                {
                    "op": "export_data_key",
                    "session_token": state.vault_client.session_token,
                },
                timeout=5.0,
            )
            if resp.get("ok"):
                key = _b64.urlsafe_b64decode(resp["data_key_b64"].encode("ascii"))
                vault.unlock_with_data_key(key)
                vault.profile = dict(state.dana_profile)
        except Exception as exc:  # noqa: BLE001
            log("Memory", f"WARNING: local vault hydrate skipped ({exc})")
        dana_vault = vault
        populate_vault_hot_cache(state.vault_client)
        log(
            "Memory",
            f"Vault unlocked via daemon session "
            f"(keys={len(state.dana_profile)}; token cached in RAM daemon).",
        )
        print(
            "[Memory] Vault unlocked via daemon session (keys cached in RAM).",
            flush=True,
        )
        notify_vault_unlocked()
        return vault

    # else: daemon locked → resolve credential (env → keyring → TTY prompt →
    # GUI modal, when a Dashboard has registered to handle vault prompts).
    from dana.tools.vault import VaultCredentialsMissing, _get_master_key

    prompt = "Enter Master Password (or pasted Recovery Key) to unlock Dānā: "
    vault_exists = os.path.isfile(MEMORY_FILE)

    def _resolve_password(gui_reason: str) -> str:
        try:
            return _get_master_key(prompt=prompt)
        except VaultCredentialsMissing:
            if not has_vault_prompt_listener():
                raise
            gui_password = request_vault_unlock(gui_reason)
            if not gui_password:
                raise VaultCredentialsMissing(
                    "Vault unlock cancelled from the Dashboard prompt."
                ) from None
            return gui_password

    try:
        password = _resolve_password(
            "Dana's memory vault needs your Master Password (or Recovery Key) "
            "to unlock — no credential found in the environment or OS keyring."
        )
    except VaultCredentialsMissing as exc:
        print(f"[Memory Error] {exc}", flush=True)
        log("Memory", f"ERROR: {exc}")
        raise SystemExit(1) from exc

    if not vault_exists:
        recovery_key = secrets.token_urlsafe(32)
        try:
            state.dana_profile = state.vault_client.unlock(
                password, create=True, recovery_key=recovery_key
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[Memory Error] Could not create vault: {exc}", flush=True)
            log("Memory", f"ERROR creating vault: {exc}")
            raise SystemExit(1) from exc
        email_recovery_key(recovery_key)
    else:
        # Wrong password from the GUI modal gets a few retries (a mistyped
        # passcode should not tear down the whole AgentLoop thread); CLI/TTY
        # callers with no listener registered keep the original fail-fast.
        max_attempts = 3
        attempt = 0
        while True:
            attempt += 1
            try:
                state.dana_profile = state.vault_client.unlock(password, create=False)
                break
            except Exception as exc:  # noqa: BLE001
                print(f"[Memory Error] {exc}", flush=True)
                log("Memory", f"ERROR: {exc} (attempt {attempt}/{max_attempts})")
                if not has_vault_prompt_listener() or attempt >= max_attempts:
                    raise SystemExit(1) from exc
                password = request_vault_unlock(
                    f"Wrong Master Password / Recovery Key ({exc}). "
                    "Try again to unlock Dana's memory vault."
                )
                if not password:
                    raise SystemExit(1) from exc

    vault = SecureMemory(path=MEMORY_FILE)
    try:
        from dana.vault_service import _rpc
        import base64 as _b64

        resp = _rpc(
            {
                "op": "export_data_key",
                "session_token": state.vault_client.session_token,
            },
            timeout=5.0,
        )
        if resp.get("ok"):
            key = _b64.urlsafe_b64decode(resp["data_key_b64"].encode("ascii"))
            vault.unlock_with_data_key(key)
            vault.profile = dict(state.dana_profile)
    except Exception as exc:  # noqa: BLE001
        log("Memory", f"WARNING: local vault hydrate failed ({exc})")

    dana_vault = vault
    populate_vault_hot_cache(state.vault_client)
    log(
        "Memory",
        f"Vault unlocked via daemon session "
        f"(keys={len(state.dana_profile)}; token cached in RAM daemon).",
    )
    notify_vault_unlocked()
    return vault


def reset_dana_vault() -> None:
    """Authorize with master/recovery credential, then wipe the encrypted vault."""
    if not os.path.isfile(MEMORY_FILE):
        print("No vault found.", flush=True)
        log("Memory", "No vault found (--reset-vault).")
        raise SystemExit(0)

    from dana.tools.vault import VaultCredentialsMissing, _get_master_key

    try:
        password = _get_master_key(
            prompt=(
                "Enter Master Password (or Recovery Key) to authorize vault deletion: "
            )
        )
    except VaultCredentialsMissing as exc:
        print(f"[Security] ACCESS DENIED. {exc}", flush=True)
        raise SystemExit(1) from exc

    vault = SecureMemory()
    try:
        vault.unlock(password)
    except ValueError:
        print("[Security] ACCESS DENIED. Incorrect password.", flush=True)
        log("Memory", "ACCESS DENIED on --reset-vault (bad credential).")
        raise SystemExit(1)
    except Exception as exc:  # noqa: BLE001
        print("[Security] ACCESS DENIED. Incorrect password.", flush=True)
        log("Memory", f"ACCESS DENIED on --reset-vault ({exc}).")
        raise SystemExit(1)

    # Credential verified — safe to wipe.
    for path in (MEMORY_FILE, MEMORY_FILE + ".tmp"):
        try:
            os.remove(path)
            log("Memory", f"Deleted {path} (--reset-vault).")
        except FileNotFoundError:
            continue
        except OSError as exc:
            print(f"[Security] Could not delete {path}: {exc}", flush=True)
            raise SystemExit(1) from exc

    print("[Security] Vault successfully wiped.", flush=True)
    log("Memory", "Vault successfully wiped after authorized --reset-vault.")
def _trigger_dual_wake_event() -> None:
    global _dual_wake_router, _dual_wake_poller
    if _dual_wake_router is not None:
        try:
            _dual_wake_router.flush()
        except Exception:
            pass
    if _dual_wake_poller is not None:
        try:
            _dual_wake_poller.router.whisper_queue = asyncio.Queue()
            _dual_wake_poller.router.standard_queue = asyncio.Queue()
        except Exception:
            pass
    if not state.engine_engaged.is_set():
        return
    if not wakeword_armed.is_set():
        return
    log("WakeWord", "Dual-threshold wake trigger fired")
    is_recording.set()
    cooldown_until = time.monotonic() + WAKE_COOLDOWN_SEC
    try:
        if state._shared_wakeword_model is not None:
            state._shared_wakeword_model.reset()
    except Exception:
        pass


def _shutdown_agent_threads(*, join_timeout: float = 8.0) -> None:
    """Signal workers to stop and wait for AgentLoop (which joins Tracker/Wake/Audio/Conv)."""
    try:
        from dana.tools.registry import cleanup_ephemeral_tools

        cleaned = cleanup_ephemeral_tools(archive=True)
        if cleaned:
            log("Main", f"Ephemeral tool GC archived {len(cleaned)} tool(s): {cleaned}")
    except Exception as exc:  # noqa: BLE001
        log("Main", f"WARNING: ephemeral tool GC failed: {exc}")
    try:
        from dana.telemetry import set_system_status, stop_dashboard_thread, write_dashboard

        set_system_status("Restarting")
        write_dashboard()
        stop_dashboard_thread()
    except Exception:
        pass
    stop_event.set()
    try:
        speech_queue.put_nowait(None)
    except queue.Full:
        pass
    try:
        tts_interrupt_event.set()
        speech_idle.set()
    except Exception:
        pass
    thread = state.get_agent_loop_thread()
    if thread is not None and thread.is_alive():
        thread.join(timeout=join_timeout)
        if thread.is_alive():
            log("Main", "WARNING: AgentLoop did not exit within join timeout.")


def _install_signal_handlers(gui: "DanaGUI") -> None:
    """Ctrl+C / SIGTERM → destroy GUI on the Tk thread (avoids hanging workers)."""

    def _handler(signum: int, _frame: Any) -> None:
        log("Main", f"Signal {signum} received — shutting down.")
        try:
            from dana.tools.registry import cleanup_ephemeral_tools

            cleaned = cleanup_ephemeral_tools(archive=True)
            if cleaned:
                log(
                    "Main",
                    f"Ephemeral tool GC archived {len(cleaned)} tool(s): {cleaned}",
                )
        except Exception as exc:  # noqa: BLE001
            log("Main", f"WARNING: ephemeral tool GC failed: {exc}")
        stop_event.set()
        # Instantly dump TTS spool + stream sentence buffer (no shutdown spam).
        try:
            from dana.agentic import reset_stream_sentence_tts

            reset_stream_sentence_tts()
        except Exception:
            pass
        try:
            dropped = flush_tts_queue()
            if dropped:
                log("TTS", f"Shutdown flushed {dropped} pending spool item(s)")
        except Exception:
            pass
        try:
            tts_interrupt_event.set()
        except Exception:
            pass
        try:
            speech_queue.put_nowait(None)
        except queue.Full:
            pass
        try:
            gui.after(0, gui.destroy)
        except Exception:
            pass

    for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
        if sig is None:
            continue
        try:
            signal.signal(sig, _handler)
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Main - agent loop (background) + GUI main thread
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline Dana voice agent (YOLO eyes + Ollama brain + Whisper + OpenWakeWord).",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help=(
            "Allow Hugging Face / OpenWakeWord to download/cache model weights (online). "
            "Omit for offline HF loads where supported. Distil-Whisper STT still "
            "auto-falls back to download on a local cache miss."
        ),
    )
    parser.add_argument(
        "--reset-audio",
        action="store_true",
        help="Delete settings.json and re-run the interactive mic/speaker setup.",
    )
    parser.add_argument(
        "--reset-vault",
        action="store_true",
        help="Delete dana_memory.enc and exit so the next run creates a fresh vault.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Headless mode: skip CustomTkinter Live Trace / tray UI.",
    )
    return parser.parse_args()


def agent_loop(args: Optional[argparse.Namespace] = None) -> int:
    # Lazy, same precedent dana.core.agent_loop already uses for its own
    # reverse dependency on dana.core_agent (see module docstring) -- these
    # stay in core_agent.py because they're shared with that module too.
    from dana.core_agent import (
        clear_injected_question,
        input_txt_ingest_worker,
        set_injected_question,
        tracker_worker,
        wakeword_worker,
    )

    if args is None:
        args = parse_args()
    local_files_only = not args.download

    log("Main", "=== CAMGRASPER Dana voice agent ===")
    import torch

    try:
        torch.backends.mkldnn.enabled = True
    except Exception:
        pass
    log("Main", f"MKLDNN enabled: {torch.backends.mkldnn.enabled}")
    if local_files_only:
        log("Main", "Mode: OFFLINE HF loads (local_files_only=True)")
    else:
        log(
            "Main",
            "Mode: DOWNLOAD - will fetch Whisper/OWW weights if missing.",
        )

    # Headless CLI has no Dashboard ENGAGE button — arm the engine so
    # ``.trigger_ask`` injects are not left stranded in soft STANDBY.
    if getattr(args, "no_gui", False):
        os.environ.setdefault("DANA_NO_GUI", "1")
        os.environ.setdefault("DANA_HEADLESS", "1")
        state.engine_engaged.set()
        log("Main", "Headless mode: engine auto-ENGAGED for trigger injects")
        try:
            from dana.graph.meta_broker_process import start_headless_telemetry_drainer

            start_headless_telemetry_drainer()
            log("Main", "Headless Meta-Broker telemetry drainer started")
        except Exception as exc:  # noqa: BLE001
            log("Main", f"WARNING: headless telemetry drainer unavailable ({exc})")

    # Unlock encrypted long-term memory before loading models / audio threads.
    try:
        unlock_dana_memory()
    except SystemExit:
        stop_event.set()
        return 1

    if args.reset_audio:
        try:
            os.remove(SETTINGS_FILE)
            print("[Audio] Deleted settings.json — interactive setup will run.", flush=True)
            log("Audio", "Removed settings.json (--reset-audio).")
        except FileNotFoundError:
            print("[Audio] settings.json not found — setup will run anyway.", flush=True)
            log("Audio", "settings.json already absent (--reset-audio).")
        except OSError as exc:
            log("Audio", f"WARNING: could not remove settings.json: {exc}")
            print(f"[Audio] WARNING: could not remove settings.json: {exc}", flush=True)

    list_input_devices()
    list_output_devices()
    mic_id, speaker_id, mic_rate = load_audio_settings()
    # Autonomous audio — always System Default regardless of settings.json.
    mic_id, speaker_id = None, None
    log(
        "Audio",
        f"Audio pipeline: mic={mic_id} speaker={speaker_id} rate={mic_rate}",
    )
    state.AUDIO_INPUT_DEVICE = None
    state.AUDIO_OUTPUT_DEVICE = None
    state.AUDIO_INPUT_RATE = mic_rate

    # Acoustic-aware: keep OS default when live; bind physical fallback when quiet.
    state.AUDIO_INPUT_DEVICE, state.AUDIO_INPUT_RATE = ensure_live_mic(
        None,
        state.AUDIO_INPUT_RATE,
        allow_fallback=True,
    )
    if not _validate_mic_id(state.AUDIO_INPUT_DEVICE):
        log("Main", "Aborting: configured microphone is not usable.")
        stop_event.set()
        return 2
    if not _validate_speaker_id(state.AUDIO_OUTPUT_DEVICE):
        log("Main", "Aborting: configured speaker is not usable.")
        stop_event.set()
        return 2

    # Adaptive noise floor before wake-word arming (3s ambient baseline).
    calibrate_noise_floor(duration_sec=3.0)

    device = select_device()
    dtype = select_dtype(device)

    # Single PortAudio InputStream producer before wake/VAD consumers start.
    ensure_mic_ingest_thread()
    if not mic_ingest_ready.wait(timeout=8.0):
        log(
            "Main",
            "WARNING: MicIngest not ready after 8s — wake/VAD will wait on the queue",
        )

    global _dual_wake_router, _dual_wake_poller
    _dual_wake_router = None
    _dual_wake_poller = None

    try:
        router = AudioRouter(
            sample_rate=16000,
            chunk_size=1280,
            sample_width=2,
            wakeword_model_path=str(resolve_wakeword_onnx()),
        )
        poller = WakePoller(
            router=router,
            whisper_model=router.whisper_model,
            standard_model=router.standard_model,
            callback=_trigger_dual_wake_event,
            threshold=0.5,
        )
        router.start()
        poller.start()
        _dual_wake_router = router
        _dual_wake_poller = poller
        log("WakeWord", "Dual-threshold wake polling listener started")
    except Exception as exc:
        log("WakeWord", f"WARNING: dual-threshold wake listener unavailable ({exc})")

    threads = [
        threading.Thread(
            target=tracker_worker,
            name="Tracker",
            args=(device,),
            daemon=True,
        ),
        threading.Thread(target=wakeword_worker, name="WakeWord", daemon=True),
        threading.Thread(
            target=conversation_worker,
            name="Conversation",
            args=(local_files_only, device, dtype),
            daemon=True,
        ),
        threading.Thread(
            target=input_txt_ingest_worker,
            name="InputIngest",
            daemon=True,
        ),
    ]
    # Centralized TTSManager owns the Piper consumer thread (speech_queue).
    try:
        _tts_mgr_thread = _tts_manager.start(worker=tts_worker)
        if _tts_mgr_thread is not None:
            threads.append(_tts_mgr_thread)
            log("Main", f"Started thread: {_tts_mgr_thread.name}")
        else:
            fallback = threading.Thread(
                target=tts_worker, name="TTSWorker", daemon=True
            )
            threads.append(fallback)
    except Exception as _tts_start_exc:  # noqa: BLE001
        log("Main", f"WARNING: TTSManager start failed ({_tts_start_exc})")
        threads.append(
            threading.Thread(target=tts_worker, name="TTSWorker", daemon=True)
        )
    for t in threads:
        if t.is_alive():
            continue
        t.start()
        log("Main", f"Started thread: {t.name}")

    # Stage 7.2 — hardware panic button (F12 / DANA_KILL_HOTKEY).
    try:
        from dana.middleware.kill_switch import start_kill_switch_listener

        if start_kill_switch_listener():
            log("Main", "Kill switch listener armed (default hotkey F12)")
        else:
            log("Main", "Kill switch listener not started")
    except Exception as exc:  # noqa: BLE001
        log("Main", f"WARNING: kill switch listener failed: {exc}")

    # Phase 1/2 — adaptive compute governor (idle → research via input.txt queue).
    try:
        from dana.middleware.idle_monitor import start_idle_monitor

        if start_idle_monitor():
            log("Main", "IdleMonitor started (USER_ACTIVE/USER_AWAY compute governor)")
        else:
            log("Main", "IdleMonitor not started")
    except Exception as exc:  # noqa: BLE001
        log("Main", f"WARNING: IdleMonitor failed: {exc}")

    # Persist conversational mode so system jobs cannot steal it later.
    try:
        set_dana_mode(get_dana_mode(), as_voice=True)
        log("Main", f"Voice session mode seeded: {get_dana_mode()}")
    except Exception as exc:  # noqa: BLE001
        log("Main", f"WARNING: voice session mode seed failed: {exc}")

    # Sidekick supervisor — eyes (vision_poller) + hands (actuator_executor).
    try:
        from dana.middleware.sidekick_supervisor import start_sidekick_supervisor

        if start_sidekick_supervisor(as_thread=True):
            log("Main", "Sidekick supervisor started (vision_poller + actuator_executor)")
        else:
            log("Main", "Sidekick supervisor not started")
    except Exception as exc:  # noqa: BLE001
        log("Main", f"WARNING: sidekick supervisor failed: {exc}")

    log(
        "Main",
        "Dana is ready. Say 'Dana' to wake. | Tray Quit / Ctrl+C=quit",
    )
    try:
        from dana.telemetry import start_dashboard_thread

        start_dashboard_thread()
        log("Main", "Live telemetry dashboard started (CAMGRASPER/dashboard.md)")
    except Exception as exc:  # noqa: BLE001
        log("Main", f"WARNING: dashboard thread failed: {exc}")
    try:
        from dana.settings import (
            get_assistant_language,
            get_whisper_language,
            is_dynamic_tool_synthesis_enabled,
            load_dana_settings,
        )

        load_dana_settings(force_reload=True)
        log(
            "Main",
            f"Language lock: assistant={get_assistant_language()} "
            f"whisper={get_whisper_language()} (English-only release)",
        )
        log(
            "Main",
            f"Tool Forge: enable_dynamic_tool_synthesis="
            f"{is_dynamic_tool_synthesis_enabled()}",
        )
    except Exception as exc:  # noqa: BLE001
        log("Main", f"WARNING: language settings unread ({exc})")

    try:
        while not stop_event.is_set():
            # File trigger for automation / remote ask.
            # Empty file => start mic listening.
            # Non-empty file => inject that text as the user transcript (skip mic).
            if os.path.isfile(TRIGGER_FILE):
                try:
                    with open(TRIGGER_FILE, "r", encoding="utf-8-sig") as fh:
                        injected = fh.read().strip()
                except OSError:
                    injected = ""
                if get_ui_state() == "idle" and not is_recording.is_set():
                    if not state.engine_engaged.is_set():
                        # Soft STANDBY — leave trigger file for retry after ENGAGE.
                        pass
                    else:
                        try:
                            os.remove(TRIGGER_FILE)
                        except OSError:
                            pass
                        if injected:
                            log("Main", f"File trigger inject -> \"{injected}\"")
                            set_injected_question(injected)
                            is_recording.set()
                        else:
                            log("Main", "File trigger -> start listening")
                            clear_injected_question()
                            is_recording.set()
                # else: leave file in place until idle so automation retries cleanly

            # PortAudio / PaErrorCode from Audio thread → soft restart before freeze.
            if audio_hardware_fault.is_set():
                detail = consume_audio_hardware_fault()
                soft_recover_audio_hardware(detail)

            time.sleep(0.1)
    except KeyboardInterrupt:
        log("Main", "Quit requested (Ctrl+C).")
    finally:
        stop_event.set()
        try:
            camera_tool.release()
            screen_tool.release()
        except Exception:
            pass
        try:
            speech_queue.put_nowait(None)
        except queue.Full:
            pass
        for t in threads:
            t.join(timeout=5.0)
        log("Main", "Shutdown complete.")

    return 0


def main() -> int:
    """GUI owns the main thread; agent_loop + tray run in background daemons."""
    try:
        from dana.stdio_boot import ensure_stdio

        ensure_stdio()
    except Exception:
        pass

    # Cwd-independent asset paths (onnx, logs, yolov8n.pt, settings.json, …).
    chdir_project_root()

    # Workspace dirs: logs/tracker/execution_jail/custom_tools + legacy migrate.
    try:
        from dana.workspace import ensure_dana_workspace

        ensure_dana_workspace(migrate=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[Workspace] WARNING: ensure_dana_workspace failed: {exc}")

    # Startup: sweep RESOLVED/FAILED tickets into patch_ledger_archive.md.
    try:
        from dana.tools.archive_ledger import archive_completed_tickets

        archive_msg = archive_completed_tickets()
        print(f"[Ledger] {archive_msg}")
    except Exception as exc:  # noqa: BLE001
        print(f"[Ledger] WARNING: archive_completed_tickets failed: {exc}")

    # Best-effort UTF-8 stdout so non-ASCII logs do not crash worker threads.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    log_path = enable_runtime_file_logging()
    log("Main", f"PROJECT_ROOT={PROJECT_ROOT}")
    try:
        from dana.paths import DANA_WORKSPACE

        log("Main", f"DANA_WORKSPACE={DANA_WORKSPACE}")
    except Exception:
        pass

    # Dual-registry boot: Git-tracked general + Desktop custom (ephemeral).
    try:
        from dana.tools.registry import (
            load_custom_tools_from_disk,
            load_general_tools_from_disk,
        )

        loaded_general = load_general_tools_from_disk()
        if loaded_general:
            log("Main", f"Loaded general tools from disk: {loaded_general!r}")
        loaded_custom = load_custom_tools_from_disk()
        if loaded_custom:
            log(
                "Main",
                f"Loaded custom/ephemeral tools from disk: {loaded_custom!r}",
            )
    except Exception as exc:  # noqa: BLE001
        log("Main", f"WARNING: load_general/custom tools from disk failed: {exc}")

    enforce_singleton()
    args = parse_args()

    if args.reset_vault:
        reset_dana_vault()
        return 0

    log("Main", f"Runtime log -> {log_path}")
    log("Main", f"Conversation log (latest) -> {CONVERSATION_LOG_PATH}")

    # Headless: no CustomTkinter / tray — agent loop owns the process.
    if getattr(args, "no_gui", False):
        try:
            from dana.ui.trace_bus import get_trace_bus

            get_trace_bus().set_enabled(False)
        except Exception:  # noqa: BLE001
            pass
        os.environ.setdefault("DANA_NO_GUI", "1")
        os.environ.setdefault("DANA_HEADLESS", "1")
        try:
            from dana.graph.meta_broker_process import start_headless_telemetry_drainer

            start_headless_telemetry_drainer()
        except Exception:  # noqa: BLE001
            pass
        log("Main", "Headless mode (--no-gui): Live Trace UI disabled.")
        return agent_loop(args)

    # Stage 8.9.7 — GUI paints in STANDBY first; heavy agent loop deferred.
    gui = DanaGUI()
    state.set_gui_instance(gui)
    # Boot visible by default; honor open_window_on_startup (tray-only when False).
    try:
        from dana.settings import is_open_window_on_startup

        _show_on_boot = bool(is_open_window_on_startup())
    except Exception:  # noqa: BLE001
        _show_on_boot = True
    if _show_on_boot:
        try:
            gui.after(150, gui.show_window)
        except Exception:  # noqa: BLE001
            try:
                gui.show_window()
            except Exception:  # noqa: BLE001
                pass
    else:
        log(
            "Main",
            "open_window_on_startup=False — tray/orb only (dashboard hidden).",
        )
    try:
        emit_trace(
            "Boot",
            "completed",
            "Live Trace UI online (STANDBY — engine unlocked)",
            mode=get_dana_mode(),
        )
        emit_trace("STT", "idle", "STT: deferred until AgentLoop warm")
        emit_trace("Router", "idle", "Router: waiting for ENGAGE")
    except Exception:  # noqa: BLE001
        pass

    # Keep tray-close UX: X hides the window; Quit lives on the system tray.
    try:
        gui.protocol("WM_DELETE_WINDOW", gui._on_close_to_tray)
    except Exception:
        pass

    _install_signal_handlers(gui)

    # Tray owns its own daemon thread — never block the CTk mainloop.
    threading.Thread(
        target=run_system_tray,
        name="SystemTray",
        args=(gui,),
        daemon=True,
    ).start()

    def _boot_agent_loop() -> None:
        """Deferred background start so Standby chrome paints instantly."""
        if state.get_agent_loop_thread() is not None:
            return
        agent_loop_thread = threading.Thread(
            target=agent_loop,
            name="AgentLoop",
            kwargs={"args": args},
            daemon=True,
        )
        state.set_agent_loop_thread(agent_loop_thread)
        agent_loop_thread.start()
        try:
            emit_trace("STT", "active", "STT: Whisper pipeline arming")
            emit_trace("Router", "active", "Router: waiting for turn")
        except Exception:  # noqa: BLE001
            pass

    try:
        # ~350ms lets Tk draw ENGAGE/STANDBY before Whisper/tracker threads.
        gui.after(350, _boot_agent_loop)
    except Exception:  # noqa: BLE001
        _boot_agent_loop()

    try:
        gui.mainloop()
    except KeyboardInterrupt:
        log("Main", "Interrupted — shutting down.")
        stop_event.set()
        try:
            speech_queue.put_nowait(None)
        except queue.Full:
            pass
    finally:
        _shutdown_agent_threads(join_timeout=8.0)
        icon = state.get_tray_icon()
        if icon is not None:
            try:
                icon.stop()
            except Exception:
                pass
        try:
            unregister_transcript_listener(gui._on_transcript_event)
        except Exception:  # noqa: BLE001
            pass
        try:
            unregister_vault_prompt_listener(gui._on_vault_unlock_request)
        except Exception:  # noqa: BLE001
            pass
        try:
            unregister_spec_approval_listener(gui._on_spec_approval_requested)
        except Exception:  # noqa: BLE001
            pass
        try:
            unregister_dictation_sessions_listener(gui._on_dictation_sessions_changed)
        except Exception:  # noqa: BLE001
            pass
        state.set_gui_instance(None)
        state.set_agent_loop_thread(None)
        log("Main", "GUI closed.")

    return 0


if __name__ == "__main__":
    try:
        from dana.stdio_boot import ensure_stdio

        ensure_stdio()
    except Exception:
        pass
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        try:
            _shutdown_agent_threads(join_timeout=5.0)
        except Exception:
            pass
        sys.exit(130)
