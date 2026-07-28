"""In-RAM vault key daemon + local client (Option B — passwordless after first unlock).

Architecture:
  - Long-lived process binds 127.0.0.1:47475 only (loopback).
  - First authenticated unlock caches the Fernet *data key* exclusively in RAM.
  - Clients receive an ephemeral session token; subsequent ops use that token.
  - No plaintext password or recovery key is written to disk by this service.

Protocol: newline-delimited JSON request/response over TCP.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import secrets
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

try:
    from dana.stdio_boot import ensure_stdio

    ensure_stdio()
except Exception:
    pass

# Absolute Windows console kill-switch: CREATE_NO_WINDOW + STARTUPINFO hide +
# mutate python.exe → pythonw.exe (class patch so asyncio can still subclass).
if os.name == "nt":
    _original_popen = subprocess.Popen
    _pythonw_path = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")

    def _coerce_pythonw_cmd(cmd0: Any) -> Any:
        if not os.path.isfile(_pythonw_path):
            return cmd0
        if isinstance(cmd0, (list, tuple)) and cmd0:
            cmd = list(cmd0)
            head = str(cmd[0])
            if (
                head == sys.executable
                or os.path.normcase(head) == os.path.normcase(sys.executable)
                or os.path.basename(head).lower() == "python.exe"
            ):
                cmd[0] = _pythonw_path
            return cmd
        if isinstance(cmd0, str):
            if cmd0 == sys.executable or cmd0.startswith(sys.executable):
                return cmd0.replace(sys.executable, _pythonw_path, 1)
            if os.path.basename(cmd0.split(" ", 1)[0]).lower() == "python.exe":
                return cmd0.replace(cmd0.split(" ", 1)[0], _pythonw_path, 1)
        return cmd0

    class _PatchedPopen(_original_popen):  # type: ignore[valid-type,misc]
        def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            # 1. Force CREATE_NO_WINDOW
            kwargs["creationflags"] = int(kwargs.get("creationflags") or 0) | 0x08000000

            # 2. Force STARTUPINFO invisibility cloak
            if "startupinfo" not in kwargs or kwargs.get("startupinfo") is None:
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = int(getattr(subprocess, "SW_HIDE", 0))
                kwargs["startupinfo"] = startupinfo

            # 3. Intercept sys.executable / python.exe and mutate to pythonw.exe
            if args:
                first = _coerce_pythonw_cmd(args[0])
                args = (first,) + args[1:]
            if "args" in kwargs and kwargs["args"] is not None:
                kwargs["args"] = _coerce_pythonw_cmd(kwargs["args"])

            super().__init__(*args, **kwargs)

    subprocess.Popen = _PatchedPopen  # type: ignore[misc, assignment]

from dana.secure_memory import SecureMemory, default_vault_path
from dana.tools.ipc import VaultRequest, VaultResponse

VAULT_DAEMON_HOST = "127.0.0.1"

# Windows: hide console for detached daemon (CREATE_NO_WINDOW = 0x08000000).
_CREATE_NO_WINDOW = int(getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000))


def windows_no_window_creationflags(*extra: int) -> int:
    """Merge CREATE_NO_WINDOW with optional Windows process flags."""
    if os.name != "nt":
        return 0
    flags = _CREATE_NO_WINDOW
    for f in extra:
        flags |= int(f)
    return flags

# ---------------------------------------------------------------------------
# Advanced Memory Lifecycle — semantic compaction & temporal decay
# ---------------------------------------------------------------------------

# Keys that are high-frequency / low-value session noise — never persist.
_TRANSIENT_KEY_RE = re.compile(
    r"(?ix)^("
    r"session[_-]?"
    r"|tmp[_-]?"
    r"|temp[_-]?"
    r"|ephemeral[_-]?"
    r"|scratch[_-]?"
    r"|nonce[_-]?"
    r"|csrf[_-]?"
    r"|request[_-]?id"
    r"|trace[_-]?id"
    r"|span[_-]?id"
    r"|pid$"
    r"|hwnd$"
    r"|window[_-]?handle"
    r").*"
)

# Value patterns that look like transient session metadata even under stable keys.
_TRANSIENT_VALUE_RE = re.compile(
    r"(?ix)^("
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"  # UUID
    r"|sess[_-]?[0-9a-f]{16,}"
    r"|Bearer\s+\S+"
    r")$"
)

# Deterministic semantic synonym groups — same family ⇒ contradiction on write.
_SEMANTIC_FAMILIES: tuple[frozenset[str], ...] = (
    frozenset(
        {
            "current_project_directory",
            "project_directory",
            "project_dir",
            "cwd",
            "working_directory",
            "workspace_path",
        }
    ),
    frozenset({"remembered_ip", "saved_ip", "current_ip", "ip_address", "local_ip"}),
    frozenset({"user_name", "username", "preferred_name", "display_name"}),
    frozenset({"active_theme", "ui_theme", "theme"}),
    frozenset({"default_language", "reply_language", "preferred_language", "lang"}),
    frozenset({"timezone", "time_zone", "tz", "local_timezone"}),
    frozenset({"home_city", "city", "hometown", "location_city"}),
    frozenset({"home_region", "region", "state", "province", "home_state"}),
    frozenset({"family_partner", "partner", "spouse", "wife", "husband"}),
    frozenset({"family_children", "children", "kids", "son", "daughter"}),
    frozenset({"family_notes", "family", "family_info"}),
)

_META_KEY = "_donna_memory_meta"


def _normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (key or "").strip().lower()).strip("_")


def _family_for_key(key: str) -> frozenset[str] | None:
    nk = _normalize_key(key)
    for family in _SEMANTIC_FAMILIES:
        if nk in family or key in family:
            return family
    return None


def _unwrap_entry(raw: Any) -> tuple[Any, dict[str, Any]]:
    """Return (value, meta) for both legacy plain values and structured entries."""
    if isinstance(raw, dict) and "value" in raw and (
        "last_updated" in raw or "status" in raw or "meta" in raw
    ):
        meta = {
            "last_updated": float(raw.get("last_updated") or 0.0),
            "status": str(raw.get("status") or "active"),
        }
        return raw.get("value"), meta
    return raw, {"last_updated": 0.0, "status": "active"}


def _wrap_entry(value: Any, *, last_updated: float | None = None, status: str = "active") -> dict[str, Any]:
    return {
        "value": value,
        "last_updated": float(last_updated if last_updated is not None else time.time()),
        "status": status,
    }


def is_transient_memory(key: str, value: Any) -> bool:
    """True when the key/value pair is high-frequency temporal noise."""
    nk = _normalize_key(key)
    if _TRANSIENT_KEY_RE.match(nk) or _TRANSIENT_KEY_RE.match(key or ""):
        return True
    if isinstance(value, str) and _TRANSIENT_VALUE_RE.match(value.strip()):
        # Only prune UUID-like values when the key also smells transient / session.
        if any(tok in nk for tok in ("session", "tmp", "temp", "token", "nonce", "csrf")):
            return True
    return False


def consolidate_vault_memory(
    profile: dict[str, Any],
    key: str,
    value: Any,
    *,
    now: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Semantic compaction pipeline for a single write.

    Returns (updated_profile, report) where report describes overrides / prunes.
    """
    ts = float(now if now is not None else time.time())
    report: dict[str, Any] = {
        "key": key,
        "action": "write",
        "overridden": [],
        "deprecated": [],
        "pruned_transient": False,
        "skipped": False,
    }

    if not key or not str(key).strip():
        raise ValueError("Vault key cannot be empty")

    clean_key = str(key).strip()
    if is_transient_memory(clean_key, value):
        report["action"] = "prune_transient"
        report["pruned_transient"] = True
        report["skipped"] = True
        return dict(profile), report

    out = dict(profile)
    # Drop reserved meta bookkeeping from user-facing keys if present as value store.
    family = _family_for_key(clean_key)
    overridden: list[str] = []
    deprecated: list[str] = []

    # Direct key overwrite (same key, possibly different value).
    if clean_key in out:
        old_val, old_meta = _unwrap_entry(out[clean_key])
        if old_val != value:
            overridden.append(clean_key)
            report["previous_value_present"] = True

    # Semantic family contradictions → deprecate / remove sibling keys.
    if family is not None:
        for existing_key in list(out.keys()):
            if existing_key == clean_key or existing_key == _META_KEY:
                continue
            if existing_key.startswith("_"):
                continue
            enk = _normalize_key(existing_key)
            if enk in family or existing_key in family:
                old_val, _old_meta = _unwrap_entry(out[existing_key])
                if old_val != value:
                    deprecated.append(existing_key)
                    del out[existing_key]
                    overridden.append(existing_key)

    out[clean_key] = _wrap_entry(value, last_updated=ts, status="active")

    # Maintain a compact index of last_updated for retrieval prioritization.
    meta_index = out.get(_META_KEY)
    if not isinstance(meta_index, dict):
        meta_index = {}
    else:
        meta_index = dict(meta_index)
    for dead in deprecated:
        meta_index.pop(dead, None)
    meta_index[clean_key] = {"last_updated": ts, "status": "active"}
    # Sweep stale index entries that no longer exist in the profile.
    for mk in list(meta_index.keys()):
        if mk not in out and mk != clean_key:
            meta_index.pop(mk, None)
    out[_META_KEY] = meta_index

    report["overridden"] = overridden
    report["deprecated"] = deprecated
    report["last_updated"] = ts
    report["action"] = "override" if overridden else "write"
    return out, report


def memory_value(profile: dict[str, Any], key: str) -> Any:
    """Read a profile key, unwrapping structured entries for callers."""
    if key not in profile:
        raise KeyError(f"Vault key not found: {key}")
    raw = profile[key]
    value, meta = _unwrap_entry(raw)
    if meta.get("status") == "deprecated":
        raise KeyError(f"Vault key deprecated: {key}")
    return value


def profile_for_prompt(profile: dict[str, Any]) -> dict[str, Any]:
    """Flatten structured entries to plain values for LLM profile summaries."""
    flat: dict[str, Any] = {}
    for k, v in profile.items():
        if k == _META_KEY or str(k).startswith("_"):
            continue
        val, meta = _unwrap_entry(v)
        if meta.get("status") == "deprecated":
            continue
        flat[k] = val
    return flat


def _vault_port() -> int:
    raw = os.environ.get("DONNA_VAULT_PORT", "").strip()
    if raw.isdigit():
        return int(raw)
    return 47475


VAULT_DAEMON_PORT = _vault_port()
VAULT_DAEMON_TIMEOUT_SEC = 2.0
SESSION_TTL_SEC = 12 * 60 * 60  # 12 hours


@dataclass
class _Session:
    token: str
    created: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)


class VaultKeyDaemon:
    """Caches vault data-key in RAM; authenticates clients via session tokens."""

    def __init__(self, vault_path: str | None = None) -> None:
        self.vault_path = vault_path or default_vault_path()
        self._lock = threading.RLock()
        self._memory = SecureMemory(path=self.vault_path)
        self._unlocked = False
        self._sessions: dict[str, _Session] = {}
        self._stop = threading.Event()

    def _purge_expired(self) -> None:
        now = time.time()
        dead = [
            tok for tok, sess in self._sessions.items() if (now - sess.last_used) > SESSION_TTL_SEC
        ]
        for tok in dead:
            self._sessions.pop(tok, None)

    def _new_session(self) -> str:
        token = secrets.token_urlsafe(32)
        self._sessions[token] = _Session(token=token)
        return token

    def _require_session(self, token: str | None) -> _Session:
        self._purge_expired()
        if not token or token not in self._sessions:
            raise PermissionError("Invalid or expired session token.")
        sess = self._sessions[token]
        sess.last_used = time.time()
        return sess

    def handle(self, req: dict[str, Any]) -> dict[str, Any]:
        """Handle a raw dict request; prefer typed ``handle_request`` for new code."""
        return self.handle_request(VaultRequest.from_dict(req)).to_dict()

    def handle_request(self, request: VaultRequest) -> VaultResponse:
        """Typed IPC entrypoint (VaultRequest → VaultResponse)."""
        op = request.op
        try:
            if op == "ping":
                return VaultResponse(ok=True, unlocked=self._unlocked)

            if op == "status":
                with self._lock:
                    return VaultResponse(
                        ok=True,
                        unlocked=self._unlocked,
                        sessions=len(self._sessions),
                        vault_exists=os.path.isfile(self.vault_path),
                    )

            if op == "unlock":
                password = request.password or ""
                create = bool(request.create)
                recovery_key = request.recovery_key or ""
                with self._lock:
                    if self._unlocked and self._memory._data_key is not None:
                        token = self._new_session()
                        return VaultResponse(
                            ok=True,
                            session_token=token,
                            profile=dict(self._memory.profile),
                            created=False,
                            already_unlocked=True,
                        )
                    if create:
                        if not password or not recovery_key:
                            raise ValueError("create requires password and recovery_key")
                        profile = self._memory.create_new(password, recovery_key)
                        created = True
                    else:
                        if not password:
                            raise ValueError("unlock requires password")
                        if not os.path.isfile(self.vault_path):
                            raise FileNotFoundError(self.vault_path)
                        profile = self._memory.unlock(password)
                        created = False
                    self._unlocked = True
                    token = self._new_session()
                    return VaultResponse(
                        ok=True,
                        session_token=token,
                        profile=profile,
                        created=created,
                        already_unlocked=False,
                    )

            if op == "get_profile":
                with self._lock:
                    self._require_session(request.session_token)
                    if not self._unlocked:
                        raise RuntimeError("Vault locked in daemon.")
                    return VaultResponse(ok=True, profile=dict(self._memory.profile))

            if op == "save_profile":
                with self._lock:
                    self._require_session(request.session_token)
                    if not self._unlocked:
                        raise RuntimeError("Vault locked in daemon.")
                    profile_obj = request.profile
                    if not isinstance(profile_obj, dict):
                        raise ValueError("profile must be an object")
                    self._memory.profile = dict(profile_obj)
                    self._memory.save()
                    return VaultResponse(ok=True)

            if op == "write_memory":
                with self._lock:
                    self._require_session(request.session_token)
                    if not self._unlocked:
                        raise RuntimeError("Vault locked in daemon.")
                    key = (request.key or "").strip()
                    if not key:
                        raise ValueError("write_memory requires key")
                    updated, report = consolidate_vault_memory(
                        dict(self._memory.profile),
                        key,
                        request.value,
                    )
                    if not report.get("skipped"):
                        self._memory.profile = updated
                        self._memory.save()
                    return VaultResponse(
                        ok=True,
                        profile=dict(self._memory.profile),
                        extra={"consolidation": report},
                    )

            if op == "export_data_key":
                with self._lock:
                    self._require_session(request.session_token)
                    if not self._unlocked:
                        raise RuntimeError("Vault locked in daemon.")
                    key = self._memory.export_data_key()
                    return VaultResponse(
                        ok=True,
                        data_key_b64=base64.urlsafe_b64encode(key).decode("ascii"),
                    )

            if op == "lock":
                with self._lock:
                    self._require_session(request.session_token)
                    self._memory.lock()
                    self._unlocked = False
                    self._sessions.clear()
                    return VaultResponse(ok=True)

            # Emergency / lockdown kill-switch: purge RAM key without a session token.
            # Bound to loopback daemon only — never exposed off-host.
            if op in ("flush", "purge"):
                with self._lock:
                    self._memory.lock()
                    self._unlocked = False
                    self._sessions.clear()
                    return VaultResponse(ok=True, unlocked=False)

            return VaultResponse(ok=False, error=f"Unknown op: {op}")
        except Exception as exc:  # noqa: BLE001
            return VaultResponse(ok=False, error=str(exc))

    def stop(self) -> None:
        self._stop.set()

    def serve_forever(self) -> None:
        global VAULT_DAEMON_PORT
        VAULT_DAEMON_PORT = _vault_port()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((VAULT_DAEMON_HOST, VAULT_DAEMON_PORT))
        sock.listen(8)
        sock.settimeout(1.0)
        print(
            f"[VaultDaemon] Listening on {VAULT_DAEMON_HOST}:{VAULT_DAEMON_PORT} "
            f"(vault={os.path.basename(self.vault_path)})",
            flush=True,
        )
        while not self._stop.is_set():
            try:
                conn, _addr = sock.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            threading.Thread(target=self._serve_conn, args=(conn,), daemon=True).start()
        try:
            sock.close()
        except OSError:
            pass

    def _serve_conn(self, conn: socket.socket) -> None:
        try:
            conn.settimeout(10.0)
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buf += chunk
                if len(buf) > 1_000_000:
                    raise ValueError("Request too large")
            if not buf:
                return
            raw_req = json.loads(buf.decode("utf-8"))
            if not isinstance(raw_req, dict):
                raise ValueError("Request must be a JSON object")
            # Typed parse; never log password / recovery_key / tokens.
            request = VaultRequest.from_dict(raw_req)
            resp = self.handle_request(request)
            payload = (json.dumps(resp.to_dict(), ensure_ascii=False) + "\n").encode("utf-8")
            conn.sendall(payload)
        except Exception as exc:  # noqa: BLE001
            try:
                err = json.dumps({"ok": False, "error": str(exc)}) + "\n"
                conn.sendall(err.encode("utf-8"))
            except OSError:
                pass
        finally:
            try:
                conn.close()
            except OSError:
                pass


def _rpc(req: dict[str, Any], timeout: float = VAULT_DAEMON_TIMEOUT_SEC) -> dict[str, Any]:
    port = _vault_port()
    typed = VaultRequest.from_dict(req)
    raw = (json.dumps(typed.to_dict(), ensure_ascii=False) + "\n").encode("utf-8")
    with socket.create_connection((VAULT_DAEMON_HOST, port), timeout=timeout) as sock:
        sock.sendall(raw)
        buf = b""
        sock.settimeout(timeout)
        while not buf.endswith(b"\n"):
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf += chunk
    if not buf:
        raise ConnectionError("Vault daemon closed connection")
    resp_raw = json.loads(buf.decode("utf-8"))
    if not isinstance(resp_raw, dict):
        raise ValueError("Invalid daemon response")
    return VaultResponse.from_dict(resp_raw).to_dict()


def daemon_reachable() -> bool:
    try:
        resp = _rpc({"op": "ping"}, timeout=0.4)
        return bool(resp.get("ok"))
    except OSError:
        return False


def ensure_daemon_running(
    python_exe: str | None = None,
    vault_path: str | None = None,
) -> bool:
    """Start the vault daemon as a detached background process if needed."""
    if daemon_reachable():
        return True
    # Prefer windowless pythonw.exe on Windows so the daemon never allocates a console.
    if python_exe:
        exe = python_exe
    elif os.name == "nt":
        pythonw = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
        exe = pythonw if os.path.isfile(pythonw) else sys.executable
    else:
        exe = sys.executable
    creationflags = 0
    startupinfo = None
    if os.name == "nt":
        creationflags = windows_no_window_creationflags(
            int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)),
            int(getattr(subprocess, "DETACHED_PROCESS", 0x00000008)),
        )
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = int(getattr(subprocess, "SW_HIDE", 0))
    cmd = [exe, "-m", "dana.vault_service", "--serve"]
    if vault_path:
        cmd.extend(["--vault", vault_path])
    env = os.environ.copy()
    env["DONNA_VAULT_PORT"] = str(_vault_port())
    from dana.paths import PROJECT_ROOT

    popen_kwargs: dict[str, Any] = {
        "cwd": str(PROJECT_ROOT),
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "stdin": subprocess.DEVNULL,
        "close_fds": True,
        "env": env,
    }
    if creationflags:
        popen_kwargs["creationflags"] = creationflags
    if startupinfo is not None:
        popen_kwargs["startupinfo"] = startupinfo
    subprocess.Popen(cmd, **popen_kwargs)
    for _ in range(40):
        time.sleep(0.1)
        if daemon_reachable():
            return True
    return False


class VaultClient:
    """Agent-facing client: unlock once via daemon, then use session token."""

    def __init__(self) -> None:
        self.session_token: str | None = None
        self.profile: dict[str, Any] = {}
        self._last_consolidation: dict[str, Any] = {}

    def ensure_ready(self, vault_path: str | None = None) -> None:
        if not ensure_daemon_running(vault_path=vault_path):
            raise RuntimeError(
                "Could not start or reach Donna vault daemon on "
                f"{VAULT_DAEMON_HOST}:{_vault_port()}"
            )

    def status(self) -> dict[str, Any]:
        self.ensure_ready()
        return _rpc({"op": "status"})

    def unlock(
        self,
        password: str,
        *,
        create: bool = False,
        recovery_key: str = "",
    ) -> dict[str, Any]:
        self.ensure_ready()
        req = VaultRequest(
            op="unlock",
            password=password,
            create=create,
            recovery_key=recovery_key or None,
        )
        resp = VaultResponse.from_dict(_rpc(req.to_dict(), timeout=30.0))
        if not resp.ok:
            raise RuntimeError(resp.error or "unlock failed")
        self.session_token = str(resp.session_token)
        self.profile = dict(resp.profile or {})
        return self.profile

    def get_profile(self) -> dict[str, Any]:
        self.ensure_ready()
        if not self.session_token:
            raise RuntimeError("No vault session; unlock first.")
        resp = VaultResponse.from_dict(
            _rpc(
                VaultRequest(op="get_profile", session_token=self.session_token).to_dict(),
                timeout=5.0,
            )
        )
        if not resp.ok:
            raise RuntimeError(resp.error or "get_profile failed")
        self.profile = dict(resp.profile or {})
        return self.profile

    def save_profile(self, profile: dict[str, Any]) -> None:
        self.ensure_ready()
        if not self.session_token:
            raise RuntimeError("No vault session; unlock first.")
        resp = VaultResponse.from_dict(
            _rpc(
                VaultRequest(
                    op="save_profile",
                    session_token=self.session_token,
                    profile=profile,
                ).to_dict(),
                timeout=5.0,
            )
        )
        if not resp.ok:
            raise RuntimeError(resp.error or "save_profile failed")
        self.profile = dict(profile)

    def try_resume_session(self) -> bool:
        """If daemon already holds the key, mint a session without a password."""
        self.ensure_ready()
        try:
            ping = _rpc({"op": "ping"}, timeout=0.5)
        except OSError:
            return False
        if not ping.get("ok"):
            return False

        try:
            st = _rpc({"op": "status"}, timeout=0.5)
        except OSError:
            st = {}
        unlocked = bool(st.get("unlocked") or ping.get("unlocked"))

        # Always ask the daemon: empty password resumes only when RAM key is hot.
        try:
            resp = VaultResponse.from_dict(
                _rpc(VaultRequest(op="unlock", password="").to_dict(), timeout=5.0)
            )
        except Exception:
            return False

        if resp.ok and resp.session_token and (
            resp.already_unlocked is True or unlocked
        ):
            self.session_token = str(resp.session_token)
            self.profile = dict(resp.profile or {})
            return True
        return False

    def lock_vault(self) -> None:
        """Purge RAM Fernet key + sessions so the next boot requires a password."""
        self.ensure_ready()
        token = self.session_token
        try:
            if token:
                resp = VaultResponse.from_dict(
                    _rpc(
                        VaultRequest(op="lock", session_token=token).to_dict(),
                        timeout=5.0,
                    )
                )
                if not resp.ok:
                    # Fall through to flush if session already stale.
                    resp = VaultResponse.from_dict(_rpc({"op": "flush"}, timeout=5.0))
            else:
                resp = VaultResponse.from_dict(_rpc({"op": "flush"}, timeout=5.0))
            if not resp.ok:
                raise RuntimeError(resp.error or "lock_vault failed")
        finally:
            self.session_token = None
            self.profile = {}

    def purge_session(self) -> None:
        """Alias for lock_vault (lockdown kill-switch)."""
        self.lock_vault()

    def read_memory(self, key: str) -> Any:
        """Read one profile key via session-authenticated daemon RPC."""
        profile = self.get_profile()
        return memory_value(profile, key)

    def write_memory(self, key: str, value: Any) -> dict[str, Any]:
        """Write via semantic compaction engine, then persist through the daemon."""
        if not key or not str(key).strip():
            raise ValueError("Vault key cannot be empty")
        self.ensure_ready()
        if not self.session_token:
            raise RuntimeError("No vault session; unlock first.")
        resp = VaultResponse.from_dict(
            _rpc(
                VaultRequest(
                    op="write_memory",
                    session_token=self.session_token,
                    key=str(key).strip(),
                    value=value,
                ).to_dict(),
                timeout=5.0,
            )
        )
        if not resp.ok:
            err = resp.error or "write_memory failed"
            # Older daemons: fall back to local consolidate + save_profile.
            if "Unknown op" in err:
                profile = self.get_profile()
                updated, report = consolidate_vault_memory(profile, str(key).strip(), value)
                self._last_consolidation = report
                if report.get("skipped"):
                    return profile
                self.save_profile(updated)
                return updated
            raise RuntimeError(err)
        self.profile = dict(resp.profile or {})
        self._last_consolidation = dict(resp.extra.get("consolidation") or {})
        return self.profile

    @property
    def last_consolidation(self) -> dict[str, Any]:
        return dict(self._last_consolidation or {})


def main(argv: list[str] | None = None) -> int:
    try:
        from dana.stdio_boot import ensure_stdio

        ensure_stdio()
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="Donna vault key daemon")
    parser.add_argument("--serve", action="store_true", help="Run the vault key daemon")
    parser.add_argument("--vault", default=None, help="Path to donna_memory.enc")
    parser.add_argument("--port", type=int, default=None, help="Override loopback port")
    args = parser.parse_args(argv)
    if args.port is not None:
        os.environ["DONNA_VAULT_PORT"] = str(args.port)
    if args.serve:
        VaultKeyDaemon(vault_path=args.vault).serve_forever()
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
