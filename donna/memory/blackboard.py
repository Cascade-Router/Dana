"""Persistent Blackboard (SQLite) for Donna session memory & reasoning traces.

LangGraph state stays minimal (``session_id`` / ``current_agent`` / ``active_intent``).
Chat turns and chain-of-thought are filed here and pulled by ID when needed.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from donna.paths import DONNA_WORKSPACE

# Runtime artifact (not under source package) — do not edit donna/paths.py.
BLACKBOARD_DB_PATH: Path = DONNA_WORKSPACE / "memory" / "blackboard.db"

_LOCK = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    current_agent TEXT NOT NULL DEFAULT '',
    active_intent TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    meta_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);
CREATE INDEX IF NOT EXISTS idx_messages_session
    ON messages(session_id, id);
CREATE TABLE IF NOT EXISTS reasoning_traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'deepseek-r1',
    think_text TEXT NOT NULL,
    clean_text TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);
CREATE INDEX IF NOT EXISTS idx_reasoning_session
    ON reasoning_traces(session_id, id);
CREATE TABLE IF NOT EXISTS sensor_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    meta_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS action_queue (
    action_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL DEFAULT '',
    tool_name TEXT NOT NULL,
    arguments TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'pending',
    result TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    is_notified INTEGER NOT NULL DEFAULT 0,
    error_context TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_action_queue_status
    ON action_queue(status, action_id);
CREATE INDEX IF NOT EXISTS idx_action_queue_unread
    ON action_queue(is_notified, status);
CREATE TABLE IF NOT EXISTS persona_mixer (
    trait_name TEXT PRIMARY KEY,
    value INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS system_state (
    key TEXT PRIMARY KEY,
    value INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dictation_sessions (
    session_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    command_text TEXT NOT NULL DEFAULT '',
    visual_state_reference TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'recorded'
);
CREATE INDEX IF NOT EXISTS idx_dictation_sessions_ts
    ON dictation_sessions(timestamp DESC);
"""

# Stage 4.1 — continuous vision publisher key on the Blackboard middleware.
LATEST_VISUAL_CONTEXT_KEY = "latest_visual_context"

# Sidekick reliability — typed perception topics (never overload one key).
PERCEPTION_OBJECTS_KEY = "perception.objects"
PERCEPTION_OCR_KEY = "perception.ocr"
PERCEPTION_FRAME_REF_KEY = "perception.frame_ref"
SCHEMA_OBJECTS_V1 = "perception.objects.v1"
SCHEMA_OCR_V1 = "perception.ocr.v1"

# Middleware heartbeats + voice-mode isolation + actuator lease.
HEARTBEAT_VISION_KEY = "heartbeat.vision_poller"
HEARTBEAT_ACTUATOR_KEY = "heartbeat.actuator"
VOICE_SESSION_MODE_KEY = "voice_session_mode"
ACTUATOR_LEASE_KEY = "actuator_lease"
ACTUATOR_LEASE_TTL_S = 120.0
HEARTBEAT_STALE_S = 45.0

# Stage 6.4 / 8.5 — Receptionist + Behavior Mixer sliders (0–100).
PERSONA_MIXER_DEFAULTS: dict[str, int] = {
    "verbosity": 50,
    "humor": 20,
    "flirt": 10,
    "technical_depth": 80,
    "autonomy": 40,
    "creativity": 50,
}
_PERSONA_OVERRIDE_MARKER = "[SYSTEM OVERRIDE: Current Persona Settings (0-100)"

# Stage 8.5 — GUI / voice dictation routing latch.
DICTATION_MODE_KEY = "dictation_mode"

# Stage 4.2 — tools enqueued for the actuator daemon (not run on the LLM turn).
HEAVY_ACTUATOR_TOOLS: frozenset[str] = frozenset(
    {
        "draft_cursor_prompt",
        "file_editor",
        "shell_execute",
        "python_repl",
        "architect_new_tool",
        "web_search",
        "read_local_file",
        "dispatch_jason_supervisor",
        "type_stealth_text",
        "navigate_and_click",
        "press_key",
    }
)

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_action_queue_columns(conn: sqlite3.Connection) -> None:
    """Migrate older blackboard DBs that lack Stage 4.3 / 6.5 columns."""
    try:
        cols = {
            str(r[1])
            for r in conn.execute("PRAGMA table_info(action_queue)").fetchall()
        }
    except sqlite3.Error:
        return
    if not cols:
        return
    if "is_notified" not in cols:
        conn.execute(
            "ALTER TABLE action_queue ADD COLUMN is_notified "
            "INTEGER NOT NULL DEFAULT 0"
        )
    if "error_context" not in cols:
        conn.execute(
            "ALTER TABLE action_queue ADD COLUMN error_context "
            "TEXT NOT NULL DEFAULT ''"
        )


def _seed_persona_mixer(conn: sqlite3.Connection) -> None:
    """Insert default slider values without overwriting existing traits."""
    for name, value in PERSONA_MIXER_DEFAULTS.items():
        conn.execute(
            "INSERT OR IGNORE INTO persona_mixer (trait_name, value) VALUES (?, ?)",
            (name, int(value)),
        )


def _seed_system_state(conn: sqlite3.Connection) -> None:
    """Ensure Stage 7.3 flags exist (``is_typing`` defaults off)."""
    now = _utc_now()
    conn.execute(
        "INSERT OR IGNORE INTO system_state (key, value, updated_at) VALUES (?, ?, ?)",
        ("is_typing", 0, now),
    )


def init_blackboard(db_path: Path | str | None = None) -> Path:
    """Ensure parent dirs + tables exist; return resolved DB path."""
    path = Path(db_path or BLACKBOARD_DB_PATH).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        with sqlite3.connect(str(path), timeout=30.0) as conn:
            # WAL so the vision poller and main graph can concurrent-read/write.
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA busy_timeout=30000")
            except sqlite3.Error:
                pass
            conn.executescript(_SCHEMA)
            _ensure_action_queue_columns(conn)
            _seed_persona_mixer(conn)
            _seed_system_state(conn)
            conn.commit()
    return path


def _clamp_persona_value(value: int | float | str) -> int:
    try:
        n = int(round(float(value)))
    except (TypeError, ValueError):
        n = 0
    return max(0, min(100, n))


def get_persona_mixer(
    db_path: Path | str | None = None,
) -> dict[str, int]:
    """Return all persona traits (defaults filled for any missing keys)."""
    path = init_blackboard(db_path)
    out = dict(PERSONA_MIXER_DEFAULTS)
    with _LOCK:
        with sqlite3.connect(str(path), timeout=30.0) as conn:
            rows = conn.execute(
                "SELECT trait_name, value FROM persona_mixer"
            ).fetchall()
    for name, value in rows:
        key = str(name or "").strip()
        if key:
            out[key] = _clamp_persona_value(value)
    return out


def set_persona_trait(
    trait_name: str,
    value: int | float | str,
    *,
    db_path: Path | str | None = None,
) -> None:
    """Upsert one persona slider value (0–100)."""
    key = (trait_name or "").strip()
    if not key:
        raise ValueError("persona trait_name must be non-empty")
    path = init_blackboard(db_path)
    clamped = _clamp_persona_value(value)
    with _LOCK:
        with sqlite3.connect(str(path), timeout=30.0) as conn:
            conn.execute(
                "INSERT INTO persona_mixer (trait_name, value) VALUES (?, ?) "
                "ON CONFLICT(trait_name) DO UPDATE SET value = excluded.value",
                (key, clamped),
            )
            conn.commit()


def set_persona_mixer(
    values: dict[str, int | float | str],
    *,
    db_path: Path | str | None = None,
) -> dict[str, int]:
    """Upsert multiple persona traits; return the full mixer state."""
    path = init_blackboard(db_path)
    with _LOCK:
        with sqlite3.connect(str(path), timeout=30.0) as conn:
            for name, value in (values or {}).items():
                key = str(name or "").strip()
                if not key:
                    continue
                conn.execute(
                    "INSERT INTO persona_mixer (trait_name, value) VALUES (?, ?) "
                    "ON CONFLICT(trait_name) DO UPDATE SET value = excluded.value",
                    (key, _clamp_persona_value(value)),
                )
            conn.commit()
    return get_persona_mixer(path)


def format_persona_mixer_override(
    db_path: Path | str | None = None,
) -> str:
    """Receptionist / Jason system-prompt block for current slider state."""
    m = get_persona_mixer(db_path)
    v = m.get("verbosity", PERSONA_MIXER_DEFAULTS["verbosity"])
    h = m.get("humor", PERSONA_MIXER_DEFAULTS["humor"])
    f = m.get("flirt", PERSONA_MIXER_DEFAULTS["flirt"])
    t = m.get("technical_depth", PERSONA_MIXER_DEFAULTS["technical_depth"])
    a = m.get("autonomy", PERSONA_MIXER_DEFAULTS["autonomy"])
    c = m.get("creativity", PERSONA_MIXER_DEFAULTS["creativity"])
    return (
        f"{_PERSONA_OVERRIDE_MARKER} - Verbosity: {v}, Humor: {h}, "
        f"Flirt: {f}, Tech Depth: {t}, Autonomy: {a}, Creativity: {c}. "
        "Strictly adapt your tone and initiative to match these levels. "
        "Do not acknowledge these settings to the user.]"
    )


def is_dictation_mode(*, db_path: Path | str | None = None) -> bool:
    """True when GUI / router has forced dictation routing on."""
    row = get_sensor_state(DICTATION_MODE_KEY, db_path=db_path)
    if not row:
        return False
    return str(row.get("value") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
        "active",
    }


def set_dictation_mode(
    active: bool,
    *,
    db_path: Path | str | None = None,
) -> bool:
    """Force dictation routing on/off (GUI Start/Stop toggle)."""
    on = bool(active)
    set_sensor_state(
        DICTATION_MODE_KEY,
        "on" if on else "off",
        meta={"publisher": "dictation", "active": on},
        db_path=db_path,
    )
    return on


def record_dictation_session(
    command_text: str,
    *,
    visual_state_reference: str = "",
    status: str = "recorded",
    session_id: str | None = None,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """INSERT one Stage 8.5 dictation learning step; return the row dict."""
    path = init_blackboard(db_path)
    sid = (session_id or "").strip() or uuid.uuid4().hex[:16]
    now = _utc_now()
    cmd = (command_text or "").strip()
    visual = (visual_state_reference or "").strip()
    st = (status or "recorded").strip() or "recorded"
    with _LOCK:
        with sqlite3.connect(str(path), timeout=30.0) as conn:
            conn.execute(
                "INSERT INTO dictation_sessions "
                "(session_id, timestamp, command_text, visual_state_reference, status) "
                "VALUES (?, ?, ?, ?, ?)",
                (sid, now, cmd, visual, st),
            )
            conn.commit()
    return {
        "session_id": sid,
        "timestamp": now,
        "command_text": cmd,
        "visual_state_reference": visual,
        "status": st,
    }


def list_dictation_sessions(
    *,
    limit: int = 50,
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Newest-first dictation sessions for the GUI list."""
    path = init_blackboard(db_path)
    n = max(1, min(500, int(limit)))
    with _LOCK:
        with sqlite3.connect(str(path), timeout=30.0) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT session_id, timestamp, command_text, "
                "visual_state_reference, status "
                "FROM dictation_sessions ORDER BY timestamp DESC LIMIT ?",
                (n,),
            ).fetchall()
    return [
        {
            "session_id": str(r["session_id"] or ""),
            "timestamp": str(r["timestamp"] or ""),
            "command_text": str(r["command_text"] or ""),
            "visual_state_reference": str(r["visual_state_reference"] or ""),
            "status": str(r["status"] or ""),
        }
        for r in rows
    ]


def behavior_mixer_prompt_weights(
    db_path: Path | str | None = None,
) -> str:
    """Compact Jason/supervisor weight block from Behavior Mixer sliders."""
    m = get_persona_mixer(db_path)
    a = int(m.get("autonomy", PERSONA_MIXER_DEFAULTS["autonomy"]))
    v = int(m.get("verbosity", PERSONA_MIXER_DEFAULTS["verbosity"]))
    c = int(m.get("creativity", PERSONA_MIXER_DEFAULTS["creativity"]))
    t = int(m.get("technical_depth", PERSONA_MIXER_DEFAULTS["technical_depth"]))
    return (
        "[BEHAVIOR MIXER] Autonomy={a}/100 Verbosity={v}/100 "
        "Creativity={c}/100 TechDepth={t}/100. "
        "Higher Autonomy → more decisive operator actions with less confirmation; "
        "higher Verbosity → longer evaluations; higher Creativity → freer phrasing; "
        "higher TechDepth → denser technical criteria."
    ).format(a=a, v=v, c=c, t=t)


def append_persona_mixer_override(
    prompt: str,
    *,
    db_path: Path | str | None = None,
) -> str:
    """Append (or refresh) the persona override block at the end of ``prompt``."""
    base = (prompt or "").rstrip()
    # Drop a prior override so live slider changes always win.
    base = re.sub(
        r"\n*\[SYSTEM OVERRIDE: Current Persona Settings \(0-100\).*?"
        r"Do not acknowledge these settings to the user\.\]\s*",
        "",
        base,
        flags=re.DOTALL,
    ).rstrip()
    block = format_persona_mixer_override(db_path)
    if not base:
        return block
    return f"{base}\n\n{block}"


# Stage 7.3 — acoustic mute flag for Ghost Typist.
IS_TYPING_KEY = "is_typing"
_IS_TYPING_CACHE: bool = False
_IS_TYPING_CACHE_MONO: float = 0.0
_IS_TYPING_CACHE_TTL_S = 0.05  # cross-process: poll SQLite at most every 50ms


def get_system_state(
    key: str,
    *,
    db_path: Path | str | None = None,
    default: int = 0,
) -> int:
    """Read an integer flag from ``system_state``."""
    path = init_blackboard(db_path)
    k = (key or "").strip()
    if not k:
        return int(default)
    with _LOCK:
        with sqlite3.connect(str(path), timeout=30.0) as conn:
            row = conn.execute(
                "SELECT value FROM system_state WHERE key = ?",
                (k,),
            ).fetchone()
    if row is None:
        return int(default)
    try:
        return int(row[0])
    except (TypeError, ValueError):
        return int(default)


def set_system_state(
    key: str,
    value: int | bool,
    *,
    db_path: Path | str | None = None,
) -> None:
    """Upsert an integer flag in ``system_state``."""
    global _IS_TYPING_CACHE, _IS_TYPING_CACHE_MONO
    path = init_blackboard(db_path)
    k = (key or "").strip()
    if not k:
        raise ValueError("system_state key must be non-empty")
    now = _utc_now()
    if isinstance(value, bool):
        n = 1 if value else 0
    else:
        try:
            n = int(value)
        except (TypeError, ValueError):
            n = 0
    with _LOCK:
        with sqlite3.connect(str(path), timeout=30.0) as conn:
            conn.execute(
                "INSERT INTO system_state (key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
                "updated_at = excluded.updated_at",
                (k, n, now),
            )
            conn.commit()
    if k == IS_TYPING_KEY:
        _IS_TYPING_CACHE = n != 0
        _IS_TYPING_CACHE_MONO = time.monotonic()


def set_is_typing(
    typing: bool,
    *,
    db_path: Path | str | None = None,
) -> None:
    """Stage 7.3 — Ghost Typist acoustic mute latch."""
    set_system_state(IS_TYPING_KEY, 1 if typing else 0, db_path=db_path)


def is_typing(*, db_path: Path | str | None = None) -> bool:
    """True while Ghost Typist is injecting keystrokes (mute mic → Whisper).

    Cached briefly so the audio thread does not hit SQLite every frame; the
    actuator process writes the durable flag that the main VAD loop reads.
    """
    global _IS_TYPING_CACHE, _IS_TYPING_CACHE_MONO
    now = time.monotonic()
    if (now - _IS_TYPING_CACHE_MONO) < _IS_TYPING_CACHE_TTL_S and db_path is None:
        return _IS_TYPING_CACHE
    val = get_system_state(IS_TYPING_KEY, db_path=db_path, default=0) != 0
    _IS_TYPING_CACHE = val
    _IS_TYPING_CACHE_MONO = now
    return val


def ensure_session(
    session_id: str | None = None,
    *,
    current_agent: str = "",
    active_intent: str = "",
    db_path: Path | str | None = None,
) -> str:
    """Create or touch a session row; return ``session_id``."""
    path = init_blackboard(db_path)
    sid = (session_id or "").strip() or uuid.uuid4().hex[:16]
    now = _utc_now()
    agent = (current_agent or "").strip()
    intent = (active_intent or "").strip()
    with _LOCK:
        with sqlite3.connect(str(path), timeout=30.0) as conn:
            row = conn.execute(
                "SELECT session_id FROM sessions WHERE session_id = ?",
                (sid,),
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO sessions "
                    "(session_id, current_agent, active_intent, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (sid, agent, intent, now, now),
                )
            else:
                conn.execute(
                    "UPDATE sessions SET current_agent = COALESCE(NULLIF(?, ''), current_agent), "
                    "active_intent = COALESCE(NULLIF(?, ''), active_intent), "
                    "updated_at = ? WHERE session_id = ?",
                    (agent, intent, now, sid),
                )
            conn.commit()
    return sid


def set_session_meta(
    session_id: str,
    *,
    current_agent: str | None = None,
    active_intent: str | None = None,
    db_path: Path | str | None = None,
) -> None:
    """Update bureaucratic pointers for a session."""
    path = init_blackboard(db_path)
    sid = (session_id or "").strip()
    if not sid:
        return
    ensure_session(sid, db_path=path)
    now = _utc_now()
    with _LOCK:
        with sqlite3.connect(str(path), timeout=30.0) as conn:
            if current_agent is not None:
                conn.execute(
                    "UPDATE sessions SET current_agent = ?, updated_at = ? "
                    "WHERE session_id = ?",
                    ((current_agent or "").strip(), now, sid),
                )
            if active_intent is not None:
                conn.execute(
                    "UPDATE sessions SET active_intent = ?, updated_at = ? "
                    "WHERE session_id = ?",
                    ((active_intent or "").strip(), now, sid),
                )
            conn.commit()


def get_session_meta(
    session_id: str,
    *,
    db_path: Path | str | None = None,
) -> dict[str, str]:
    """Return ``{session_id, current_agent, active_intent}`` (empty strings if missing)."""
    path = init_blackboard(db_path)
    sid = (session_id or "").strip()
    if not sid:
        return {"session_id": "", "current_agent": "", "active_intent": ""}
    with _LOCK:
        with sqlite3.connect(str(path), timeout=30.0) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT session_id, current_agent, active_intent FROM sessions "
                "WHERE session_id = ?",
                (sid,),
            ).fetchone()
    if row is None:
        return {"session_id": sid, "current_agent": "", "active_intent": ""}
    return {
        "session_id": str(row["session_id"]),
        "current_agent": str(row["current_agent"] or ""),
        "active_intent": str(row["active_intent"] or ""),
    }


def append_message(
    session_id: str,
    role: str,
    content: str,
    *,
    meta: dict[str, Any] | None = None,
    db_path: Path | str | None = None,
) -> int:
    """Append one role/content turn; return row id."""
    path = init_blackboard(db_path)
    sid = ensure_session(session_id, db_path=path)
    role_s = (role or "user").strip().lower() or "user"
    body = content if content is not None else ""
    meta_json = json.dumps(meta or {}, ensure_ascii=False)
    with _LOCK:
        with sqlite3.connect(str(path), timeout=30.0) as conn:
            cur = conn.execute(
                "INSERT INTO messages (session_id, role, content, meta_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (sid, role_s, body, meta_json, _utc_now()),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (_utc_now(), sid),
            )
            conn.commit()
            return int(cur.lastrowid or 0)


def load_messages(
    session_id: str,
    *,
    limit: int | None = None,
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Return messages for a session (oldest first). ``limit`` keeps the newest N."""
    path = init_blackboard(db_path)
    sid = (session_id or "").strip()
    if not sid:
        return []
    with _LOCK:
        with sqlite3.connect(str(path), timeout=30.0) as conn:
            conn.row_factory = sqlite3.Row
            if limit is not None and int(limit) > 0:
                rows = conn.execute(
                    "SELECT id, role, content, meta_json, created_at FROM ("
                    "  SELECT id, role, content, meta_json, created_at "
                    "  FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?"
                    ") ORDER BY id ASC",
                    (sid, int(limit)),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, role, content, meta_json, created_at "
                    "FROM messages WHERE session_id = ? ORDER BY id ASC",
                    (sid,),
                ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        try:
            meta = json.loads(row["meta_json"] or "{}")
        except Exception:  # noqa: BLE001
            meta = {}
        out.append(
            {
                "id": int(row["id"]),
                "role": str(row["role"] or ""),
                "content": str(row["content"] or ""),
                "meta": meta if isinstance(meta, dict) else {},
                "created_at": str(row["created_at"] or ""),
            }
        )
    return out


def append_reasoning_trace(
    session_id: str,
    think_text: str,
    *,
    clean_text: str = "",
    source: str = "deepseek-r1",
    db_path: Path | str | None = None,
) -> int:
    """File a ``<think>`` / CoT block to the Blackboard; return row id."""
    path = init_blackboard(db_path)
    sid = ensure_session(session_id, db_path=path)
    with _LOCK:
        with sqlite3.connect(str(path), timeout=30.0) as conn:
            cur = conn.execute(
                "INSERT INTO reasoning_traces "
                "(session_id, source, think_text, clean_text, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    sid,
                    (source or "deepseek-r1").strip() or "deepseek-r1",
                    think_text or "",
                    clean_text or "",
                    _utc_now(),
                ),
            )
            conn.commit()
            return int(cur.lastrowid or 0)


def load_reasoning_traces(
    session_id: str,
    *,
    limit: int | None = None,
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Return reasoning traces for a session (oldest first)."""
    path = init_blackboard(db_path)
    sid = (session_id or "").strip()
    if not sid:
        return []
    with _LOCK:
        with sqlite3.connect(str(path), timeout=30.0) as conn:
            conn.row_factory = sqlite3.Row
            if limit is not None and int(limit) > 0:
                rows = conn.execute(
                    "SELECT id, source, think_text, clean_text, created_at FROM ("
                    "  SELECT id, source, think_text, clean_text, created_at "
                    "  FROM reasoning_traces WHERE session_id = ? "
                    "  ORDER BY id DESC LIMIT ?"
                    ") ORDER BY id ASC",
                    (sid, int(limit)),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, source, think_text, clean_text, created_at "
                    "FROM reasoning_traces WHERE session_id = ? ORDER BY id ASC",
                    (sid,),
                ).fetchall()
    return [dict(r) for r in rows]


def set_sensor_state(
    key: str,
    value: str,
    *,
    meta: dict[str, Any] | None = None,
    db_path: Path | str | None = None,
) -> None:
    """Upsert a continuous sensor value (ROS-style blackboard topic)."""
    path = init_blackboard(db_path)
    k = (key or "").strip()
    if not k:
        raise ValueError("sensor_state key must be non-empty")
    meta_json = json.dumps(meta or {}, ensure_ascii=False)
    now = _utc_now()
    with _LOCK:
        with sqlite3.connect(str(path), timeout=30.0) as conn:
            conn.execute(
                "INSERT INTO sensor_state (key, value, updated_at, meta_json) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET "
                "value = excluded.value, "
                "updated_at = excluded.updated_at, "
                "meta_json = excluded.meta_json",
                (k, value if value is not None else "", now, meta_json),
            )
            conn.commit()


def get_sensor_state(
    key: str,
    *,
    db_path: Path | str | None = None,
) -> dict[str, Any] | None:
    """Return ``{key, value, updated_at, meta}`` or ``None`` if missing."""
    path = init_blackboard(db_path)
    k = (key or "").strip()
    if not k:
        return None
    with _LOCK:
        with sqlite3.connect(str(path), timeout=30.0) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT key, value, updated_at, meta_json FROM sensor_state "
                "WHERE key = ?",
                (k,),
            ).fetchone()
    if row is None:
        return None
    try:
        meta = json.loads(row["meta_json"] or "{}")
    except Exception:  # noqa: BLE001
        meta = {}
    return {
        "key": str(row["key"] or ""),
        "value": str(row["value"] or ""),
        "updated_at": str(row["updated_at"] or ""),
        "meta": meta if isinstance(meta, dict) else {},
    }


def _new_frame_id() -> str:
    return uuid.uuid4().hex[:12]


def publish_perception_objects(
    text: str,
    *,
    producer: str,
    model: str = "yolov8",
    boxes: list[Any] | None = None,
    frame_id: str = "",
    latency_ms: float | None = None,
    skipped: bool = False,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Publish YOLO / object-detection envelope to ``perception.objects``.

    Also mirrors the human-readable ``text`` onto legacy
    ``latest_visual_context`` so Chat ambient reads stay compatible.
    """
    body = (text or "").strip()
    fid = (frame_id or "").strip() or _new_frame_id()
    meta: dict[str, Any] = {
        "schema": SCHEMA_OBJECTS_V1,
        "producer": (producer or "").strip() or "unknown",
        "model": (model or "").strip() or "yolov8",
        "frame_id": fid,
        "boxes": list(boxes or []),
        "skipped_inference": bool(skipped),
        "kind": "objects",
    }
    if latency_ms is not None:
        meta["latency_ms"] = float(latency_ms)
    set_sensor_state(PERCEPTION_OBJECTS_KEY, body, meta=meta, db_path=db_path)
    # Legacy mirror for Chat / older readers (objects only — never OCR).
    set_sensor_state(
        LATEST_VISUAL_CONTEXT_KEY,
        body,
        meta={**meta, "mirrored_from": PERCEPTION_OBJECTS_KEY},
        db_path=db_path,
    )
    return {"key": PERCEPTION_OBJECTS_KEY, "text": body, "meta": meta}


def publish_perception_ocr(
    text: str,
    *,
    producer: str,
    model: str = "florence-2",
    boxes: list[Any] | None = None,
    frame_id: str = "",
    latency_ms: float | None = None,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Publish Florence OCR envelope to ``perception.ocr`` (never YOLO prose)."""
    body = (text or "").strip()
    fid = (frame_id or "").strip() or _new_frame_id()
    meta: dict[str, Any] = {
        "schema": SCHEMA_OCR_V1,
        "producer": (producer or "").strip() or "unknown",
        "model": (model or "").strip() or "florence-2",
        "frame_id": fid,
        "boxes": list(boxes or []),
        "kind": "ocr",
    }
    if latency_ms is not None:
        meta["latency_ms"] = float(latency_ms)
    set_sensor_state(PERCEPTION_OCR_KEY, body, meta=meta, db_path=db_path)
    return {"key": PERCEPTION_OCR_KEY, "text": body, "meta": meta}


def publish_perception_frame_ref(
    path: str,
    *,
    producer: str = "debug_vision_live",
    frame_id: str = "",
    db_path: Path | str | None = None,
) -> None:
    """Optional debug frame path/hash — never treated as OCR/object text."""
    set_sensor_state(
        PERCEPTION_FRAME_REF_KEY,
        (path or "").strip(),
        meta={
            "schema": "perception.frame_ref.v1",
            "producer": (producer or "").strip() or "unknown",
            "frame_id": (frame_id or "").strip() or _new_frame_id(),
            "kind": "frame_ref",
        },
        db_path=db_path,
    )


def _parse_sensor_age_seconds(updated_at: str) -> float | None:
    text = (updated_at or "").strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds())
    except ValueError:
        return None


def get_perception_topic(
    key: str,
    *,
    expected_schema: str,
    db_path: Path | str | None = None,
) -> dict[str, Any] | None:
    """Return typed perception row or ``None`` when missing / wrong schema."""
    row = get_sensor_state(key, db_path=db_path)
    if not row:
        return None
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    schema = str(meta.get("schema") or "").strip()
    if schema != expected_schema:
        return None
    age = _parse_sensor_age_seconds(str(row.get("updated_at") or ""))
    return {
        "key": str(row.get("key") or key),
        "text": str(row.get("value") or "").strip(),
        "updated_at": str(row.get("updated_at") or ""),
        "age_seconds": age,
        "meta": meta,
        "schema": schema,
    }


def read_perception_objects(
    *,
    db_path: Path | str | None = None,
    max_age_s: float | None = None,
) -> dict[str, Any] | None:
    """Read ``perception.objects``; reject wrong schema / optional staleness."""
    row = get_perception_topic(
        PERCEPTION_OBJECTS_KEY,
        expected_schema=SCHEMA_OBJECTS_V1,
        db_path=db_path,
    )
    if row is None:
        return None
    if max_age_s is not None:
        age = row.get("age_seconds")
        if age is None or float(age) > float(max_age_s):
            return None
    return row


def read_perception_ocr(
    *,
    db_path: Path | str | None = None,
    max_age_s: float | None = None,
) -> dict[str, Any] | None:
    """Read ``perception.ocr``; reject YOLO / wrong schema / optional staleness."""
    row = get_perception_topic(
        PERCEPTION_OCR_KEY,
        expected_schema=SCHEMA_OCR_V1,
        db_path=db_path,
    )
    if row is None:
        return None
    text = str(row.get("text") or "")
    # Fail closed: never treat YOLO object prose as OCR corpus.
    if text.lstrip().startswith("[Vision Output]"):
        return None
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    if str(meta.get("kind") or "").strip() == "objects":
        return None
    if max_age_s is not None:
        age = row.get("age_seconds")
        if age is None or float(age) > float(max_age_s):
            return None
    return row


def read_perception_ocr_text(*, db_path: Path | str | None = None) -> str:
    """OCR corpus string, or empty when unavailable / schema-mismatched."""
    row = read_perception_ocr(db_path=db_path)
    if not row:
        return ""
    return str(row.get("text") or "").strip()


def read_visual_state(*, db_path: Path | str | None = None) -> str:
    """Chat ambient visual line — prefers typed objects, falls back to legacy key."""
    row = read_perception_objects(db_path=db_path)
    if row and str(row.get("text") or "").strip():
        return str(row.get("text") or "").strip()
    legacy = get_sensor_state(LATEST_VISUAL_CONTEXT_KEY, db_path=db_path)
    if not legacy:
        return ""
    return str(legacy.get("value") or "").strip()


def publish_heartbeat(
    key: str,
    *,
    publisher: str,
    pid: int | None = None,
    ok: bool = True,
    detail: str = "",
    db_path: Path | str | None = None,
) -> None:
    """Write a middleware heartbeat topic for the sidekick supervisor."""
    import os as _os

    meta = {
        "publisher": (publisher or "").strip() or "middleware",
        "pid": int(pid if pid is not None else _os.getpid()),
        "ok": bool(ok),
        "detail": (detail or "").strip(),
        "last_ok_at": _utc_now(),
    }
    set_sensor_state(key, "ok" if ok else "degraded", meta=meta, db_path=db_path)


def read_heartbeat(
    key: str,
    *,
    db_path: Path | str | None = None,
    stale_s: float = HEARTBEAT_STALE_S,
) -> dict[str, Any]:
    """Return heartbeat health ``{alive, age_seconds, meta, value}``."""
    row = get_sensor_state(key, db_path=db_path)
    if not row:
        return {"alive": False, "age_seconds": None, "meta": {}, "value": ""}
    age = _parse_sensor_age_seconds(str(row.get("updated_at") or ""))
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    alive = bool(meta.get("ok", True)) and age is not None and float(age) <= float(stale_s)
    return {
        "alive": alive,
        "age_seconds": age,
        "meta": meta,
        "value": str(row.get("value") or ""),
    }


def sidekick_health(*, db_path: Path | str | None = None) -> dict[str, Any]:
    """Aggregate eyes/hands health for Chat degraded-mode messaging."""
    vision = read_heartbeat(HEARTBEAT_VISION_KEY, db_path=db_path)
    actuator = read_heartbeat(HEARTBEAT_ACTUATOR_KEY, db_path=db_path)
    return {
        "vision_alive": bool(vision.get("alive")),
        "actuator_alive": bool(actuator.get("alive")),
        "vision": vision,
        "actuator": actuator,
        "degraded": (not vision.get("alive")) or (not actuator.get("alive")),
    }


def get_voice_session_mode(*, db_path: Path | str | None = None) -> str:
    """Durable conversational mode (not stolen by system job escalations)."""
    row = get_sensor_state(VOICE_SESSION_MODE_KEY, db_path=db_path)
    if not row:
        return "chat"
    mode = str(row.get("value") or "chat").strip().lower()
    if mode == "agent":
        return "developer"
    if mode in {"chat", "developer", "vision", "research"}:
        return mode
    return "chat"


def set_voice_session_mode(
    mode: str,
    *,
    db_path: Path | str | None = None,
) -> str:
    """Persist the user's conversational mode on the Blackboard."""
    raw = (mode or "").strip().lower()
    if raw == "agent":
        raw = "developer"
    if raw not in {"chat", "developer", "vision", "research"}:
        raw = "chat"
    set_sensor_state(
        VOICE_SESSION_MODE_KEY,
        raw,
        meta={"publisher": "mode_manager", "scope": "voice"},
        db_path=db_path,
    )
    return raw


def try_acquire_actuator_lease(
    owner: str,
    *,
    ttl_s: float = ACTUATOR_LEASE_TTL_S,
    db_path: Path | str | None = None,
) -> bool:
    """Single foreground desktop owner. Returns False if another lease is fresh."""
    path = init_blackboard(db_path)
    owner_s = (owner or "").strip() or "actuator"
    now = _utc_now()
    with _LOCK:
        with sqlite3.connect(str(path), timeout=30.0) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT value, updated_at, meta_json FROM sensor_state WHERE key = ?",
                (ACTUATOR_LEASE_KEY,),
            ).fetchone()
            if row is not None:
                age = _parse_sensor_age_seconds(str(row["updated_at"] or ""))
                try:
                    meta = json.loads(row["meta_json"] or "{}")
                except Exception:  # noqa: BLE001
                    meta = {}
                holder = str((meta or {}).get("owner") or row["value"] or "").strip()
                if (
                    holder
                    and holder != owner_s
                    and age is not None
                    and float(age) < float(ttl_s)
                ):
                    return False
            meta_json = json.dumps(
                {"owner": owner_s, "acquired_at": now, "ttl_s": float(ttl_s)},
                ensure_ascii=False,
            )
            conn.execute(
                "INSERT INTO sensor_state (key, value, updated_at, meta_json) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET "
                "value = excluded.value, "
                "updated_at = excluded.updated_at, "
                "meta_json = excluded.meta_json",
                (ACTUATOR_LEASE_KEY, owner_s, now, meta_json),
            )
            conn.commit()
    return True


def release_actuator_lease(
    owner: str,
    *,
    db_path: Path | str | None = None,
) -> None:
    """Release lease when ``owner`` still holds it."""
    path = init_blackboard(db_path)
    owner_s = (owner or "").strip() or "actuator"
    with _LOCK:
        with sqlite3.connect(str(path), timeout=30.0) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT meta_json FROM sensor_state WHERE key = ?",
                (ACTUATOR_LEASE_KEY,),
            ).fetchone()
            if row is None:
                return
            try:
                meta = json.loads(row["meta_json"] or "{}")
            except Exception:  # noqa: BLE001
                meta = {}
            holder = str((meta or {}).get("owner") or "").strip()
            if holder and holder != owner_s:
                return
            conn.execute(
                "DELETE FROM sensor_state WHERE key = ?",
                (ACTUATOR_LEASE_KEY,),
            )
            conn.commit()


def is_heavy_actuator_tool(tool_name: str) -> bool:
    """True when the graph should enqueue instead of executing inline."""
    return (tool_name or "").strip() in HEAVY_ACTUATOR_TOOLS


def enqueue_action(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    *,
    session_id: str = "",
    db_path: Path | str | None = None,
) -> int:
    """INSERT a pending action; return ``action_id``."""
    path = init_blackboard(db_path)
    name = (tool_name or "").strip()
    if not name:
        raise ValueError("tool_name must be non-empty")
    args_json = json.dumps(arguments or {}, ensure_ascii=False)
    now = _utc_now()
    with _LOCK:
        with sqlite3.connect(str(path), timeout=30.0) as conn:
            cur = conn.execute(
                "INSERT INTO action_queue "
                "(session_id, tool_name, arguments, status, result, "
                " created_at, updated_at) "
                "VALUES (?, ?, ?, 'pending', '', ?, ?)",
                (
                    (session_id or "").strip(),
                    name,
                    args_json,
                    now,
                    now,
                ),
            )
            conn.commit()
            return int(cur.lastrowid or 0)


def claim_next_pending(
    *,
    db_path: Path | str | None = None,
) -> dict[str, Any] | None:
    """Atomically claim the oldest pending row (``pending`` → ``running``)."""
    path = init_blackboard(db_path)
    now = _utc_now()
    with _LOCK:
        with sqlite3.connect(str(path), timeout=30.0) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT action_id, session_id, tool_name, arguments, status, "
                "result, created_at, updated_at, is_notified, error_context "
                "FROM action_queue WHERE status = 'pending' "
                "ORDER BY action_id ASC LIMIT 1"
            ).fetchone()
            if row is None:
                conn.rollback()
                return None
            aid = int(row["action_id"])
            cur = conn.execute(
                "UPDATE action_queue SET status = 'running', updated_at = ? "
                "WHERE action_id = ? AND status = 'pending'",
                (now, aid),
            )
            if int(cur.rowcount or 0) != 1:
                conn.rollback()
                return None
            conn.commit()
    return _action_row_to_dict(dict(row), status_override="running", updated_at=now)


def resolve_action(
    action_id: int,
    *,
    status: str,
    result: str = "",
    error_context: str = "",
    db_path: Path | str | None = None,
) -> None:
    """Mark an action ``completed``, ``failed``, or ``cancelled``."""
    path = init_blackboard(db_path)
    st = (status or "").strip().lower()
    if st not in {"completed", "failed", "cancelled"}:
        raise ValueError("status must be 'completed', 'failed', or 'cancelled'")
    now = _utc_now()
    err = (error_context or "") if st in {"failed", "cancelled"} else ""
    with _LOCK:
        with sqlite3.connect(str(path), timeout=30.0) as conn:
            conn.execute(
                "UPDATE action_queue SET status = ?, result = ?, "
                "error_context = ?, updated_at = ? WHERE action_id = ?",
                (st, result if result is not None else "", err, now, int(action_id)),
            )
            conn.commit()


def cancel_open_actions(
    *,
    db_path: Path | str | None = None,
    reason: str = "halted by GLOBAL_HALT_EVENT",
) -> int:
    """Stage 7.2 — mark pending/running/in_progress rows as ``cancelled``."""
    path = init_blackboard(db_path)
    now = _utc_now()
    reason_s = (reason or "halted by GLOBAL_HALT_EVENT").strip()
    with _LOCK:
        with sqlite3.connect(str(path), timeout=30.0) as conn:
            cur = conn.execute(
                "UPDATE action_queue SET status = 'cancelled', "
                "result = CASE WHEN TRIM(COALESCE(result, '')) = '' "
                "THEN 'cancelled by kill switch' ELSE result END, "
                "error_context = CASE WHEN TRIM(COALESCE(error_context, '')) = '' "
                "THEN ? ELSE error_context END, "
                "updated_at = ? "
                "WHERE status IN ('pending', 'running', 'in_progress')",
                (reason_s, now),
            )
            conn.commit()
            return int(cur.rowcount or 0)


def get_action(
    action_id: int,
    *,
    db_path: Path | str | None = None,
) -> dict[str, Any] | None:
    """Return one action_queue row as a dict, or ``None``."""
    path = init_blackboard(db_path)
    with _LOCK:
        with sqlite3.connect(str(path), timeout=30.0) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT action_id, session_id, tool_name, arguments, status, "
                "result, created_at, updated_at, is_notified, error_context "
                "FROM action_queue WHERE action_id = ?",
                (int(action_id),),
            ).fetchone()
    if row is None:
        return None
    return _action_row_to_dict(dict(row))


def get_and_clear_unread_notifications(
    session_id: str | None = None,
    *,
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Select completed/failed rows with ``is_notified=0``, then mark them read.

    When ``session_id`` is non-empty, returns that session's unread rows plus
    rows with an empty ``session_id`` (global). When empty, returns all unread.
    """
    path = init_blackboard(db_path)
    sid = (session_id or "").strip()
    with _LOCK:
        with sqlite3.connect(str(path), timeout=30.0) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE")
            if sid:
                rows = conn.execute(
                    "SELECT action_id, session_id, tool_name, arguments, status, "
                    "result, created_at, updated_at, is_notified, error_context "
                    "FROM action_queue "
                    "WHERE is_notified = 0 "
                    "AND status IN ('completed', 'failed') "
                    "AND (session_id = ? OR session_id = '') "
                    "ORDER BY action_id ASC",
                    (sid,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT action_id, session_id, tool_name, arguments, status, "
                    "result, created_at, updated_at, is_notified, error_context "
                    "FROM action_queue "
                    "WHERE is_notified = 0 "
                    "AND status IN ('completed', 'failed') "
                    "ORDER BY action_id ASC"
                ).fetchall()
            ids = [int(r["action_id"]) for r in rows]
            if ids:
                placeholders = ",".join("?" for _ in ids)
                conn.execute(
                    f"UPDATE action_queue SET is_notified = 1, updated_at = ? "
                    f"WHERE action_id IN ({placeholders})",
                    (_utc_now(), *ids),
                )
            conn.commit()
    return [_action_row_to_dict(dict(r)) for r in rows]


def format_background_system_alert(notifications: list[dict[str, Any]]) -> str:
    """Build the Chat Node piggyback block from unread action rows."""
    if not notifications:
        return ""
    bits: list[str] = []
    for row in notifications:
        tool = str(row.get("tool_name") or "task").strip() or "task"
        st = str(row.get("status") or "").strip().lower()
        if st == "completed":
            bits.append(
                f"The {tool} task the user requested earlier has successfully finished."
            )
        else:
            err = str(row.get("error_context") or row.get("result") or "").strip()
            if err:
                bits.append(
                    f"The {tool} task the user requested earlier has failed "
                    f"(Andon: {err[:180]})."
                )
            else:
                bits.append(
                    f"The {tool} task the user requested earlier has failed."
                )
    return "[BACKGROUND SYSTEM ALERT: " + " ".join(bits) + "]"


def _action_row_to_dict(
    row: dict[str, Any],
    *,
    status_override: str | None = None,
    updated_at: str | None = None,
) -> dict[str, Any]:
    try:
        args = json.loads(row.get("arguments") or "{}")
    except Exception:  # noqa: BLE001
        args = {}
    if not isinstance(args, dict):
        args = {}
    notified_raw = row.get("is_notified", 0)
    try:
        is_notified = bool(int(notified_raw))
    except (TypeError, ValueError):
        is_notified = bool(notified_raw)
    return {
        "action_id": int(row.get("action_id") or 0),
        "session_id": str(row.get("session_id") or ""),
        "tool_name": str(row.get("tool_name") or ""),
        "arguments": args,
        "status": status_override
        if status_override is not None
        else str(row.get("status") or ""),
        "result": str(row.get("result") or ""),
        "error_context": str(row.get("error_context") or ""),
        "created_at": str(row.get("created_at") or ""),
        "updated_at": updated_at
        if updated_at is not None
        else str(row.get("updated_at") or ""),
        "is_notified": is_notified,
    }
