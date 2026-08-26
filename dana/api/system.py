"""REST API for the frontend's Environment Viewer panel — a read-only,
masked window into the routing/provider configuration Dana is actually
running with.

``GET /api/system/env`` deliberately does NOT scan ``os.environ`` with a
keyword/substring filter (e.g. "contains API or KEY"): this machine's real
process environment can hold plenty of variables that have nothing to do
with Dana — other tools' tokens, cloud CLI credentials, whatever the user's
shell profile exports globally — and a substring match is one typo away
from picking one of those up too. Instead this is a fixed ALLOWLIST of the
exact variable names Dana's own code reads (see ``dana.core.model_provider``,
``dana.cascade_router``, and the Cascade-Router gateway's own ``.env``
surface) — a name not on this list is never reachable through this endpoint
no matter what it's called or what it contains.

Every sensitive entry is masked (first 3 chars + "***" + last 2, or "***"
outright if too short to leave anything genuinely hidden) INSIDE this
module, before the response dict is ever built — raw values never reach the
route handler's return statement, let alone serialization.
"""

from __future__ import annotations

import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from dana.paths import ENV_PATH

router = APIRouter()

# Holds a real credential — always masked.
_SENSITIVE_VARS = frozenset(
    {
        "GROQ_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",  # read by dana.core.model_provider's "anthropic" branch —
        # missing from this allowlist before was a real gap, not intentional.
        "LLM_GATEWAY_API_KEY",
        "HF_TOKEN_ALTEREGO",
        "HF_TOKEN_DEEPRESEARCH",
        "PUSHOVER_TOKEN",
        "PUSHOVER_USER",
        "SENDGRID_API_KEY",
        "SERPER_API_KEY",
    }
)

# Routing/model config only — no credential material, safe to show verbatim.
_NON_SENSITIVE_VARS = frozenset(
    {
        "CASCADE_PROVIDERS",
        "DANA_CLOUD_PRIMARY",
        "DANA_GROQ_MODEL",
        "DANA_LOCAL_MODEL",
        "DANA_VISION_MODEL",
        "DANA_REASONER_MODEL",
        "DANA_CASCADE_EXTERNAL",
        "DANA_CASCADE_MODEL",
        "DANA_FORCE_LOCAL",
        "ENABLE_ML",
        "OLLAMA_URL",
        "LLM_GATEWAY_URL",
        "GROQ_MODEL",
        "GROQ_HOST",
        "GROQ_BASE_PATH",
        "GROQ_PORT",
        "GROQ_CASCADE_CODE_MODEL",
        "GEMINI_MODEL",
        "OPENAI_MODEL",
    }
)

_ALLOWLIST = _SENSITIVE_VARS | _NON_SENSITIVE_VARS

_MASK_PREFIX_LEN = 3
_MASK_SUFFIX_LEN = 2


def _mask(value: str) -> str:
    """``gsk_abc...xyz12`` -> ``gsk***12``. Too short to leave anything
    hidden (<= prefix+suffix chars) -> ``"***"`` outright."""
    if len(value) <= _MASK_PREFIX_LEN + _MASK_SUFFIX_LEN:
        return "***"
    return f"{value[:_MASK_PREFIX_LEN]}***{value[-_MASK_SUFFIX_LEN:]}"


def _env_snapshot() -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for name in sorted(_ALLOWLIST):
        raw = os.environ.get(name)
        if not raw:
            continue
        snapshot[name] = _mask(raw) if name in _SENSITIVE_VARS else raw
    return snapshot


@router.get("/api/system/env")
def get_system_env() -> dict[str, dict[str, str]]:
    return {"env": _env_snapshot()}


class SaveEnvKeyRequest(BaseModel):
    key: str
    value: str


def _update_dotenv_line(path: Path, key: str, value: str) -> None:
    """Idempotent ``KEY=value`` upsert into a ``.env`` file: replaces an
    existing line for ``key`` in place (every other line, its own order,
    comments, all untouched) or appends a new line if ``key`` isn't present
    yet. Deliberately line-based rather than a full .env parser/writer
    library — this only ever touches one already-allowlisted key at a time.
    """
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    pattern = re.compile(rf"^{re.escape(key)}\s*=")
    new_line = f"{key}={value}"
    for i, line in enumerate(lines):
        if pattern.match(line):
            lines[i] = new_line
            break
    else:
        lines.append(new_line)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# One-shot "does the provider accept this key" probe, per recognized
# credential — a network failure/timeout is reported as unverified, NEVER
# conflated with "the key is invalid" (only an explicit 401/403 means that);
# a transient connectivity issue must never block saving a key the user
# typed correctly.
_VALIDATION_TIMEOUT_S = 6.0


def _validate_key(name: str, value: str) -> tuple[bool, str]:
    try:
        if name == "GROQ_API_KEY":
            req = urllib.request.Request(
                "https://api.groq.com/openai/v1/models", headers={"Authorization": f"Bearer {value}"}
            )
        elif name == "OPENAI_API_KEY":
            req = urllib.request.Request(
                "https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {value}"}
            )
        elif name == "ANTHROPIC_API_KEY":
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/models",
                headers={"x-api-key": value, "anthropic-version": "2023-06-01"},
            )
        elif name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            req = urllib.request.Request(
                f"https://generativelanguage.googleapis.com/v1beta/models?key={urllib.parse.quote(value)}"
            )
        else:
            return True, "saved (no live validation defined for this key)"
        with urllib.request.urlopen(req, timeout=_VALIDATION_TIMEOUT_S) as resp:
            if 200 <= resp.status < 300:
                return True, "valid — provider accepted the key"
            return False, f"provider returned HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return False, f"provider rejected the key (HTTP {exc.code})"
        return True, f"saved — validation inconclusive (HTTP {exc.code})"
    except Exception as exc:  # noqa: BLE001 — network/timeout is unverifiable, not "invalid"
        return True, f"saved — validation unreachable ({exc})"


@router.post("/api/system/env")
def save_system_env(body: SaveEnvKeyRequest) -> dict[str, Any]:
    """Saves one recognized credential to ``.env`` (upsert) and to THIS
    process's live ``os.environ`` (so a call made right after saving — no
    backend restart needed — already sees it), then runs a best-effort live
    probe against the actual provider. Only ever writes a key already on
    ``_SENSITIVE_VARS`` — this is the credential allowlist, not an arbitrary
    env-var setter.
    """
    key = body.key.strip()
    if key not in _SENSITIVE_VARS:
        raise HTTPException(status_code=400, detail=f"{key!r} is not a recognized/writable credential key")
    value = body.value.strip()
    if not value:
        raise HTTPException(status_code=400, detail="value must not be empty")

    _update_dotenv_line(ENV_PATH, key, value)
    os.environ[key] = value

    valid, detail = _validate_key(key, value)
    return {"ok": True, "key": key, "valid": valid, "detail": detail, "env": _env_snapshot()}


class ValidateEnvKeyRequest(BaseModel):
    key: str


@router.post("/api/system/env/validate")
def validate_system_env(body: ValidateEnvKeyRequest) -> dict[str, Any]:
    """Re-runs ``_validate_key`` against whatever value is CURRENTLY live in
    ``os.environ`` for ``key`` — powers the Environment panel's per-provider
    Valid/Invalid badges on open (and on manual re-check) without ever
    round-tripping a raw secret back to the frontend a second time, unlike
    ``POST /api/system/env`` which necessarily receives one to save it.
    """
    key = body.key.strip()
    if key not in _SENSITIVE_VARS:
        raise HTTPException(status_code=400, detail=f"{key!r} is not a recognized credential key")
    value = os.environ.get(key, "").strip()
    if not value:
        return {"ok": True, "key": key, "configured": False, "valid": False, "detail": "not configured"}
    valid, detail = _validate_key(key, value)
    return {"ok": True, "key": key, "configured": True, "valid": valid, "detail": detail}
