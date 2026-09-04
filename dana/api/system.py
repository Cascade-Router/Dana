"""REST API for the frontend's Environment Viewer panel — a read-only,
masked window into the routing/provider configuration Dana is actually
running with.

``GET /api/system/env`` deliberately does NOT scan ``os.environ`` with a
keyword/substring filter (e.g. "contains API or KEY"): this machine's real
process environment can hold plenty of variables that have nothing to do
with Dana — other tools' tokens, cloud CLI credentials, whatever the user's
shell profile exports globally — and a substring match is one typo away
from picking one of those up too. Instead this is a fixed ALLOWLIST of the
exact variable names Dana's own code reads (see ``dana.core.model_provider``
and ``dana.cascade_router``) — a name not on this list is never reachable
through this endpoint no matter what it's called or what it contains.

Every sensitive entry is masked (first 3 chars + "***" + last 2, or "***"
outright if too short to leave anything genuinely hidden) INSIDE this
module, before the response dict is ever built — raw values never reach the
route handler's return statement, let alone serialization.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from dotenv import set_key
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
        "OPENROUTER_API_KEY",  # read by dana.core.model_provider._resolve_openai_endpoint's
        # "openrouter" branch — the Settings modal's Cloud Provider Manager
        # needs this alongside OpenAI/Gemini to cover all three of its
        # provider choices.
        "HF_TOKEN_ALTEREGO",
        "HF_TOKEN_DEEPRESEARCH",
        "PUSHOVER_TOKEN",
        "PUSHOVER_USER",
        "SENDGRID_API_KEY",
        "SERPER_API_KEY",
    }
)

# Routing/model config only — no credential material, safe to show verbatim.
# Also the Settings modal's Model Priority Manager's writable non-credential
# fields (DANA_CLOUD_PROVIDER, DANA_LOCAL_MODEL, DANA_OPENROUTER_MODEL,
# DANA_GEMINI_MODEL) — see save_system_env below, which now accepts anything
# in _ALLOWLIST, not just _SENSITIVE_VARS.
_NON_SENSITIVE_VARS = frozenset(
    {
        "DANA_CLOUD_PRIMARY",
        "DANA_CLOUD_PROVIDER",
        "DANA_GROQ_MODEL",
        "DANA_LOCAL_MODEL",
        "DANA_OPENROUTER_MODEL",
        "DANA_GEMINI_MODEL",
        "DANA_VISION_MODEL",
        "DANA_REASONER_MODEL",
        "DANA_CASCADE_EXTERNAL",
        "DANA_CASCADE_MODEL",
        "DANA_FORCE_LOCAL",
        "OLLAMA_URL",
        "GROQ_MODEL",
        "GROQ_HOST",
        "GROQ_BASE_PATH",
        "GROQ_PORT",
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
    """Idempotent ``KEY=value`` upsert into a ``.env`` file via python-dotenv's
    own ``set_key`` — replaces an existing line for ``key`` in place (every
    other line, its own order, comments, all untouched) or appends a new
    line if ``key`` isn't present yet. ``quote_mode="never"`` keeps the
    written value bare (matching every existing unquoted line in this
    project's ``.env``) rather than ``set_key``'s own default of always
    wrapping values in double quotes. ``touch`` first — ``set_key`` raises
    if the file doesn't exist yet, unlike the hand-rolled line editor this
    replaced.
    """
    path.touch(exist_ok=True)
    set_key(str(path), key, value, quote_mode="never")


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
        elif name == "OPENROUTER_API_KEY":
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/auth/key", headers={"Authorization": f"Bearer {value}"}
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
    """Saves one recognized env var to ``.env`` (upsert, via python-dotenv's
    ``set_key`` — see ``_update_dotenv_line``) and to THIS process's live
    ``os.environ`` — so a call made right after saving already sees it, with
    no backend restart needed, since every reader in this codebase
    (``dana.core.model_provider``'s ``local_model_name``/``cloud_provider_name``/
    etc.) re-reads ``os.environ`` fresh on every call rather than caching it.
    Only ever writes a key already on ``_ALLOWLIST`` — this is a fixed
    allowlist, not an arbitrary env-var setter. A live probe against the
    actual provider only runs for a recognized CREDENTIAL (``_SENSITIVE_VARS``)
    — the Settings modal's Cloud Provider Manager also uses this same
    endpoint to write plain routing config (``DANA_CLOUD_PROVIDER``,
    ``DANA_LOCAL_MODEL``, ...), which ``_validate_key`` has no provider
    endpoint to probe.
    """
    key = body.key.strip()
    if key not in _ALLOWLIST:
        raise HTTPException(status_code=400, detail=f"{key!r} is not a recognized/writable setting")
    value = body.value.strip()
    if not value:
        raise HTTPException(status_code=400, detail="value must not be empty")

    _update_dotenv_line(ENV_PATH, key, value)
    os.environ[key] = value

    if key in _SENSITIVE_VARS:
        valid, detail = _validate_key(key, value)
    else:
        valid, detail = True, "saved"
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
