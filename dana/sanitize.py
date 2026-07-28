"""Redact secrets from log lines and ReAct traces before stdout / files."""

from __future__ import annotations

import re
from typing import Any

# High-sensitivity field names (vault / crypto / session).
_SENSITIVE_KEYS = (
    "password",
    "recovery_key",
    "session_token",
    "data_key",
    "data_key_b64",
    "master_password",
    "token",
)

_KV_SECRET_RE = re.compile(
    r"(?i)\b("
    + "|".join(re.escape(k) for k in _SENSITIVE_KEYS)
    + r")\b(\s*[:=]\s*)([^\s,;\]\}\"']+)",
)

_B64_LONG_RE = re.compile(r"\b[A-Za-z0-6+/_-]{48,}={0,2}\b")
_DATA_URI_RE = re.compile(r"data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=\s]+", re.I)
# Vault tool observations often embed key=value payloads.
_VAULT_VALUE_RE = re.compile(
    r"(?i)\b(write_vault_memory|read_vault_memory|saved|OK:)\b([^\n]{0,40}?)(=)([^\n,]{1,200})"
)


def sanitize_log_message(message: str) -> str:
    """Strip passwords, tokens, long base64 blobs, and vault value payloads from logs."""
    if not message:
        return message
    out = _DATA_URI_RE.sub("data:image/...;base64[REDACTED]", message)
    out = _KV_SECRET_RE.sub(r"\1\2***", out)
    out = _B64_LONG_RE.sub("[REDACTED_B64]", out)
    # Redact observation value tails for vault tools in agentic traces.
    out = re.sub(
        r"(?i)(write_vault_memory|read_vault_memory)([^\n]*?)(value=)([^\s,\]]+)",
        r"\1\2\3***",
        out,
    )
    out = re.sub(
        r"(?i)(OK: saved \w+=)('[^']*'|\"[^\"]*\"|\S+)",
        r"\1***",
        out,
    )
    out = re.sub(
        r"(?i)(OK: \w+=)('[^']*'|\"[^\"]*\"|\S+)",
        r"\1***",
        out,
    )
    return out


def sanitize_tool_trace(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a log-safe copy of a ReAct tool_trace."""
    safe: list[dict[str, Any]] = []
    for item in trace:
        row = dict(item)
        if "args" in row and isinstance(row["args"], dict):
            args = dict(row["args"])
            if "value" in args:
                args["value"] = "***"
            if "password" in args:
                args["password"] = "***"
            if "text" in args and row.get("tool") == "inject_keystrokes":
                raw = str(args["text"])
                args["text"] = f"[REDACTED chars={len(raw)}]"
            row["args"] = args
        if "observation" in row and isinstance(row["observation"], str):
            row["observation"] = sanitize_log_message(row["observation"])
        if "error" in row and isinstance(row["error"], str):
            row["error"] = sanitize_log_message(row["error"])
        safe.append(row)
    return safe
