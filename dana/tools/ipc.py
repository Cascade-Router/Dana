"""Typed vault daemon IPC payloads (newline-delimited JSON over loopback TCP)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class VaultRequest:
    """Client → daemon request. Sensitive fields must never be logged."""

    op: str
    session_token: str | None = None
    password: str | None = None
    recovery_key: str | None = None
    create: bool = False
    profile: dict[str, Any] | None = None
    key: str | None = None
    value: Any = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"op": self.op}
        if self.session_token is not None:
            payload["session_token"] = self.session_token
        if self.password is not None:
            payload["password"] = self.password
        if self.recovery_key is not None:
            payload["recovery_key"] = self.recovery_key
        if self.create:
            payload["create"] = True
        if self.profile is not None:
            payload["profile"] = self.profile
        if self.key is not None:
            payload["key"] = self.key
        if self.value is not None:
            payload["value"] = self.value
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VaultRequest:
        return cls(
            op=str(data.get("op") or ""),
            session_token=_opt_str(data.get("session_token")),
            password=_opt_str(data.get("password")),
            recovery_key=_opt_str(data.get("recovery_key")),
            create=bool(data.get("create")),
            profile=data.get("profile") if isinstance(data.get("profile"), dict) else None,
            key=_opt_str(data.get("key")),
            value=data.get("value"),
        )

    def redacted_dict(self) -> dict[str, Any]:
        """Safe for logs / diagnostics — secrets stripped."""
        out = self.to_dict()
        for secret in ("password", "recovery_key", "session_token"):
            if secret in out and out[secret]:
                out[secret] = "***"
        if "profile" in out:
            out["profile"] = {"_keys": list((out["profile"] or {}).keys())}
        if "value" in out:
            out["value"] = "***"
        return out


@dataclass
class VaultResponse:
    """Daemon → client response."""

    ok: bool
    error: str | None = None
    unlocked: bool | None = None
    sessions: int | None = None
    vault_exists: bool | None = None
    session_token: str | None = None
    profile: dict[str, Any] | None = None
    created: bool | None = None
    already_unlocked: bool | None = None
    data_key_b64: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"ok": self.ok}
        if self.error is not None:
            payload["error"] = self.error
        if self.unlocked is not None:
            payload["unlocked"] = self.unlocked
        if self.sessions is not None:
            payload["sessions"] = self.sessions
        if self.vault_exists is not None:
            payload["vault_exists"] = self.vault_exists
        if self.session_token is not None:
            payload["session_token"] = self.session_token
        if self.profile is not None:
            payload["profile"] = self.profile
        if self.created is not None:
            payload["created"] = self.created
        if self.already_unlocked is not None:
            payload["already_unlocked"] = self.already_unlocked
        if self.data_key_b64 is not None:
            payload["data_key_b64"] = self.data_key_b64
        payload.update(self.extra)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VaultResponse:
        known = {
            "ok",
            "error",
            "unlocked",
            "sessions",
            "vault_exists",
            "session_token",
            "profile",
            "created",
            "already_unlocked",
            "data_key_b64",
        }
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(
            ok=bool(data.get("ok")),
            error=_opt_str(data.get("error")),
            unlocked=_opt_bool(data.get("unlocked")),
            sessions=int(data["sessions"]) if data.get("sessions") is not None else None,
            vault_exists=_opt_bool(data.get("vault_exists")),
            session_token=_opt_str(data.get("session_token")),
            profile=data.get("profile") if isinstance(data.get("profile"), dict) else None,
            created=_opt_bool(data.get("created")),
            already_unlocked=_opt_bool(data.get("already_unlocked")),
            data_key_b64=_opt_str(data.get("data_key_b64")),
            extra=extra,
        )

    def redacted_dict(self) -> dict[str, Any]:
        out = self.to_dict()
        if out.get("session_token"):
            out["session_token"] = "***"
        if out.get("data_key_b64"):
            out["data_key_b64"] = "***"
        if isinstance(out.get("profile"), dict):
            out["profile"] = {"_keys": list(out["profile"].keys())}
        return out


def _opt_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _opt_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


# Silence unused import warning for asdict re-export convenience.
_ = asdict
