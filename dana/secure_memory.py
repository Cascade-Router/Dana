"""Encrypted long-term profile vault (PBKDF2 + Fernet).

Shared by the Dana agent and the in-RAM vault key daemon.
No plaintext credentials are written to disk.
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any, Optional

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

DEFAULT_VAULT_NAME = "dana_memory.enc"
DEFAULT_SALT = b"dana_secure_salt"
DEFAULT_PBKDF2_ITERATIONS = 390_000


def default_vault_path() -> str:
    from dana.paths import VAULT_PATH

    return str(VAULT_PATH)


class SecureMemory:
    """Password-gated encrypted JSON profile stored in dana_memory.enc.

    New vaults use a dual-slot format so either the master password or a
    backup recovery key can unlock the same ciphertext. Legacy single-Fernet
    files still unlock.
    """

    VAULT_VERSION = 2

    def __init__(
        self,
        path: Optional[str] = None,
        salt: bytes = DEFAULT_SALT,
        iterations: int = DEFAULT_PBKDF2_ITERATIONS,
    ) -> None:
        self.path = path or default_vault_path()
        self.salt = salt
        self.iterations = iterations
        self._fernet: Optional[Fernet] = None
        self._data_key: Optional[bytes] = None  # RAM-only session material
        self._vault_meta: Optional[dict[str, Any]] = None
        self.profile: dict[str, Any] = {}

    def _derive_fernet_key(self, password: str) -> bytes:
        if not password:
            raise ValueError("Master password / recovery key cannot be empty.")
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=self.iterations,
        )
        digest = kdf.derive(password.encode("utf-8"))
        return base64.urlsafe_b64encode(digest)

    def _password_fernet(self, password: str) -> Fernet:
        return Fernet(self._derive_fernet_key(password))

    def create_new(self, master_password: str, recovery_key: str) -> dict[str, Any]:
        """Create an empty dual-slot vault unlockable by master or recovery key."""
        data_key = Fernet.generate_key()
        self._data_key = data_key
        self._fernet = Fernet(data_key)
        self.profile = {}
        master_wrap = self._password_fernet(master_password).encrypt(data_key)
        recovery_wrap = self._password_fernet(recovery_key).encrypt(data_key)
        self._vault_meta = {
            "version": self.VAULT_VERSION,
            "keys": {
                "master": base64.urlsafe_b64encode(master_wrap).decode("ascii"),
                "recovery": base64.urlsafe_b64encode(recovery_wrap).decode("ascii"),
            },
            "ciphertext": "",
        }
        self.save()
        return self.profile

    def unlock(self, password: str) -> dict[str, Any]:
        """Decrypt an existing vault with master password or recovery key."""
        if not os.path.isfile(self.path):
            raise FileNotFoundError(self.path)

        with open(self.path, "rb") as fh:
            raw = fh.read()

        try:
            meta = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            meta = None

        if isinstance(meta, dict) and int(meta.get("version", 0)) == self.VAULT_VERSION:
            return self._unlock_v2(password, meta)
        return self._unlock_legacy(password, raw)

    def unlock_with_data_key(self, data_key: bytes, meta: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Rehydrate an unlocked vault from a RAM-cached data key (daemon path)."""
        if not os.path.isfile(self.path):
            raise FileNotFoundError(self.path)
        with open(self.path, "rb") as fh:
            raw = fh.read()
        try:
            disk_meta = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            disk_meta = None

        self._data_key = data_key
        self._fernet = Fernet(data_key)
        if isinstance(disk_meta, dict) and int(disk_meta.get("version", 0)) == self.VAULT_VERSION:
            cipher_b64 = disk_meta.get("ciphertext") or ""
            cipher = base64.urlsafe_b64decode(cipher_b64.encode("ascii"))
            plain = self._fernet.decrypt(cipher)
            data = json.loads(plain.decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError("Vault payload is not a JSON object.")
            self.profile = data
            self._vault_meta = meta or disk_meta
            return self.profile

        plain = self._fernet.decrypt(raw)
        data = json.loads(plain.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Vault payload is not a JSON object.")
        self.profile = data
        self._vault_meta = None
        return self.profile

    def _unlock_v2(self, password: str, meta: dict[str, Any]) -> dict[str, Any]:
        keys = meta.get("keys") or {}
        wrapper = self._password_fernet(password)
        data_key: Optional[bytes] = None
        for slot_name in ("master", "recovery"):
            wrapped_b64 = keys.get(slot_name)
            if not wrapped_b64:
                continue
            try:
                wrapped = base64.urlsafe_b64decode(wrapped_b64.encode("ascii"))
                data_key = wrapper.decrypt(wrapped)
                break
            except InvalidToken:
                continue
        if data_key is None:
            raise ValueError(
                "Wrong master password / recovery key (or vault corrupted). Decryption failed."
            )

        try:
            cipher_b64 = meta.get("ciphertext") or ""
            cipher = base64.urlsafe_b64decode(cipher_b64.encode("ascii"))
            self._data_key = data_key
            self._fernet = Fernet(data_key)
            plain = self._fernet.decrypt(cipher)
            data = json.loads(plain.decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError("Vault payload is not a JSON object.")
            self.profile = data
            self._vault_meta = meta
            return self.profile
        except InvalidToken as exc:
            raise ValueError("Vault data key worked but ciphertext is invalid.") from exc
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            raise ValueError(f"Vault decrypt succeeded but JSON is invalid: {exc}") from exc

    def _unlock_legacy(self, password: str, raw: bytes) -> dict[str, Any]:
        try:
            key = self._derive_fernet_key(password)
            self._data_key = key
            self._fernet = Fernet(key)
            plain = self._fernet.decrypt(raw)
            data = json.loads(plain.decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError("Vault payload is not a JSON object.")
            self.profile = data
            self._vault_meta = None
            return self.profile
        except InvalidToken as exc:
            raise ValueError(
                "Wrong master password / recovery key (or vault corrupted). Decryption failed."
            ) from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"Vault decrypt succeeded but JSON is invalid: {exc}") from exc

    def export_data_key(self) -> bytes:
        if self._data_key is None:
            raise RuntimeError("Vault is locked; no data key in RAM.")
        return self._data_key

    def save(self) -> None:
        if self._fernet is None:
            raise RuntimeError("Vault is locked; call unlock()/create_new() first.")
        payload = json.dumps(self.profile, ensure_ascii=False, indent=2).encode("utf-8")
        token = self._fernet.encrypt(payload)

        if self._vault_meta is not None:
            envelope = dict(self._vault_meta)
            envelope["version"] = self.VAULT_VERSION
            envelope["ciphertext"] = base64.urlsafe_b64encode(token).decode("ascii")
            blob = json.dumps(envelope, indent=2).encode("utf-8")
            self._vault_meta = envelope
        else:
            blob = token

        tmp = self.path + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(blob)
        os.replace(tmp, self.path)

    def prompt_summary(self) -> str:
        if not self.profile:
            return "No long-term user profile stored yet."
        try:
            return json.dumps(self.profile, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            return str(self.profile)

    def lock(self) -> None:
        """Wipe RAM key material (overwrite bytes before drop)."""
        key = self._data_key
        self._fernet = None
        self._data_key = None
        self.profile = {}
        self._vault_meta = None
        if key is not None:
            try:
                buf = bytearray(key)
                for i in range(len(buf)):
                    buf[i] = 0
            except Exception:  # noqa: BLE001
                pass
            del key
