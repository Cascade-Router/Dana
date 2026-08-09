"""Vault master-key resolution: env → OS keyring → interactive prompt."""

from __future__ import annotations

import getpass
import logging
import os
import sys

KEYRING_SERVICE = "dana_agent"
KEYRING_USERNAME = "vault_master_key"
ENV_VAR = "DANA_VAULT_KEY"

_log = logging.getLogger("dana.vault")

DEFAULT_PROMPT = "Enter Master Password (or pasted Recovery Key) to unlock Dana: "


class VaultCredentialsMissing(RuntimeError):
    """No vault master key available without blocking on stdin."""


def _get_master_key(*, prompt: str | None = None) -> str:
    """Resolve the vault unlock credential without hanging headless boots.

    Precedence:
      1. ``DANA_VAULT_KEY`` environment variable
      2. OS keyring (``dana_agent`` / ``vault_master_key``)
      3. Interactive ``getpass`` when stdin is a TTY (then persist to keyring)
      4. Raise ``VaultCredentialsMissing`` when headless and no credential exists
    """
    env_key = os.environ.get(ENV_VAR, "").strip()
    if env_key:
        return env_key

    try:
        import keyring

        stored = keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME)
        if stored:
            return stored
    except Exception as exc:  # noqa: BLE001
        _log.warning("OS keyring unavailable while reading vault master key: %s", exc)

    if sys.stdin.isatty():
        password = getpass.getpass(prompt or DEFAULT_PROMPT)
        if not password:
            raise VaultCredentialsMissing("Password cannot be empty.")
        try:
            import keyring

            keyring.set_password(KEYRING_SERVICE, KEYRING_USERNAME, password)
            _log.info(
                "Vault master key stored in OS keyring (%s / %s)",
                KEYRING_SERVICE,
                KEYRING_USERNAME,
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning("Could not persist vault master key to OS keyring: %s", exc)
        return password

    msg = (
        "Missing Vault credentials for headless boot. "
        f"Set {ENV_VAR} or store the key in the OS keyring "
        f"({KEYRING_SERVICE}/{KEYRING_USERNAME}) by running once interactively."
    )
    _log.critical(msg)
    raise VaultCredentialsMissing(msg)
