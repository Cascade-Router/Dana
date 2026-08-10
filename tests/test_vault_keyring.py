"""Tests for OS-keyring vault master-key resolution."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from dana.tools import vault


def test_env_var_takes_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(vault.ENV_VAR, "from-env")
    with patch.object(vault, "keyring", create=True):
        assert vault._get_master_key() == "from-env"


def test_keyring_used_when_env_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(vault.ENV_VAR, raising=False)
    fake = MagicMock()
    fake.get_password.return_value = "from-keyring"
    with patch.dict(sys.modules, {"keyring": fake}):
        assert vault._get_master_key() == "from-keyring"
    fake.get_password.assert_called_once_with(
        vault.KEYRING_SERVICE, vault.KEYRING_USERNAME
    )


def test_interactive_prompt_persists_to_keyring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(vault.ENV_VAR, raising=False)
    fake = MagicMock()
    fake.get_password.return_value = None
    with (
        patch.dict(sys.modules, {"keyring": fake}),
        patch.object(sys.stdin, "isatty", return_value=True),
        patch.object(vault.getpass, "getpass", return_value="typed-secret"),
    ):
        assert vault._get_master_key(prompt="pw: ") == "typed-secret"
    fake.set_password.assert_called_once_with(
        vault.KEYRING_SERVICE, vault.KEYRING_USERNAME, "typed-secret"
    )


def test_headless_missing_credentials_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(vault.ENV_VAR, raising=False)
    fake = MagicMock()
    fake.get_password.return_value = None
    with (
        patch.dict(sys.modules, {"keyring": fake}),
        patch.object(sys.stdin, "isatty", return_value=False),
        patch.object(vault.getpass, "getpass") as gp,
    ):
        with pytest.raises(vault.VaultCredentialsMissing):
            vault._get_master_key()
    gp.assert_not_called()


def test_none_stdin_treated_as_non_interactive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GUI-launched processes (e.g. pythonw) can have ``sys.stdin is None`` —
    must fail fast with VaultCredentialsMissing, not crash on ``.isatty()``.
    """
    monkeypatch.delenv(vault.ENV_VAR, raising=False)
    fake = MagicMock()
    fake.get_password.return_value = None
    with (
        patch.dict(sys.modules, {"keyring": fake}),
        patch.object(vault.sys, "stdin", None),
        patch.object(vault.getpass, "getpass") as gp,
    ):
        with pytest.raises(vault.VaultCredentialsMissing):
            vault._get_master_key()
    gp.assert_not_called()
