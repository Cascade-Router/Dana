#!/usr/bin/env python3
"""CLI: Reset Dana's local encrypted memory vault (dev-only, lost-credential path).

Use this only when the master password AND recovery key are both lost — the
vault's PBKDF2+Fernet encryption has no backdoor, so its contents cannot be
recovered without one of them. This does not "unlock" the vault; it backs up
the unreadable ciphertext and removes it so the next launch falls through to
Dana's existing "no vault found -> create new vault" flow, which will prompt
for a brand-new master password and recovery key.

The backup is kept (still encrypted, still unreadable without the old
credential) in case the password is found later.

Usage (from repo root)::

    python scripts/reset_vault.py            # prompts for confirmation
    python scripts/reset_vault.py --yes       # skip confirmation
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _flush_daemon() -> None:
    """Best-effort: purge any RAM-cached data key so a stale session can't
    keep serving the old profile after the on-disk vault is replaced."""
    try:
        from dana.vault_service import _rpc

        _rpc({"op": "purge"}, timeout=1.0)
    except Exception:
        pass


def reset_vault(*, assume_yes: bool) -> int:
    from dana.paths import VAULT_PATH

    vault_path = Path(VAULT_PATH)
    if not vault_path.is_file():
        print(f"No vault at {vault_path}; nothing to reset.")
        return 0

    print(f"Vault found: {vault_path}")
    print(
        "This cannot decrypt or recover the existing profile — it only backs up "
        "the ciphertext and clears the slot so Dana creates a fresh vault next run."
    )
    if not assume_yes:
        reply = input("Proceed with backup + reset? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            print("Aborted.")
            return 1

    backup_path = vault_path.with_suffix(f".enc.bak.{int(time.time())}")
    shutil.copy2(vault_path, backup_path)
    print(f"Backed up (still encrypted) to {backup_path}")

    _flush_daemon()

    vault_path.unlink()
    print(f"Removed {vault_path}")
    print("Run `python run.py` and follow the prompt to set a new master password + recovery key.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yes", action="store_true", help="Skip the interactive confirmation prompt."
    )
    args = parser.parse_args()
    return reset_vault(assume_yes=args.yes)


if __name__ == "__main__":
    raise SystemExit(main())
