"""Sandboxed file reads for Tool Forge generated tools.



Generated tools MUST call ``sandbox_read(filepath)`` instead of native ``open()``.

Reads are restricted to ``DANA_WORKSPACE/execution_jail`` (CAMGRASPER/execution_jail).

"""



from __future__ import annotations



import os

from pathlib import Path



from dana.paths import PROJECT_ROOT, EXECUTION_JAIL_DIR



# Absolute jail root — generated tools cannot escape this tree.

_SANDBOX_READ_ROOT = EXECUTION_JAIL_DIR.resolve()





class SandboxReadError(PermissionError):

    """Raised when a path escapes the sandbox jail."""





def sandbox_read_root() -> Path:

    return _SANDBOX_READ_ROOT





def resolve_sandbox_path(filepath: str | os.PathLike[str]) -> Path:

    """Resolve ``filepath`` under the sandbox jail; raise if it escapes."""

    raw = Path(str(filepath)).expanduser()

    if not raw.is_absolute():

        parts = raw.parts

        # Tolerate leading ``execution_jail/`` or legacy ``docs/`` prefixes.

        if parts and parts[0].lower() in ("execution_jail", "sandbox", "docs"):

            raw = Path(*parts[1:]) if len(parts) > 1 else Path(".")

        candidate = (_SANDBOX_READ_ROOT / raw).resolve()

    else:

        candidate = raw.resolve()

    try:

        candidate.relative_to(_SANDBOX_READ_ROOT)

    except ValueError as exc:

        raise SandboxReadError(

            f"sandbox_read refused path outside {_SANDBOX_READ_ROOT}: {filepath!r}"

        ) from exc

    return candidate





def resolve_safe_path(filepath: str | os.PathLike[str]) -> Path:

    """Jailed path resolver for image/binary loaders (e.g. ``PIL.Image.open``).



    Returns a ``Path`` guaranteed to live inside the sandbox jail. Generated

    tools must wrap any loader path in this (or ``sandbox_read``) so the AST

    Gatekeeper can prove the access is sandboxed.

    """

    return resolve_sandbox_path(filepath)





def sandbox_read(filepath: str | os.PathLike[str], *, encoding: str = "utf-8") -> str:

    """Read a text file strictly inside the CAMGRASPER/execution_jail jail."""

    path = resolve_sandbox_path(filepath)

    if not path.is_file():

        raise FileNotFoundError(f"sandbox_read: file not found: {path}")

    return path.read_text(encoding=encoding, errors="replace")





def sandbox_write_probe() -> str:

    """Return a short capability string for self-tests (does not write)."""

    return f"sandbox_read_root={_SANDBOX_READ_ROOT}"





# Keep PROJECT_ROOT import used for documentation / future jail expansion.

_ = PROJECT_ROOT


