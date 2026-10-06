"""Create the Tool Forge package directories the tool registry loads from:
``custom_tools/`` (ephemeral, on ``sys.path``) and ``dana/tools/general/``.

The CAMGRASPER-era workspace bootstrap and one-shot legacy migration that
used to live here were removed on 2026-10-06; no entry point had run them
since a8b40ed.
"""



from __future__ import annotations





from pathlib import Path




from dana.paths import CUSTOM_TOOLS_DIR, GENERAL_TOOLS_DIR, ensure_workspace_on_syspath





def ensure_custom_tools_package() -> Path:

    """Create Desktop ``custom_tools/`` with ``__init__.py`` and put it on ``sys.path``."""

    CUSTOM_TOOLS_DIR.mkdir(parents=True, exist_ok=True)

    init_path = CUSTOM_TOOLS_DIR / "__init__.py"

    if not init_path.is_file():

        init_path.write_text(

            '"""Hot-loaded Tool Forge modules (CAMGRASPER/custom_tools)."""\n',

            encoding="utf-8",

        )

    ensure_workspace_on_syspath()

    return CUSTOM_TOOLS_DIR





def ensure_general_tools_package() -> Path:

    """Ensure repo ``dana/tools/general/`` exists with ``__init__.py``."""

    GENERAL_TOOLS_DIR.mkdir(parents=True, exist_ok=True)

    init_path = GENERAL_TOOLS_DIR / "__init__.py"

    if not init_path.is_file():

        init_path.write_text(

            '"""Promoted general-purpose tools (Git-tracked)."""\n',

            encoding="utf-8",

        )

    return GENERAL_TOOLS_DIR





