"""The PyInstaller build (build_dana.py, mirrored by Dana.spec) bundles every
data file the code reads at runtime, from paths that exist, and nothing
local to a developer's checkout."""

from __future__ import annotations

import ast
import shutil
import subprocess
import sys

import pytest

from dana import paths
from dana.paths import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT))
import build_dana  # noqa: E402


def _bundled() -> dict[str, str]:
    return {src.relative_to(PROJECT_ROOT).as_posix(): dest for src, dest in build_dana.data_files()}


def test_every_bundled_source_exists() -> None:
    missing = [str(src) for src, _ in build_dana.data_files() if not src.is_file()]
    assert not missing


@pytest.mark.parametrize(
    "required",
    [
        "dana/tools/tools.json",
        "dana/plugins/manufacturing/materials.json",
        "routing_config.yaml",
        "ARCHITECTURE.md",
        *sorted(p.relative_to(PROJECT_ROOT).as_posix() for p in (PROJECT_ROOT / "dana/plugins/freecad/templates").glob("*.jinja2")),
        *sorted(p.relative_to(PROJECT_ROOT).as_posix() for p in (PROJECT_ROOT / "dana/plugins").glob("*/manifest.json")),
    ],
)
def test_runtime_data_is_bundled_where_the_code_looks(required: str) -> None:
    """Each file lands at its own path relative to the bundle root, which is
    PROJECT_ROOT in the frozen app (dana/paths.py's parent's parent)."""
    bundled = _bundled()
    assert required in bundled, f"{required} is not bundled"
    parent = str((PROJECT_ROOT / required).parent.relative_to(PROJECT_ROOT).as_posix())
    assert bundled[required] == parent


@pytest.mark.parametrize(
    "constant", ["TOOLS_JSON", "SECURITY_POLICY_PATH", "MODELS_REGISTRY_PATH", "ROUTING_CONFIG_PATH", "ARCHITECTURE_MD"]
)
def test_files_named_in_dana_paths_ship(constant: str) -> None:
    path = getattr(paths, constant)
    assert path.relative_to(PROJECT_ROOT).as_posix() in _bundled()


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git to list tracked files")
def test_only_tracked_files_ship() -> None:
    """A git-ignored file that matches the allowlist (a developer's own
    dana/settings.json) must not end up in a build made from their checkout."""
    listed = subprocess.run(["git", "ls-files"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
    tracked = set(listed.stdout.split())
    untracked = sorted(set(_bundled()) - tracked)
    assert not untracked


def test_dana_spec_mirrors_build_dana() -> None:
    """Dana.spec is the hand-kept copy of build_dana.py's data rules: same
    suffix allowlist, same root files, same local-only exclusions."""
    tree = ast.parse((PROJECT_ROOT / "Dana.spec").read_text(encoding="utf-8"))
    string_sets = [
        {e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)}
        for node in ast.walk(tree)
        if isinstance(node, (ast.Set, ast.Tuple)) and node.elts
    ]
    assert set(build_dana.PACKAGE_DATA_SUFFIXES) in string_sets
    assert set(build_dana.ROOT_DATA_FILES) in string_sets
    assert set(build_dana.LOCAL_ONLY_NAMES) in string_sets
