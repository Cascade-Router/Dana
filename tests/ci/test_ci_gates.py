"""scripts/ci/check_boundaries.py and scripts/ci/check_dead_code_incremental.py.

The boundary check runs on small synthetic trees and on the real repo (which
must stay within its baseline). The dead-code check runs end to end on a
throwaway git repo, so the base-vs-working-tree diff is exercised with real
git and real vulture.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

CI_DIR = Path(__file__).resolve().parents[2] / "scripts" / "ci"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"ci_{name}", CI_DIR / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


boundaries = _load("check_boundaries")
dead_code = _load("check_dead_code_incremental")


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# -- check_boundaries ----------------------------------------------------------------


def test_every_import_form_that_reaches_a_plugin_is_caught(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "dana/core/sub/mod.py",
        "import dana.plugins.freecad.engine\n"
        "from dana.plugins.hardware.printer_tools import pause_print, emergency_stop\n"
        "from dana import plugins\n"
        "from ...plugins.os import file_system\n"
        "from ...manufacturing import checks\n"
        "def f():\n"
        "    from dana.manufacturing.slicer import run\n"
        # allowed:
        "from dana.tools import registry\n"
        "from . import sibling\n"
        "from ..other import helper\n"
        "import dana.pluginsx\n",
    )
    found = {(v.line, v.module) for v in boundaries.find_violations(tmp_path)}
    assert found == {
        (1, "dana.plugins.freecad.engine"),
        (2, "dana.plugins.hardware.printer_tools"),  # the module, not each imported name
        (3, "dana.plugins"),
        (4, "dana.plugins.os"),
        (5, "dana.manufacturing"),
        (7, "dana.manufacturing.slicer"),
    }


def test_files_outside_dana_core_are_not_checked(tmp_path: Path) -> None:
    _write(tmp_path, "dana/api/server.py", "from dana.plugins.hardware import watchdog\n")
    _write(tmp_path, "dana/core/__init__.py", "")
    assert boundaries.find_violations(tmp_path) == []


@pytest.fixture
def fake_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(boundaries, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(boundaries, "BASELINE_PATH", tmp_path / "baseline.txt")
    _write(tmp_path, "dana/core/a.py", "from dana.plugins.web.research import search_web\n")
    return tmp_path


def test_new_edge_fails_and_baselined_edge_passes(fake_repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert boundaries.main([]) == 1
    assert "dana/core/a.py:1: dana/core imports dana.plugins.web.research" in capsys.readouterr().out

    assert boundaries.main(["--update-baseline"]) == 0
    assert "dana/core/a.py dana.plugins.web.research" in (fake_repo / "baseline.txt").read_text(encoding="utf-8")
    assert boundaries.main([]) == 0

    # More names from a baselined module: same edge, still passes.
    _write(fake_repo, "dana/core/a.py", "from dana.plugins.web.research import search_web, read_webpage\n")
    assert boundaries.main([]) == 0
    # A new plugin module from the same file: new edge, fails.
    _write(fake_repo, "dana/core/a.py", "from dana.plugins.web.research import search_web\nimport dana.plugins.os\n")
    assert boundaries.main([]) == 1


def test_removed_edge_is_reported_as_stale_without_failing(
    fake_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    boundaries.main(["--update-baseline"])
    _write(fake_repo, "dana/core/a.py", "import json\n")
    capsys.readouterr()
    assert boundaries.main([]) == 0
    assert "dana/core/a.py dana.plugins.web.research" in capsys.readouterr().out


def test_the_repo_itself_has_no_unbaselined_core_to_plugin_imports() -> None:
    edges = {v.edge for v in boundaries.find_violations()}
    assert edges - boundaries.load_baseline() == set()


# -- check_dead_code_incremental -------------------------------------------------------


def test_moved_findings_are_not_new_but_extra_copies_are() -> None:
    base = [("a.py", 10, "unused function 'old'"), ("b.py", 3, "unused import 'os'")]
    head = [
        ("a.py", 40, "unused function 'old'"),  # moved
        ("a.py", 50, "unused function 'fresh'"),
        ("b.py", 3, "unused import 'os'"),
        ("b.py", 9, "unused import 'os'"),  # a second one
        ("c.py", 1, "unused function 'untouched'"),  # file not in the change
    ]
    assert dead_code.new_findings(base, head, ["a.py", "b.py"]) == [
        ("a.py", 50, "unused function 'fresh'"),
        ("b.py", 9, "unused import 'os'"),
    ]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=ci", "-c", "user.email=ci@example.invalid", "-c", "commit.gpgsign=false", *args],
        cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.strip()


@pytest.fixture
def git_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str]:
    if shutil.which("git") is None:
        pytest.skip("git not installed")
    pytest.importorskip("vulture")
    repo = tmp_path / "repo"
    _write(repo, "pyproject.toml", '[tool.vulture]\npaths = ["pkg"]\nmin_confidence = 60\n')
    _write(repo, "pkg/__init__.py", "")
    _write(
        repo,
        "pkg/legacy.py",
        "def legacy_dead():\n    return 1\n\n\ndef used():\n    return 2\n",
    )
    _write(repo, "pkg/main.py", "from pkg.legacy import used\n\nprint(used())\n")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    monkeypatch.setattr(dead_code, "REPO_ROOT", repo)
    return repo, _git(repo, "rev-parse", "HEAD")


def test_touching_a_legacy_file_does_not_fail_on_its_old_dead_code(
    git_repo: tuple[Path, str], capsys: pytest.CaptureFixture[str]
) -> None:
    repo, base = git_repo
    # Shift legacy_dead's line number too: findings match on message, not line.
    _write(repo, "pkg/legacy.py", "# touched\n\n\ndef legacy_dead():\n    return 1\n\n\ndef used():\n    return 3\n")
    assert dead_code.main(["--base", base]) == 0
    assert "OK" in capsys.readouterr().out


def test_new_dead_function_fails_but_a_new_cross_module_one_does_not(
    git_repo: tuple[Path, str], capsys: pytest.CaptureFixture[str]
) -> None:
    repo, base = git_repo
    # helper is only called from main.py: scanning helpers.py alone would call
    # it unused; the whole-tree scan must not.
    _write(repo, "pkg/helpers.py", "def helper():\n    return 4\n\n\ndef orphan():\n    return 5\n")
    _write(repo, "pkg/main.py", "from pkg.legacy import used\nfrom pkg.helpers import helper\n\nprint(used(), helper())\n")
    assert dead_code.main(["--base", base]) == 1
    out = capsys.readouterr().out
    assert "pkg/helpers.py:5: unused function 'orphan'" in out
    assert "'helper'" not in out
    assert "legacy_dead" not in out  # legacy.py wasn't touched


def test_a_gitignored_local_file_cannot_hide_new_dead_code(
    git_repo: tuple[Path, str], capsys: pytest.CaptureFixture[str]
) -> None:
    """CI never sees gitignored files, so a local one that still calls a name
    must not make that name look used (the 2026-10-06 mic_ingest_ready miss)."""
    repo, base = git_repo
    _write(repo, ".gitignore", "pkg/scratch.py\n")
    _write(repo, "pkg/scratch.py", "from pkg.helpers import orphan\n\norphan()\n")
    _write(repo, "pkg/helpers.py", "def orphan():\n    return 5\n")
    assert dead_code.main(["--base", base]) == 1
    assert "pkg/helpers.py:1: unused function 'orphan'" in capsys.readouterr().out


def test_committed_change_is_compared_to_the_given_base(git_repo: tuple[Path, str]) -> None:
    repo, base = git_repo
    _write(repo, "pkg/legacy.py", "def legacy_dead():\n    return 1\n\n\ndef used():\n    return 2\n\n\ndef new_dead():\n    return 6\n")
    _git(repo, "commit", "-q", "-am", "add dead code")
    assert dead_code.main(["--base", base]) == 1
    # A push's all-zero "before" falls back to HEAD~1, which is the same base here.
    assert dead_code.main(["--base", "0" * 40]) == 1
    assert dead_code.main(["--base", "HEAD"]) == 0
