"""Incremental dead-code check: fail only on unused code a change ADDS.

Runs vulture over the whole scan set (``[tool.vulture] paths`` in
pyproject.toml) twice, on the base revision and on the working tree, keeps
only the findings in ``.py`` files the change touched, and fails if the
working tree has findings the base didn't. Findings are compared by
(file, message), not line number, so code that merely moved isn't new.

Why whole-tree and not just the changed files: vulture only sees usages in
the files it is given, so scanning a changed file alone reports every
function it exports to other modules as unused. And why diff against the
base: running vulture on a legacy file reports all the dead code already in
it, which would fail whoever happens to touch that file next.

Base revision, first that applies:
  1. ``--base REF``
  2. ``git merge-base HEAD origin/main``, unless that is HEAD itself (a push
     to main) — then ``HEAD~1``.

Usage:
    python scripts/ci/check_dead_code_incremental.py [--base REF]
Exit status 1 if the change adds unused code.
"""

from __future__ import annotations

import argparse
import io
import re
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_FINDING = re.compile(r"^(?P<file>.+?\.py):(?P<line>\d+): (?P<message>.+)$")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()


def _ref_exists(ref: str) -> bool:
    return subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"], cwd=REPO_ROOT,
                          capture_output=True).returncode == 0


def resolve_base(explicit: str | None) -> str:
    # GitHub sends an all-zero "before" SHA for a branch's first push.
    if explicit and explicit.strip("0") and _ref_exists(explicit):
        return _git("rev-parse", explicit)
    head = _git("rev-parse", "HEAD")
    if _ref_exists("origin/main"):
        base = _git("merge-base", "HEAD", "origin/main")
        if base != head:
            return base
    return _git("rev-parse", "HEAD~1")


def changed_python_files(base: str) -> list[str]:
    # Working tree vs base, plus untracked files: a local run checks
    # uncommitted work too. (In CI the tree is clean and both match the commit.)
    out = _git("diff", "--name-only", "--diff-filter=AMR", base, "--", "*.py")
    untracked = _git("ls-files", "--others", "--exclude-standard", "--", "*.py")
    return sorted({line for line in (out.splitlines() + untracked.splitlines()) if line})


def vulture_settings() -> tuple[list[str], list[str]]:
    """(scan paths, extra CLI args) from pyproject.toml's [tool.vulture]."""
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")).get("tool", {}).get("vulture", {})
    paths = list(config.get("paths") or ["dana"])
    args = ["--min-confidence", str(config.get("min_confidence", 0))]
    if config.get("ignore_decorators"):
        args += ["--ignore-decorators", ",".join(config["ignore_decorators"])]
    if config.get("ignore_names"):
        args += ["--ignore-names", ",".join(config["ignore_names"])]
    return paths, args


def visible_python_files(paths: list[str]) -> list[str]:
    """The working tree's .py files under ``paths`` that git tracks or would
    track: gitignored local files (scratch scripts, old copies) are left out,
    so a local run scans what CI scans and can't be masked by a stray file
    that still uses a name."""
    out = _git("ls-files", "--cached", "--others", "--exclude-standard", "--", *paths)
    return sorted(f for f in out.splitlines() if f.endswith(".py") and (REPO_ROOT / f).is_file())


def run_vulture(root: Path, paths: list[str], args: list[str]) -> list[tuple[str, int, str]]:
    """Findings as (posix path relative to root, line, message)."""
    present = [p for p in paths if (root / p).exists()]
    if not present:
        return []
    proc = subprocess.run(
        [sys.executable, "-m", "vulture", *present, *args],
        cwd=root, capture_output=True, text=True, encoding="utf-8",
    )
    if proc.returncode not in (0, 3):  # 3 = dead code found; anything else is a vulture failure
        raise RuntimeError(f"vulture failed in {root} (exit {proc.returncode}):\n{proc.stderr or proc.stdout}")
    findings = []
    for line in proc.stdout.splitlines():
        match = _FINDING.match(line.strip())
        if match:
            findings.append((Path(match["file"]).as_posix(), int(match["line"]), match["message"]))
    return findings


def export_revision(rev: str, paths: list[str], dest: Path) -> None:
    """Extract ``paths`` as they were at ``rev`` into ``dest``."""
    tracked = [p for p in paths if _git("ls-tree", "--name-only", rev, "--", p)]
    if not tracked:
        return
    archive = subprocess.run(["git", "archive", "--format=tar", rev, "--", *tracked], cwd=REPO_ROOT,
                             check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(dest, filter="data")


def new_findings(
    base_findings: list[tuple[str, int, str]], head_findings: list[tuple[str, int, str]], changed: list[str]
) -> list[tuple[str, int, str]]:
    changed_set = set(changed)
    remaining = Counter((f, msg) for f, _line, msg in base_findings if f in changed_set)
    added = []
    for f, line, msg in head_findings:
        if f not in changed_set:
            continue
        if remaining[(f, msg)] > 0:
            remaining[(f, msg)] -= 1
        else:
            added.append((f, line, msg))
    return added


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", help="revision to compare against (default: merge-base with origin/main)")
    args = parser.parse_args(argv)

    base = resolve_base(args.base)
    changed = changed_python_files(base)
    if not changed:
        print(f"OK: no Python files changed since {base[:12]}.")
        return 0

    paths, vulture_args = vulture_settings()
    head_findings = run_vulture(REPO_ROOT, visible_python_files(paths), vulture_args)
    with tempfile.TemporaryDirectory() as tmp:
        export_revision(base, paths, Path(tmp))
        # The base tree has no pyproject.toml; settings come in as CLI args.
        base_findings = run_vulture(Path(tmp), paths, vulture_args)

    added = new_findings(base_findings, head_findings, changed)
    for f, line, msg in added:
        print(f"{f}:{line}: {msg}")
    if added:
        print(
            f"FAIL: {len(added)} new unused item(s) in {len(changed)} changed file(s) since {base[:12]}. "
            "Delete them or use them; for a real false positive (e.g. called only via getattr), "
            "add the name to ignore_names under [tool.vulture] in pyproject.toml."
        )
        return 1
    print(f"OK: no new unused code in {len(changed)} changed Python file(s) since {base[:12]}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
