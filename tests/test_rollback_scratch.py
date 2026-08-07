"""rollback_scratch_workspace unit tests."""

from __future__ import annotations

from pathlib import Path

from dana.graph.runtime_harness import (
    begin_epic_artifact_tracking,
    rollback_scratch_workspace,
)


def test_rollback_deletes_new_and_restores_existing(tmp_path: Path) -> None:
    ws = tmp_path
    existed = ws / "keep_me.py"
    existed.write_text("OLD = 1\n", encoding="utf-8")
    begin_epic_artifact_tracking(
        str(ws),
        ["keep_me.py", "new_broken.py"],
        run_key="epic-t",
    )
    existed.write_text("NEW = 2\n", encoding="utf-8")
    created = ws / "new_broken.py"
    created.write_text("bad\n", encoding="utf-8")

    rb = rollback_scratch_workspace(str(ws), run_key="epic-t")
    assert "new_broken.py" in rb["deleted"]
    assert not created.exists()
    assert "keep_me.py" in rb["restored"]
    assert existed.read_text(encoding="utf-8") == "OLD = 1\n"


def test_rollback_skips_protected_dana_package(tmp_path: Path) -> None:
    ws = tmp_path
    protected = ws / "dana" / "core_agent.py"
    protected.parent.mkdir()
    protected.write_text("x=1\n", encoding="utf-8")
    begin_epic_artifact_tracking(
        str(ws), ["dana/core_agent.py"], run_key="epic-p"
    )
    rb = rollback_scratch_workspace(str(ws), run_key="epic-p")
    assert protected.exists()
    assert "dana/core_agent.py" in rb["skipped"] or not rb["deleted"]
