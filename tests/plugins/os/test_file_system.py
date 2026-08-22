"""Tests for dana.plugins.os.file_system — the sandboxed read/write layer
backing the real "os_tools" capability domain (dana.core.react_dispatch's
_OS_TOOLS_TOOL_IDS). Every test redirects the sandbox root to a throwaway
temp directory (see the autouse `_sandbox` fixture) — none of these ever
touch the real AGENT_WORKSPACE_DIR on disk.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dana.plugins.os import file_system


@pytest.fixture(autouse=True)
def _sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "agent_workspace"
    # exist_ok=True: tests/conftest.py's own global _isolate_os_tools_sandbox
    # autouse fixture already creates this same tmp_path/agent_workspace
    # directory (and runs first, per pytest's conftest-before-module
    # autouse-fixture ordering) — this fixture's OWN monkeypatch below still
    # needs to run regardless, so tolerate the directory already existing
    # rather than raising on a bare mkdir().
    root.mkdir(exist_ok=True)
    monkeypatch.setattr(file_system, "_SANDBOX_ROOT", root)
    return root


# --------------------------------------------------------------------------
# Path traversal rejection
# --------------------------------------------------------------------------


def test_rejects_parent_traversal(_sandbox: Path) -> None:
    with pytest.raises(file_system.PathEscapeError):
        file_system.resolve_sandboxed_path("../outside.txt")


def test_rejects_deeply_nested_traversal(_sandbox: Path) -> None:
    with pytest.raises(file_system.PathEscapeError):
        file_system.resolve_sandboxed_path("a/b/../../../outside.txt")


def test_rejects_absolute_posix_style_path(_sandbox: Path) -> None:
    with pytest.raises(file_system.PathEscapeError):
        file_system.resolve_sandboxed_path("/etc/passwd")


def test_rejects_absolute_windows_style_path(_sandbox: Path, tmp_path: Path) -> None:
    outside = tmp_path / "elsewhere.txt"
    with pytest.raises(file_system.PathEscapeError):
        file_system.resolve_sandboxed_path(str(outside))


def test_rejects_empty_path(_sandbox: Path) -> None:
    with pytest.raises(file_system.PathEscapeError):
        file_system.resolve_sandboxed_path("")


def test_symlink_escape_is_rejected(_sandbox: Path, tmp_path: Path) -> None:
    """A symlink created INSIDE the sandbox but pointing OUTSIDE it must
    still be caught — resolve_sandboxed_path follows symlinks (Path.resolve)
    before checking containment, not after."""
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    (outside_dir / "secret.txt").write_text("top secret")
    link = _sandbox / "escape_link"
    try:
        link.symlink_to(outside_dir, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation not permitted in this environment")
    with pytest.raises(file_system.PathEscapeError):
        file_system.resolve_sandboxed_path("escape_link/secret.txt")


def test_list_directory_reports_clear_error_on_traversal(_sandbox: Path) -> None:
    result = file_system.list_directory("../../etc")
    assert result["ok"] is False
    assert "outside the sandbox" in result["error"]


def test_read_file_reports_clear_error_on_traversal(_sandbox: Path) -> None:
    result = file_system.read_file("../secret.txt")
    assert result["ok"] is False
    assert "outside the sandbox" in result["error"]


def test_write_file_reports_clear_error_on_traversal_and_writes_nothing(
    _sandbox: Path, tmp_path: Path
) -> None:
    result = file_system.write_file("../evil.txt", "pwned")
    assert result["ok"] is False
    assert "outside the sandbox" in result["error"]
    assert not (tmp_path / "evil.txt").exists()


# --------------------------------------------------------------------------
# Read/write success inside the sandbox
# --------------------------------------------------------------------------


def test_write_then_read_file_round_trips(_sandbox: Path) -> None:
    write_result = file_system.write_file("notes/todo.txt", "buy milk")
    assert write_result["ok"] is True
    assert write_result["bytes_written"] == len("buy milk".encode("utf-8"))

    read_result = file_system.read_file("notes/todo.txt")
    assert read_result["ok"] is True
    assert read_result["content"] == "buy milk"


def test_write_file_creates_parent_directories(_sandbox: Path) -> None:
    result = file_system.write_file("a/b/c/deep.txt", "hello")
    assert result["ok"] is True
    assert (_sandbox / "a" / "b" / "c" / "deep.txt").is_file()


def test_list_directory_lists_files_and_subdirs(_sandbox: Path) -> None:
    (_sandbox / "file_a.txt").write_text("a")
    (_sandbox / "subdir").mkdir()
    result = file_system.list_directory(".")
    assert result["ok"] is True
    names = {e["name"]: e["type"] for e in result["entries"]}
    assert names == {"file_a.txt": "file", "subdir": "directory"}


def test_read_file_handles_invalid_utf8_gracefully(_sandbox: Path) -> None:
    (_sandbox / "binary.dat").write_bytes(b"\xff\xfe not valid utf-8 \x80")
    result = file_system.read_file("binary.dat")
    assert result["ok"] is True
    assert "�" in result["content"]  # replacement char, not a crash


def test_read_file_missing_file_reports_error_not_crash(_sandbox: Path) -> None:
    result = file_system.read_file("does_not_exist.txt")
    assert result["ok"] is False
    assert "does not exist" in result["error"]


def test_list_directory_missing_path_reports_error_not_crash(_sandbox: Path) -> None:
    result = file_system.list_directory("nope")
    assert result["ok"] is False


def test_write_file_rejects_directory_target(_sandbox: Path) -> None:
    (_sandbox / "adir").mkdir()
    result = file_system.write_file("adir", "oops")
    assert result["ok"] is False


def test_read_file_rejects_directory_target(_sandbox: Path) -> None:
    (_sandbox / "adir").mkdir()
    result = file_system.read_file("adir")
    assert result["ok"] is False


# --------------------------------------------------------------------------
# Dynamic Workspace Mounting — resolve_sandboxed_path's allowed_mounts param
# (dana.api.workspace.mount_workspace_directory registers these; this module
# only enforces the resulting trust decision). A mock mounted directory is
# just another tmp_path subdirectory, deliberately OUTSIDE `_sandbox` — the
# whole point is proving these are trusted despite living outside the
# sandbox root, and that nothing else outside BOTH roots is.
# --------------------------------------------------------------------------


@pytest.fixture
def _mount(tmp_path: Path) -> Path:
    mount_dir = tmp_path / "mounted_project"
    mount_dir.mkdir()
    return mount_dir


def test_absolute_path_inside_a_mount_is_allowed(_sandbox: Path, _mount: Path) -> None:
    (_mount / "main.py").write_text("print('hi')")
    resolved = file_system.resolve_sandboxed_path(str(_mount / "main.py"), allowed_mounts=[str(_mount)])
    assert resolved == (_mount / "main.py").resolve()


def test_absolute_path_outside_every_mount_is_still_rejected(_sandbox: Path, _mount: Path, tmp_path: Path) -> None:
    """A mount registered for a DIFFERENT directory must not grant access
    to some other, unrelated absolute path — only the specific registered
    mount(s) are trusted, not "absolute paths in general" once any mount
    exists."""
    unrelated = tmp_path / "unrelated_dir"
    unrelated.mkdir()
    with pytest.raises(file_system.PathEscapeError):
        file_system.resolve_sandboxed_path(str(unrelated / "secret.txt"), allowed_mounts=[str(_mount)])


def test_absolute_path_with_no_mounts_registered_is_rejected(_sandbox: Path, _mount: Path) -> None:
    """allowed_mounts=None (or empty) must behave exactly like before this
    feature existed — an absolute path is rejected outright."""
    with pytest.raises(file_system.PathEscapeError):
        file_system.resolve_sandboxed_path(str(_mount / "main.py"), allowed_mounts=None)
    with pytest.raises(file_system.PathEscapeError):
        file_system.resolve_sandboxed_path(str(_mount / "main.py"), allowed_mounts=[])


def test_traversal_that_escapes_the_mount_is_rejected(_sandbox: Path, _mount: Path) -> None:
    """A '..' that walks back OUT of a registered mount must still fail —
    being inside a mount grants access to that mount's own subtree, not to
    its parent directory or siblings."""
    escape = str(_mount / ".." / "sibling_dir" / "secret.txt")
    with pytest.raises(file_system.PathEscapeError):
        file_system.resolve_sandboxed_path(escape, allowed_mounts=[str(_mount)])


def test_symlink_escape_from_within_a_mount_is_rejected(_sandbox: Path, _mount: Path, tmp_path: Path) -> None:
    """Same guarantee as the sandbox's own symlink test, but for a
    dynamically-trusted mount: a symlink INSIDE the mount pointing OUTSIDE
    every trusted root must still be caught post-resolve()."""
    outside_dir = tmp_path / "outside_the_mount"
    outside_dir.mkdir()
    (outside_dir / "secret.txt").write_text("top secret")
    link = _mount / "escape_link"
    try:
        link.symlink_to(outside_dir, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation not permitted in this environment")
    with pytest.raises(file_system.PathEscapeError):
        file_system.resolve_sandboxed_path(str(link / "secret.txt"), allowed_mounts=[str(_mount)])


def test_sandbox_root_still_works_unaffected_by_an_unrelated_mount(_sandbox: Path, _mount: Path) -> None:
    """Registering a mount must not change how a plain, relative,
    sandbox-relative path resolves — the two trust roots are additive, not
    a replacement for each other."""
    resolved = file_system.resolve_sandboxed_path("notes.txt", allowed_mounts=[str(_mount)])
    assert resolved == (_sandbox / "notes.txt").resolve()


def test_list_directory_reads_inside_a_mounted_directory(_sandbox: Path, _mount: Path) -> None:
    (_mount / "a.py").write_text("a")
    (_mount / "sub").mkdir()
    result = file_system.list_directory(str(_mount), allowed_mounts=[str(_mount)])
    assert result["ok"] is True
    names = {e["name"]: e["type"] for e in result["entries"]}
    assert names == {"a.py": "file", "sub": "directory"}


def test_read_file_reads_inside_a_mounted_directory(_sandbox: Path, _mount: Path) -> None:
    (_mount / "notes.txt").write_text("mounted content")
    result = file_system.read_file(str(_mount / "notes.txt"), allowed_mounts=[str(_mount)])
    assert result["ok"] is True
    assert result["content"] == "mounted content"


def test_write_then_read_file_round_trips_inside_a_mounted_directory(_sandbox: Path, _mount: Path) -> None:
    target = str(_mount / "output" / "result.txt")
    write_result = file_system.write_file(target, "computed value", allowed_mounts=[str(_mount)])
    assert write_result["ok"] is True
    assert (_mount / "output" / "result.txt").is_file()

    read_result = file_system.read_file(target, allowed_mounts=[str(_mount)])
    assert read_result["ok"] is True
    assert read_result["content"] == "computed value"


def test_write_file_inside_mount_rejected_without_the_mount_registered(_sandbox: Path, _mount: Path) -> None:
    """The exact same absolute path succeeds with the mount registered
    (test_write_then_read_file_round_trips_inside_a_mounted_directory) and
    must fail identically to any other unauthorized absolute path once the
    mount is not in allowed_mounts — proving the tool functions genuinely
    gate on it, not just resolve_sandboxed_path in isolation."""
    result = file_system.write_file(str(_mount / "result.txt"), "computed value", allowed_mounts=None)
    assert result["ok"] is False
    assert "outside the sandbox" in result["error"]
    assert not (_mount / "result.txt").exists()


def test_dispatch_tool_call_write_file_reaches_a_mounted_directory(_sandbox: Path, _mount: Path) -> None:
    """End-to-end through react_dispatch.dispatch_tool_call's own
    allowed_mounts param — not just the underlying file_system function —
    proving the ReAct wiring actually threads a session's registered mounts
    down to the tool."""
    import dana.core.react_dispatch as rd
    from dana.tools.schema import ToolCall

    call = ToolCall(tool_id="write_file", arguments={"path": str(_mount / "hi.txt"), "content": "hello mount"})
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None, allowed_mounts=[str(_mount)])
    assert result.ok is True
    assert (_mount / "hi.txt").read_text() == "hello mount"


# --------------------------------------------------------------------------
# Surgical File Editing — edit_file. A strict str.count()/str.replace(),
# deliberately not a diffing/AST library (see edit_file's own docstring for
# why): search_block must match the file's CURRENT content exactly once, or
# the edit is rejected rather than guessing.
# --------------------------------------------------------------------------


def test_edit_file_replaces_the_single_exact_match(_sandbox: Path) -> None:
    (_sandbox / "greet.py").write_text("def greet():\n    print('hello')\n")
    result = file_system.edit_file("greet.py", "print('hello')", "print('goodbye')")
    assert result["ok"] is True
    assert (_sandbox / "greet.py").read_text() == "def greet():\n    print('goodbye')\n"


def test_edit_file_preserves_the_rest_of_a_larger_file(_sandbox: Path) -> None:
    """The whole point of edit_file over write_file: everything OUTSIDE
    search_block must survive untouched."""
    original = "\n".join(f"line {i}" for i in range(1, 21))
    (_sandbox / "big.txt").write_text(original)
    result = file_system.edit_file("big.txt", "line 10", "line TEN")
    assert result["ok"] is True
    updated = (_sandbox / "big.txt").read_text()
    assert "line TEN" in updated
    assert "line 10" not in updated
    for i in (1, 5, 9, 11, 15, 20):
        assert f"line {i}" in updated


def test_edit_file_fails_with_zero_matches(_sandbox: Path) -> None:
    (_sandbox / "notes.txt").write_text("the quick brown fox")
    result = file_system.edit_file("notes.txt", "the slow brown fox", "replacement")
    assert result["ok"] is False
    assert result["error"] == "Search block not found or multiple matches found. Provide a more unique search block."
    assert (_sandbox / "notes.txt").read_text() == "the quick brown fox"  # untouched


def test_edit_file_fails_with_multiple_matches(_sandbox: Path) -> None:
    (_sandbox / "dupes.txt").write_text("retry()\nretry()\nretry()")
    result = file_system.edit_file("dupes.txt", "retry()", "retry_once()")
    assert result["ok"] is False
    assert result["error"] == "Search block not found or multiple matches found. Provide a more unique search block."
    assert (_sandbox / "dupes.txt").read_text() == "retry()\nretry()\nretry()"  # untouched


def test_edit_file_matches_multi_line_blocks_exactly(_sandbox: Path) -> None:
    original = "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n"
    # write_bytes, not write_text: on Windows, write_text's platform-default
    # text mode would translate every bare "\n" to "\r\n" on disk, and a
    # search_block with an EMBEDDED "\n" (unlike every other test above,
    # whose search_block is single-line) would then no longer match those
    # now-"\r\n" line breaks — write_bytes keeps this fixture's on-disk
    # bytes exactly the literal "\n"-only text below, on every platform.
    (_sandbox / "math_ops.py").write_bytes(original.encode("utf-8"))
    search = "def add(a, b):\n    return a + b"
    replace = "def add(a, b):\n    return a + b  # type: ignore"
    result = file_system.edit_file("math_ops.py", search, replace)
    assert result["ok"] is True
    assert (_sandbox / "math_ops.py").read_text() == original.replace(search, replace)


def test_edit_file_reports_clear_error_on_traversal_and_writes_nothing(_sandbox: Path, tmp_path: Path) -> None:
    result = file_system.edit_file("../evil.txt", "a", "b")
    assert result["ok"] is False
    assert "outside the sandbox" in result["error"]
    assert not (tmp_path / "evil.txt").exists()


def test_edit_file_missing_file_reports_error_not_crash(_sandbox: Path) -> None:
    result = file_system.edit_file("does_not_exist.txt", "a", "b")
    assert result["ok"] is False
    assert "does not exist" in result["error"]


def test_edit_file_rejects_directory_target(_sandbox: Path) -> None:
    (_sandbox / "adir").mkdir()
    result = file_system.edit_file("adir", "a", "b")
    assert result["ok"] is False


def test_edit_file_works_inside_a_mounted_directory(_sandbox: Path, _mount: Path) -> None:
    (_mount / "config.py").write_text("DEBUG = False\n")
    result = file_system.edit_file(
        str(_mount / "config.py"), "DEBUG = False", "DEBUG = True", allowed_mounts=[str(_mount)]
    )
    assert result["ok"] is True
    assert (_mount / "config.py").read_text() == "DEBUG = True\n"


def test_edit_file_inside_mount_rejected_without_the_mount_registered(_sandbox: Path, _mount: Path) -> None:
    (_mount / "config.py").write_text("DEBUG = False\n")
    result = file_system.edit_file(str(_mount / "config.py"), "DEBUG = False", "DEBUG = True", allowed_mounts=None)
    assert result["ok"] is False
    assert "outside the sandbox" in result["error"]
    assert (_mount / "config.py").read_text() == "DEBUG = False\n"  # untouched


def test_dispatch_tool_call_edit_file_end_to_end(_sandbox: Path) -> None:
    """dispatch_tool_call itself (not just the handler) must produce a
    successful ToolResult for a real edit — proving the wiring, not just
    the underlying file_system function, works."""
    import dana.core.react_dispatch as rd
    from dana.tools.schema import ToolCall

    (_sandbox / "hi.txt").write_text("hello world")
    call = ToolCall(
        tool_id="edit_file", arguments={"path": "hi.txt", "search_block": "hello world", "replace_block": "hi there"}
    )
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is True
    assert (_sandbox / "hi.txt").read_text() == "hi there"


def test_dispatch_tool_call_edit_file_ambiguous_match_is_digested_not_crashed(_sandbox: Path) -> None:
    """A rejected (0 or 2+ match) edit must come back as a normal digested
    failure, not an uncaught exception — same contract as write_file's own
    traversal-rejection test below."""
    import dana.core.react_dispatch as rd
    from dana.tools.schema import ToolCall

    (_sandbox / "dupes.txt").write_text("x = 1\nx = 1\n")
    call = ToolCall(
        tool_id="edit_file", arguments={"path": "dupes.txt", "search_block": "x = 1", "replace_block": "x = 2"}
    )
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "more unique search block" in result.payload.get("raw_error", "")


def test_dispatch_tool_call_edit_file_reaches_a_mounted_directory(_sandbox: Path, _mount: Path) -> None:
    """End-to-end through react_dispatch.dispatch_tool_call's own
    allowed_mounts param, mirroring write_file's own equivalent test —
    edit_file must be threaded through _TOOLS_NEEDING_MOUNTS the same way."""
    import dana.core.react_dispatch as rd
    from dana.tools.schema import ToolCall

    (_mount / "config.py").write_text("DEBUG = False\n")
    call = ToolCall(
        tool_id="edit_file",
        arguments={"path": str(_mount / "config.py"), "search_block": "DEBUG = False", "replace_block": "DEBUG = True"},
    )
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None, allowed_mounts=[str(_mount)])
    assert result.ok is True
    assert (_mount / "config.py").read_text() == "DEBUG = True\n"


# --------------------------------------------------------------------------
# Codebase Search — search_files. Pure pathlib + string matching (no grep/
# ripgrep subprocess), case-insensitive substring match, capped results.
# --------------------------------------------------------------------------


def test_search_files_finds_a_match_in_a_single_file(_sandbox: Path) -> None:
    (_sandbox / "app.py").write_text("def handler():\n    return connect_to_database()\n")
    result = file_system.search_files(".", "connect_to_database")
    assert result["ok"] is True
    assert result["matches"] == [{"file": "app.py", "line": 2, "content": "return connect_to_database()"}]
    assert result["truncated"] is False


def test_search_files_is_case_insensitive(_sandbox: Path) -> None:
    (_sandbox / "app.py").write_text("MyVariable = 42\n")
    result = file_system.search_files(".", "myvariable")
    assert result["ok"] is True
    assert len(result["matches"]) == 1
    assert result["matches"][0]["content"] == "MyVariable = 42"


def test_search_files_recurses_into_nested_subdirectories(_sandbox: Path) -> None:
    nested = _sandbox / "src" / "utils" / "deep"
    nested.mkdir(parents=True)
    (nested / "helpers.py").write_text("def target_function():\n    pass\n")
    (_sandbox / "unrelated.py").write_text("nothing to see here\n")

    result = file_system.search_files(".", "target_function")
    assert result["ok"] is True
    assert result["matches"] == [
        {"file": "src/utils/deep/helpers.py", "line": 1, "content": "def target_function():"}
    ]


def test_search_files_reports_every_matching_line_in_a_file(_sandbox: Path) -> None:
    (_sandbox / "app.py").write_text("token = 1\nprint(token)\ntoken = token + 1\n")
    result = file_system.search_files(".", "token")
    assert result["ok"] is True
    assert [m["line"] for m in result["matches"]] == [1, 2, 3]


def test_search_files_skips_binary_files_gracefully(_sandbox: Path) -> None:
    (_sandbox / "photo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x01\x02\xff\xfe TARGET_TEXT \x80\x81")
    (_sandbox / "readme.txt").write_text("TARGET_TEXT appears here too\n")

    result = file_system.search_files(".", "TARGET_TEXT")
    assert result["ok"] is True
    files_matched = {m["file"] for m in result["matches"]}
    assert files_matched == {"readme.txt"}  # the .png is silently skipped, not errored


def test_search_files_ignores_git_directory(_sandbox: Path) -> None:
    git_dir = _sandbox / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("SECRET_TOKEN = abc123\n")
    (_sandbox / "app.py").write_text("SECRET_TOKEN = abc123\n")

    result = file_system.search_files(".", "SECRET_TOKEN")
    assert result["ok"] is True
    assert {m["file"] for m in result["matches"]} == {"app.py"}


def test_search_files_ignores_other_hidden_directories(_sandbox: Path) -> None:
    hidden = _sandbox / ".vscode"
    hidden.mkdir()
    (hidden / "settings.json").write_text('{"needle": true}\n')

    result = file_system.search_files(".", "needle")
    assert result["ok"] is True
    assert result["matches"] == []


def test_search_files_ignores_node_modules_pycache_and_venv(_sandbox: Path) -> None:
    for skip_dir in ("node_modules", "__pycache__", "venv"):
        d = _sandbox / skip_dir
        d.mkdir()
        (d / "generated.txt").write_text("needle_value = 1\n")
    (_sandbox / "real_source.py").write_text("needle_value = 1\n")

    result = file_system.search_files(".", "needle_value")
    assert result["ok"] is True
    assert {m["file"] for m in result["matches"]} == {"real_source.py"}


def test_search_files_caps_results_and_reports_truncated(_sandbox: Path) -> None:
    for i in range(60):
        (_sandbox / f"file_{i:02d}.txt").write_text("needle\n")

    result = file_system.search_files(".", "needle")
    assert result["ok"] is True
    assert len(result["matches"]) == 50
    assert result["truncated"] is True


def test_search_files_no_matches_is_still_ok_true(_sandbox: Path) -> None:
    (_sandbox / "app.py").write_text("nothing relevant\n")
    result = file_system.search_files(".", "does_not_exist_anywhere")
    assert result["ok"] is True
    assert result["matches"] == []
    assert result["truncated"] is False


def test_search_files_rejects_traversal(_sandbox: Path) -> None:
    result = file_system.search_files("../../etc", "root")
    assert result["ok"] is False
    assert "outside the sandbox" in result["error"]


def test_search_files_missing_directory_reports_error_not_crash(_sandbox: Path) -> None:
    result = file_system.search_files("nope", "query")
    assert result["ok"] is False
    assert "does not exist" in result["error"]


def test_search_files_rejects_a_file_target(_sandbox: Path) -> None:
    (_sandbox / "a_file.txt").write_text("content")
    result = file_system.search_files("a_file.txt", "content")
    assert result["ok"] is False
    assert "not a directory" in result["error"]


def test_search_files_rejects_empty_query(_sandbox: Path) -> None:
    result = file_system.search_files(".", "")
    assert result["ok"] is False


def test_search_files_searches_inside_a_mounted_directory(_sandbox: Path, _mount: Path) -> None:
    (_mount / "service.py").write_text("class PaymentProcessor:\n    pass\n")
    result = file_system.search_files(str(_mount), "PaymentProcessor", allowed_mounts=[str(_mount)])
    assert result["ok"] is True
    assert len(result["matches"]) == 1
    # A mount-relative match's "file" is an ABSOLUTE path (not relative to
    # the sandbox root, which it isn't under) — directly reusable as-is by
    # read_file/edit_file's own `path` argument.
    assert Path(result["matches"][0]["file"]).resolve() == (_mount / "service.py").resolve()


def test_search_files_inside_mount_rejected_without_the_mount_registered(_sandbox: Path, _mount: Path) -> None:
    (_mount / "service.py").write_text("class PaymentProcessor:\n    pass\n")
    result = file_system.search_files(str(_mount), "PaymentProcessor", allowed_mounts=None)
    assert result["ok"] is False
    assert "outside the sandbox" in result["error"]


def test_dispatch_tool_call_search_files_end_to_end(_sandbox: Path) -> None:
    """dispatch_tool_call itself (not just the handler) must produce a
    successful ToolResult for a real search — proving the wiring, not just
    the underlying file_system function, works."""
    import dana.core.react_dispatch as rd
    from dana.tools.schema import ToolCall

    (_sandbox / "app.py").write_text("def target():\n    pass\n")
    call = ToolCall(tool_id="search_files", arguments={"directory_path": ".", "query": "target"})
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is True
    assert result.payload["matches"] == [{"file": "app.py", "line": 1, "content": "def target():"}]


def test_dispatch_tool_call_search_files_reaches_a_mounted_directory(_sandbox: Path, _mount: Path) -> None:
    """End-to-end through react_dispatch.dispatch_tool_call's own
    allowed_mounts param, mirroring write_file/edit_file's own equivalent
    tests — search_files must be threaded through _TOOLS_NEEDING_MOUNTS the
    same way."""
    import dana.core.react_dispatch as rd
    from dana.tools.schema import ToolCall

    (_mount / "service.py").write_text("class PaymentProcessor:\n    pass\n")
    call = ToolCall(tool_id="search_files", arguments={"directory_path": str(_mount), "query": "PaymentProcessor"})
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None, allowed_mounts=[str(_mount)])
    assert result.ok is True
    assert len(result.payload["matches"]) == 1


# --------------------------------------------------------------------------
# HITL wiring — write_file/edit_file must require approval; list_directory/
# read_file/search_files must not.
# --------------------------------------------------------------------------


def test_write_file_is_mutating_and_requires_hitl() -> None:
    import dana.core.react_dispatch as rd

    assert rd.is_mutating_tool("write_file") is True


def test_edit_file_is_mutating_and_requires_hitl() -> None:
    import dana.core.react_dispatch as rd

    assert rd.is_mutating_tool("edit_file") is True


def test_list_directory_and_read_file_and_search_files_are_not_mutating() -> None:
    import dana.core.react_dispatch as rd

    assert rd.is_mutating_tool("list_directory") is False
    assert rd.is_mutating_tool("read_file") is False
    assert rd.is_mutating_tool("search_files") is False


def test_all_five_os_tools_registered_and_in_os_tools_domain() -> None:
    import dana.core.react_dispatch as rd

    for tool_id in ("list_directory", "read_file", "write_file", "edit_file", "search_files"):
        assert tool_id in rd.TOOL_HANDLERS, tool_id
        assert tool_id in rd._OS_TOOLS_TOOL_IDS, tool_id


def test_edit_file_and_search_files_are_threaded_through_tools_needing_mounts() -> None:
    import dana.core.react_dispatch as rd

    assert "edit_file" in rd._TOOLS_NEEDING_MOUNTS
    assert "search_files" in rd._TOOLS_NEEDING_MOUNTS


def test_dispatch_tool_call_write_file_end_to_end(_sandbox: Path) -> None:
    """dispatch_tool_call itself (not just the handler) must produce a
    successful ToolResult for a real write — proving the wiring, not just
    the underlying file_system function, works."""
    import dana.core.react_dispatch as rd
    from dana.tools.schema import ToolCall

    call = ToolCall(tool_id="write_file", arguments={"path": "hi.txt", "content": "hello world"})
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is True
    assert (_sandbox / "hi.txt").read_text() == "hello world"


def test_dispatch_tool_call_write_file_traversal_is_digested_not_crashed(_sandbox: Path) -> None:
    """A path-escape error must come back as a normal digested failure
    (dispatch_tool_call's usual error shape), not an uncaught exception."""
    import dana.core.react_dispatch as rd
    from dana.tools.schema import ToolCall

    call = ToolCall(tool_id="write_file", arguments={"path": "../escape.txt", "content": "x"})
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "outside the sandbox" in result.payload.get("raw_error", "")
