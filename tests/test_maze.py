"""Verify BFS maze pathfinding against a known grid (Epic 3 artifact)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maze_solver import Maze, bfs, bfs_path


def test_bfs_known_grid_shortest_path() -> None:
    grid = [
        ["S", ".", "1", "."],
        [".", ".", "1", "."],
        ["1", ".", ".", "E"],
    ]
    path = bfs_path(grid)
    assert path[0] == (0, 0)
    assert path[-1] == (2, 3)
    # Exactly one shortest-route length for this layout (ties allowed on route).
    assert len(path) == 6
    # Every step is a 4-connected move on a walkable cell.
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        assert abs(r0 - r1) + abs(c0 - c1) == 1
        assert grid[r1][c1] != "1"


def test_bfs_alias_and_maze_wrapper() -> None:
    grid = [
        ["S", "0", "1"],
        ["0", "0", "0"],
        ["1", "0", "E"],
    ]
    via_alias = bfs(grid, (0, 0), (2, 2))
    via_maze = Maze(grid).solve()
    assert via_alias[0] == (0, 0)
    assert via_alias[-1] == (2, 2)
    assert via_maze == via_alias


def test_bfs_blocked_raises() -> None:
    grid = [
        ["S", "1", "E"],
    ]
    with pytest.raises(ValueError, match="no path"):
        bfs_path(grid)
