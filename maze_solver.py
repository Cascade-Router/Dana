"""BFS maze pathfinding (Epic 2 artifact)."""

from __future__ import annotations

from collections import deque
from typing import Hashable, Iterable, Sequence

Grid = Sequence[Sequence[Hashable]]
Cell = tuple[int, int]

# Walkable cell markers used by generated / hand-written tests.
_WALKABLE = frozenset({".", " ", "0", "S", "E", "s", "e", 0})
_START = frozenset({"S", "s"})
_END = frozenset({"E", "e"})


def _in_bounds(grid: Grid, r: int, c: int) -> bool:
    return 0 <= r < len(grid) and 0 <= c < len(grid[0])


def _is_walkable(cell: Hashable) -> bool:
    return cell in _WALKABLE


def find_marker(grid: Grid, markers: Iterable[Hashable]) -> Cell | None:
    want = set(markers)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            if val in want:
                return (r, c)
    return None


def bfs_path(
    grid: Grid,
    start: Cell | None = None,
    end: Cell | None = None,
) -> list[Cell]:
    """Return a shortest path from start→end using BFS (4-connected).

    Cells marked ``1`` (or other non-walkable tokens) are walls.
    Raises ``ValueError`` when start/end are missing or no path exists.
    """
    if not grid or not grid[0]:
        raise ValueError("grid must be non-empty")

    start_cell = start or find_marker(grid, _START)
    end_cell = end or find_marker(grid, _END)
    if start_cell is None:
        raise ValueError("start cell 'S' not found")
    if end_cell is None:
        raise ValueError("end cell 'E' not found")

    q: deque[Cell] = deque([start_cell])
    parent: dict[Cell, Cell | None] = {start_cell: None}
    while q:
        r, c = q.popleft()
        if (r, c) == end_cell:
            break
        for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)
            if nxt in parent:
                continue
            if not _in_bounds(grid, nr, nc):
                continue
            if not _is_walkable(grid[nr][nc]):
                continue
            parent[nxt] = (r, c)
            q.append(nxt)
    else:
        raise ValueError("no path from start to end")

    path: list[Cell] = []
    cur: Cell | None = end_cell
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    path.reverse()
    if not path or path[0] != start_cell:
        raise ValueError("no path from start to end")
    return path


class Maze:
    """Thin wrapper around a grid that exposes BFS solving."""

    def __init__(self, grid: Grid) -> None:
        self.grid = [list(row) for row in grid]
        self.rows = len(self.grid)
        self.cols = len(self.grid[0]) if self.grid else 0

    def solve(self) -> list[Cell]:
        return bfs_path(self.grid)


# Alias expected by some broker-authored tests.
bfs = bfs_path

__all__ = ("Maze", "bfs", "bfs_path", "find_marker")
