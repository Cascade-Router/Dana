"""BFS path planner over a GridMap."""
import queue
from collections import deque
from grid_map import GridMap

class RobotRouter:
    """Plan 4-connected paths on a ``GridMap`` using BFS with a visited set."""

    def plan_path(self, grid: GridMap, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
        """Return a list of ``(x, y)`` steps from ``start`` to ``goal``, or None.

        ``start == goal`` returns ``[start]``. Unreachable goals (blocked cells
        or no path) return ``None``. A ``visited`` set prevents infinite loops
        when exploring blocked or cyclic neighborhoods.
        """
        if not isinstance(grid, GridMap):
            raise TypeError('grid must be a GridMap instance')
        sx, sy = start
        gx, gy = goal
        if not grid.is_valid_cell(sx, sy) or not grid.is_valid_cell(gx, gy):
            return None
        if start == goal:
            return [start]
        queue: deque[tuple[tuple[int, int], list[tuple[int, int]]]] = deque()
        queue.append((start, [start]))
        visited: set[tuple[int, int]] = {start}
        while queue:
            current, path = queue.popleft()
            for neighbor in grid.get_neighbors(*current):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                next_path = path + [neighbor]
                if neighbor == goal:
                    return next_path
                queue.append((neighbor, next_path))
        return None
__all__ = ('RobotRouter',)
