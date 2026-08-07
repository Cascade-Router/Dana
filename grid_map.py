"""Grid map with obstacle-aware neighbor queries."""

class GridMap:

    def __init__(self, width, height, obstacles=None):
        self.width = int(width)
        self.height = int(height)
        self.obstacles = set(obstacles) if obstacles else set()

    def is_valid_cell(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height and ((x, y) not in self.obstacles)

    def get_neighbors(self, x, y):
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = (x + dx, y + dy)
            if self.is_valid_cell(nx, ny):
                neighbors.append((nx, ny))
        return neighbors
__all__ = ('GridMap',)
