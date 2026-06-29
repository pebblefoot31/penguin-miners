"""
navigation/occupancy_grid.py
============================

A 2D occupancy grid built from detected obstacles. Provides world<->grid
coordinate conversion and obstacle inflation (Minkowski-style padding by the
robot footprint radius) so the planner can treat the robot as a point.

Single responsibility: maintain the occupancy grid and its coordinate mapping.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np

from utils.logger import get_logger
from utils.types import Obstacle

FREE = 0
OCCUPIED = 1


class OccupancyGrid:
    """A discretised 2D map of free/occupied cells."""

    def __init__(
        self,
        resolution: float,
        width: int,
        height: int,
        origin: Tuple[float, float],
        inflation_cells: int = 0,
    ) -> None:
        self._log = get_logger("navigation.occupancy_grid")
        self._res = resolution
        self._w = width
        self._h = height
        self._origin = np.array(origin, dtype=float)
        self._inflation = inflation_cells
        self._grid = np.zeros((height, width), dtype=np.uint8)

    # --------------------------------------------------------- coordinate maps
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world (x, y) metres to grid (col, row) indices."""
        col = int((x - self._origin[0]) / self._res)
        row = int((y - self._origin[1]) / self._res)
        return col, row

    def grid_to_world(self, col: int, row: int) -> Tuple[float, float]:
        """Convert grid (col, row) to the world (x, y) of the cell centre."""
        x = self._origin[0] + (col + 0.5) * self._res
        y = self._origin[1] + (row + 0.5) * self._res
        return x, y

    def in_bounds(self, col: int, row: int) -> bool:
        return 0 <= col < self._w and 0 <= row < self._h

    def is_free(self, col: int, row: int) -> bool:
        """True if the cell is in bounds and not occupied."""
        return self.in_bounds(col, row) and self._grid[row, col] == FREE

    # ----------------------------------------------------------------- updates
    def clear(self) -> None:
        """Reset all cells to free."""
        self._grid.fill(FREE)

    def update_from_obstacles(self, obstacles: Iterable[Obstacle]) -> None:
        """Rebuild the grid from a fresh set of obstacles, then inflate."""
        self.clear()
        for obstacle in obstacles:
            self._mark_obstacle(obstacle)
        if self._inflation > 0:
            self._inflate(self._inflation)

    def _mark_obstacle(self, obstacle: Obstacle) -> None:
        """Mark all cells within the obstacle's radius as occupied."""
        cx, cy = self.world_to_grid(obstacle.centroid.x, obstacle.centroid.y)
        radius_cells = int(np.ceil(obstacle.radius / self._res))
        for d_row in range(-radius_cells, radius_cells + 1):
            for d_col in range(-radius_cells, radius_cells + 1):
                if d_col * d_col + d_row * d_row > radius_cells * radius_cells:
                    continue
                col, row = cx + d_col, cy + d_row
                if self.in_bounds(col, row):
                    self._grid[row, col] = OCCUPIED

    def _inflate(self, cells: int) -> None:
        """Dilate occupied cells by ``cells`` to account for robot radius."""
        occupied = np.argwhere(self._grid == OCCUPIED)
        inflated = self._grid.copy()
        for row, col in occupied:
            r0, r1 = max(0, row - cells), min(self._h, row + cells + 1)
            c0, c1 = max(0, col - cells), min(self._w, col + cells + 1)
            inflated[r0:r1, c0:c1] = OCCUPIED
        self._grid = inflated

    # ------------------------------------------------------------------ access
    @property
    def shape(self) -> Tuple[int, int]:
        return (self._h, self._w)

    @property
    def matrix(self) -> np.ndarray:
        """The raw occupancy matrix (rows, cols)."""
        return self._grid

    def occupied_cell_count(self) -> int:
        return int(np.count_nonzero(self._grid))
