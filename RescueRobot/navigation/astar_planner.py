"""
navigation/astar_planner.py
===========================

A* path planner operating on an :class:`OccupancyGrid`. Returns a list of
world-frame waypoints from start to goal, or an empty list if no path exists.

Single responsibility: compute a collision-free grid path with A*.
"""

from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Tuple

from navigation.occupancy_grid import OccupancyGrid
from utils.logger import get_logger

Cell = Tuple[int, int]  # (col, row)


class AStarPlanner:
    """Grid-based A* planner with optional diagonal moves."""

    def __init__(
        self,
        grid: OccupancyGrid,
        allow_diagonal: bool = True,
        heuristic_weight: float = 1.0,
    ) -> None:
        self._log = get_logger("navigation.astar")
        self._grid = grid
        self._allow_diagonal = allow_diagonal
        self._h_weight = heuristic_weight

    def plan(
        self, start_world: Tuple[float, float], goal_world: Tuple[float, float]
    ) -> List[Tuple[float, float]]:
        """
        Plan a path between two world-frame points. Returns a list of world
        waypoints (including the goal), or ``[]`` if unreachable.
        """
        start = self._grid.world_to_grid(*start_world)
        goal = self._grid.world_to_grid(*goal_world)

        if not self._grid.in_bounds(*start) or not self._grid.in_bounds(*goal):
            self._log.warning("Start or goal outside grid bounds.")
            return []
        goal = self._nearest_free(goal)
        if goal is None:
            self._log.warning("Goal region fully blocked; no path.")
            return []

        cell_path = self._astar(start, goal)
        if not cell_path:
            self._log.info("A* found no path from %s to %s.", start, goal)
            return []

        world_path = [self._grid.grid_to_world(col, row) for col, row in cell_path]
        return self._simplify(world_path)

    # --------------------------------------------------------------- internals
    def _astar(self, start: Cell, goal: Cell) -> List[Cell]:
        """Core A* search returning a list of grid cells."""
        open_heap: List[Tuple[float, Cell]] = [(0.0, start)]
        came_from: Dict[Cell, Cell] = {}
        g_score: Dict[Cell, float] = {start: 0.0}
        closed = set()

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current == goal:
                return self._reconstruct(came_from, current)
            if current in closed:
                continue
            closed.add(current)

            for neighbour, step_cost in self._neighbours(current):
                if neighbour in closed:
                    continue
                tentative = g_score[current] + step_cost
                if tentative < g_score.get(neighbour, math.inf):
                    came_from[neighbour] = current
                    g_score[neighbour] = tentative
                    f = tentative + self._h_weight * self._heuristic(neighbour, goal)
                    heapq.heappush(open_heap, (f, neighbour))
        return []

    def _neighbours(self, cell: Cell):
        """Yield free neighbouring cells and their step costs."""
        col, row = cell
        moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        if self._allow_diagonal:
            moves += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        for d_col, d_row in moves:
            n = (col + d_col, row + d_row)
            if not self._grid.is_free(*n):
                continue
            # Prevent corner-cutting between two diagonal obstacles.
            if d_col != 0 and d_row != 0:
                if not (
                    self._grid.is_free(col + d_col, row)
                    and self._grid.is_free(col, row + d_row)
                ):
                    continue
            cost = math.hypot(d_col, d_row)
            yield n, cost

    def _heuristic(self, a: Cell, b: Cell) -> float:
        """Octile distance heuristic (admissible with diagonal moves)."""
        dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
        if self._allow_diagonal:
            return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)
        return dx + dy

    @staticmethod
    def _reconstruct(came_from: Dict[Cell, Cell], current: Cell) -> List[Cell]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def _nearest_free(self, cell: Cell, max_radius: int = 10) -> Optional[Cell]:
        """Find the nearest free cell to ``cell`` (goal may sit on inflation)."""
        if self._grid.is_free(*cell):
            return cell
        col, row = cell
        for radius in range(1, max_radius + 1):
            for d_col in range(-radius, radius + 1):
                for d_row in range(-radius, radius + 1):
                    candidate = (col + d_col, row + d_row)
                    if self._grid.is_free(*candidate):
                        return candidate
        return None

    @staticmethod
    def _simplify(
        path: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Drop collinear intermediate waypoints to shorten the waypoint list."""
        if len(path) <= 2:
            return path
        simplified = [path[0]]
        for i in range(1, len(path) - 1):
            ax, ay = simplified[-1]
            bx, by = path[i]
            cx, cy = path[i + 1]
            # Cross product of AB and BC; ~0 => collinear, skip B.
            cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
            if abs(cross) > 1e-6:
                simplified.append(path[i])
        simplified.append(path[-1])
        return simplified
