"""
navigation/path_manager.py
==========================

Owns the current plan and decides when to replan. It rebuilds the occupancy
grid from the latest obstacles, invokes A*, and detects when newly observed
obstacles invalidate the active path (satisfying "continuously update the path
if new obstacles appear").

Single responsibility: maintain a valid, up-to-date path to the goal.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from navigation.astar_planner import AStarPlanner
from navigation.occupancy_grid import OccupancyGrid
from utils.logger import get_logger
from utils.types import Obstacle, WorldModel

Waypoint = Tuple[float, float]


class PathManager:
    """Maintains and revalidates the path from the robot to its goal."""

    def __init__(self, grid: OccupancyGrid, planner: AStarPlanner) -> None:
        self._log = get_logger("navigation.path_manager")
        self._grid = grid
        self._planner = planner
        self._path: List[Waypoint] = []
        self._goal: Optional[Waypoint] = None

    @property
    def path(self) -> List[Waypoint]:
        return list(self._path)

    @property
    def goal(self) -> Optional[Waypoint]:
        return self._goal

    def set_goal(self, goal: Waypoint) -> None:
        """Set a new navigation goal (forces a replan on next update)."""
        self._goal = goal
        self._path = []

    def update(self, world: WorldModel) -> List[Waypoint]:
        """
        Refresh the occupancy grid from current obstacles and (re)plan if the
        path is missing or now blocked. Returns the current waypoint list.
        """
        if self._goal is None:
            return []

        self._grid.update_from_obstacles(world.obstacles)
        start = (world.robot_pose.x, world.robot_pose.y)

        if not self._path or self._path_blocked():
            self._replan(start, self._goal, world.obstacles)
        return list(self._path)

    def replan_now(self, world: WorldModel) -> List[Waypoint]:
        """Force an immediate replan (e.g. on a safety-triggered REPLAN)."""
        if self._goal is None:
            return []
        self._grid.update_from_obstacles(world.obstacles)
        self._replan(
            (world.robot_pose.x, world.robot_pose.y), self._goal, world.obstacles
        )
        return list(self._path)

    # --------------------------------------------------------------- internals
    def _replan(
        self, start: Waypoint, goal: Waypoint, obstacles: List[Obstacle]
    ) -> None:
        self._path = self._planner.plan(start, goal)
        if self._path:
            self._log.info(
                "Replanned: %d waypoint(s) avoiding %d obstacle(s).",
                len(self._path),
                len(obstacles),
            )
        else:
            self._log.warning("Replan failed: no collision-free path to goal.")

    def _path_blocked(self) -> bool:
        """True if any waypoint on the current path now lies in an occupied cell."""
        for x, y in self._path:
            col, row = self._grid.world_to_grid(x, y)
            if not self._grid.is_free(col, row):
                self._log.info("Active path blocked by a new obstacle; replanning.")
                return True
        return False
