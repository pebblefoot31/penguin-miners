"""
navigation/waypoint_follower.py
===============================

Drives the mobile base along a sequence of waypoints using a simple
pure-pursuit-style controller: steer toward the current lookahead waypoint,
advancing through the list as each is reached.

Single responsibility: convert a waypoint list + current pose into base
velocity commands. It does not plan paths.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from simulation.robot import RescueRobot
from utils.geometry import angle_to, euclidean_distance, wrap_to_pi
from utils.logger import get_logger
from utils.types import Pose2D

Waypoint = Tuple[float, float]


class WaypointFollower:
    """Pure-pursuit waypoint follower for the mobile base."""

    def __init__(
        self,
        robot: RescueRobot,
        max_linear_speed: float,
        max_angular_speed: float,
        arrival_tolerance: float,
        lookahead_distance: float,
    ) -> None:
        self._log = get_logger("navigation.waypoint_follower")
        self._robot = robot
        self._max_lin = max_linear_speed
        self._max_ang = max_angular_speed
        self._arrival_tol = arrival_tolerance
        self._lookahead = lookahead_distance
        self._path: List[Waypoint] = []
        self._index = 0

    def set_path(self, path: List[Waypoint]) -> None:
        """
        Provide a path to follow. If it is identical to the current path,
        progress (the waypoint index) is preserved — callers may push the same
        path every tick without resetting the controller. A genuinely new path
        resets progress to the first waypoint.
        """
        new_path = list(path)
        if new_path == self._path:
            return
        self._path = new_path
        self._index = 0

    def is_finished(self) -> bool:
        """True once the final waypoint has been reached."""
        return self._index >= len(self._path)

    def step(self, pose: Pose2D, dt: float) -> bool:
        """
        Advance the controller one tick. Issues a base velocity command toward
        the current lookahead waypoint. Returns ``True`` when the path is done.
        """
        if self.is_finished() or not self._path:
            self._robot.stop_base()
            return True

        target = self._current_target(pose)
        if target is None:
            self._robot.stop_base()
            return True

        distance = euclidean_distance((pose.x, pose.y), target)
        desired_heading = angle_to(target, (pose.x, pose.y))
        heading_error = wrap_to_pi(desired_heading - pose.yaw)

        angular = max(-self._max_ang, min(self._max_ang, 2.0 * heading_error))
        # Slow down while turning sharply and as we approach the waypoint.
        turn_scale = max(0.0, math.cos(heading_error))
        linear = min(self._max_lin, 1.5 * distance) * turn_scale

        self._robot.command_base_velocity(linear, angular, dt)
        return self.is_finished()

    # --------------------------------------------------------------- internals
    def _current_target(self, pose: Pose2D) -> Optional[Waypoint]:
        """
        Return the active lookahead waypoint, advancing the index past any
        waypoints already within the arrival tolerance.
        """
        while self._index < len(self._path):
            waypoint = self._path[self._index]
            if euclidean_distance((pose.x, pose.y), waypoint) <= self._arrival_tol:
                self._index += 1
                continue
            return waypoint
        return None

    @property
    def remaining_waypoints(self) -> int:
        return max(0, len(self._path) - self._index)
