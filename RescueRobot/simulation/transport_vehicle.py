"""
simulation/transport_vehicle.py
===============================

Models the transport vehicle the robot rides on until deployment. Provides the
deploy (undock) operation that releases the robot into the disaster environment
and reports whether the robot is still mounted.

Single responsibility: manage the mounted/deployed state and the undock action.
"""

from __future__ import annotations

from typing import Any, Optional

from utils.logger import get_logger
from utils.types import Pose2D


class TransportVehicle:
    """Carries the robot to the scene and deploys it on command."""

    def __init__(self, deploy_pose: Pose2D, mock: bool = False) -> None:
        self._log = get_logger("transport_vehicle")
        self._deploy_pose = deploy_pose
        self._mock = mock
        self._robot_mounted = True

    @property
    def robot_mounted(self) -> bool:
        """True while the robot is still on the transport vehicle."""
        return self._robot_mounted

    def deploy(self, robot: Any) -> Pose2D:
        """
        Undock the robot from the vehicle and place it at the deploy pose.

        Returns the world pose the robot was deployed to. After this call the
        mission may begin autonomous operation.
        """
        if not self._robot_mounted:
            self._log.info("Robot already deployed; deploy() is a no-op.")
            return self._deploy_pose

        self._log.info(
            "Deploying robot at (%.2f, %.2f, yaw=%.2f).",
            self._deploy_pose.x,
            self._deploy_pose.y,
            self._deploy_pose.yaw,
        )
        self._place_robot(robot)
        self._robot_mounted = False
        return self._deploy_pose

    def _place_robot(self, robot: Any) -> None:
        """Set the robot's world pose to the deploy pose (best effort)."""
        articulation: Optional[Any] = getattr(robot, "articulation", None)
        if articulation is None:
            # Mock mode: nudge the robot's odometry origin to the deploy pose.
            if hasattr(robot, "_pose"):
                robot._pose = Pose2D(
                    self._deploy_pose.x, self._deploy_pose.y, self._deploy_pose.yaw
                )
            return
        try:  # pragma: no cover - requires Isaac runtime
            import numpy as np

            half = self._deploy_pose.yaw / 2.0
            quat = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
            articulation.set_world_pose(
                position=np.array([self._deploy_pose.x, self._deploy_pose.y, 0.0]),
                orientation=quat,
            )
        except Exception as exc:  # pragma: no cover
            self._log.warning("Could not set robot world pose (%s).", exc)
