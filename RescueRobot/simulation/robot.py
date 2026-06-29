"""
simulation/robot.py
===================

Robot abstraction layer. Wraps the Isaac Sim articulation and exposes a clean
command API used by the controllers:

* base velocity / pose control (mobile base)
* arm joint control + end-effector pose targets
* odometry / current pose readout

Single responsibility: translate high-level controller commands into Isaac Sim
articulation actions (or mock state updates). It contains **no** planning logic.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional

import numpy as np

from utils.logger import get_logger
from utils.types import Pose2D, Position3D


class RescueRobot:
    """Mobile-manipulator robot: a wheeled base plus an N-DOF arm."""

    def __init__(
        self,
        prim_path: str,
        arm_dof_count: int,
        arm_home_config: List[float],
        max_linear_speed: float,
        max_angular_speed: float,
        mock: bool = False,
    ) -> None:
        self._log = get_logger("robot")
        self._prim_path = prim_path
        self._arm_dof_count = arm_dof_count
        self._arm_home = list(arm_home_config)
        self._max_lin = max_linear_speed
        self._max_ang = max_angular_speed
        self._mock = mock

        self._articulation: Optional[Any] = None
        self._base_controller: Optional[Any] = None

        # Mock-mode kinematic state (also used as the odometry source).
        self._pose = Pose2D(0.0, 0.0, 0.0)
        self._arm_config = list(arm_home_config)
        self._ee_position = Position3D(0.4, 0.0, 0.6)

        if not mock:
            self._initialise()

    def _initialise(self) -> None:
        try:
            from isaacsim.core.prims import Articulation  # type: ignore

            self._articulation = Articulation(self._prim_path)
            self._articulation.initialize()
            self._log.info("Robot articulation bound at %s.", self._prim_path)
        except Exception as exc:  # pragma: no cover
            self._log.warning("Robot mock fallback (%s).", exc)
            self._mock = True

    # ------------------------------------------------------------------ base
    def command_base_velocity(self, linear: float, angular: float, dt: float) -> None:
        """
        Command the mobile base with a (clamped) linear and angular velocity.
        In mock mode this integrates a simple unicycle model for odometry.
        """
        linear = float(np.clip(linear, -self._max_lin, self._max_lin))
        angular = float(np.clip(angular, -self._max_ang, self._max_ang))

        if self._mock or self._articulation is None:
            self._pose.yaw += angular * dt
            self._pose.x += linear * math.cos(self._pose.yaw) * dt
            self._pose.y += linear * math.sin(self._pose.yaw) * dt
            return

        try:  # pragma: no cover - requires Isaac runtime
            from isaacsim.robot.wheeled_robots.controllers.differential_controller import (  # type: ignore
                DifferentialController,
            )

            if self._base_controller is None:
                self._base_controller = DifferentialController(
                    name="base_ctrl", wheel_radius=0.1, wheel_base=0.5
                )
            action = self._base_controller.forward(np.array([linear, angular]))
            self._articulation.apply_action(action)
        except Exception as exc:  # pragma: no cover
            self._log.warning("Base command failed (%s).", exc)

    def stop_base(self) -> None:
        """Immediately halt base motion."""
        self.command_base_velocity(0.0, 0.0, 0.0)

    def get_pose(self) -> Pose2D:
        """Return the current planar base pose (odometry)."""
        if self._mock or self._articulation is None:
            return Pose2D(self._pose.x, self._pose.y, self._pose.yaw)
        try:  # pragma: no cover
            position, orientation = self._articulation.get_world_pose()
            yaw = self._quat_to_yaw(orientation)
            return Pose2D(float(position[0]), float(position[1]), yaw)
        except Exception:  # pragma: no cover
            return Pose2D(self._pose.x, self._pose.y, self._pose.yaw)

    @staticmethod
    def _quat_to_yaw(quat: np.ndarray) -> float:
        """Extract yaw from a (w, x, y, z) quaternion."""
        w, x, y, z = (float(v) for v in quat)
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    # ------------------------------------------------------------------- arm
    def set_arm_joint_targets(self, joint_targets: List[float]) -> None:
        """Command the arm joints to the given configuration (radians)."""
        if len(joint_targets) != self._arm_dof_count:
            raise ValueError(
                f"Expected {self._arm_dof_count} joint targets, "
                f"got {len(joint_targets)}."
            )
        self._arm_config = list(joint_targets)
        if self._mock or self._articulation is None:
            return
        try:  # pragma: no cover
            from isaacsim.core.utils.types import ArticulationAction  # type: ignore

            action = ArticulationAction(joint_positions=np.array(joint_targets))
            self._articulation.apply_action(action)
        except Exception as exc:  # pragma: no cover
            self._log.warning("Arm command failed (%s).", exc)

    def move_arm_home(self) -> None:
        """Return the arm to its stowed home configuration."""
        self.set_arm_joint_targets(self._arm_home)

    def get_arm_config(self) -> List[float]:
        """Current arm joint configuration."""
        return list(self._arm_config)

    def get_end_effector_position(self) -> Position3D:
        """Return the current end-effector position (world frame)."""
        return self._ee_position

    def set_end_effector_position(self, position: Position3D) -> None:
        """Mock helper used by the arm controller to track the EE pose."""
        self._ee_position = position

    @property
    def articulation(self) -> Optional[Any]:
        """The underlying Isaac Sim articulation (None in mock mode)."""
        return self._articulation
