"""
manipulation/capsule_loader.py
==============================

Moves a successfully grasped target into the rescue/transport capsule and
verifies the load. Used after a successful lift in the manipulation pipeline.

Single responsibility: place the held target into the capsule and confirm it.
"""

from __future__ import annotations

import math

import numpy as np

from manipulation.arm_controller import ArmController
from manipulation.torque_monitor import TorqueMonitor
from simulation.robot import RescueRobot
from utils.logger import get_logger
from utils.types import Position3D


class CapsuleLoader:
    """Loads the grasped target into the transport capsule."""

    def __init__(
        self,
        arm: ArmController,
        torque_monitor: TorqueMonitor,
        capsule_load_pose: Position3D,
        robot: RescueRobot,
    ) -> None:
        self._log = get_logger("manipulation.capsule_loader")
        self._arm = arm
        self._torque = torque_monitor
        # The capsule rides on the robot, so its load pose is expressed in the
        # robot base frame and transformed to world at load time.
        self._load_pose_base = capsule_load_pose
        self._robot = robot

    def load(self) -> bool:
        """
        Carry the held target to the capsule load pose, release it, and verify.
        Returns ``True`` if loading is verified, ``False`` otherwise.
        """
        if not self._arm.is_holding:
            self._log.warning("Cannot load: arm is not holding a target.")
            return False

        load_world = self._load_pose_world()
        self._log.info(
            "Moving target to capsule load pose (world %.2f, %.2f, %.2f).",
            load_world.x,
            load_world.y,
            load_world.z,
        )
        if not self._arm.move_to_position(load_world):
            self._log.warning("Capsule load pose unreachable.")
            return False

        if self._torque.is_unsafe():
            self._log.warning("Torque unsafe during capsule transfer; aborting load.")
            return False

        self._arm.release()
        verified = self._verify_load()
        if verified:
            self._log.info("Target successfully loaded into capsule.")
        else:
            self._log.warning("Capsule load verification failed.")
        self._arm.stow()
        return verified

    def _load_pose_world(self) -> Position3D:
        """Transform the base-relative capsule load pose into the world frame."""
        pose = self._robot.get_pose()
        bx, by, bz = (
            self._load_pose_base.x,
            self._load_pose_base.y,
            self._load_pose_base.z,
        )
        wx = pose.x + bx * math.cos(pose.yaw) - by * math.sin(pose.yaw)
        wy = pose.y + bx * math.sin(pose.yaw) + by * math.cos(pose.yaw)
        return Position3D(wx, wy, bz)

    def _verify_load(self) -> bool:
        """
        Verify the target settled in the capsule. After release the arm should
        feel near-zero residual torque (load transferred to the capsule).
        """
        reading = self._torque.sample()
        residual = reading.max_abs
        settled = residual < self._torque.safety_threshold * 0.3
        self._log.debug("Post-release residual torque: %.1f Nm.", residual)
        return bool(settled and np.isfinite(residual))
