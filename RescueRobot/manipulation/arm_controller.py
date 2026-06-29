"""
manipulation/arm_controller.py
==============================

Executes manipulation primitives on the robot arm: move to a pre-grasp pose,
close on a handle, and lift — all while polling the torque monitor so the lift
is aborted the moment torque exceeds the safety threshold.

Single responsibility: execute arm motions safely. It does not decide *whether*
to grasp (the mission controller does) or *where* (the grasp planner does).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from manipulation.grasp_planner import GraspPose
from manipulation.torque_monitor import TorqueMonitor
from simulation.robot import RescueRobot
from utils.logger import get_logger
from utils.types import Position3D


class GraspResult(Enum):
    """Outcome of a grasp-and-lift attempt."""

    SUCCESS = auto()
    TORQUE_ABORT = auto()      # torque exceeded safety threshold
    NO_CONTACT = auto()        # never felt the handle
    UNREACHABLE = auto()       # IK / motion could not reach the pose


@dataclass
class GraspExecution:
    """Detailed result of a grasp attempt for logging/decision-making."""

    result: GraspResult
    peak_torque_nm: float
    handle_name: str


class ArmController:
    """Drives the arm through pre-grasp → grasp → lift with torque guarding."""

    def __init__(
        self,
        robot: RescueRobot,
        torque_monitor: TorqueMonitor,
        lift_height: float,
        settle_steps: int = 30,
        workspace_radius: float = 1.5,
        sim_contact_load_nm: float = 6.0,
        sim_lift_load_nm: float = 18.0,
    ) -> None:
        self._log = get_logger("manipulation.arm_controller")
        self._robot = robot
        self._torque = torque_monitor
        self._lift_height = lift_height
        self._settle_steps = settle_steps
        self._workspace_radius = workspace_radius
        # Mock-only nominal loads (ignored by a real torque sensor).
        self._sim_contact_load = sim_contact_load_nm
        self._sim_lift_load = sim_lift_load_nm
        self._holding = False

    @property
    def is_holding(self) -> bool:
        """True if the arm currently believes it is holding the target."""
        return self._holding

    def execute_grasp(self, grasp: GraspPose) -> GraspExecution:
        """
        Run the full grasp-and-lift sequence for a single handle. Aborts
        immediately if torque exceeds the safety threshold at any point.
        """
        self._log.info("Executing grasp on '%s'.", grasp.handle_name)
        peak = 0.0

        # 1) Approach the pre-grasp standoff pose.
        if not self._move_to(grasp.pre_grasp):
            return GraspExecution(GraspResult.UNREACHABLE, peak, grasp.handle_name)

        # 2) Descend onto the handle until contact is felt.
        if not self._move_to(grasp.grasp):
            return GraspExecution(GraspResult.UNREACHABLE, peak, grasp.handle_name)

        # Gripper now in contact with the handle (mock-only load injection).
        self._torque.simulate_load(self._sim_contact_load)
        peak = max(peak, self._settle_and_check_contact())
        if not self._torque.has_contact():
            self._log.warning("No contact torque detected on '%s'.", grasp.handle_name)
            self._torque.simulate_load(0.0)
            return GraspExecution(GraspResult.NO_CONTACT, peak, grasp.handle_name)

        # 3) Lift, monitoring torque continuously. The lifted weight raises the
        # felt load (mock-only injection); a real sensor reports physics torque.
        self._torque.simulate_load(self._sim_lift_load)
        lift_result, lift_peak = self._lift(grasp.grasp)
        peak = max(peak, lift_peak)
        if lift_result is GraspResult.TORQUE_ABORT:
            self.release()
            return GraspExecution(GraspResult.TORQUE_ABORT, peak, grasp.handle_name)

        self._holding = True
        self._log.info(
            "Grasp on '%s' succeeded (peak torque %.1f Nm).",
            grasp.handle_name,
            peak,
        )
        return GraspExecution(GraspResult.SUCCESS, peak, grasp.handle_name)

    def move_to_position(self, position: Position3D) -> bool:
        """Public helper to move the end-effector to a world position."""
        return self._move_to(position)

    def release(self) -> None:
        """Open the gripper / drop any held load and stow torque state."""
        self._holding = False
        self._torque.simulate_load(0.0)  # load transferred away (mock-only)
        self._robot.set_arm_joint_targets(self._robot.get_arm_config())
        self._log.info("Gripper released.")

    def stow(self) -> None:
        """Return the arm to its home configuration."""
        self._robot.move_arm_home()

    # --------------------------------------------------------------- internals
    def _move_to(self, position: Position3D) -> bool:
        """
        Move the end-effector toward a world-frame position. A full IK solve is
        delegated to Isaac Sim's motion generation in a real deployment; here we
        update the tracked EE pose and apply a nominal joint command.
        """
        target = position.as_array()
        ee = self._robot.get_end_effector_position().as_array()
        if not np.all(np.isfinite(target)):
            return False
        # Reachability is evaluated relative to the robot BASE, not the world
        # origin — the arm's workspace moves with the robot.
        base = self._robot.get_pose()
        reach = np.linalg.norm(target[:2] - np.array([base.x, base.y]))
        if reach > self._workspace_radius:
            self._log.debug(
                "Target %.2f m from base exceeds workspace %.2f m.",
                reach,
                self._workspace_radius,
            )
            return False
        self._robot.set_end_effector_position(position)
        # In a real system: solve IK and command joints toward the solution.
        self._step_torque()
        _ = ee  # current EE retained for future trajectory interpolation
        return True

    def _settle_and_check_contact(self) -> float:
        """Hold position briefly so contact torque can build, return peak."""
        peak = 0.0
        for _ in range(self._settle_steps):
            reading = self._torque.sample()
            peak = max(peak, reading.max_abs)
            if self._torque.is_unsafe():
                break
            time.sleep(0.0)  # cooperative yield; real loop is physics-stepped
        return peak

    def _lift(self, grasp_position: Position3D):
        """Lift the load while continuously checking torque. Returns (result, peak)."""
        peak = 0.0
        lifted = grasp_position.as_array() + np.array([0.0, 0.0, self._lift_height])
        steps = 40
        for i in range(steps):
            fraction = (i + 1) / steps
            intermediate = grasp_position.as_array() + np.array(
                [0.0, 0.0, self._lift_height * fraction]
            )
            self._robot.set_end_effector_position(
                Position3D.from_array(intermediate)
            )
            reading = self._torque.sample()
            peak = max(peak, reading.max_abs)
            if self._torque.is_unsafe():
                self._log.warning(
                    "Lift aborted: torque %.1f Nm > %.1f Nm.",
                    reading.max_abs,
                    self._torque.safety_threshold,
                )
                return GraspResult.TORQUE_ABORT, peak
        self._robot.set_end_effector_position(Position3D.from_array(lifted))
        return GraspResult.SUCCESS, peak

    def _step_torque(self) -> None:
        """Take one torque sample (keeps the monitor's reading fresh)."""
        self._torque.sample()
