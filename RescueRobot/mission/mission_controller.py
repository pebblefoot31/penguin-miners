"""
mission/mission_controller.py
=============================

The top-level mission orchestrator. Each control tick it:

1. Updates perception -> fused world model.
2. Asks the high-level LLM planner for the next action (given structured state).
3. Enforces safety rules (target loss, obstacle replanning, torque abort).
4. Executes the chosen action via the deterministic controllers.
5. Drives the mission state machine accordingly.

The LLM proposes intent; this controller has final authority and owns all
safety enforcement. Every decision and transition is logged.

Single responsibility: orchestrate the perception → planning → control loop and
enforce mission/safety policy.
"""

from __future__ import annotations

import time
from typing import Optional

from ai.llm_planner import LLMPlanner, PlannerDecision
from manipulation.arm_controller import ArmController, GraspResult
from manipulation.capsule_loader import CapsuleLoader
from manipulation.grasp_planner import GraspPlanner
from manipulation.torque_monitor import TorqueMonitor
from mission.state_machine import StateMachine
from mission.states import HighLevelAction, MissionState
from navigation.path_manager import PathManager
from navigation.waypoint_follower import WaypointFollower
from perception.world_model import WorldModelBuilder
from simulation.robot import RescueRobot
from utils.geometry import angle_to, euclidean_distance, wrap_to_pi
from utils.logger import DecisionLogger, get_logger
from utils.types import WorldModel


class MissionController:
    """Coordinates perception, planning, navigation, and manipulation."""

    def __init__(
        self,
        *,
        robot: RescueRobot,
        world_builder: WorldModelBuilder,
        path_manager: PathManager,
        waypoint_follower: WaypointFollower,
        grasp_planner: GraspPlanner,
        arm_controller: ArmController,
        torque_monitor: TorqueMonitor,
        capsule_loader: CapsuleLoader,
        planner: LLMPlanner,
        state_machine: StateMachine,
        decision_logger: DecisionLogger,
        config,
    ) -> None:
        self._log = get_logger("mission.controller")
        self._robot = robot
        self._world_builder = world_builder
        self._path = path_manager
        self._follower = waypoint_follower
        self._grasp_planner = grasp_planner
        self._arm = arm_controller
        self._torque = torque_monitor
        self._capsule = capsule_loader
        self._planner = planner
        self._fsm = state_machine
        self._decisions = decision_logger
        self._cfg = config

        # Tunables.
        self._approach_distance = config.get(
            "navigation.waypoint_follower.approach_distance", 1.0
        )
        self._search_rotate_speed = config.get("mission.search_rotate_speed", 0.4)
        self._abort_on_target_loss = config.get("mission.abort_on_target_loss", True)
        self._planner_period = config.get("ai.request_period_s", 1.0)

        # Runtime state.
        self._last_world: Optional[WorldModel] = None
        self._last_decision: Optional[PlannerDecision] = None
        self._last_decision_state: Optional[MissionState] = None
        self._last_grasp_summary: str = "none"
        self._last_planner_time = 0.0
        self._mission_start = time.time()

    # ------------------------------------------------------------------ tick
    def tick(self, dt: float) -> MissionState:
        """
        Run one mission control cycle. Returns the current mission state. The
        caller is responsible for stepping the simulation.
        """
        world = self._world_builder.update(self._robot.get_pose())
        self._last_world = world

        # Safety enforcement runs every tick, independent of the planner.
        if self._enforce_safety(world):
            return self._fsm.state

        decision = self._consult_planner(world)
        self._apply_action(decision.action, world, dt)
        return self._fsm.state

    # ------------------------------------------------------------- safety
    def _enforce_safety(self, world: WorldModel) -> bool:
        """
        Enforce hard safety rules. Returns ``True`` if a safety action
        pre-empted normal planning this tick.

        Rules:
          * Stop movement / handle target loss in motion states.
          * Replan when the active path is blocked by a new obstacle.
          * Abort manipulation if torque exceeds the configured limit.
        """
        state = self._fsm.state

        # Torque abort during grasp/lift/transfer.
        if state in (MissionState.GRASP, MissionState.VERIFY_LOAD, MissionState.LOAD_CAPSULE):
            if self._torque.is_unsafe():
                self._decisions.log(
                    "safety_abort",
                    {"reason": "torque_exceeded", "max_nm": self._torque.max_torque},
                )
                self._arm.release()
                self._abort("Torque exceeded safety threshold.")
                return True

        # Target loss while pursuing the target.
        if state in (MissionState.NAVIGATE, MissionState.APPROACH) and not world.has_target:
            self._robot.stop_base()
            self._decisions.log("safety_event", {"reason": "target_lost", "state": state.value})
            if self._abort_on_target_loss and state is MissionState.APPROACH:
                self._abort("Target lost during final approach.")
            else:
                self._fsm.transition(MissionState.SEARCH, "Target lost; resume search.")
            return True

        # Obstacle-driven replanning while navigating.
        if state is MissionState.NAVIGATE:
            new_path = self._path.update(world)
            if not new_path:
                self._decisions.log("safety_event", {"reason": "no_path"})
            self._follower.set_path(new_path)
        return False

    # ------------------------------------------------------------- planner
    def _consult_planner(self, world: WorldModel) -> PlannerDecision:
        """
        Ask the high-level planner for the next action, throttled to the
        configured period. The decision is refreshed immediately whenever the
        mission state changes (a stale action from a previous state must never
        be replayed), otherwise the last decision is reused between calls.
        """
        now = time.time()
        state_changed = self._last_decision_state is not self._fsm.state
        if (
            self._last_decision is None
            or state_changed
            or (now - self._last_planner_time) >= self._planner_period
        ):
            self._last_decision = self._planner.decide(
                self._fsm.state,
                world,
                path_length=len(self._path.path),
                last_grasp_summary=self._last_grasp_summary,
            )
            self._last_planner_time = now
            self._last_decision_state = self._fsm.state
        return self._last_decision

    # ------------------------------------------------------------- actions
    def _apply_action(
        self, action: HighLevelAction, world: WorldModel, dt: float
    ) -> None:
        """Translate a high-level action into deterministic controller calls."""
        handler = {
            HighLevelAction.CONTINUE_SEARCH: self._do_search,
            HighLevelAction.NAVIGATE_TO_TARGET: self._do_navigate,
            HighLevelAction.REPLAN_PATH: self._do_replan,
            HighLevelAction.APPROACH_TARGET: self._do_approach,
            HighLevelAction.BEGIN_GRASP: self._do_grasp,
            HighLevelAction.VERIFY_LOAD: self._do_verify_load,
            HighLevelAction.LOAD_CAPSULE: self._do_load_capsule,
            HighLevelAction.COMPLETE_MISSION: self._do_complete,
            HighLevelAction.ABORT_MISSION: lambda w, d: self._abort("Planner requested abort."),
        }.get(action)
        if handler is None:
            self._log.warning("No handler for action %s.", action)
            return
        handler(world, dt)

    def _do_search(self, world: WorldModel, dt: float) -> None:
        """Rotate in place scanning for the rescue target."""
        if world.has_target:
            self._fsm.transition(MissionState.TARGET_FOUND, "Valid target acquired.")
            self._robot.stop_base()
            return
        self._robot.command_base_velocity(0.0, self._search_rotate_speed, dt)

    def _do_navigate(self, world: WorldModel, dt: float) -> None:
        """Plan (if needed) and follow the path toward the target."""
        # Advance SEARCH -> TARGET_FOUND -> NAVIGATE as needed (the FSM only
        # permits one hop per transition, so step through intermediate states).
        if self._fsm.state is MissionState.SEARCH and world.has_target:
            self._fsm.transition(MissionState.TARGET_FOUND, "Valid target acquired.")
        if self._fsm.state is MissionState.TARGET_FOUND:
            self._fsm.transition(MissionState.NAVIGATE, "Begin navigation to target.")
        if not world.has_target or world.target is None:
            return
        goal = (world.target.position.x, world.target.position.y)
        if self._path.goal is None or self._goal_moved(goal):
            self._path.set_goal(goal)
        path = self._path.update(world)
        self._follower.set_path(path)
        self._follower.step(world.robot_pose, dt)

        if self._within_approach(world):
            self._robot.stop_base()
            self._fsm.transition(MissionState.APPROACH, "Reached approach range.")

    def _do_replan(self, world: WorldModel, dt: float) -> None:
        """Force a path replan and continue following."""
        path = self._path.replan_now(world)
        self._follower.set_path(path)
        self._follower.step(world.robot_pose, dt)

    def _do_approach(self, world: WorldModel, dt: float) -> None:
        """Fine approach: align with and close on the target to grasp range."""
        if self._fsm.state is MissionState.NAVIGATE:
            self._fsm.transition(MissionState.APPROACH, "Switch to fine approach.")
        if not world.has_target or world.target is None:
            return
        goal = (world.target.position.x, world.target.position.y)
        distance = euclidean_distance((world.robot_pose.x, world.robot_pose.y), goal)
        heading = angle_to(goal, (world.robot_pose.x, world.robot_pose.y))
        heading_error = wrap_to_pi(heading - world.robot_pose.yaw)

        if distance <= self._approach_distance and abs(heading_error) < 0.2:
            self._robot.stop_base()
            self._fsm.transition(MissionState.GRASP, "In grasp range and aligned.")
            return
        # Creep forward while aligning.
        linear = 0.2 if distance > self._approach_distance else 0.0
        self._robot.command_base_velocity(linear, 1.5 * heading_error, dt)

    def _do_grasp(self, world: WorldModel, dt: float) -> None:
        """Plan and execute the grasp + lift, guarded by the torque monitor."""
        if self._fsm.state is MissionState.APPROACH:
            self._fsm.transition(MissionState.GRASP, "Begin grasp.")
        if not world.has_target or world.target is None:
            self._abort("Lost target before grasp.")
            return

        grasps = self._grasp_planner.plan(world.target)
        if not grasps:
            self._abort("No feasible grasp pose.")
            return

        execution = self._arm.execute_grasp(grasps[0])
        self._last_grasp_summary = (
            f"{execution.handle_name}:{execution.result.name}:"
            f"{execution.peak_torque_nm:.1f}Nm"
        )
        self._decisions.log(
            "grasp_attempt",
            {
                "handle": execution.handle_name,
                "result": execution.result.name,
                "peak_torque_nm": round(execution.peak_torque_nm, 2),
            },
        )

        if execution.result is GraspResult.SUCCESS:
            self._fsm.transition(MissionState.VERIFY_LOAD, "Grasp succeeded.")
        elif execution.result is GraspResult.TORQUE_ABORT:
            self._abort("Torque abort during lift.")
        else:
            # No contact / unreachable -> retry by re-approaching.
            self._fsm.transition(MissionState.VERIFY_LOAD, "Grasp inconclusive; verify.")

    def _do_verify_load(self, world: WorldModel, dt: float) -> None:
        """Confirm the lift was successful before transferring to the capsule."""
        if not self._arm.is_holding:
            self._decisions.log("verify_load", {"holding": False})
            self._fsm.transition(MissionState.GRASP, "Not holding; retry grasp.")
            return
        if self._torque.is_unsafe():
            self._abort("Unsafe torque at verify_load.")
            return
        self._decisions.log("verify_load", {"holding": True})
        self._fsm.transition(MissionState.LOAD_CAPSULE, "Lift verified.")

    def _do_load_capsule(self, world: WorldModel, dt: float) -> None:
        """Place the target into the capsule and verify the load."""
        # Advance VERIFY_LOAD -> LOAD_CAPSULE before performing the transfer so
        # the subsequent COMPLETE transition is legal.
        if self._fsm.state is MissionState.VERIFY_LOAD:
            self._fsm.transition(MissionState.LOAD_CAPSULE, "Begin capsule load.")
        if self._capsule.load():
            self._fsm.transition(MissionState.COMPLETE, "Target loaded into capsule.")
        else:
            self._abort("Capsule load failed verification.")

    def _do_complete(self, world: WorldModel, dt: float) -> None:
        """Finalise a successful mission."""
        if self._fsm.state is not MissionState.COMPLETE:
            self._fsm.transition(MissionState.COMPLETE, "Mission complete.")
        self._robot.stop_base()
        self._arm.stow()

    # ------------------------------------------------------------- helpers
    def _abort(self, reason: str) -> None:
        """Force the mission into ABORT and bring the robot to a safe state."""
        self._robot.stop_base()
        self._arm.release()
        self._fsm.force(MissionState.ABORT, reason)

    def _goal_moved(self, goal, tolerance: float = 0.5) -> bool:
        if self._path.goal is None:
            return True
        return euclidean_distance(self._path.goal, goal) > tolerance

    def _within_approach(self, world: WorldModel) -> bool:
        if not world.has_target or world.target is None:
            return False
        goal = (world.target.position.x, world.target.position.y)
        return (
            euclidean_distance((world.robot_pose.x, world.robot_pose.y), goal)
            <= self._approach_distance
        )

    @property
    def state(self) -> MissionState:
        return self._fsm.state

    @property
    def is_done(self) -> bool:
        return self._fsm.is_terminal()
