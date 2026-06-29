"""
ai/prompt_builder.py
====================

Builds the structured robot-state summary that is fed to the high-level LLM
planner. The summary is deliberately compact and *structured* (target position,
obstacle-map summary, torque estimate, mission status) so the planner reasons
over a clean abstraction rather than raw sensor data.

Single responsibility: serialise the robot/world state into planner inputs
(a system prompt + a JSON state payload).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from manipulation.torque_monitor import TorqueMonitor
from mission.states import VALID_ACTIONS_BY_STATE, HighLevelAction, MissionState
from utils.geometry import euclidean_distance
from utils.types import WorldModel

# Stable system prompt — kept byte-identical across requests so it can be cached.
SYSTEM_PROMPT = """\
You are the high-level mission planner for an autonomous disaster-rescue robot \
operating in NVIDIA Isaac Sim. The robot must detect a person wearing a \
rescue suit, navigate to them while avoiding obstacles, grasp them by the suit \
handles, and load them into a transport capsule.

You are a HIGH-LEVEL planner only. You select the next high-level ACTION from a \
fixed list. You never produce motor commands, joint angles, or velocities — \
deterministic controllers handle all motion. Safety rules (torque limits, \
target-loss handling, obstacle replanning) are enforced by the robot \
independently of you; your job is to choose the sensible next intent given the \
structured state.

Respond ONLY with a single action from the allowed list for the current \
mission state, plus a brief reason. Prefer progress, but choose ABORT_MISSION \
when the state indicates an unrecoverable safety condition.\
"""


class PromptBuilder:
    """Serialises robot/world state into the planner's structured input."""

    def __init__(self, torque_monitor: TorqueMonitor, approach_distance: float) -> None:
        self._torque = torque_monitor
        self._approach_distance = approach_distance

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def build_state_payload(
        self,
        state: MissionState,
        world: WorldModel,
        path_length: int,
        last_grasp_summary: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build the structured state dict given to the planner. Includes the
        target position, an obstacle-map summary, the torque estimate, the
        mission status, and the actions allowed from the current state.
        """
        payload: Dict[str, Any] = {
            "mission_state": state.value,
            "target": self._target_summary(world),
            "obstacles": self._obstacle_summary(world),
            "torque": self._torque_summary(),
            "navigation": {
                "active_path_waypoints": path_length,
                "approach_distance_m": self._approach_distance,
            },
            "robot_pose": {
                "x": round(world.robot_pose.x, 3),
                "y": round(world.robot_pose.y, 3),
                "yaw": round(world.robot_pose.yaw, 3),
            },
            "last_grasp": last_grasp_summary or "none",
            "allowed_actions": self._allowed_actions(state),
        }
        return payload

    def build_user_message(self, payload: Dict[str, Any]) -> str:
        """Render the state payload as the user message text for the planner."""
        return (
            "Current structured robot state:\n"
            f"{json.dumps(payload, indent=2)}\n\n"
            "Choose exactly one next high-level action from `allowed_actions`."
        )

    # --------------------------------------------------------------- internals
    def _target_summary(self, world: WorldModel) -> Dict[str, Any]:
        if not world.has_target or world.target is None:
            return {"visible": False}
        pos = world.target.position
        distance = euclidean_distance(
            (world.robot_pose.x, world.robot_pose.y), (pos.x, pos.y)
        )
        return {
            "visible": True,
            "position": {"x": round(pos.x, 3), "y": round(pos.y, 3), "z": round(pos.z, 3)},
            "distance_m": round(distance, 3),
            "confidence": round(world.target.confidence, 3),
            "age_s": round(world.target.age(), 3),
            "within_grasp_range": distance <= self._approach_distance,
        }

    def _obstacle_summary(self, world: WorldModel) -> Dict[str, Any]:
        obstacles = world.obstacles
        nearest = None
        if obstacles:
            nearest = min(
                euclidean_distance(
                    (world.robot_pose.x, world.robot_pose.y),
                    (o.centroid.x, o.centroid.y),
                )
                for o in obstacles
            )
        return {
            "count": len(obstacles),
            "nearest_distance_m": round(nearest, 3) if nearest is not None else None,
        }

    def _torque_summary(self) -> Dict[str, Any]:
        return {
            "max_nm": round(self._torque.max_torque, 2),
            "safety_threshold_nm": self._torque.safety_threshold,
            "warning": self._torque.is_warning(),
            "unsafe": self._torque.is_unsafe(),
        }

    @staticmethod
    def _allowed_actions(state: MissionState) -> List[str]:
        return [a.value for a in VALID_ACTIONS_BY_STATE.get(state, set())]
