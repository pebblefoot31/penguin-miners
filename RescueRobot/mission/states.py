"""
mission/states.py
=================

Defines the mission state machine's states and the closed set of high-level
actions the LLM planner is allowed to return. Keeping these as enums gives the
rest of the system a single, authoritative vocabulary and lets the LLM output be
validated against ``HighLevelAction``.

Single responsibility: enumerate mission states and allowed high-level actions.
"""

from __future__ import annotations

from enum import Enum


class MissionState(str, Enum):
    """The lifecycle states of a rescue mission."""

    SEARCH = "SEARCH"               # scanning the environment for the target
    TARGET_FOUND = "TARGET_FOUND"   # a valid rescue-suit target is detected
    NAVIGATE = "NAVIGATE"           # planning/following a path to the target
    APPROACH = "APPROACH"           # fine approach to grasping range
    GRASP = "GRASP"                 # executing the grasp + lift
    VERIFY_LOAD = "VERIFY_LOAD"     # confirming the lift before transfer
    LOAD_CAPSULE = "LOAD_CAPSULE"   # placing the target into the capsule
    COMPLETE = "COMPLETE"           # mission finished successfully
    ABORT = "ABORT"                 # mission aborted for safety/failure


class HighLevelAction(str, Enum):
    """
    The closed set of high-level actions the LLM planner may return. The LLM
    chooses *intent*; deterministic controllers translate intent into motion.
    The LLM never commands motors directly.
    """

    CONTINUE_SEARCH = "CONTINUE_SEARCH"
    APPROACH_TARGET = "APPROACH_TARGET"
    NAVIGATE_TO_TARGET = "NAVIGATE_TO_TARGET"
    REPLAN_PATH = "REPLAN_PATH"
    BEGIN_GRASP = "BEGIN_GRASP"
    VERIFY_LOAD = "VERIFY_LOAD"
    LOAD_CAPSULE = "LOAD_CAPSULE"
    COMPLETE_MISSION = "COMPLETE_MISSION"
    ABORT_MISSION = "ABORT_MISSION"


# Mapping used to validate LLM suggestions against the current state. The
# mission controller retains final authority; this only bounds what the planner
# may propose from a given state.
VALID_ACTIONS_BY_STATE = {
    MissionState.SEARCH: {
        HighLevelAction.CONTINUE_SEARCH,
        HighLevelAction.NAVIGATE_TO_TARGET,
        HighLevelAction.ABORT_MISSION,
    },
    MissionState.TARGET_FOUND: {
        HighLevelAction.NAVIGATE_TO_TARGET,
        HighLevelAction.ABORT_MISSION,
    },
    MissionState.NAVIGATE: {
        HighLevelAction.NAVIGATE_TO_TARGET,
        HighLevelAction.REPLAN_PATH,
        HighLevelAction.APPROACH_TARGET,
        HighLevelAction.CONTINUE_SEARCH,
        HighLevelAction.ABORT_MISSION,
    },
    MissionState.APPROACH: {
        HighLevelAction.APPROACH_TARGET,
        HighLevelAction.BEGIN_GRASP,
        HighLevelAction.REPLAN_PATH,
        HighLevelAction.ABORT_MISSION,
    },
    MissionState.GRASP: {
        HighLevelAction.BEGIN_GRASP,
        HighLevelAction.VERIFY_LOAD,
        HighLevelAction.ABORT_MISSION,
    },
    MissionState.VERIFY_LOAD: {
        HighLevelAction.VERIFY_LOAD,
        HighLevelAction.LOAD_CAPSULE,
        HighLevelAction.BEGIN_GRASP,
        HighLevelAction.ABORT_MISSION,
    },
    MissionState.LOAD_CAPSULE: {
        HighLevelAction.LOAD_CAPSULE,
        HighLevelAction.COMPLETE_MISSION,
        HighLevelAction.ABORT_MISSION,
    },
    MissionState.COMPLETE: set(),
    MissionState.ABORT: set(),
}
