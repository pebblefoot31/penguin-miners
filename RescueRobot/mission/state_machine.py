"""
mission/state_machine.py
========================

A small generic finite-state machine that enforces legal transitions and logs
every transition to the decision log (satisfying "log every ... state
transition"). The mission controller drives it.

Single responsibility: manage and record state transitions.
"""

from __future__ import annotations

from typing import Dict, Optional, Set

from mission.states import MissionState
from utils.logger import DecisionLogger, get_logger

# Legal state transitions for the rescue mission.
_TRANSITIONS: Dict[MissionState, Set[MissionState]] = {
    MissionState.SEARCH: {MissionState.TARGET_FOUND, MissionState.ABORT},
    MissionState.TARGET_FOUND: {MissionState.NAVIGATE, MissionState.ABORT},
    MissionState.NAVIGATE: {
        MissionState.APPROACH,
        MissionState.SEARCH,        # target lost during navigation
        MissionState.ABORT,
    },
    MissionState.APPROACH: {
        MissionState.GRASP,
        MissionState.NAVIGATE,      # drifted out of range / replan
        MissionState.SEARCH,        # target lost
        MissionState.ABORT,
    },
    MissionState.GRASP: {
        MissionState.VERIFY_LOAD,
        MissionState.ABORT,         # torque abort
    },
    MissionState.VERIFY_LOAD: {
        MissionState.LOAD_CAPSULE,
        MissionState.GRASP,         # retry grasp
        MissionState.ABORT,
    },
    MissionState.LOAD_CAPSULE: {MissionState.COMPLETE, MissionState.ABORT},
    MissionState.COMPLETE: set(),
    MissionState.ABORT: set(),
}


class StateMachine:
    """Finite-state machine with legal-transition enforcement and logging."""

    def __init__(
        self,
        decision_logger: DecisionLogger,
        initial: MissionState = MissionState.SEARCH,
    ) -> None:
        self._log = get_logger("mission.state_machine")
        self._decisions = decision_logger
        self._state = initial
        self._decisions.log("state_init", {"state": initial.value})

    @property
    def state(self) -> MissionState:
        return self._state

    def can_transition(self, target: MissionState) -> bool:
        """Whether a transition from the current state to ``target`` is legal."""
        return target in _TRANSITIONS.get(self._state, set())

    def transition(self, target: MissionState, reason: str = "") -> bool:
        """
        Attempt to transition to ``target``. Logs the transition (or rejection)
        and returns whether it succeeded.
        """
        if target == self._state:
            return True
        if not self.can_transition(target):
            self._log.warning(
                "Illegal transition %s -> %s rejected.", self._state.value, target.value
            )
            self._decisions.log(
                "state_transition_rejected",
                {"from": self._state.value, "to": target.value, "reason": reason},
            )
            return False
        previous = self._state
        self._state = target
        self._log.info("State %s -> %s (%s).", previous.value, target.value, reason)
        self._decisions.log(
            "state_transition",
            {"from": previous.value, "to": target.value, "reason": reason},
        )
        return True

    def force(self, target: MissionState, reason: str = "") -> None:
        """
        Force a transition regardless of legality (reserved for emergency
        ABORT). Still logged for auditability.
        """
        previous = self._state
        self._state = target
        self._log.warning("Forced state %s -> %s (%s).", previous.value, target.value, reason)
        self._decisions.log(
            "state_forced",
            {"from": previous.value, "to": target.value, "reason": reason},
        )

    def is_terminal(self) -> bool:
        """True if the FSM is in a terminal state (COMPLETE or ABORT)."""
        return self._state in (MissionState.COMPLETE, MissionState.ABORT)
