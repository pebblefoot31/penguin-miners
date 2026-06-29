"""
ai/llm_planner.py
=================

The high-level planner. It calls the Claude API (official ``anthropic`` SDK)
with the structured robot-state summary and uses **structured outputs** so the
returned value is always a valid :class:`HighLevelAction`. The model only
selects intent — it never emits motor commands.

If no API key is configured (or the SDK/network is unavailable), the planner
transparently falls back to a deterministic rule-based policy so the mission
still runs. The mission controller retains final authority over every action.

Single responsibility: map structured state -> a validated high-level action.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ai.prompt_builder import PromptBuilder
from mission.states import VALID_ACTIONS_BY_STATE, HighLevelAction, MissionState
from utils.logger import DecisionLogger, get_logger


class PlannerDecision(BaseModel):
    """Structured output schema returned by the LLM planner."""

    action: HighLevelAction = Field(
        description="The single next high-level action to take."
    )
    reasoning: str = Field(
        description="One short sentence justifying the chosen action."
    )
    confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Planner confidence in this action, 0..1.",
    )


class LLMPlanner:
    """Claude-backed high-level planner with a deterministic fallback."""

    def __init__(
        self,
        prompt_builder: PromptBuilder,
        decision_logger: DecisionLogger,
        model: str = "claude-opus-4-8",
        effort: str = "medium",
        max_tokens: int = 1024,
        timeout_s: float = 30.0,
        enable_llm: bool = True,
    ) -> None:
        self._log = get_logger("ai.llm_planner")
        self._prompt = prompt_builder
        self._decisions = decision_logger
        self._model = model
        self._effort = effort
        self._max_tokens = max_tokens
        self._timeout = timeout_s
        self._enable_llm = enable_llm
        self._client: Optional[Any] = None
        if enable_llm:
            self._init_client()

    def _init_client(self) -> None:
        """Create the Anthropic client if a key is present and the SDK loads."""
        if not os.environ.get("ANTHROPIC_API_KEY"):
            self._log.warning(
                "ANTHROPIC_API_KEY not set; using deterministic fallback planner."
            )
            self._enable_llm = False
            return
        try:
            import anthropic  # type: ignore

            self._client = anthropic.Anthropic(timeout=self._timeout)
            self._log.info("LLM planner ready (model=%s).", self._model)
        except Exception as exc:
            self._log.warning("Anthropic SDK unavailable (%s); using fallback.", exc)
            self._enable_llm = False

    def decide(
        self,
        state: MissionState,
        world,
        path_length: int,
        last_grasp_summary: Optional[str] = None,
    ) -> PlannerDecision:
        """
        Return the next high-level action for the current state. Tries the LLM
        first, then validates and (on any problem) falls back to the rule-based
        policy. The result is always a legal action for ``state``.
        """
        payload = self._prompt.build_state_payload(
            state, world, path_length, last_grasp_summary
        )

        decision: Optional[PlannerDecision] = None
        if self._enable_llm and self._client is not None:
            decision = self._decide_with_llm(payload)

        if decision is None or not self._is_legal(state, decision.action):
            decision = self._fallback_policy(state, payload)
            source = "fallback"
        else:
            source = "llm"

        self._decisions.log(
            "llm_action",
            {
                "source": source,
                "state": state.value,
                "action": decision.action.value,
                "reasoning": decision.reasoning,
                "confidence": decision.confidence,
            },
        )
        return decision

    # --------------------------------------------------------------- LLM call
    def _decide_with_llm(self, payload: Dict[str, Any]) -> Optional[PlannerDecision]:
        """Query Claude with structured outputs; return None on any failure."""
        try:
            response = self._client.messages.parse(
                model=self._model,
                max_tokens=self._max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": self._prompt.system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                output_config={"effort": self._effort},
                messages=[
                    {
                        "role": "user",
                        "content": self._prompt.build_user_message(payload),
                    }
                ],
                output_format=PlannerDecision,
            )
            return response.parsed_output
        except Exception as exc:  # network/SDK/parse error -> fallback
            self._log.warning("LLM planner call failed (%s); using fallback.", exc)
            return None

    # ------------------------------------------------------- deterministic policy
    def _fallback_policy(
        self, state: MissionState, payload: Dict[str, Any]
    ) -> PlannerDecision:
        """
        Deterministic, safety-first policy used when the LLM is unavailable or
        returns an illegal action. Mirrors the intended high-level logic.
        """
        target = payload.get("target", {})
        torque = payload.get("torque", {})
        nav = payload.get("navigation", {})

        if torque.get("unsafe"):
            return self._mk(HighLevelAction.ABORT_MISSION, "Torque exceeds safety limit.")

        if state is MissionState.SEARCH:
            if target.get("visible"):
                return self._mk(
                    HighLevelAction.NAVIGATE_TO_TARGET, "Rescue target detected."
                )
            return self._mk(HighLevelAction.CONTINUE_SEARCH, "No target yet; keep scanning.")

        if state is MissionState.TARGET_FOUND:
            return self._mk(HighLevelAction.NAVIGATE_TO_TARGET, "Proceed to target.")

        if state is MissionState.NAVIGATE:
            if not target.get("visible"):
                return self._mk(HighLevelAction.CONTINUE_SEARCH, "Target lost; re-search.")
            if target.get("within_grasp_range"):
                return self._mk(HighLevelAction.APPROACH_TARGET, "Within approach range.")
            if nav.get("active_path_waypoints", 0) == 0:
                return self._mk(HighLevelAction.REPLAN_PATH, "No active path; replan.")
            return self._mk(HighLevelAction.NAVIGATE_TO_TARGET, "Continue to target.")

        if state is MissionState.APPROACH:
            if not target.get("visible"):
                return self._mk(HighLevelAction.ABORT_MISSION, "Target lost during approach.")
            if target.get("within_grasp_range"):
                return self._mk(HighLevelAction.BEGIN_GRASP, "In range; begin grasp.")
            return self._mk(HighLevelAction.APPROACH_TARGET, "Close the final distance.")

        if state is MissionState.GRASP:
            return self._mk(HighLevelAction.VERIFY_LOAD, "Grasp executed; verify load.")

        if state is MissionState.VERIFY_LOAD:
            return self._mk(HighLevelAction.LOAD_CAPSULE, "Load verified; load capsule.")

        if state is MissionState.LOAD_CAPSULE:
            return self._mk(HighLevelAction.COMPLETE_MISSION, "Target loaded; complete.")

        return self._mk(HighLevelAction.ABORT_MISSION, "No valid action for state.")

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _is_legal(state: MissionState, action: HighLevelAction) -> bool:
        return action in VALID_ACTIONS_BY_STATE.get(state, set())

    @staticmethod
    def _mk(action: HighLevelAction, reason: str) -> PlannerDecision:
        return PlannerDecision(action=action, reasoning=reason, confidence=0.6)

    def allowed_actions(self, state: MissionState) -> List[HighLevelAction]:
        return list(VALID_ACTIONS_BY_STATE.get(state, set()))
