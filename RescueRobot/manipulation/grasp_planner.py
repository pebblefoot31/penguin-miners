"""
manipulation/grasp_planner.py
=============================

Plans grasp poses on the rescue suit's predefined handles. Given the target's
estimated 3D position, it computes world-frame pre-grasp (standoff) and grasp
poses for each handle, ordered by reachability.

Single responsibility: compute *where* to grasp. Executing the motion is the
arm controller's job.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from utils.logger import get_logger
from utils.types import Position3D, TargetEstimate


@dataclass
class GraspPose:
    """A planned grasp on a single suit handle."""

    handle_name: str
    pre_grasp: Position3D    # standoff pose above the handle
    grasp: Position3D        # handle contact pose


class GraspPlanner:
    """Computes grasp poses on the rescue suit's handles."""

    def __init__(
        self,
        handle_offsets: List[List[float]],
        pre_grasp_standoff: float,
        handle_names: List[str] = None,
    ) -> None:
        self._log = get_logger("manipulation.grasp_planner")
        self._handle_offsets = [np.array(o, dtype=float) for o in handle_offsets]
        self._standoff = pre_grasp_standoff
        self._names = handle_names or [
            f"handle_{i}" for i in range(len(handle_offsets))
        ]

    def plan(self, target: TargetEstimate) -> List[GraspPose]:
        """
        Return grasp poses for each suit handle, sorted by proximity to the
        target centroid (closest/most-reachable first).
        """
        centroid = target.position.as_array()
        grasps: List[GraspPose] = []
        for name, offset in zip(self._names, self._handle_offsets):
            handle_world = centroid + offset
            pre = handle_world + np.array([0.0, 0.0, self._standoff])
            grasps.append(
                GraspPose(
                    handle_name=name,
                    pre_grasp=Position3D.from_array(pre),
                    grasp=Position3D.from_array(handle_world),
                )
            )
        grasps.sort(key=lambda g: np.linalg.norm(g.grasp.as_array() - centroid))
        self._log.debug("Planned %d grasp pose(s).", len(grasps))
        return grasps
