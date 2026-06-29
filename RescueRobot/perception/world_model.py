"""
perception/world_model.py
=========================

Sensor-fusion hub. Combines RGB-based rescue-suit detection, depth-based 3D
target localisation, and depth-based obstacle detection into a single unified
:class:`~utils.types.WorldModel`. Also maintains short-term temporal memory of
the target so brief detection dropouts don't immediately lose it.

Single responsibility: fuse perception outputs into one coherent world view.
"""

from __future__ import annotations

import time
from typing import List, Optional

from perception.depth_camera import DepthCamera
from perception.obstacle_detector import ObstacleDetector
from perception.rgb_camera import RGBCamera
from perception.suit_detector import SuitDetector
from perception.target_localizer import TargetLocalizer
from utils.logger import get_logger
from utils.types import Detection, Obstacle, Pose2D, TargetEstimate, WorldModel


class WorldModelBuilder:
    """Builds and maintains the fused world model each perception cycle."""

    def __init__(
        self,
        rgb_camera: RGBCamera,
        depth_camera: DepthCamera,
        suit_detector: SuitDetector,
        target_localizer: TargetLocalizer,
        obstacle_detector: ObstacleDetector,
        max_target_age_s: float,
        target_exclusion_radius_m: float = 0.7,
    ) -> None:
        self._log = get_logger("perception.world_model")
        self._rgb = rgb_camera
        self._depth = depth_camera
        self._detector = suit_detector
        self._localizer = target_localizer
        self._obstacles_detector = obstacle_detector
        self._max_target_age = max_target_age_s
        self._target_exclusion_radius = target_exclusion_radius_m
        self._last_target: Optional[TargetEstimate] = None

    def update(self, robot_pose: Pose2D) -> WorldModel:
        """
        Run one full perception cycle and return the fused world model.

        Steps: read RGB+depth once → detect suit → localise target in 3D →
        detect obstacles → fuse + apply temporal target memory.
        """
        rgb_frame = self._rgb.get_frame()
        depth_frame = self._depth.get_depth()

        detections = self._detector.detect(rgb_frame)
        target = self._fuse_target(detections, robot_pose, depth_frame)
        obstacles = self._obstacles_detector.detect(robot_pose, depth_frame)
        # The rescue subject is itself a depth return; don't let it be treated
        # as an obstacle to avoid, or navigation can never reach it.
        obstacles = self._exclude_target_from_obstacles(obstacles, target)

        model = WorldModel(
            target=target,
            obstacles=obstacles,
            robot_pose=robot_pose,
            timestamp=time.time(),
        )
        self._log.debug(
            "World model: target=%s, obstacles=%d.",
            "yes" if model.has_target else "no",
            len(obstacles),
        )
        return model

    # --------------------------------------------------------------- internals
    def _exclude_target_from_obstacles(
        self, obstacles: List[Obstacle], target: Optional[TargetEstimate]
    ) -> List[Obstacle]:
        """Drop obstacle clusters that coincide with the rescue target."""
        if target is None:
            return obstacles
        tx, ty = target.position.x, target.position.y
        kept = [
            o
            for o in obstacles
            if ((o.centroid.x - tx) ** 2 + (o.centroid.y - ty) ** 2) ** 0.5
            > self._target_exclusion_radius
        ]
        dropped = len(obstacles) - len(kept)
        if dropped:
            self._log.debug("Excluded %d obstacle(s) coinciding with target.", dropped)
        return kept

    def _fuse_target(
        self,
        detections: List[Detection],
        robot_pose: Pose2D,
        depth_frame,
    ) -> Optional[TargetEstimate]:
        """
        Choose the best detection, localise it, and apply temporal memory so a
        single-frame dropout doesn't immediately drop the target.
        """
        best = self._select_best_detection(detections)
        if best is not None:
            estimate = self._localizer.localize(best, robot_pose, depth_frame)
            if estimate is not None:
                self._last_target = estimate
                return estimate

        # No fresh estimate: keep the last one if it's still recent enough.
        if self._last_target is not None:
            if self._last_target.age() <= self._max_target_age:
                return self._last_target
            self._log.info("Target memory expired (age > %.1fs).", self._max_target_age)
            self._last_target = None
        return None

    @staticmethod
    def _select_best_detection(
        detections: List[Detection],
    ) -> Optional[Detection]:
        """Pick the highest-confidence validated rescue-suit detection."""
        suit_dets = [d for d in detections if d.is_rescue_suit]
        if not suit_dets:
            return None
        return max(suit_dets, key=lambda d: d.confidence)

    def clear_target_memory(self) -> None:
        """Forget any remembered target (used on ABORT / mission reset)."""
        self._last_target = None
