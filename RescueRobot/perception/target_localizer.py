"""
perception/target_localizer.py
==============================

Estimates the 3D position of a detected rescue target by sampling the depth
image inside the detection bounding box and transforming the camera-frame point
into the world frame using the robot pose and camera extrinsics.

Single responsibility: convert a 2D detection + depth into a world-frame 3D
target estimate.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from perception.depth_camera import DepthCamera
from utils.geometry import (
    euler_to_rotation_matrix,
    make_transform,
    transform_point,
)
from utils.logger import get_logger
from utils.types import Detection, Pose2D, Position3D, TargetEstimate


class TargetLocalizer:
    """Computes a world-frame 3D estimate for a rescue-suit detection."""

    def __init__(
        self,
        depth_camera: DepthCamera,
        camera_mount_xyz=(0.30, 0.0, 0.80),
        camera_mount_rpy=(-1.5708, 0.0, -1.5708),
    ) -> None:
        """
        ``camera_mount_xyz`` / ``camera_mount_rpy`` describe the depth camera's
        fixed pose in the robot base frame. The default RPY rotates the optical
        frame (z-forward, x-right, y-down) into the robot frame (x-forward).
        """
        self._log = get_logger("perception.target_localizer")
        self._depth = depth_camera
        self._cam_xyz = np.array(camera_mount_xyz, dtype=float)
        self._cam_rpy = camera_mount_rpy

    def localize(
        self,
        detection: Detection,
        robot_pose: Pose2D,
        depth: Optional[np.ndarray] = None,
    ) -> Optional[TargetEstimate]:
        """
        Return a world-frame :class:`TargetEstimate` for ``detection`` or
        ``None`` if depth inside the bounding box is unusable.
        """
        if depth is None:
            depth = self._depth.get_depth()

        cam_point = self._sample_bbox_depth(detection, depth)
        if cam_point is None:
            self._log.debug("No valid depth within target bbox; localisation failed.")
            return None

        world_point = self._camera_to_world(cam_point, robot_pose)
        estimate = TargetEstimate(
            position=Position3D.from_array(world_point),
            confidence=detection.confidence,
            detection=detection,
        )
        self._log.debug(
            "Target localised at world (%.2f, %.2f, %.2f).",
            world_point[0],
            world_point[1],
            world_point[2],
        )
        return estimate

    # --------------------------------------------------------------- internals
    def _sample_bbox_depth(
        self, detection: Detection, depth: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Robustly sample the target's camera-frame point: take the median depth
        of valid pixels in the bbox center region, then back-project the centre
        pixel at that depth.
        """
        bbox = detection.bbox
        cx, cy = bbox.center
        # Central region (inner 50%) is more likely to be on the body, not edges.
        half_w = max(1, bbox.width // 4)
        half_h = max(1, bbox.height // 4)
        x1, x2 = max(0, cx - half_w), min(depth.shape[1], cx + half_w)
        y1, y2 = max(0, cy - half_h), min(depth.shape[0], cy + half_h)
        patch = depth[y1:y2, x1:x2]
        valid = patch[np.isfinite(patch) & (patch > 0)]
        if valid.size == 0:
            return None
        median_depth = float(np.median(valid))
        fx, fy, ccx, ccy = self._depth.intrinsics
        x = (cx - ccx) * median_depth / fx
        y = (cy - ccy) * median_depth / fy
        return np.array([x, y, median_depth], dtype=float)

    def _camera_to_world(
        self, cam_point: np.ndarray, robot_pose: Pose2D
    ) -> np.ndarray:
        """Transform a camera-frame point into the world frame."""
        # Camera -> robot base.
        cam_rot = euler_to_rotation_matrix(*self._cam_rpy)
        cam_to_base = make_transform(cam_rot, self._cam_xyz)
        base_point = transform_point(cam_to_base, cam_point)

        # Robot base -> world (planar).
        base_rot = euler_to_rotation_matrix(0.0, 0.0, robot_pose.yaw)
        base_to_world = make_transform(
            base_rot, np.array([robot_pose.x, robot_pose.y, 0.0])
        )
        return transform_point(base_to_world, base_point)
