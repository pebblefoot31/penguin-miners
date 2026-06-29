"""
simulation/mock_scene.py
========================

A lightweight *ground-truth* world used only in mock mode (when Isaac Sim is not
available). It holds world-fixed entities — the rescue target and some static
obstacles — and projects them into the robot's camera using the exact same
pinhole + extrinsic model the perception stack inverts. This makes the mock
self-consistent: as the robot drives toward the target, the target's apparent
range shrinks and perception recovers its true world position, so the full
mission (search → navigate → approach → grasp → load) actually converges.

Single responsibility: provide world-fixed ground truth and project it to the
camera image plane for the mock sensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from utils.geometry import euler_to_rotation_matrix, make_transform
from utils.types import Pose2D


@dataclass
class MockEntity:
    """A world-fixed object the mock sensors can 'see'."""

    world_xyz: Tuple[float, float, float]
    kind: str                 # "target" | "obstacle"
    radius_px: int            # rendered blob radius in the image


@dataclass
class Projection:
    """An entity projected into the current camera image."""

    u: int
    v: int
    depth: float
    kind: str
    radius_px: int


# Camera extrinsics — MUST match perception's TargetLocalizer / ObstacleDetector
# defaults so the mock and the perception math are mutually consistent.
CAMERA_MOUNT_XYZ = (0.30, 0.0, 0.80)
CAMERA_MOUNT_RPY = (-1.5708, 0.0, -1.5708)


class MockScene:
    """Holds world-fixed entities and projects them to the camera frame."""

    def __init__(
        self,
        intrinsics: Tuple[float, float, float, float],
        resolution: Tuple[int, int],
        min_range: float,
        max_range: float,
        pose_provider: Callable[[], Pose2D],
    ) -> None:
        self._fx, self._fy, self._cx, self._cy = intrinsics
        self._width, self._height = resolution
        self._min_range = min_range
        self._max_range = max_range
        self._pose_provider = pose_provider
        self._entities: List[MockEntity] = []

        # Precompute the (constant) camera->base transform.
        cam_rot = euler_to_rotation_matrix(*CAMERA_MOUNT_RPY)
        self._cam_to_base = make_transform(cam_rot, np.array(CAMERA_MOUNT_XYZ))

    # ------------------------------------------------------------- scene setup
    def add_entity(self, entity: MockEntity) -> None:
        self._entities.append(entity)

    def add_target(self, world_xyz: Tuple[float, float, float]) -> None:
        self.add_entity(MockEntity(world_xyz, "target", radius_px=45))

    def add_obstacle(self, world_xyz: Tuple[float, float, float]) -> None:
        self.add_entity(MockEntity(world_xyz, "obstacle", radius_px=55))

    # -------------------------------------------------------------- projection
    def project_all(self) -> List[Projection]:
        """Project every visible entity into the current camera image."""
        pose = self._pose_provider()
        base_to_world = make_transform(
            euler_to_rotation_matrix(0.0, 0.0, pose.yaw),
            np.array([pose.x, pose.y, 0.0]),
        )
        world_to_cam = np.linalg.inv(base_to_world @ self._cam_to_base)

        projections: List[Projection] = []
        for entity in self._entities:
            proj = self._project_entity(entity, world_to_cam)
            if proj is not None:
                projections.append(proj)
        return projections

    def _project_entity(
        self, entity: MockEntity, world_to_cam: np.ndarray
    ) -> Optional[Projection]:
        """Project a single entity, returning None if not in view."""
        world_h = np.array([*entity.world_xyz, 1.0])
        cam = (world_to_cam @ world_h)[:3]
        z = float(cam[2])
        if z < self._min_range or z > self._max_range:
            return None  # behind the camera or out of range
        u = int(round(self._cx + self._fx * cam[0] / z))
        v = int(round(self._cy + self._fy * cam[1] / z))
        if not (0 <= u < self._width and 0 <= v < self._height):
            return None  # outside the image (out of field of view)
        return Projection(u=u, v=v, depth=z, kind=entity.kind, radius_px=entity.radius_px)
