"""
perception/obstacle_detector.py
===============================

Detects static obstacles from the depth camera's point cloud. The ground plane
is removed (using a height band), and remaining points are clustered with
Open3D (DBSCAN) into discrete obstacles, each summarised by a world-frame
centroid and a conservative bounding radius.

Single responsibility: turn a depth frame into a list of world-frame obstacles.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from perception.depth_camera import DepthCamera
from utils.geometry import euler_to_rotation_matrix, make_transform, transform_point
from utils.logger import get_logger
from utils.types import Obstacle, Pose2D, Position3D


class ObstacleDetector:
    """Extracts static obstacles from depth point clouds."""

    def __init__(
        self,
        depth_camera: DepthCamera,
        voxel_size: float,
        obstacle_min_height: float,
        obstacle_max_height: float,
        ground_plane_tolerance: float,
        camera_mount_xyz=(0.30, 0.0, 0.80),
        camera_mount_rpy=(-1.5708, 0.0, -1.5708),
        max_cloud_points: int = 20000,
    ) -> None:
        self._log = get_logger("perception.obstacle_detector")
        self._depth = depth_camera
        self._voxel = voxel_size
        self._min_h = obstacle_min_height
        self._max_h = obstacle_max_height
        self._ground_tol = ground_plane_tolerance
        self._cam_xyz = np.array(camera_mount_xyz, dtype=float)
        self._cam_rpy = camera_mount_rpy
        self._max_points = max_cloud_points

    def detect(
        self, robot_pose: Pose2D, depth: Optional[np.ndarray] = None
    ) -> List[Obstacle]:
        """Return world-frame obstacles visible in the current depth frame."""
        if depth is None:
            depth = self._depth.get_depth()
        cam_points = self._depth.get_point_cloud(depth)
        if cam_points.shape[0] == 0:
            return []

        cam_points = self._subsample(cam_points)
        world_points = self._camera_cloud_to_world(cam_points, robot_pose)
        obstacle_points = self._filter_by_height(world_points)
        if obstacle_points.shape[0] == 0:
            return []

        clusters = self._cluster(obstacle_points)
        obstacles = [self._summarise_cluster(c) for c in clusters if c.shape[0] > 0]
        self._log.debug("Detected %d obstacle(s).", len(obstacles))
        return obstacles

    # --------------------------------------------------------------- internals
    def _subsample(self, points: np.ndarray) -> np.ndarray:
        """Randomly cap the cloud size so clustering stays real-time."""
        if points.shape[0] <= self._max_points:
            return points
        idx = np.random.choice(points.shape[0], self._max_points, replace=False)
        return points[idx]

    def _camera_cloud_to_world(
        self, cam_points: np.ndarray, robot_pose: Pose2D
    ) -> np.ndarray:
        """Vectorised camera-frame -> world-frame transform for a point cloud."""
        cam_rot = euler_to_rotation_matrix(*self._cam_rpy)
        cam_to_base = make_transform(cam_rot, self._cam_xyz)
        base_rot = euler_to_rotation_matrix(0.0, 0.0, robot_pose.yaw)
        base_to_world = make_transform(
            base_rot, np.array([robot_pose.x, robot_pose.y, 0.0])
        )
        full = base_to_world @ cam_to_base  # camera -> world

        # Compute in float64 and silence harmless BLAS edge-case warnings.
        points64 = cam_points.astype(np.float64)
        homogeneous = np.hstack((points64, np.ones((points64.shape[0], 1))))
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            world = (full @ homogeneous.T).T[:, :3]
        return world

    def _filter_by_height(self, world_points: np.ndarray) -> np.ndarray:
        """Keep points within the obstacle height band (drops ground/ceiling)."""
        z = world_points[:, 2]
        keep = (z > (self._min_h + self._ground_tol)) & (z < self._max_h)
        return world_points[keep]

    def _cluster(self, points: np.ndarray) -> List[np.ndarray]:
        """Cluster obstacle points with Open3D DBSCAN (grid fallback)."""
        try:
            import open3d as o3d  # type: ignore

            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(points)
            if self._voxel > 0:
                cloud = cloud.voxel_down_sample(self._voxel)
            labels = np.array(
                cloud.cluster_dbscan(eps=0.3, min_points=10, print_progress=False)
            )
            pts = np.asarray(cloud.points)
            clusters: List[np.ndarray] = []
            for label in set(labels.tolist()):
                if label < 0:
                    continue  # noise
                clusters.append(pts[labels == label])
            return clusters
        except Exception as exc:  # pragma: no cover
            self._log.debug("Open3D clustering unavailable (%s); grid fallback.", exc)
            return self._grid_cluster(points)

    def _grid_cluster(self, points: np.ndarray) -> List[np.ndarray]:
        """Cheap fallback: bucket points into a 2D grid and group by cell."""
        cell = 0.3
        keys = np.floor(points[:, :2] / cell).astype(int)
        clusters: dict = {}
        for point, key in zip(points, map(tuple, keys)):
            clusters.setdefault(key, []).append(point)
        return [np.array(v) for v in clusters.values() if len(v) >= 5]

    def _summarise_cluster(self, cluster: np.ndarray) -> Obstacle:
        """Summarise a point cluster as a centroid + conservative radius."""
        centroid = cluster.mean(axis=0)
        radii = np.linalg.norm(cluster[:, :2] - centroid[:2], axis=1)
        radius = float(np.percentile(radii, 95)) if radii.size else 0.2
        return Obstacle(
            centroid=Position3D.from_array(centroid),
            radius=max(radius, 0.1),
        )
