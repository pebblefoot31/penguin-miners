"""
perception/depth_camera.py
==========================

Perception-side reader for the 3D depth camera. Provides depth images and,
using the camera intrinsics, back-projects them into organised point clouds for
obstacle detection and target localisation.

Single responsibility: produce depth images and point clouds. No segmentation
or fusion happens here.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from simulation.sensors import DepthCameraSensor
from utils.logger import get_logger


class DepthCamera:
    """High-level depth + point-cloud provider used by the perception pipeline."""

    def __init__(self, sensor: DepthCameraSensor) -> None:
        self._log = get_logger("perception.depth")
        self._sensor = sensor

    @property
    def intrinsics(self) -> Tuple[float, float, float, float]:
        """Pinhole intrinsics (fx, fy, cx, cy)."""
        return self._sensor.intrinsics

    def get_depth(self) -> np.ndarray:
        """Return the latest depth image (H, W) in metres (NaN = invalid)."""
        return self._sensor.read()

    def get_point_cloud(self, depth: np.ndarray = None) -> np.ndarray:
        """
        Back-project a depth image into an organised (N, 3) point cloud in the
        camera frame (z forward). Invalid pixels are dropped.
        """
        if depth is None:
            depth = self.get_depth()
        fx, fy, cx, cy = self.intrinsics
        height, width = depth.shape

        us, vs = np.meshgrid(np.arange(width), np.arange(height))
        z = depth
        valid = np.isfinite(z) & (z > 0)

        x = (us - cx) * z / fx
        y = (vs - cy) * z / fy

        points = np.stack(
            (x[valid], y[valid], z[valid]), axis=-1
        ).astype(np.float32)
        return points

    def pixel_to_3d(self, u: int, v: int, depth: np.ndarray = None) -> np.ndarray:
        """Back-project a single pixel to a 3D camera-frame point."""
        if depth is None:
            depth = self.get_depth()
        fx, fy, cx, cy = self.intrinsics
        z = float(depth[v, u])
        if not np.isfinite(z) or z <= 0:
            return np.array([np.nan, np.nan, np.nan], dtype=np.float32)
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return np.array([x, y, z], dtype=np.float32)
