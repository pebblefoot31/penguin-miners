"""
utils/geometry.py
=================

Pure-function geometry helpers: coordinate transforms between the camera,
robot, and world frames, plus small math utilities used by perception and
navigation.

Single responsibility: stateless geometric math. No I/O, no Isaac Sim.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return a 3x3 rotation matrix from intrinsic XYZ Euler angles (radians)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Compose a 4x4 homogeneous transform from a 3x3 R and 3-vector t."""
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to a 3D point."""
    homogeneous = np.append(point, 1.0)
    return (transform @ homogeneous)[:3]


def pixel_to_camera(
    u: float,
    v: float,
    depth: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """
    Back-project an image pixel + depth into the camera frame using the pinhole
    model. Returns an (x, y, z) point in metres (z forward).
    """
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    return np.array([x, y, depth], dtype=float)


def intrinsics_from_fov(
    width: int, height: int, horizontal_fov_deg: float
) -> Tuple[float, float, float, float]:
    """
    Derive (fx, fy, cx, cy) from image size and a horizontal field of view.
    Assumes square pixels (fx == fy).
    """
    cx = width / 2.0
    cy = height / 2.0
    fx = (width / 2.0) / math.tan(math.radians(horizontal_fov_deg) / 2.0)
    fy = fx
    return fx, fy, cx, cy


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two points."""
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def angle_to(target_xy: Tuple[float, float], from_xy: Tuple[float, float]) -> float:
    """Heading (radians) from ``from_xy`` toward ``target_xy``."""
    dx = target_xy[0] - from_xy[0]
    dy = target_xy[1] - from_xy[1]
    return math.atan2(dy, dx)


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle to the range (-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi
