"""
utils/types.py
==============

Small, typed data classes shared across the perception → planning → control
pipeline. Keeping these in one place means modules pass around well-defined
objects (with units documented) instead of bare NumPy arrays or tuples.

Single responsibility: define the common data vocabulary of the system.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Pose2D:
    """A planar pose in the world frame (metres, radians)."""

    x: float
    y: float
    yaw: float

    def as_xy(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class Position3D:
    """A 3D point in the world frame (metres)."""

    x: float
    y: float
    z: float

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    @staticmethod
    def from_array(arr: np.ndarray) -> "Position3D":
        return Position3D(float(arr[0]), float(arr[1]), float(arr[2]))


@dataclass
class BoundingBox:
    """Axis-aligned 2D image bounding box in pixels (x1, y1, x2, y2)."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


@dataclass
class Detection:
    """A single rescue-suit detection from the perception stack."""

    bbox: BoundingBox
    confidence: float
    class_name: str
    color_match_fraction: float = 0.0
    is_rescue_suit: bool = False


@dataclass
class TargetEstimate:
    """Fused 3D estimate of the rescue target."""

    position: Position3D                 # world-frame position (metres)
    confidence: float
    detection: Optional[Detection] = None
    timestamp: float = field(default_factory=time.time)

    def age(self, now: Optional[float] = None) -> float:
        """Seconds since this estimate was produced."""
        return (now if now is not None else time.time()) - self.timestamp


@dataclass
class Obstacle:
    """A static obstacle approximated by a world-frame footprint centroid."""

    centroid: Position3D
    radius: float                        # metres, conservative bounding radius


@dataclass
class TorqueReading:
    """A single multi-joint torque sample from the arm (Newton-metres)."""

    values_nm: List[float]
    timestamp: float = field(default_factory=time.time)

    @property
    def max_abs(self) -> float:
        return max((abs(v) for v in self.values_nm), default=0.0)


@dataclass
class WorldModel:
    """
    Unified world representation produced by sensor fusion and consumed by
    navigation, manipulation, and the high-level planner.
    """

    target: Optional[TargetEstimate] = None
    obstacles: List[Obstacle] = field(default_factory=list)
    robot_pose: Pose2D = field(default_factory=lambda: Pose2D(0.0, 0.0, 0.0))
    timestamp: float = field(default_factory=time.time)

    @property
    def has_target(self) -> bool:
        return self.target is not None
