"""
perception/rgb_camera.py
========================

Perception-side reader for the RGB stream. Wraps the simulation sensor and
applies light, perception-oriented preprocessing (color-space conversion).

Single responsibility: deliver clean RGB/BGR frames to the detector.
"""

from __future__ import annotations

import cv2
import numpy as np

from simulation.sensors import RGBCameraSensor
from utils.logger import get_logger


class RGBCamera:
    """High-level RGB frame provider used by the perception pipeline."""

    def __init__(self, sensor: RGBCameraSensor) -> None:
        self._log = get_logger("perception.rgb")
        self._sensor = sensor

    def get_frame(self) -> np.ndarray:
        """Return the latest RGB frame as an (H, W, 3) uint8 array (RGB order)."""
        frame = self._sensor.read()
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Unexpected RGB frame shape: {frame.shape}")
        return frame

    def get_bgr_frame(self) -> np.ndarray:
        """Return the frame in OpenCV's native BGR order for HSV gating."""
        return cv2.cvtColor(self.get_frame(), cv2.COLOR_RGB2BGR)
