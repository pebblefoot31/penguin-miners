"""
simulation/sensors.py
======================

Thin wrappers around the robot's simulated sensors:

* :class:`RGBCameraSensor`  — Zivid M60 RGB stream.
* :class:`DepthCameraSensor` — Zivid M60 depth stream + intrinsics.
* :class:`TorqueSensor`     — per-joint torque feedback from the arm.

Each wrapper exposes a tiny, hardware-agnostic ``read()`` API. When Isaac Sim is
unavailable they synthesise plausible data so the perception/manipulation stacks
remain exercisable.

Single responsibility: expose raw sensor data; no interpretation happens here.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from utils.geometry import intrinsics_from_fov
from utils.logger import get_logger
from utils.types import TorqueReading

# Type-only import; avoids a hard dependency when not in mock mode.
try:  # pragma: no cover - convenience import
    from simulation.mock_scene import MockScene
except Exception:  # pragma: no cover
    MockScene = Any  # type: ignore


class RGBCameraSensor:
    """Reads RGB frames from an Isaac Sim camera (Zivid M60)."""

    def __init__(
        self,
        prim_path: str,
        resolution: Tuple[int, int],
        horizontal_fov_deg: float,
        mock: bool = False,
    ) -> None:
        self._log = get_logger("rgb_camera")
        self._prim_path = prim_path
        self._width, self._height = resolution
        self._fov = horizontal_fov_deg
        self._mock = mock
        self._camera: Optional[Any] = None
        self._scene: Optional[MockScene] = None
        if not mock:
            self._initialise()

    def bind_scene(self, scene: "MockScene") -> None:
        """Attach the mock ground-truth scene used to render synthetic frames."""
        self._scene = scene

    def _initialise(self) -> None:
        try:
            from isaacsim.sensors.camera import Camera  # type: ignore

            self._camera = Camera(
                prim_path=self._prim_path,
                resolution=(self._width, self._height),
            )
            self._camera.initialize()
            self._camera.add_motion_vectors_to_frame()
            self._log.info("RGB camera initialised at %s.", self._prim_path)
        except Exception as exc:  # pragma: no cover
            self._log.warning("RGB camera mock fallback (%s).", exc)
            self._mock = True

    def read(self) -> np.ndarray:
        """Return the current RGB frame as an (H, W, 3) uint8 array (RGB)."""
        if self._mock or self._camera is None:
            return self._synthetic_frame()
        frame = self._camera.get_rgba()[:, :, :3]
        return frame.astype(np.uint8)

    def _synthetic_frame(self) -> np.ndarray:
        """
        Render a grey frame, painting the rescue target as a high-visibility
        orange blob (RGB orange = 230, 110, 20) and obstacles as dark grey, at
        the pixel locations given by the world-fixed mock scene.
        """
        frame = np.full((self._height, self._width, 3), 90, dtype=np.uint8)
        if self._scene is None:
            return frame
        for proj in self._scene.project_all():
            color = (230, 110, 20) if proj.kind == "target" else (40, 40, 40)
            self._paint_blob(frame, proj.u, proj.v, proj.radius_px, color)
        return frame

    @staticmethod
    def _paint_blob(frame: np.ndarray, u: int, v: int, radius: int, color) -> None:
        y0, y1 = max(0, v - radius), min(frame.shape[0], v + radius)
        x0, x1 = max(0, u - radius), min(frame.shape[1], u + radius)
        frame[y0:y1, x0:x1] = color

    @property
    def resolution(self) -> Tuple[int, int]:
        return (self._width, self._height)


class DepthCameraSensor:
    """Reads depth frames and exposes pinhole intrinsics for back-projection."""

    def __init__(
        self,
        prim_path: str,
        resolution: Tuple[int, int],
        min_range: float,
        max_range: float,
        horizontal_fov_deg: float = 54.0,
        mock: bool = False,
    ) -> None:
        self._log = get_logger("depth_camera")
        self._prim_path = prim_path
        self._width, self._height = resolution
        self._min_range = min_range
        self._max_range = max_range
        self._mock = mock
        self._camera: Optional[Any] = None
        self._scene: Optional[MockScene] = None
        self._fx, self._fy, self._cx, self._cy = intrinsics_from_fov(
            self._width, self._height, horizontal_fov_deg
        )
        if not mock:
            self._initialise()

    def bind_scene(self, scene: "MockScene") -> None:
        """Attach the mock ground-truth scene used to render synthetic depth."""
        self._scene = scene

    def _initialise(self) -> None:
        try:
            from isaacsim.sensors.camera import Camera  # type: ignore

            self._camera = Camera(
                prim_path=self._prim_path,
                resolution=(self._width, self._height),
            )
            self._camera.initialize()
            self._camera.add_distance_to_image_plane_to_frame()
            self._log.info("Depth camera initialised at %s.", self._prim_path)
        except Exception as exc:  # pragma: no cover
            self._log.warning("Depth camera mock fallback (%s).", exc)
            self._mock = True

    def read(self) -> np.ndarray:
        """Return a depth image (H, W) in metres, with NaN where invalid."""
        if self._mock or self._camera is None:
            return self._synthetic_depth()
        depth = self._camera.get_depth()
        depth = np.where(
            (depth < self._min_range) | (depth > self._max_range), np.nan, depth
        )
        return depth.astype(np.float32)

    def _synthetic_depth(self) -> np.ndarray:
        """
        Render depth from the world-fixed mock scene: open space (NaN, no
        return) everywhere except blobs at the projected target and obstacle
        pixels, each at its true range. Keeping the cloud sparse keeps the mock
        fast, and projecting from world ground truth keeps it self-consistent
        with the perception math (so the mission actually converges).
        """
        depth = np.full((self._height, self._width), np.nan, dtype=np.float32)
        if self._scene is None:
            return depth
        for proj in self._scene.project_all():
            r = proj.radius_px
            y0, y1 = max(0, proj.v - r), min(self._height, proj.v + r)
            x0, x1 = max(0, proj.u - r), min(self._width, proj.u + r)
            depth[y0:y1, x0:x1] = proj.depth
        return depth

    @property
    def intrinsics(self) -> Tuple[float, float, float, float]:
        """Pinhole intrinsics (fx, fy, cx, cy)."""
        return (self._fx, self._fy, self._cx, self._cy)

    @property
    def resolution(self) -> Tuple[int, int]:
        return (self._width, self._height)


class TorqueSensor:
    """Reads per-joint torque feedback from the manipulator joints."""

    def __init__(self, joint_paths: List[str], mock: bool = False) -> None:
        self._log = get_logger("torque_sensor")
        self._joint_paths = joint_paths
        self._mock = mock
        self._articulation: Optional[Any] = None
        self._sim_load_nm = 0.0  # mock-mode externally driven "felt" load

    def bind_articulation(self, articulation: Any) -> None:
        """Attach the robot articulation so real torques can be queried."""
        self._articulation = articulation
        self._mock = articulation is None

    def set_mock_load(self, load_nm: float) -> None:
        """Mock helper: set the load the arm currently 'feels' (Newton-metres)."""
        self._sim_load_nm = load_nm

    def read(self) -> TorqueReading:
        """Return the latest multi-joint torque reading."""
        if self._mock or self._articulation is None:
            base = self._sim_load_nm
            values = [base + float(np.random.normal(0, 0.3)) for _ in self._joint_paths]
            return TorqueReading(values_nm=values)
        try:
            efforts = self._articulation.get_measured_joint_efforts()
            values = [float(v) for v in np.atleast_1d(efforts)]
        except Exception as exc:  # pragma: no cover
            self._log.warning("Torque read failed (%s); returning zeros.", exc)
            values = [0.0 for _ in self._joint_paths]
        return TorqueReading(values_nm=values)
