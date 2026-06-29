"""
manipulation/torque_monitor.py
==============================

Continuously monitors the arm torque sensor during grasping and lifting. Exposes
the latest reading, a warning flag, and a hard safety-violation flag. The
manipulation controllers poll this monitor and abort the lift if torque exceeds
the configurable safety threshold.

Single responsibility: interpret raw torque readings against safety limits.
"""

from __future__ import annotations

from typing import Optional

from simulation.sensors import TorqueSensor
from utils.logger import get_logger
from utils.types import TorqueReading


class TorqueMonitor:
    """Tracks arm torque and flags warning / unsafe conditions."""

    def __init__(
        self,
        sensor: TorqueSensor,
        safety_threshold_nm: float,
        warning_threshold_nm: float,
        contact_threshold_nm: float,
    ) -> None:
        self._log = get_logger("manipulation.torque_monitor")
        self._sensor = sensor
        self._safety = safety_threshold_nm
        self._warning = warning_threshold_nm
        self._contact = contact_threshold_nm
        self._last: Optional[TorqueReading] = None

    def simulate_load(self, load_nm: float) -> None:
        """
        Mock-only hook: in the absence of a physics engine, tell the underlying
        torque sensor what load the gripper is currently bearing so the contact
        / lift / safety logic can be exercised. A no-op against a real sensor.
        """
        if hasattr(self._sensor, "set_mock_load"):
            self._sensor.set_mock_load(load_nm)

    def sample(self) -> TorqueReading:
        """Take a fresh torque reading and cache it."""
        self._last = self._sensor.read()
        if self.is_unsafe():
            self._log.warning(
                "Torque %.1f Nm exceeds safety threshold %.1f Nm.",
                self._last.max_abs,
                self._safety,
            )
        return self._last

    @property
    def last_reading(self) -> Optional[TorqueReading]:
        return self._last

    @property
    def max_torque(self) -> float:
        """Maximum absolute torque from the most recent reading (Nm)."""
        return self._last.max_abs if self._last else 0.0

    def is_unsafe(self) -> bool:
        """True if the latest reading exceeds the hard safety threshold."""
        return self._last is not None and self._last.max_abs > self._safety

    def is_warning(self) -> bool:
        """True if torque is in the warning band (approaching the limit)."""
        return self._last is not None and self._last.max_abs > self._warning

    def has_contact(self) -> bool:
        """True if torque indicates the gripper has contacted a handle/load."""
        return self._last is not None and self._last.max_abs >= self._contact

    @property
    def safety_threshold(self) -> float:
        return self._safety
