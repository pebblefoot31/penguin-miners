"""
simulation/isaac_app.py
=======================

Bootstraps the Isaac Sim application. The ``SimulationApp`` *must* be created
before any other ``omni.*`` / ``isaacsim.*`` import — this module centralises
that lifecycle so nothing else has to worry about import ordering.

Single responsibility: own the Isaac Sim app + World lifecycle.

If Isaac Sim is not installed, ``IsaacApp.available`` is ``False`` and the
caller should switch to mock mode.
"""

from __future__ import annotations

from typing import Any, Optional

from utils.logger import get_logger


class IsaacApp:
    """Owns the Isaac Sim ``SimulationApp`` and the physics ``World``."""

    def __init__(
        self,
        headless: bool = False,
        physics_dt: float = 1.0 / 60.0,
        rendering_dt: float = 1.0 / 30.0,
        stage_units_in_meters: float = 1.0,
    ) -> None:
        self._log = get_logger("isaac_app")
        self._headless = headless
        self._physics_dt = physics_dt
        self._rendering_dt = rendering_dt
        self._stage_units = stage_units_in_meters
        self._app: Optional[Any] = None
        self._world: Optional[Any] = None
        self._available = False
        self._start()

    def _start(self) -> None:
        """Create the SimulationApp and World, tolerating a missing install."""
        try:
            from isaacsim import SimulationApp  # type: ignore

            self._app = SimulationApp({"headless": self._headless})

            # Imports that require a live SimulationApp:
            from isaacsim.core.api import World  # type: ignore

            self._world = World(
                physics_dt=self._physics_dt,
                rendering_dt=self._rendering_dt,
                stage_units_in_meters=self._stage_units,
            )
            self._available = True
            self._log.info("Isaac Sim initialised (headless=%s).", self._headless)
        except Exception as exc:  # pragma: no cover - depends on Isaac install
            self._log.warning(
                "Isaac Sim unavailable (%s). Falling back to mock mode.", exc
            )
            self._available = False

    @property
    def available(self) -> bool:
        """True when a real Isaac Sim app is running."""
        return self._available

    @property
    def world(self) -> Any:
        return self._world

    def reset(self) -> None:
        """Reset the physics world to its initial state."""
        if self._world is not None:
            self._world.reset()

    def step(self, render: bool = True) -> None:
        """Advance the simulation by one step."""
        if self._world is not None:
            self._world.step(render=render)

    def is_running(self) -> bool:
        """Whether the underlying app window/process is still alive."""
        return bool(self._app is not None and self._app.is_running())

    def shutdown(self) -> None:
        """Cleanly close the Isaac Sim application."""
        if self._app is not None:
            self._log.info("Shutting down Isaac Sim.")
            self._app.close()
            self._app = None
