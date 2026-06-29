# RescueRobot — Autonomous Disaster Rescue Robot (NVIDIA Isaac Sim)

A modular Python project simulating an autonomous rescue robot in a disaster
environment using NVIDIA Isaac Sim. The robot is transported on a vehicle until
deployment, then autonomously **detects → navigates to → retrieves → loads** a
rescue target (a person wearing a predefined rescue suit) into a transport
capsule.

The codebase follows the classic robotics pipeline:

```
        Perception  ──►  Planning / Navigation  ──►  Control / Manipulation
            ▲                      │                          │
            └──────────  World Model / Mission FSM  ◄─────────┘
                                   │
                          LLM High-Level Planner
```

## Design principles

- **One responsibility per file.** Each class lives in its own module with a
  clear interface and comprehensive docstrings.
- **Object-oriented.** Components communicate through small, typed data classes
  (see `utils/types.py`) rather than passing raw arrays around.
- **The LLM never touches motors.** The LLM is a *high-level* planner: it
  receives a structured robot-state summary and returns one of a fixed set of
  high-level actions. Motor commands are always produced by deterministic
  controllers.
- **Safety first.** Torque limits, target-loss handling, and obstacle replanning
  are enforced by the mission controller, independent of the LLM.

## Package layout

```
RescueRobot/
├── main.py                       # Entry point — wires everything together
├── requirements.txt
├── config/
│   ├── config.yaml               # All tunable parameters
│   └── rescue_suit.yaml          # Rescue-suit detection signature
├── utils/
│   ├── logger.py                 # Structured logging + decision log
│   ├── config_loader.py          # YAML config loading
│   ├── types.py                  # Shared dataclasses (Detection, Pose, ...)
│   └── geometry.py               # Coordinate transforms & math helpers
├── simulation/
│   ├── isaac_app.py              # Isaac Sim SimulationApp bootstrap
│   ├── sensors.py                # RGB / depth / torque sensor wrappers
│   ├── robot.py                  # Robot articulation + base/arm command API
│   ├── transport_vehicle.py      # Deploy/undock from the transport vehicle
│   └── mock_scene.py             # World-fixed ground truth for mock mode
├── perception/
│   ├── rgb_camera.py             # Read RGB frames (Zivid M60)
│   ├── depth_camera.py           # Read depth frames + point clouds
│   ├── suit_detector.py          # YOLO rescue-suit detector
│   ├── target_localizer.py       # 3D target position from depth
│   ├── obstacle_detector.py      # Static obstacles from depth
│   └── world_model.py            # Fuse RGB + depth into a world model
├── navigation/
│   ├── occupancy_grid.py         # Build occupancy grid from depth
│   ├── astar_planner.py          # A* path planning
│   ├── path_manager.py           # Re-plan on new obstacles
│   └── waypoint_follower.py      # Drive the base along waypoints
├── manipulation/
│   ├── arm_controller.py         # Move arm to grasp poses
│   ├── torque_monitor.py         # Continuous torque safety monitor
│   ├── grasp_planner.py          # Plan grasp on suit handles
│   └── capsule_loader.py         # Load target into the capsule
├── ai/
│   ├── prompt_builder.py         # Build structured planner prompt
│   └── llm_planner.py            # Claude-based high-level planner
└── mission/
    ├── states.py                 # MissionState enum + action enum
    ├── state_machine.py          # Generic FSM with transition logging
    └── mission_controller.py     # Orchestrates the full mission
```

## Running

This project targets **Python 3.11** and **NVIDIA Isaac Sim**. Run it from
inside the Isaac Sim Python environment (so the `omni.*` / `isaacsim.*` modules
are importable):

```bash
# From the Isaac Sim install:
./python.sh /path/to/RescueRobot/main.py --config config/config.yaml
```

If Isaac Sim is not available, the simulation layer falls back to lightweight
stub interfaces so the perception/navigation/mission logic can still be
exercised and unit-tested (`--mock` flag, enabled automatically when the Isaac
modules are missing).

### LLM planner

The high-level planner calls the Claude API via the official `anthropic` SDK and
uses **structured outputs** so the returned action is always one of the allowed
high-level actions. Set your key first:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

The planner model defaults to `claude-opus-4-8` and is configurable in
`config/config.yaml` (`ai.model`). If no API key is present, the planner falls
back to a deterministic rule-based policy so the mission still runs.

## Dependencies

See `requirements.txt`. The core third-party libraries are NumPy, OpenCV,
Open3D, Ultralytics YOLO, PyYAML, and the Anthropic SDK. Isaac Sim provides its
own bundled Python packages.
