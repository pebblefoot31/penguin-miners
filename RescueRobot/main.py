"""
main.py
=======

Entry point for the autonomous rescue-robot simulation. Wires the full
perception → planning → control pipeline together and runs the mission loop:

    deploy from transport vehicle
        └─► [perceive → plan (LLM) → enforce safety → act → step sim] ──► COMPLETE/ABORT

Run from inside the Isaac Sim Python environment::

    ./python.sh main.py --config config/config.yaml

If Isaac Sim is not importable, the program automatically runs in ``--mock``
mode so the pipeline and logic can still be exercised end-to-end.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Ensure the package root is importable when launched directly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai.llm_planner import LLMPlanner  # noqa: E402
from ai.prompt_builder import PromptBuilder  # noqa: E402
from manipulation.arm_controller import ArmController  # noqa: E402
from manipulation.capsule_loader import CapsuleLoader  # noqa: E402
from manipulation.grasp_planner import GraspPlanner  # noqa: E402
from manipulation.torque_monitor import TorqueMonitor  # noqa: E402
from mission.mission_controller import MissionController  # noqa: E402
from mission.state_machine import StateMachine  # noqa: E402
from mission.states import MissionState  # noqa: E402
from navigation.astar_planner import AStarPlanner  # noqa: E402
from navigation.occupancy_grid import OccupancyGrid  # noqa: E402
from navigation.path_manager import PathManager  # noqa: E402
from navigation.waypoint_follower import WaypointFollower  # noqa: E402
from perception.depth_camera import DepthCamera  # noqa: E402
from perception.obstacle_detector import ObstacleDetector  # noqa: E402
from perception.rgb_camera import RGBCamera  # noqa: E402
from perception.suit_detector import SuitDetector  # noqa: E402
from perception.target_localizer import TargetLocalizer  # noqa: E402
from perception.world_model import WorldModelBuilder  # noqa: E402
from simulation.isaac_app import IsaacApp  # noqa: E402
from simulation.robot import RescueRobot  # noqa: E402
from simulation.sensors import (  # noqa: E402
    DepthCameraSensor,
    RGBCameraSensor,
    TorqueSensor,
)
from simulation.transport_vehicle import TransportVehicle  # noqa: E402
from utils.config_loader import Config  # noqa: E402
from utils.logger import DecisionLogger, get_logger  # noqa: E402
from utils.types import Position3D, Pose2D  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous rescue-robot simulation.")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "config", "config.yaml"),
        help="Path to the master YAML config.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force mock mode (skip Isaac Sim even if available).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2000,
        help="Safety cap on simulation steps.",
    )
    return parser.parse_args()


def build_system(config: Config, force_mock: bool):
    """Construct and wire every component. Returns (controller, isaac_app, mock)."""
    log = get_logger("main")

    # --- Simulation / Isaac Sim app ---------------------------------------
    sim_cfg = config.section("simulation")
    isaac = IsaacApp(
        headless=sim_cfg.get("headless", False),
        physics_dt=sim_cfg.get("physics_dt", 1.0 / 60.0),
        rendering_dt=sim_cfg.get("rendering_dt", 1.0 / 30.0),
        stage_units_in_meters=sim_cfg.get("stage_units_in_meters", 1.0),
    )
    mock = force_mock or not isaac.available
    if mock:
        log.info("Running in MOCK mode (no live Isaac Sim).")

    # --- Decision / event logging -----------------------------------------
    decisions = DecisionLogger(
        config.get("logging.decision_log_path", "logs/decisions.jsonl"),
        also_console=config.get("logging.console", True),
    )

    # --- Sensors -----------------------------------------------------------
    rgb_cfg = config.get("sensors.rgb_camera", {})
    depth_cfg = config.get("sensors.depth_camera", {})
    torque_cfg = config.get("sensors.torque_sensor", {})

    rgb_sensor = RGBCameraSensor(
        prim_path=rgb_cfg.get("prim_path", ""),
        resolution=tuple(rgb_cfg.get("resolution", [1280, 720])),
        horizontal_fov_deg=rgb_cfg.get("horizontal_fov_deg", 54.0),
        mock=mock,
    )
    depth_sensor = DepthCameraSensor(
        prim_path=depth_cfg.get("prim_path", ""),
        resolution=tuple(depth_cfg.get("resolution", [1280, 720])),
        min_range=depth_cfg.get("min_range", 0.3),
        max_range=depth_cfg.get("max_range", 8.0),
        horizontal_fov_deg=rgb_cfg.get("horizontal_fov_deg", 54.0),
        mock=mock,
    )
    torque_sensor = TorqueSensor(
        joint_paths=torque_cfg.get("joint_paths", ["j0"]),
        mock=mock,
    )

    # --- Robot + transport vehicle ----------------------------------------
    robot_cfg = config.section("robot")
    robot = RescueRobot(
        prim_path=sim_cfg.get("robot_prim_path", "/World/RescueRobot"),
        arm_dof_count=robot_cfg.get("arm_dof_count", 6),
        arm_home_config=robot_cfg.get("arm_home_config", [0, 0, 0, 0, 0, 0]),
        max_linear_speed=robot_cfg.get("base_max_linear_speed", 0.8),
        max_angular_speed=robot_cfg.get("base_max_angular_speed", 1.2),
        mock=mock,
    )
    torque_sensor.bind_articulation(robot.articulation)
    vehicle = TransportVehicle(deploy_pose=Pose2D(0.0, 0.0, 0.0), mock=mock)

    # In mock mode, attach a world-fixed ground-truth scene so the synthetic
    # cameras render a consistent target + obstacles that the mission can solve.
    if mock:
        from simulation.mock_scene import MockScene  # local import: mock-only

        scene = MockScene(
            intrinsics=depth_sensor.intrinsics,
            resolution=tuple(depth_cfg.get("resolution", [1280, 720])),
            min_range=depth_cfg.get("min_range", 0.3),
            max_range=depth_cfg.get("max_range", 8.0),
            pose_provider=robot.get_pose,
        )
        scene.add_target((3.0, 0.0, 0.9))      # rescue target ~3 m dead ahead
        scene.add_obstacle((1.5, 1.4, 0.4))    # rubble beside the corridor
        rgb_sensor.bind_scene(scene)
        depth_sensor.bind_scene(scene)

    # --- Perception --------------------------------------------------------
    suit_signature = Config.load(
        os.path.join(os.path.dirname(__file__), "config", "rescue_suit.yaml")
    ).get("rescue_suit", {})
    yolo_cfg = config.get("perception.yolo", {})
    depth_perc = config.get("perception.depth", {})

    rgb_camera = RGBCamera(rgb_sensor)
    depth_camera = DepthCamera(depth_sensor)
    suit_detector = SuitDetector(
        weights=yolo_cfg.get("weights", "yolo.pt"),
        target_class_name=yolo_cfg.get("target_class_name", "rescue_suit"),
        confidence_threshold=yolo_cfg.get("confidence_threshold", 0.55),
        iou_threshold=yolo_cfg.get("iou_threshold", 0.45),
        device=yolo_cfg.get("device", "cpu"),
        suit_signature=suit_signature,
        mock=mock,
    )
    target_localizer = TargetLocalizer(depth_camera)
    obstacle_detector = ObstacleDetector(
        depth_camera=depth_camera,
        voxel_size=depth_perc.get("voxel_size", 0.05),
        obstacle_min_height=depth_perc.get("obstacle_min_height", 0.10),
        obstacle_max_height=depth_perc.get("obstacle_max_height", 2.0),
        ground_plane_tolerance=depth_perc.get("ground_plane_tolerance", 0.04),
    )
    world_builder = WorldModelBuilder(
        rgb_camera=rgb_camera,
        depth_camera=depth_camera,
        suit_detector=suit_detector,
        target_localizer=target_localizer,
        obstacle_detector=obstacle_detector,
        max_target_age_s=config.get("perception.fusion.max_target_age_s", 1.0),
        target_exclusion_radius_m=config.get(
            "perception.fusion.target_exclusion_radius_m", 0.7
        ),
    )

    # --- Navigation --------------------------------------------------------
    grid_cfg = config.get("navigation.occupancy_grid", {})
    astar_cfg = config.get("navigation.astar", {})
    follow_cfg = config.get("navigation.waypoint_follower", {})

    grid = OccupancyGrid(
        resolution=grid_cfg.get("resolution", 0.10),
        width=grid_cfg.get("width", 200),
        height=grid_cfg.get("height", 200),
        origin=tuple(grid_cfg.get("origin", [-10.0, -10.0])),
        inflation_cells=astar_cfg.get("obstacle_inflation_cells", 5),
    )
    planner_astar = AStarPlanner(
        grid=grid,
        allow_diagonal=astar_cfg.get("allow_diagonal", True),
        heuristic_weight=astar_cfg.get("heuristic_weight", 1.0),
    )
    path_manager = PathManager(grid=grid, planner=planner_astar)
    waypoint_follower = WaypointFollower(
        robot=robot,
        max_linear_speed=robot_cfg.get("base_max_linear_speed", 0.8),
        max_angular_speed=robot_cfg.get("base_max_angular_speed", 1.2),
        arrival_tolerance=follow_cfg.get("arrival_tolerance", 0.15),
        lookahead_distance=follow_cfg.get("lookahead_distance", 0.5),
    )

    # --- Manipulation ------------------------------------------------------
    grasp_cfg = config.get("manipulation.grasp", {})
    torque_lim = config.get("manipulation.torque", {})
    capsule_cfg = config.get("manipulation.capsule", {})

    torque_monitor = TorqueMonitor(
        sensor=torque_sensor,
        safety_threshold_nm=torque_lim.get("safety_threshold_nm", 35.0),
        warning_threshold_nm=torque_lim.get("warning_threshold_nm", 28.0),
        contact_threshold_nm=torque_lim.get("grasp_contact_threshold_nm", 4.0),
    )
    grasp_planner = GraspPlanner(
        handle_offsets=grasp_cfg.get("handle_offsets", [[0, 0.18, 0.05], [0, -0.18, 0.05]]),
        pre_grasp_standoff=grasp_cfg.get("pre_grasp_standoff", 0.12),
        handle_names=["left_shoulder_loop", "right_shoulder_loop"],
    )
    arm_controller = ArmController(
        robot=robot,
        torque_monitor=torque_monitor,
        lift_height=grasp_cfg.get("lift_height", 0.4),
    )
    capsule_loader = CapsuleLoader(
        arm=arm_controller,
        torque_monitor=torque_monitor,
        capsule_load_pose=Position3D(*capsule_cfg.get("load_pose", [0.0, 0.0, 0.6])),
        robot=robot,
    )

    # --- AI high-level planner --------------------------------------------
    ai_cfg = config.section("ai")
    prompt_builder = PromptBuilder(
        torque_monitor=torque_monitor,
        approach_distance=follow_cfg.get("approach_distance", 1.0),
    )
    llm_planner = LLMPlanner(
        prompt_builder=prompt_builder,
        decision_logger=decisions,
        model=ai_cfg.get("model", "claude-opus-4-8"),
        effort=ai_cfg.get("effort", "medium"),
        max_tokens=ai_cfg.get("max_tokens", 1024),
        timeout_s=ai_cfg.get("timeout_s", 30),
        enable_llm=ai_cfg.get("enable_llm", True),
    )

    # --- Mission FSM + controller -----------------------------------------
    state_machine = StateMachine(decisions, initial=MissionState.SEARCH)
    controller = MissionController(
        robot=robot,
        world_builder=world_builder,
        path_manager=path_manager,
        waypoint_follower=waypoint_follower,
        grasp_planner=grasp_planner,
        arm_controller=arm_controller,
        torque_monitor=torque_monitor,
        capsule_loader=capsule_loader,
        planner=llm_planner,
        state_machine=state_machine,
        decision_logger=decisions,
        config=config,
    )

    # Deploy the robot from the transport vehicle before the mission begins.
    decisions.log("deploy", {"from": "transport_vehicle"})
    vehicle.deploy(robot)

    return controller, isaac, mock


def run(controller: MissionController, isaac: IsaacApp, mock: bool, max_steps: int):
    """Run the mission loop until COMPLETE/ABORT or the step cap is hit."""
    log = get_logger("main")
    physics_dt = 1.0 / 60.0

    if not mock:
        isaac.reset()

    log.info("Starting rescue mission.")
    for step in range(max_steps):
        if not mock and not isaac.is_running():
            log.info("Isaac Sim window closed; stopping.")
            break

        state = controller.tick(physics_dt)

        if not mock:
            isaac.step(render=True)
        else:
            time.sleep(0.0)  # cooperative; no real-time wait needed in mock mode

        if controller.is_done:
            log.info("Mission finished in state %s after %d steps.", state.value, step)
            break
    else:
        log.warning("Reached max steps (%d) without terminal state.", max_steps)

    final = controller.state
    get_logger("main").info("FINAL MISSION STATE: %s", final.value)
    return final


def main() -> int:
    args = parse_args()
    config = Config.load(args.config)
    log = get_logger(
        "main",
        level=config.get("logging.level", "INFO"),
        console=config.get("logging.console", True),
    )

    controller, isaac, mock = build_system(config, force_mock=args.mock)
    try:
        final_state = run(controller, isaac, mock, args.max_steps)
    finally:
        if not mock:
            isaac.shutdown()

    return 0 if final_state is MissionState.COMPLETE else 1


if __name__ == "__main__":
    raise SystemExit(main())
