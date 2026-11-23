"""Launches the real-time scene analysis pipeline.

    ros2 launch src/ros2_nodes/launch/scene_analysis.launch.py                # live sensors
    ros2 launch src/ros2_nodes/launch/scene_analysis.launch.py \\
        kitti_split_dir:=/data/kitti/training use_slam:=false               # KITTI frames
    ros2 launch src/ros2_nodes/launch/scene_analysis.launch.py \\
        kitti_raw_dir:=/data/raw/2011_09_26/2011_09_26_drive_0009_sync \\
        rviz_config:=config/rviz/kitti_drive.rviz use_slam:=false           # KITTI raw drive

Arguments:
    params_file      ROS parameters (default config/ros2_pipeline.yaml)
    kitti_split_dir  If set, KITTI Object frames are replayed instead of live sensors
    kitti_raw_dir    If set, a KITTI raw drive is replayed (with ego motion as TF)
    use_slam         Start FAST-LIO2 (needs a LiDAR with IMU; not with KITTI replay)
    fast_lio_config  FAST-LIO2 config file name inside the fast_lio package
    rviz             Start RViz2
    rviz_config      RViz2 config (kitti_drive.rviz follows the car in the world frame)
    python           Interpreter that has the project installed (e.g. a venv)

Project nodes are started with ``python -m`` so the project needs no colcon build;
only third-party packages (FAST-LIO2, sensor drivers) live in a ROS 2 workspace.
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.substitutions import FindPackageShare

PROJECT_ROOT = Path(__file__).resolve().parents[3]
KITTI_CALIBRATION = "/tmp/kitti_calibration.yaml"


def python_node(module: str, *extra_args, condition=None) -> ExecuteProcess:
    return ExecuteProcess(
        cmd=[
            LaunchConfiguration("python"),
            "-m",
            f"multimodal_transformer.ros2_nodes.{module}",
            "--ros-args",
            "--params-file",
            LaunchConfiguration("params_file"),
            *extra_args,
        ],  # fmt: skip
        cwd=str(PROJECT_ROOT),
        output="screen",
        condition=condition,
    )


def generate_launch_description() -> LaunchDescription:
    replaying = PythonExpression(
        [
            "'", LaunchConfiguration("kitti_split_dir"), "' != '' or '",
            LaunchConfiguration("kitti_raw_dir"), "' != ''",
        ]
    )  # fmt: skip
    replay = IfCondition(replaying)
    live = UnlessCondition(replaying)

    args = [
        DeclareLaunchArgument(
            "params_file", default_value=str(PROJECT_ROOT / "config" / "ros2_pipeline.yaml")
        ),
        DeclareLaunchArgument("kitti_split_dir", default_value=""),
        DeclareLaunchArgument("kitti_raw_dir", default_value=""),
        DeclareLaunchArgument("use_slam", default_value="true"),
        DeclareLaunchArgument("fast_lio_config", default_value="mid360.yaml"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument(
            "rviz_config",
            default_value=str(PROJECT_ROOT / "config" / "rviz" / "scene_analysis.rviz"),
        ),
        DeclareLaunchArgument("python", default_value="python3"),
    ]

    # Paths are quoted so that an empty (unused) source is still a valid string parameter.
    player = python_node(
        "kitti_player_node",
        "-p", ["kitti_split_dir:='", LaunchConfiguration("kitti_split_dir"), "'"],
        "-p", ["kitti_raw_dir:='", LaunchConfiguration("kitti_raw_dir"), "'"],
        "-p", f"calibration_out:={KITTI_CALIBRATION}",
        condition=replay,
    )  # fmt: skip
    roi = python_node("roi_extraction_node")
    inference_live = python_node("inference_node", condition=live)
    # With replay, the calibration file is produced by the player at start-up, so the
    # inference node is delayed until it exists.
    inference_replay = TimerAction(
        period=3.0,
        actions=[
            python_node(
                "inference_node", "-p", f"calibration_file:={KITTI_CALIBRATION}", condition=replay
            )
        ],
    )

    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("fast_lio"), "launch", "mapping.launch.py"])
        ),
        launch_arguments={
            "config_file": LaunchConfiguration("fast_lio_config"),
            "rviz": "false",
        }.items(),
        condition=IfCondition(LaunchConfiguration("use_slam")),
    )

    rviz = ExecuteProcess(
        cmd=["rviz2", "-d", LaunchConfiguration("rviz_config")],
        output="screen",
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    return LaunchDescription(args + [player, slam, roi, inference_live, inference_replay, rviz])
