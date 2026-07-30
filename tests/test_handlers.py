from __future__ import annotations

from forge_msgs import JointCommand

from mujoco_sim.config import RobotConfig
from mujoco_sim.handlers import build_robot_command


def _robot() -> RobotConfig:
    return RobotConfig(
        id="piper",
        joints=["joint1", "joint2", "gripper"],
        actuator_order=["motor1", "motor2", "gripper_motor"],
        joint_to_actuator={
            "joint1": "motor1",
            "joint2": "motor2",
            "gripper": "gripper_motor",
        },
    )


def test_arm_command_does_not_add_or_zero_gripper() -> None:
    mapped = build_robot_command(
        JointCommand(name=["joint2", "joint1"], position=[0.2, 0.1]),
        _robot(),
    )

    assert mapped == JointCommand(
        name=["motor1", "motor2"],
        position=[0.1, 0.2],
    )


def test_gripper_command_does_not_add_or_zero_arm() -> None:
    mapped = build_robot_command(
        JointCommand(name=["gripper"], position=[0.04]),
        _robot(),
    )

    assert mapped == JointCommand(name=["gripper_motor"], position=[0.04])


def test_unrelated_robot_receives_no_command() -> None:
    assert (
        build_robot_command(
            JointCommand(name=["other_joint"], position=[1.0]),
            _robot(),
        )
        is None
    )


def test_sparse_mapping_preserves_mode_and_populated_fields() -> None:
    mapped = build_robot_command(
        JointCommand(
            name=["joint1"],
            mode="hybrid",
            position=[0.1],
            velocity=[0.2],
            effort=[0.3],
            kp=[20.0],
            kd=[1.0],
        ),
        _robot(),
    )

    assert mapped == JointCommand(
        name=["motor1"],
        mode="hybrid",
        position=[0.1],
        velocity=[0.2],
        effort=[0.3],
        kp=[20.0],
        kd=[1.0],
    )
