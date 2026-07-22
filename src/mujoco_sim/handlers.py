"""JointCommand/JointState conversion logic for MuJoCo."""

from __future__ import annotations

from forge_msgs import JointCommand, JointState

from .config import MuJoCoNodeConfig, RobotConfig


def _values_by_name(message: JointCommand | JointState, field: str) -> dict[str, float]:
    values = getattr(message, field)
    if not values:
        return {}
    return dict(zip(message.name, values, strict=True))


def build_robot_command(command: JointCommand, robot: RobotConfig) -> JointCommand:
    """Map global joint command names to this robot's actuator names."""

    source = {
        "position": _values_by_name(command, "position"),
        "velocity": _values_by_name(command, "velocity"),
        "effort": _values_by_name(command, "effort"),
        "kp": _values_by_name(command, "kp"),
        "kd": _values_by_name(command, "kd"),
    }

    names: list[str] = []
    position: list[float] = []
    velocity: list[float] = []
    effort: list[float] = []
    kp: list[float] = []
    kd: list[float] = []

    for joint_name, actuator_name in robot.joint_to_actuator.items():
        names.append(actuator_name)
        position.append(source["position"].get(joint_name, 0.0))
        velocity.append(source["velocity"].get(joint_name, 0.0))
        effort.append(source["effort"].get(joint_name, 0.0))
        kp.append(source["kp"].get(joint_name, 0.0))
        kd.append(source["kd"].get(joint_name, 0.0))

    return JointCommand(
        name=names,
        position=position,
        velocity=velocity,
        effort=effort,
        kp=kp,
        kd=kd,
    )


def build_joint_state(
    robot_states: dict[str, JointState],
    config: MuJoCoNodeConfig,
) -> JointState:
    """Aggregate per-robot actuator-keyed states into a global JointState."""

    positions: dict[str, float] = {}
    for robot in config.robots:
        state = robot_states.get(robot.id)
        if not state:
            continue

        values = _values_by_name(state, "position")
        actuator_to_joint = {v: k for k, v in robot.joint_to_actuator.items()}
        for actuator_name, value in values.items():
            joint_name = actuator_to_joint.get(actuator_name)
            if joint_name is not None:
                positions[joint_name] = value

    names = [name for name in config.joint_order if name in positions]
    return JointState(
        name=names,
        position=[positions[name] for name in names],
        velocity=[],
        effort=[],
    )


def sim_state_to_robot_state(
    sim_state: JointState,
    robot: RobotConfig,
) -> JointState:
    """Normalize simulator state into actuator-keyed JointState."""

    values = _values_by_name(sim_state, "position")
    actuator_to_joint = {v: k for k, v in robot.joint_to_actuator.items()}
    names: list[str] = []
    position: list[float] = []

    for actuator_name in robot.actuator_order:
        if actuator_name in values:
            names.append(actuator_name)
            position.append(values[actuator_name])
            continue
        joint_name = actuator_to_joint.get(actuator_name)
        if joint_name is not None and joint_name in values:
            names.append(actuator_name)
            position.append(values[joint_name])

    return JointState(name=names, position=position, velocity=[], effort=[])


def robot_state_to_robot_command(
    state: JointState,
    robot: RobotConfig,
) -> JointCommand:
    """Convert master JointState into actuator-keyed JointCommand."""

    values = _values_by_name(state, "position")
    actuator_to_joint = {v: k for k, v in robot.joint_to_actuator.items()}
    names: list[str] = []
    position: list[float] = []

    for actuator_name in robot.actuator_order:
        if actuator_name in values:
            names.append(actuator_name)
            position.append(values[actuator_name])
            continue
        joint_name = actuator_to_joint.get(actuator_name)
        if joint_name is not None and joint_name in values:
            names.append(actuator_name)
            position.append(values[joint_name])

    return JointCommand(
        name=names,
        position=position,
        velocity=[],
        effort=[],
        kp=[],
        kd=[],
    )
