"""Action -> RobotAction 与 RobotState -> ProprioState 转换逻辑。"""

from __future__ import annotations

from forge_msgs import Action, ActuatorValue, JointValue, ProprioState, RobotAction, RobotState

from config import MuJoCoNodeConfig, RobotConfig


def build_robot_action(action: Action, robot: RobotConfig) -> RobotAction:
    """根据 Action 和机器人配置，生成该机器人的 RobotAction。"""
    actuators: dict[str, ActuatorValue] = {}

    for joint_name, actuator_name in robot.joint_to_actuator.items():
        if joint_name in action.joints:
            jv = action.joints[joint_name]
            actuators[actuator_name] = ActuatorValue(
                value=jv.value,
                mode=jv.mode,
                unit=jv.unit,
            )
        else:
            actuators[actuator_name] = ActuatorValue(
                value=0.0,
                mode="position",
                unit="radians",
            )

    return RobotAction(
        timestamp=action.ref_timestamp,
        actuators=actuators,
    )


def build_proprio_state(
    robot_states: dict[str, RobotState],
    config: MuJoCoNodeConfig,
) -> ProprioState:
    """将各机器人的 RobotState 聚合为 ProprioState。"""
    joints: dict[str, JointValue] = {}
    latest_timestamp = 0.0

    for robot in config.robots:
        state = robot_states.get(robot.id)
        if not state:
            continue

        actuator_to_joint = {v: k for k, v in robot.joint_to_actuator.items()}

        for actuator_name, av in state.actuators.items():
            joint_name = actuator_to_joint.get(actuator_name)
            if joint_name is not None:
                joints[joint_name] = JointValue(
                    value=av.value,
                    mode=av.mode,
                    unit=av.unit,
                )

        if state.timestamp > latest_timestamp:
            latest_timestamp = state.timestamp

    return ProprioState(timestamp=latest_timestamp, joints=joints)


def sim_state_to_robot_state(
    sim_state: RobotState,
    robot: RobotConfig,
) -> RobotState:
    """
    将 MuJoCoSimulator 返回的 joint-keyed RobotState 转为 actuator-keyed，
    供 build_proprio_state 使用。
    """
    actuator_to_joint = {v: k for k, v in robot.joint_to_actuator.items()}
    actuators: dict[str, ActuatorValue] = {}

    for actuator_name in robot.actuator_order:
        joint_name = actuator_to_joint.get(actuator_name)
        if joint_name is not None and joint_name in sim_state.actuators:
            actuators[actuator_name] = sim_state.actuators[joint_name]

    return RobotState(timestamp=sim_state.timestamp, actuators=actuators)
