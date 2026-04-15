"""Action -> RobotAction 与 RobotState -> ProprioState 转换逻辑。"""

from __future__ import annotations

import pyarrow as pa
from forge_msgs import Action, ActuatorValue, JointValue, ProprioState, RobotAction, RobotState
from forge_msgs.value import ensure_record_batch

from config import MuJoCoNodeConfig, RobotConfig


def master_state_arrow_parse_order(
    value: pa.RecordBatch | pa.Table | bytes,
    robot: RobotConfig,
) -> list[str]:
    """为 master_state 的 RobotState.from_arrow 选择 Arrow 列名（与 schema 一致、长度同 actuator_order）。

    上游常见两种表头：与 ``actuator_order`` 一致的 **actuator 短名**（如 joint1、gripper），
    或与 ``joint_to_actuator`` 的 key 一致的 **MuJoCo/配置关节全路径**（如 robot_left/joint1）。
    对每个自由度依次：若表中存在 actuator 列则用之（主路径，与 vr_device ``*/state`` 一致）；
    否则若存在该自由度对应的 joint 列则用之；否则仍填 actuator 名，由 ``from_arrow`` 用默认 0。

    ``RobotState.from_arrow`` 会用本列表的字符串既读列又写入 ``actuators`` 的 key；
    ``robot_state_to_robot_action`` 已同时支持 actuator key 与 joint key。
    """
    batch = ensure_record_batch(value)
    names = set(batch.schema.names)
    order: list[str] = []
    for act_name in robot.actuator_order:
        if act_name in names:
            order.append(act_name)
            continue
        joint_name = next(
            (j for j, a in robot.joint_to_actuator.items() if a == act_name),
            None,
        )
        if joint_name is not None and joint_name in names:
            order.append(joint_name)
        else:
            order.append(act_name)
    return order


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

    return RobotAction(actuators=actuators)


def build_proprio_state(
    robot_states: dict[str, RobotState],
    config: MuJoCoNodeConfig,
) -> ProprioState:
    """将各机器人的 RobotState 聚合为 ProprioState。"""
    joints: dict[str, JointValue] = {}

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

    return ProprioState(joints=joints)


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
        # 优先读取模型侧同名 actuator/joint（MuJoCo 内部常见）
        if actuator_name in sim_state.actuators:
            actuators[actuator_name] = sim_state.actuators[actuator_name]
            continue
        # 兼容旧行为：读取 policy joint 名
        joint_name = actuator_to_joint.get(actuator_name)
        if joint_name is not None and joint_name in sim_state.actuators:
            actuators[actuator_name] = sim_state.actuators[joint_name]

    return RobotState(actuators=actuators)


def robot_state_to_robot_action(
    state: RobotState,
    robot: RobotConfig,
) -> RobotAction:
    """将 RobotState 转换为该机器人的 RobotAction。

    兼容两种输入格式：
    1) state.actuators 使用 actuator 名称（默认）
    2) state.actuators 使用 joint 名称（多臂场景常见），通过 joint_to_actuator 映射
    """
    actuators: dict[str, ActuatorValue] = {}
    actuator_to_joint = {v: k for k, v in robot.joint_to_actuator.items()}
    for name in robot.actuator_order:
        if name in state.actuators:
            av = state.actuators[name]
            actuators[name] = ActuatorValue(
                value=av.value,
                mode=av.mode,
                unit=av.unit,
            )
            continue
        # 兼容 joint-name keyed RobotState
        joint_name = actuator_to_joint.get(name)
        if joint_name and joint_name in state.actuators:
            av = state.actuators[joint_name]
            actuators[name] = ActuatorValue(
                value=av.value,
                mode=av.mode,
                unit=av.unit,
            )
    return RobotAction(actuators=actuators)
