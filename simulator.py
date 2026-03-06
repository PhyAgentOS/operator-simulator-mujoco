"""MuJoCo 仿真逻辑：接收 RobotAction，输出 RobotState。不依赖 forge_robots_core。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco

from forge_msgs import ActuatorValue, RobotAction, RobotState
from forge_common import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class JointSpec:
    """关节规格，仅 name 与 mode（用于单位推断）。"""

    name: str
    mode: str = "position"


@dataclass(frozen=True)
class ActuatorSpec:
    """执行器规格，用于限位与单位。"""

    name: str
    id: int
    control_mode: str
    min_value: float
    max_value: float
    unit: str = "radians"


class MuJoCoSimulator:
    """
    MuJoCo 仿真节点逻辑。

    直接使用 MuJoCo model/data，通过可选 prefix 将逻辑关节/执行器名映射到模型名。
    单位（msgs 接口）：角度弧度，距离毫米；MuJoCo 内部弧度/米，边界处转换。
    """

    def __init__(
        self,
        model: Any,
        data: Any,
        joints: list[JointSpec] | None = None,
        actuators: list[ActuatorSpec] | None = None,
        derived_state_cfg: dict[str, dict[str, Any]] | None = None,
        prefix: str = "",
    ):
        self._model = model
        self._data = data
        self._joints = joints or []
        self._actuators = actuators or []
        self._joint_map = {j.name: j for j in self._joints}
        self._actuator_map = {a.name: a for a in self._actuators}

        if prefix and not prefix.endswith("/"):
            prefix += "/"
        self._prefix = prefix

        logger.info("[MuJoCoSimulator] Initializing with prefix: '%s'", prefix)

        self._qpos_addrs: dict[str, int] = {}
        self._joint_qpos_addrs: dict[str, int] = {}
        self._ctrl_indices: dict[str, int] = {}

        for joint in self._joints:
            logical_name = joint.name
            prefixed_name = prefix + logical_name
            try:
                joint_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, prefixed_name
                )
                qpos_addr = model.jnt_qposadr[joint_id]
                self._qpos_addrs[logical_name] = qpos_addr
                self._joint_qpos_addrs[logical_name] = qpos_addr
                logger.info(
                    "  - Mapped Joint '%s' -> '%s' at qpos_addr: %s",
                    logical_name,
                    prefixed_name,
                    qpos_addr,
                )
            except Exception:
                logger.warning(
                    "  - WARNING: Joint '%s' not found in mjModel.",
                    prefixed_name,
                )

        for actuator in self._actuators:
            logical_name = actuator.name
            prefixed_name = prefix + logical_name
            try:
                act_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefixed_name
                )
                self._ctrl_indices[logical_name] = act_id
                logger.info(
                    "  - Mapped Actuator '%s' -> '%s' with ctrl_idx: %s",
                    logical_name,
                    prefixed_name,
                    act_id,
                )
            except Exception:
                logger.warning(
                    "  - WARNING: Actuator '%s' not found in mjModel.",
                    prefixed_name,
                )

        self._derived_state_cfg: dict[str, dict[str, Any]] = {}
        if derived_state_cfg:
            for act_name, cfg in derived_state_cfg.items():
                joint_names = cfg.get("joint_names") or []
                expr = cfg.get("expr")
                scale = cfg.get("scale", 1.0)

                if not joint_names:
                    continue

                joint_qpos_addrs: list[int] = []
                for j_name in joint_names:
                    if j_name in self._joint_qpos_addrs:
                        joint_qpos_addrs.append(self._joint_qpos_addrs[j_name])
                        continue

                    prefixed_joint_name = prefix + j_name
                    try:
                        joint_id = mujoco.mj_name2id(
                            model, mujoco.mjtObj.mjOBJ_JOINT, prefixed_joint_name
                        )
                        qpos_addr = model.jnt_qposadr[joint_id]
                        self._joint_qpos_addrs[j_name] = qpos_addr
                        joint_qpos_addrs.append(qpos_addr)
                        logger.info(
                            "  - DerivedState: joint '%s' -> '%s' at qpos_addr: %s",
                            j_name,
                            prefixed_joint_name,
                            qpos_addr,
                        )
                    except Exception:
                        logger.warning(
                            "  - WARNING: DerivedState joint '%s' (prefixed '%s') "
                            "not found in mjModel.",
                            j_name,
                            prefixed_joint_name,
                        )

                if not joint_qpos_addrs:
                    continue

                self._derived_state_cfg[act_name] = {
                    "joint_qpos_addrs": joint_qpos_addrs,
                    "expr": expr,
                    "scale": float(scale),
                }

    @property
    def actuator_order(self) -> list[str]:
        """执行器名称顺序（用于 RobotState / RobotAction）。"""
        return [a.name for a in self._actuators]

    def reset(self) -> None:
        """将仿真状态重置为模型初始（qpos0，零速度）。"""
        for logical_name, addr in self._qpos_addrs.items():
            self._data.qpos[addr] = self._model.qpos0[addr]
        self._data.qvel[:] = 0
        logger.debug("[MuJoCoSimulator] Reset to initial state")

    def get_state(self) -> RobotState:
        """从 MuJoCo 读取状态，返回 RobotState（弧度/毫米）。"""
        qpos = self._data.qpos
        actuator_values: dict[str, ActuatorValue] = {}

        for logical_name, addr in self._qpos_addrs.items():
            actuator = self._actuator_map.get(logical_name)
            joint = self._joint_map.get(logical_name)
            if actuator:
                unit = actuator.unit
            elif joint:
                unit = "radians" if joint.mode == "position" else "millimeters"
            else:
                logger.warning("Joint/actuator '%s' not found.", logical_name)
                continue

            qpos_value = qpos[addr]
            if unit in ("millimeters", "millimeters/s"):
                value = float(qpos_value) * 1000.0
            else:
                value = float(qpos_value)
            actuator_values[logical_name] = ActuatorValue(
                value=value,
                mode="position",
                unit=unit,
            )

        for act_name, cfg in self._derived_state_cfg.items():
            joint_qpos_addrs = cfg.get("joint_qpos_addrs", [])
            if not joint_qpos_addrs:
                continue

            expr = cfg.get("expr")
            scale = float(cfg.get("scale", 1.0))

            values = [float(qpos[addr]) for addr in joint_qpos_addrs]
            if not values:
                continue

            if expr == "diff" and len(values) >= 2:
                raw = values[0] - values[1]
            elif expr == "sum":
                raw = sum(values)
            else:
                raw = values[0]

            value = raw * scale

            actuator = self._actuator_map.get(act_name)
            if actuator:
                mode = actuator.control_mode
                unit = actuator.unit
            else:
                mode = "position"
                unit = "radians"

            actuator_values[act_name] = ActuatorValue(
                value=value,
                mode=mode,
                unit=unit,
            )

        return RobotState(actuators=actuator_values)

    def set_action(self, action: RobotAction) -> None:
        """下发 RobotAction（弧度/毫米），内部裁剪到限位后写入 ctrl。"""
        safe_action = self._get_safe_action(action)
        ctrl = self._data.ctrl

        for logical_name, act_val in safe_action.actuators.items():
            actuator = self._actuator_map.get(logical_name)
            if not actuator:
                logger.warning(
                    "Actuator '%s' not found in actuator_map. Skipping.",
                    logical_name,
                )
                continue

            if logical_name not in self._ctrl_indices:
                logger.warning(
                    "Actuator '%s' not found in ctrl_indices. Skipping.",
                    logical_name,
                )
                continue

            if actuator.control_mode not in ("position", "prismatic", "velocity"):
                logger.warning(
                    "Unsupported control mode '%s' for actuator '%s'. Skipping.",
                    actuator.control_mode,
                    logical_name,
                )
                continue

            ctrl_index = self._ctrl_indices[logical_name]
            if act_val.unit in ("millimeters", "millimeters/s"):
                ctrl[ctrl_index] = act_val.value / 1000.0
            else:
                ctrl[ctrl_index] = act_val.value

    def _get_safe_action(self, action: RobotAction) -> RobotAction:
        """按执行器限位裁剪 action。"""
        safe_actuators = {}
        for name, act_val in action.actuators.items():
            actuator = self._actuator_map.get(name)
            if not actuator:
                continue
            clipped_val = max(
                min(act_val.value, actuator.max_value), actuator.min_value
            )
            safe_actuators[name] = ActuatorValue(
                value=clipped_val,
                mode=act_val.mode,
                unit=act_val.unit,
            )
        return RobotAction(actuators=safe_actuators)
