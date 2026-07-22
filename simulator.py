"""MuJoCo simulation logic for JointCommand/JointState."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import mujoco

from forge_msgs import JointCommand, JointState
from forge_common import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class JointSpec:
    """关节规格，仅 name 与 mode（用于单位推断）。"""

    name: str
    mode: str = "position"
    #: 若与 name 不同：MuJoCo 模型中的关节名后缀（prefix 后路径），如 robot_left/joint1
    mj_name: str | None = None


@dataclass(frozen=True)
class ActuatorSpec:
    """执行器规格，用于限位与单位。"""

    name: str
    id: int
    control_mode: str
    min_value: float
    max_value: float
    unit: str = "radians"
    #: 若与 name 不同：MuJoCo 模型中的 actuator 名后缀（prefix 后路径）
    mj_name: str | None = None


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
            mj_suffix = joint.mj_name if joint.mj_name is not None else joint.name
            prefixed_name = prefix + mj_suffix
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, prefixed_name
            )
            if joint_id >= 0:
                qpos_addr = model.jnt_qposadr[joint_id]
                self._qpos_addrs[logical_name] = qpos_addr
                self._joint_qpos_addrs[logical_name] = qpos_addr
                logger.info(
                    "  - Mapped Joint '%s' -> '%s' at qpos_addr: %s",
                    logical_name,
                    prefixed_name,
                    qpos_addr,
                )
            else:
                if derived_state_cfg and logical_name in derived_state_cfg:
                    logger.info(
                        "  - Joint '%s' is derived, skipping direct qpos mapping.",
                        logical_name,
                    )
                else:
                    logger.warning(
                        "  - WARNING: Joint '%s' not found in mjModel.",
                        prefixed_name,
                    )

        for actuator in self._actuators:
            logical_name = actuator.name
            mj_suffix = (
                actuator.mj_name if actuator.mj_name is not None else actuator.name
            )
            prefixed_name = prefix + mj_suffix
            act_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefixed_name
            )
            if act_id >= 0:
                self._ctrl_indices[logical_name] = act_id
                runtime_actuator = actuator
                if bool(model.actuator_ctrllimited[act_id]):
                    min_value = float(model.actuator_ctrlrange[act_id][0])
                    max_value = float(model.actuator_ctrlrange[act_id][1])
                    if actuator.unit in ("millimeters", "millimeters/s"):
                        min_value *= 1000.0
                        max_value *= 1000.0
                    runtime_actuator = replace(
                        actuator,
                        min_value=min_value,
                        max_value=max_value,
                    )
                    self._actuator_map[logical_name] = runtime_actuator
                logger.info(
                    "  - Mapped Actuator '%s' -> '%s' with ctrl_idx: %s, range: [%s, %s] %s",
                    logical_name,
                    prefixed_name,
                    act_id,
                    runtime_actuator.min_value,
                    runtime_actuator.max_value,
                    runtime_actuator.unit,
                )
            else:
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
                    joint_id = mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_JOINT, prefixed_joint_name
                    )
                    if joint_id >= 0:
                        qpos_addr = model.jnt_qposadr[joint_id]
                        self._joint_qpos_addrs[j_name] = qpos_addr
                        joint_qpos_addrs.append(qpos_addr)
                        logger.info(
                            "  - DerivedState: joint '%s' -> '%s' at qpos_addr: %s",
                            j_name,
                            prefixed_joint_name,
                            qpos_addr,
                        )
                    else:
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
        """执行器名称顺序（用于 JointState / JointCommand）。"""
        return [a.name for a in self._actuators]

    def reset(self) -> None:
        """将本 robot 状态重置为模型 qpos0，并同步 actuator 目标。"""
        qpos_addrs = set(self._qpos_addrs.values())
        for cfg in self._derived_state_cfg.values():
            qpos_addrs.update(cfg.get("joint_qpos_addrs", []))
        for addr in qpos_addrs:
            self._data.qpos[addr] = self._model.qpos0[addr]
        self._data.qvel[:] = 0
        mujoco.mj_forward(self._model, self._data)
        self.sync_actuator_targets()
        logger.debug("[MuJoCoSimulator] Reset to model initial state")

    def apply_initial_positions(self, positions: dict[str, float]) -> None:
        """应用 actuator-keyed 初始位置。

        ``positions`` 使用消息侧单位（弧度、米或毫米）。调用方应先执行
        ``mj_resetData``；未显式配置的关节保留模型 ``qpos0``。写入全部 robot
        后须执行 ``mj_forward`` 和 :meth:`sync_actuator_targets`。
        """
        mapped_names = set(self._qpos_addrs) | set(self._derived_state_cfg)
        unknown = sorted(set(positions) - mapped_names)
        if unknown:
            raise ValueError(
                f"初始位置包含未映射到 MuJoCo joint 的 actuator: {unknown}"
            )

        for logical_name, value in positions.items():
            actuator = self._actuator_map.get(logical_name)
            if actuator is None:
                raise ValueError(f"初始位置 actuator '{logical_name}' 未配置")

            qpos_addr = self._qpos_addrs.get(logical_name)
            if qpos_addr is not None:
                qpos_value = value
                if actuator.unit == "millimeters":
                    qpos_value /= 1000.0
                self._data.qpos[qpos_addr] = qpos_value
                continue

            cfg = self._derived_state_cfg[logical_name]
            qpos_addrs = cfg["joint_qpos_addrs"]
            scale = float(cfg.get("scale", 1.0))
            if scale == 0.0:
                raise ValueError(
                    f"初始位置 {logical_name} 无法应用：state_scale 不能为 0"
                )
            target_raw = value / scale
            current = [float(self._data.qpos[addr]) for addr in qpos_addrs]
            expr = cfg.get("expr")
            if expr == "diff" and len(qpos_addrs) >= 2:
                delta = (target_raw - (current[0] - current[1])) / 2.0
                self._data.qpos[qpos_addrs[0]] += delta
                self._data.qpos[qpos_addrs[1]] -= delta
            elif expr == "sum":
                delta = (target_raw - sum(current)) / len(qpos_addrs)
                for addr in qpos_addrs:
                    self._data.qpos[addr] += delta
            else:
                self._data.qpos[qpos_addrs[0]] = target_raw

    def sync_actuator_targets(self) -> None:
        """根据 forward 后的 actuator length 同步 position/velocity ctrl。"""
        for logical_name, actuator in self._actuator_map.items():
            ctrl_index = self._ctrl_indices.get(logical_name)
            if ctrl_index is None:
                continue
            if actuator.control_mode in ("position", "prismatic"):
                target = float(self._data.actuator_length[ctrl_index])
                if bool(self._model.actuator_ctrllimited[ctrl_index]):
                    min_ctrl = float(self._model.actuator_ctrlrange[ctrl_index][0])
                    max_ctrl = float(self._model.actuator_ctrlrange[ctrl_index][1])
                    if target < min_ctrl or target > max_ctrl:
                        raise ValueError(
                            f"初始位置 {logical_name} 对应 actuator target={target} "
                            f"超出模型 ctrlrange [{min_ctrl}, {max_ctrl}]"
                        )
                self._data.ctrl[ctrl_index] = target
            elif actuator.control_mode == "velocity":
                self._data.ctrl[ctrl_index] = 0.0

    def get_state(self) -> JointState:
        """从 MuJoCo 读取状态，返回 JointState（弧度/毫米）。"""
        qpos = self._data.qpos
        names: list[str] = []
        position: list[float] = []

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
            # 状态始终报当前位置（qpos）；速度型关节的 unit 为 radians/s 等，报告时用位置单位
            if unit in ("millimeters", "millimeters/s"):
                report_unit = "millimeters"
                value = float(qpos_value) * 1000.0
            elif unit in ("meters", "meters/s"):
                report_unit = "meters"
                value = float(qpos_value)
            else:
                report_unit = "radians"
                value = float(qpos_value)
            _ = report_unit
            names.append(logical_name)
            position.append(value)

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

            _ = (mode, unit)
            names.append(act_name)
            position.append(value)

        return JointState(name=names, position=position, velocity=[], effort=[])

    def set_command(self, command: JointCommand) -> None:
        """下发 JointCommand（弧度/毫米），内部裁剪到限位后写入 ctrl。"""
        safe_command = self._get_safe_command(command)
        ctrl = self._data.ctrl

        position = dict(zip(safe_command.name, safe_command.position, strict=True)) if safe_command.position else {}
        velocity = dict(zip(safe_command.name, safe_command.velocity, strict=True)) if safe_command.velocity else {}
        effort = dict(zip(safe_command.name, safe_command.effort, strict=True)) if safe_command.effort else {}

        for logical_name in safe_command.name:
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
            if actuator.control_mode == "velocity":
                value = velocity.get(logical_name, position.get(logical_name, 0.0))
            elif actuator.control_mode == "effort":
                value = effort.get(logical_name, position.get(logical_name, 0.0))
            else:
                value = position.get(logical_name, 0.0)

            # 长度单位统一为米写入 ctrl；角度/角速度（radians、radians/s）原样写入
            if actuator.unit in ("millimeters", "millimeters/s"):
                ctrl[ctrl_index] = value / 1000.0
            else:
                # position(rad)、velocity(rad/s) 等与 MuJoCo 一致，直接赋值（如速度 1 -> ctrl=1）
                ctrl[ctrl_index] = value

    def _get_safe_command(self, command: JointCommand) -> JointCommand:
        """按执行器限位裁剪 command。"""
        source = {
            "position": dict(zip(command.name, command.position, strict=True)) if command.position else {},
            "velocity": dict(zip(command.name, command.velocity, strict=True)) if command.velocity else {},
            "effort": dict(zip(command.name, command.effort, strict=True)) if command.effort else {},
            "kp": dict(zip(command.name, command.kp, strict=True)) if command.kp else {},
            "kd": dict(zip(command.name, command.kd, strict=True)) if command.kd else {},
        }
        names: list[str] = []
        position: list[float] = []
        velocity: list[float] = []
        effort: list[float] = []
        kp: list[float] = []
        kd: list[float] = []

        for name in command.name:
            actuator = self._actuator_map.get(name)
            if not actuator:
                continue
            names.append(name)
            for field, target in (
                ("position", position),
                ("velocity", velocity),
                ("effort", effort),
                ("kp", kp),
                ("kd", kd),
            ):
                value = source[field].get(name, 0.0)
                if field in {"position", "velocity", "effort"}:
                    value = max(min(value, actuator.max_value), actuator.min_value)
                target.append(value)

        return JointCommand(
            name=names,
            position=position,
            velocity=velocity,
            effort=effort,
            kp=kp,
            kd=kd,
        )
