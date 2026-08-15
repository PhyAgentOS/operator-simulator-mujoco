"""MuJoCo 节点配置加载与校验。"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass(init=False)
class JointConfig:
    """Minimal joint configuration used by the MuJoCo runtime node."""

    name: str
    mode: str
    unit: str

    def __init__(
        self,
        name: str,
        mode: str = "position",
        unit: str = "radians",
        **_: Any,
    ) -> None:
        self.name = name
        self.mode = mode
        self.unit = unit


@dataclass(frozen=True)
class KinematicPlanarBaseConfig:
    """由 LocomotionCommand 驱动的 MuJoCo mocap 平面底盘。"""

    body: str
    command_timeout_sec: float | None = None


@dataclass
class RobotConfig:
    """单个机器人的 joint -> actuator 映射配置。"""

    id: str
    joints: list[str] = field(default_factory=list)
    actuator_order: list[str] = field(default_factory=list)
    joint_to_actuator: dict[str, str] = field(default_factory=dict)
    prefix: str = ""  # MuJoCo 模型中的名称前缀，如 item_1/；多机时通常由 id 推导

    def __post_init__(self) -> None:
        if self.joints:
            if not self.actuator_order:
                self.actuator_order = list(self.joints)
            if not self.joint_to_actuator:
                self.joint_to_actuator = {j: j for j in self.joints}

        if not self.actuator_order:
            raise ValueError(f"Robot '{self.id}': actuator_order 或 joints 不能为空")
        for joint, act in self.joint_to_actuator.items():
            if act not in self.actuator_order:
                raise ValueError(
                    f"Robot '{self.id}': joint_to_actuator 中的 actuator '{act}' "
                    f"不在 actuator_order 中"
                )


@dataclass
class InitialStateConfig:
    """场景初始状态；后续可扩展物体位姿、速度或 MuJoCo keyframe。"""

    joint_positions: dict[str, float] = field(default_factory=dict)


@dataclass
class DerivedActuatorStateConfig:
    """某个逻辑关节/执行器的派生状态配置。

    用于描述：该关节/执行器的状态如何由一个或多个 MuJoCo joint 的 qpos 合成。
    """

    joint_names: list[str] = field(default_factory=list)
    state_expr: Literal["diff", "sum"] | None = None
    state_scale: float = 1.0


@dataclass
class CameraConfig:
    """单个相机的输出配置。"""

    output_id: str
    image_type: Literal["image", "depth"] = "image"
    fps: float | None = None  # None 表示随 tick 输出
    width: int | None = None
    height: int | None = None


def _parse_image_type(value: Any) -> Literal["image", "depth"]:
    """解析 image_forward[].type，默认/彩色图统一为 image。"""
    if value is None:
        return "image"

    image_type = str(value).strip().lower()
    if image_type in {"image", "rgb", "color"}:
        return "image"
    if image_type == "depth":
        return "depth"
    raise ValueError(f"image_forward type 必须是 image/depth，当前: {value}")


def _parse_image_forward_item(value: Any) -> CameraConfig:
    """解析 image_forward 的单个输出条目。"""
    if isinstance(value, str):
        return CameraConfig(output_id=value, fps=None)
    if isinstance(value, dict):
        fps = value.get("fps", value.get("hz"))  # 兼容 hz
        image_type = _parse_image_type(value.get("type"))
        output_id = str(value.get("output", value.get("output_id", ""))).strip()
        if not output_id:
            raise ValueError(f"image_forward 条目缺少 output/output_id: {value}")
        return CameraConfig(
            output_id=output_id,
            image_type=image_type,
            fps=fps,
            width=value.get("width"),
            height=value.get("height"),
        )
    raise ValueError(f"image_forward 条目格式无效: {value}")


def _parse_image_forward(value: Any) -> list[CameraConfig]:
    """解析 image_forward 单个相机配置，支持简写、完整形式或输出列表。"""
    if isinstance(value, list):
        return [_parse_image_forward_item(item) for item in value]
    return [_parse_image_forward_item(value)]


@dataclass
class MuJoCoNodeConfig:
    """MuJoCo 节点完整配置。"""

    model_path: str
    joints: list[JointConfig]
    robots: list[RobotConfig]
    # 场景初始状态；当前支持关节位置，单位与对应 joints[].unit 一致。
    initial_state: InitialStateConfig = field(default_factory=InitialStateConfig)
    # name -> DerivedActuatorStateConfig，用于如 gripper 由 joint7/joint8 合成的场景
    derived_state_cfg: dict[str, DerivedActuatorStateConfig] = field(
        default_factory=dict
    )
    image_forward: dict[str, list[CameraConfig]] = field(default_factory=dict)
    image_format: Literal["raw", "jpeg", "png"] = "raw"
    image_jpeg_quality: int = 90
    control_hz: float = 50.0  # 控制周期频率，与 dataflow 中 mujoco 的 tick 间隔一致（如 50 → 20ms）
    # 可选：输出任意对象的平面位姿（PoseSet），按 {id: body_name} 配置
    # - id: 下游识别用的唯一标识（推荐用 robot_id，但也可用自定义 id）
    # - body_name: MuJoCo XML 中 body 的 name；若 id 恰好等于某个 robot_id，则允许使用该 robot 的 prefix 拼接
    pose_targets: dict[str, str] = field(default_factory=dict)
    # 可选：通过 mocap body 消费 LocomotionCommand，按机体系速度做平面运动。
    kinematic_planar_base: KinematicPlanarBaseConfig | None = None

    @property
    def control_period_sec(self) -> float:
        """控制周期（秒），由 control_hz 推导。"""
        return 1.0 / self.control_hz

    @property
    def camera_outputs(self) -> list[tuple[str, CameraConfig]]:
        """展开后的相机输出列表，允许同一个 MuJoCo camera 输出多种图像。"""
        return [
            (cam_name, cam_cfg)
            for cam_name, cam_configs in self.image_forward.items()
            for cam_cfg in cam_configs
        ]

    @property
    def joint_order(self) -> list[str]:
        """关节名称顺序，用于 JointCommand/JointState 向量排列。"""
        return [j.name for j in self.joints]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MuJoCoNodeConfig":
        """从字典解析配置。"""
        model_path = data.get("model_path", "")
        if not model_path:
            raise ValueError("model_path 不能为空")

        joints_raw = data.get("joints", [])

        derived_state_cfg: dict[str, DerivedActuatorStateConfig] = {}
        joints: list[JointConfig] = []
        for j in joints_raw:
            if not isinstance(j, dict):
                continue

            # 提取派生状态相关字段（若存在）
            state_from_joints = j.get("state_from_joints")
            state_expr = j.get("state_expr")
            state_scale = j.get("state_scale", 1.0)

            if state_from_joints is not None:
                # 规范化 joint 名列表
                if not isinstance(state_from_joints, list):
                    raise ValueError(
                        f"state_from_joints 必须是字符串列表，当前: {state_from_joints}"
                    )
                joint_names = [str(name) for name in state_from_joints]

                expr: Literal["diff", "sum"] | None
                if state_expr is None:
                    expr = None
                else:
                    if state_expr not in ("diff", "sum"):
                        raise ValueError(
                            f"state_expr 必须是 'diff' 或 'sum'，当前: {state_expr}"
                        )
                    expr = state_expr  # type: ignore[assignment]

                try:
                    scale = float(state_scale)
                except Exception as exc:  # noqa: BLE001
                    raise ValueError(
                        f"state_scale 必须是数字，当前: {state_scale}"
                    ) from exc

                name = j.get("name")
                if not name:
                    raise ValueError(
                        "带有 state_from_joints 的关节/执行器必须设置 name 字段"
                    )
                derived_state_cfg[str(name)] = DerivedActuatorStateConfig(
                    joint_names=joint_names,
                    state_expr=expr,
                    state_scale=scale,
                )

            # 传给 JointConfig 的字段需要剥离派生配置，避免未知字段报错
            cleaned = {
                key: value
                for key, value in j.items()
                if key not in {"state_from_joints", "state_expr", "state_scale"}
            }
            joints.append(JointConfig(**cleaned))

        robots_raw = data.get("robots", [])
        robots: list[RobotConfig] = []
        for r in robots_raw:
            robots.append(
                RobotConfig(
                    id=r["id"],
                    joints=r.get("joints", []),
                    actuator_order=r.get("actuator_order", []),
                    joint_to_actuator=r.get("joint_to_actuator", {}),
                    prefix=r.get("prefix", ""),
                )
            )

        # 如果全局 joints 为空，从 robots 推导
        if not joints:
            all_joint_names = []
            for r in robots:
                for j_name in r.joint_to_actuator.keys():
                    if j_name not in all_joint_names:
                        all_joint_names.append(j_name)
            joints = [JointConfig(name=n) for n in all_joint_names]

        if "initial_positions" in data or "positions" in data:
            raise ValueError(
                "initial_positions/positions 已移除，请改用 "
                "initial_state.joint_positions"
            )

        initial_state_raw = data.get("initial_state", {})
        if not isinstance(initial_state_raw, dict):
            raise ValueError("initial_state 必须是字典")

        unknown_state_fields = sorted(
            str(field_name)
            for field_name in initial_state_raw
            if field_name != "joint_positions"
        )
        if unknown_state_fields:
            raise ValueError(
                "initial_state 包含尚未支持的字段: "
                f"{unknown_state_fields}，可用值: ['joint_positions']"
            )

        joint_positions_raw = initial_state_raw.get("joint_positions", {})
        if not isinstance(joint_positions_raw, dict):
            raise ValueError(
                "initial_state.joint_positions 必须是 "
                "{joint_name: position} 的字典"
            )

        known_joint_names = {joint.name for joint in joints}
        joint_positions: dict[str, float] = {}
        for raw_name, raw_value in joint_positions_raw.items():
            name = str(raw_name).strip()
            value_path = f"initial_state.joint_positions.{name}"
            if name not in known_joint_names:
                raise ValueError(
                    f"initial_state.joint_positions 包含未知关节 '{name}'，"
                    f"可用值: {sorted(known_joint_names)}"
                )
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise ValueError(f"{value_path} 必须是数字，当前: {raw_value!r}")
            joint = next(joint for joint in joints if joint.name == name)
            if joint.mode not in ("position", "prismatic"):
                raise ValueError(
                    f"{value_path} 仅支持 position/prismatic 关节，"
                    f"当前 mode: {joint.mode}"
                )
            derived = derived_state_cfg.get(name)
            if derived is not None and (
                not math.isfinite(derived.state_scale) or derived.state_scale == 0.0
            ):
                raise ValueError(f"{value_path} 要求 state_scale 为有限非零数字")
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError(f"{value_path} 必须是有限数字，当前: {raw_value!r}")
            joint_positions[name] = value

        initial_state = InitialStateConfig(joint_positions=joint_positions)

        image_forward_raw = data.get("image_forward", {})
        if not isinstance(image_forward_raw, dict):
            image_forward_raw = {}

        image_forward: dict[str, list[CameraConfig]] = {}
        for cam_name, val in image_forward_raw.items():
            image_forward[cam_name] = _parse_image_forward(val)

        image_format = data.get("image_format", "raw")
        if image_format not in {"raw", "jpeg", "png"}:
            raise ValueError(
                f"image_format 必须是 raw/jpeg/png 之一，当前: {image_format}"
            )
        image_jpeg_quality = int(data.get("image_jpeg_quality", 90))
        if image_jpeg_quality < 1 or image_jpeg_quality > 100:
            raise ValueError("image_jpeg_quality 必须在 [1, 100] 范围内")

        control_hz = float(data.get("control_hz", 50.0))
        if control_hz <= 0:
            raise ValueError("control_hz 必须大于 0")

        kinematic_planar_base_raw = data.get("kinematic_planar_base")
        kinematic_planar_base: KinematicPlanarBaseConfig | None = None
        if kinematic_planar_base_raw is not None:
            if not isinstance(kinematic_planar_base_raw, dict):
                raise ValueError("kinematic_planar_base 必须是字典")

            body_raw = kinematic_planar_base_raw.get("body")
            if not isinstance(body_raw, str) or not body_raw.strip():
                raise ValueError("kinematic_planar_base.body 必须是非空字符串")

            timeout_raw = kinematic_planar_base_raw.get("command_timeout_sec")
            timeout: float | None = None
            if timeout_raw is not None:
                if isinstance(timeout_raw, bool) or not isinstance(
                    timeout_raw, (int, float)
                ):
                    raise ValueError(
                        "kinematic_planar_base.command_timeout_sec 必须是数字"
                    )
                timeout = float(timeout_raw)
                if not math.isfinite(timeout) or timeout <= 0.0:
                    raise ValueError(
                        "kinematic_planar_base.command_timeout_sec 必须是有限正数"
                    )

            kinematic_planar_base = KinematicPlanarBaseConfig(
                body=body_raw.strip(),
                command_timeout_sec=timeout,
            )

        pose_targets_raw = data.get("pose_targets", {})
        pose_targets: dict[str, str] = {}
        if pose_targets_raw:
            if not isinstance(pose_targets_raw, dict):
                raise ValueError("pose_targets 必须是 {id: body_name} 的字典")
            for k, v in pose_targets_raw.items():
                rk = str(k).strip()
                rv = str(v).strip()
                if not rk or not rv:
                    continue
                pose_targets[rk] = rv
        # pose_set 输出口固定为 "pose_set"

        return cls(
            model_path=model_path,
            joints=joints,
            robots=robots,
            initial_state=initial_state,
            derived_state_cfg=derived_state_cfg,
            image_forward=image_forward,
            image_format=image_format,
            image_jpeg_quality=image_jpeg_quality,
            control_hz=control_hz,
            pose_targets=pose_targets,
            kinematic_planar_base=kinematic_planar_base,
        )

    @classmethod
    def from_yaml_path(cls, path: str | Path) -> "MuJoCoNodeConfig":
        """从 YAML 文件加载配置。"""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"配置文件不存在: {p}")
        with open(p, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not data:
            raise ValueError(f"配置文件为空: {p}")
        return cls.from_dict(data)


def load_config(
    config_path: str | Path | None = None,
    dataflow_config: dict[str, Any] | None = None,
    config_base_dir: Path | None = None,
) -> MuJoCoNodeConfig:
    """
    加载配置，支持多种来源（优先级从高到低）：
    1. dataflow_config 中的 config_path 或内联 config
    2. config_path 参数
    3. 环境变量 MUJOCO_NODE_CONFIG

    Args:
        config_path: 可选，YAML 配置文件路径
        dataflow_config: 可选，dora 节点 config 字段

    Returns:
        MuJoCoNodeConfig
    """
    data: dict[str, Any] | None = None

    if dataflow_config:
        if "config" in dataflow_config:
            data = dataflow_config["config"]
        elif "config_path" in dataflow_config:
            p = dataflow_config["config_path"]
            cfg = MuJoCoNodeConfig.from_yaml_path(p)
            base = Path(p).parent
            model_p = Path(cfg.model_path)
            if not model_p.is_absolute():
                cfg.model_path = str((base / model_p).resolve())
            return cfg

    path = config_path
    if path is None and data is None:
        path = os.environ.get("MUJOCO_NODE_CONFIG")

    if path:
        cfg = MuJoCoNodeConfig.from_yaml_path(path)
        base = config_base_dir if config_base_dir is not None else Path(path).parent
        model_p = Path(cfg.model_path)
        if not model_p.is_absolute():
            cfg.model_path = str((base / model_p).resolve())
        return cfg

    if data:
        return MuJoCoNodeConfig.from_dict(data)

    raise ValueError(
        "未找到配置。请设置 MUJOCO_NODE_CONFIG 环境变量、"
        "或通过 --config 指定配置文件、或使用 dataflow config。"
    )
