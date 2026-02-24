"""MuJoCo 节点配置加载与校验。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass
class RobotConfig:
    """单个机器人的 joint -> actuator 映射配置。"""

    id: str
    actuator_order: list[str]
    joint_to_actuator: dict[str, str]
    prefix: str = ""  # MuJoCo 模型中的名称前缀，如 item_1/；多机时通常由 id 推导

    def __post_init__(self) -> None:
        if not self.actuator_order:
            raise ValueError(f"Robot '{self.id}': actuator_order 不能为空")
        for joint, act in self.joint_to_actuator.items():
            if act not in self.actuator_order:
                raise ValueError(
                    f"Robot '{self.id}': joint_to_actuator 中的 actuator '{act}' "
                    f"不在 actuator_order 中"
                )


@dataclass
class CameraConfig:
    """单个相机的输出配置。"""

    output_id: str
    hz: float | None = None  # None 表示随 tick 输出
    width: int | None = None
    height: int | None = None


def _parse_image_forward(value: Any) -> CameraConfig:
    """解析 image_forward 单个条目，支持简写或完整形式。"""
    if isinstance(value, str):
        return CameraConfig(output_id=value, hz=None)
    if isinstance(value, dict):
        return CameraConfig(
            output_id=value.get("output", value.get("output_id", "")),
            hz=value.get("hz"),
            width=value.get("width"),
            height=value.get("height"),
        )
    raise ValueError(f"image_forward 条目格式无效: {value}")


@dataclass
class MuJoCoNodeConfig:
    """MuJoCo 节点完整配置。"""

    model_path: str
    joint_order: list[str]
    robots: list[RobotConfig]
    image_forward: dict[str, CameraConfig] = field(default_factory=dict)
    image_format: Literal["raw", "jpeg", "png"] = "raw"
    image_jpeg_quality: int = 90

    @property
    def camera_names(self) -> set[str]:
        """需输出的相机名称集合。"""
        return set(self.image_forward.keys())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MuJoCoNodeConfig":
        """从字典解析配置。"""
        model_path = data.get("model_path", "")
        if not model_path:
            raise ValueError("model_path 不能为空")

        joint_order = data.get("joint_order", [])
        if not joint_order:
            raise ValueError("joint_order 不能为空")

        robots_raw = data.get("robots", [])
        robots: list[RobotConfig] = []
        for r in robots_raw:
            robots.append(
                RobotConfig(
                    id=r["id"],
                    actuator_order=r["actuator_order"],
                    joint_to_actuator=r.get("joint_to_actuator", {}),
                    prefix=r.get("prefix", ""),
                )
            )

        image_forward_raw = data.get("image_forward", {})
        if not isinstance(image_forward_raw, dict):
            image_forward_raw = {}

        image_forward: dict[str, CameraConfig] = {}
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

        return cls(
            model_path=model_path,
            joint_order=joint_order,
            robots=robots,
            image_forward=image_forward,
            image_format=image_format,
            image_jpeg_quality=image_jpeg_quality,
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
