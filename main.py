#!/usr/bin/env python3
"""MuJoCo dora-rs 节点：接收 tick 驱动仿真，接收 Action 控制 ctrl，输出 ProprioState 和相机图像。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from dora import Node
from forge_msgs import (
    Action,
    CompressedImage,
    Image,
)
from forge_robots_core import BaseActuator, BaseJoint
from forge_simulators_mujoco import MuJoCoSimulator

from config import (
    MuJoCoNodeConfig,
    load_config,
)
from handlers import build_proprio_state, build_robot_action, sim_state_to_robot_state


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MuJoCo dora 节点")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML 配置文件路径",
    )
    return parser.parse_args()


def _build_joints_and_actuators(robot_config) -> tuple[list[BaseJoint], list[BaseActuator]]:
    """从 RobotConfig 构建 BaseJoint 和 BaseActuator 列表。"""
    joints = [
        BaseJoint(name=name, mode="position")
        for name in robot_config.joint_to_actuator.keys()
    ]
    actuators = [
        BaseActuator(
            name=name,
            id=i,
            control_mode="position",
            min_value=-3.14,
            max_value=3.14,
            unit="radians",
        )
        for i, name in enumerate(robot_config.actuator_order)
    ]
    return joints, actuators


def _create_renderer(model: mujoco.MjModel, height: int = 480, width: int = 640):
    """创建 MuJoCo 离屏渲染器。"""
    return mujoco.Renderer(model, height, width)


def _build_image_payload(
    pixels: np.ndarray,
    config: MuJoCoNodeConfig,
    cv2_module: object | None,
):
    """根据配置将渲染图像编码为 raw Image 或 CompressedImage。"""
    height, width = int(pixels.shape[0]), int(pixels.shape[1])
    if config.image_format == "raw":
        return Image(
            width=width,
            height=height,
            channels=3,
            encoding="rgb8",
            data=pixels.tobytes(),
        ).to_arrow()

    if cv2_module is None:
        raise RuntimeError("image_format 为 jpeg/png 时需要安装 opencv-python")

    cv2 = cv2_module
    # OpenCV 编码器默认按 BGR 解释 3 通道输入，这里将 RGB 转为 BGR 再压缩。
    bgr = pixels[..., ::-1]
    if config.image_format == "jpeg":
        ok, encoded = cv2.imencode(
            ".jpg",
            bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(config.image_jpeg_quality)],
        )
        if not ok:
            raise RuntimeError("JPEG 编码失败")
        return CompressedImage(
            width=width,
            height=height,
            format="jpeg",
            data=encoded.tobytes(),
        ).to_arrow()

    ok, encoded = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("PNG 编码失败")
    return CompressedImage(
        width=width,
        height=height,
        format="png",
        data=encoded.tobytes(),
    ).to_arrow()


def main() -> int:
    args = _parse_args()
    config = load_config(config_path=args.config)

    # 解析 model_path（配置文件加载时已相对配置目录解析，否则相对 cwd）
    model_path = Path(config.model_path)
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path
    if not model_path.exists():
        print(f"[mujoco] 模型文件不存在: {model_path}", file=sys.stderr)
        return 1

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    # 为每个 robot 创建 MuJoCoSimulator
    simulators: dict[str, MuJoCoSimulator] = {}
    for robot in config.robots:
        joints, actuators = _build_joints_and_actuators(robot)
        prefix = robot.prefix
        sim = MuJoCoSimulator(
            model=model,
            data=data,
            joints=joints,
            actuators=actuators,
            prefix=prefix,
        )
        simulators[robot.id] = sim

    # 相机渲染器（如有配置）
    renderer = None
    if config.image_forward:
        try:
            renderer = _create_renderer(model)
        except Exception as e:
            print(f"[mujoco] 创建渲染器失败，将跳过相机输出: {e}", file=sys.stderr)

    cv2_module = None
    if config.image_format in {"jpeg", "png"}:
        try:
            import cv2  # type: ignore

            cv2_module = cv2
        except Exception as e:
            print(
                f"[mujoco] image_format={config.image_format} 需要 opencv-python: {e}",
                file=sys.stderr,
            )
            return 1

    # 相机节流：记录上次输出时间
    camera_last_time: dict[str, float] = {name: 0.0 for name in config.camera_names}

    node = Node()

    for event in node:
        match event["type"]:
            case "INPUT":
                input_id = event["id"]

                match input_id:
                    case "tick":
                        # 仿真步进
                        mujoco.mj_step(model, data)

                        # 读取状态并输出 ProprioState
                        robot_states: dict[str, object] = {}
                        for robot in config.robots:
                            sim = simulators[robot.id]
                            sim_state = sim.get_state()
                            robot_states[robot.id] = sim_state_to_robot_state(
                                sim_state, robot
                            )
                        proprio_state = build_proprio_state(robot_states, config)
                        node.send_output("proprio_state", proprio_state.to_arrow(config.joint_order))

                        # 相机输出（按 hz 节流）
                        if renderer is not None:
                            now = time.time()
                            for cam_name, cam_config in config.image_forward.items():
                                output_id = cam_config.output_id
                                if cam_config.hz is not None:
                                    interval = 1.0 / cam_config.hz
                                    if now - camera_last_time[cam_name] < interval:
                                        continue
                                    camera_last_time[cam_name] = now

                                try:
                                    renderer.update_scene(data, camera=cam_name)
                                    pixels = renderer.render()
                                    payload = _build_image_payload(
                                        pixels, config, cv2_module
                                    )
                                    node.send_output(output_id, payload)
                                except Exception as e:
                                    print(
                                        f"[mujoco] 渲染相机 {cam_name} 失败: {e}",
                                        file=sys.stderr,
                                    )

                    case "action":
                        action = Action.from_arrow(event["value"], config.joint_order)
                        for robot in config.robots:
                            robot_action = build_robot_action(action, robot)
                            sim = simulators[robot.id]
                            sim.set_action(robot_action)

            case "STOP":
                break

            case "ERROR":
                print(
                    f"[mujoco] error: {event.get('error', 'unknown')}",
                    file=sys.stderr,
                )
                break

            case _:
                pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
