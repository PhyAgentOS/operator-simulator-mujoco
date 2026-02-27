#!/usr/bin/env python3
"""MuJoCo dora-rs 节点：接收 tick 驱动仿真，接收 Action 控制 ctrl，输出 ProprioState 和相机图像。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# 渲染后端：须在 import mujoco 之前设置。默认 egl 便于无显示器/SSH/服务器；
# 若需 GLFW 窗口可设 MUJOCO_GL=glfw；纯软件渲染可设 MUJOCO_GL=osmesa（需 libOSMesa）。
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import mujoco
import numpy as np
from dora import Node
from forge_msgs import Action, Image, RobotState
from forge_robots_core import BaseActuator, BaseJoint
from forge_simulators_mujoco import MuJoCoSimulator

from config import (
    MuJoCoNodeConfig,
    load_config,
)
from handlers import build_proprio_state, build_robot_action, sim_state_to_robot_state, robot_state_to_robot_action

from forge_common import get_logger

logger = get_logger(__name__)


SIM_STATUS_CODE = {
    "UNINITIALIZED": 0,
    "INITIALIZED": 1,
    "RUNNING": 2,
    "PAUSED": 3,
    "STOPPED": 4,
    "RESETTING": 5,
}


def _patch_mujoco_renderer_close() -> None:
    """Renderer.__init__ 在 make_current() 失败时不会设置 _mjr_context，析构时 close() 会 AttributeError。"""
    def _safe_close(self) -> None:
        try:
            if getattr(self, "_gl_context", None):
                self._gl_context.free()
            if getattr(self, "_mjr_context", None):
                self._mjr_context.free()
        except Exception:
            pass
        try:
            object.__setattr__(self, "_gl_context", None)
            object.__setattr__(self, "_mjr_context", None)
        except Exception:
            pass

    mujoco.Renderer.close = _safe_close


_patch_mujoco_renderer_close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MuJoCo dora 节点")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML 配置文件路径",
    )
    return parser.parse_args()


def _actuator_limits(unit: str) -> tuple[float, float]:
    """按单位返回合理的 min/max（用于限位裁剪）。"""
    if unit in ("millimeters", "millimeters/s"):
        return 0.0, 50.0
    if unit in ("meters", "meters/s"):
        return 0.0, 0.05
    # radians 等
    return -3.14, 3.14


def _build_joints_and_actuators(
    node_config: MuJoCoNodeConfig,
    robot_config,
) -> tuple[list[BaseJoint], list[BaseActuator]]:
    """从节点配置的 JointConfig 与 RobotConfig 构建 BaseJoint 和 BaseActuator 列表。"""
    joint_specs = {j.name: j for j in node_config.joints}
    actuator_to_joint = {
        act: joint for joint, act in robot_config.joint_to_actuator.items()
    }

    joints = []
    for name in robot_config.joint_to_actuator.keys():
        spec = joint_specs.get(name)
        mode = spec.mode if spec else "position"
        joints.append(BaseJoint(name=name, mode=mode))

    actuators = []
    for i, actuator_name in enumerate(robot_config.actuator_order):
        joint_name = actuator_to_joint.get(actuator_name)
        spec = joint_specs.get(joint_name) if joint_name else None
        unit = spec.unit if spec else "radians"
        mode = spec.mode if spec else "position"
        min_val, max_val = _actuator_limits(unit)
        actuators.append(
            BaseActuator(
                name=actuator_name,
                id=i,
                control_mode=mode,
                min_value=min_val,
                max_value=max_val,
                unit=unit,
            )
        )
    return joints, actuators


def _create_renderer(model: mujoco.MjModel, height: int = 480, width: int = 640):
    """创建 MuJoCo 离屏渲染器。"""
    return mujoco.Renderer(model, height, width)


def _build_image_payload(
    pixels: np.ndarray,
    config: MuJoCoNodeConfig,
    cv2_module: object | None,
):
    """根据配置将渲染图像编码为 Image（raw 或 jpeg/png 压缩）。"""
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
        return Image(
            width=width,
            height=height,
            channels=0,
            encoding="jpeg",
            data=encoded.tobytes(),
        ).to_arrow()

    ok, encoded = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("PNG 编码失败")
    return Image(
        width=width,
        height=height,
        channels=0,
        encoding="png",
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
        joints, actuators = _build_joints_and_actuators(config, robot)
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
        # 如果有相机配置了宽高，则统一使用第一组非空宽高；否则使用默认 480x640
        render_height, render_width = 480, 640
        for cam_cfg in config.image_forward.values():
            if cam_cfg.width is not None and cam_cfg.height is not None:
                render_width = int(cam_cfg.width)
                render_height = int(cam_cfg.height)
                break
        try:
            renderer = _create_renderer(model, height=render_height, width=render_width)
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

    # 仿真状态机：默认沿用现有行为直接 RUNNING，便于不接 sim_control 时保持兼容
    sim_mode = "RUNNING"
    current_step = 0
    episode_index = 0

    for event in node:
        match event["type"]:
            case "INPUT":
                input_id = event["id"]

                match input_id:
                    case "sim_control":
                        # 解析来自 control_gateway 的控制指令
                        action: str | None = None
                        try:
                            raw = event.get("value")
                            payload = json.loads(bytes(raw).decode("utf-8")) if raw is not None else {}
                            if isinstance(payload, dict):
                                action = payload.get("action")
                        except Exception:
                            action = None

                        if action == "START":
                            if sim_mode in ("UNINITIALIZED", "STOPPED"):
                                current_step = 0
                                episode_index += 1
                            sim_mode = "RUNNING"
                        elif action == "PAUSE" and sim_mode == "RUNNING":
                            sim_mode = "PAUSED"
                        elif action == "RESUME" and sim_mode == "PAUSED":
                            sim_mode = "RUNNING"
                        elif action == "RESET":
                            current_step = 0
                            episode_index += 1
                            sim_mode = "RUNNING"
                        elif action == "STOP":
                            sim_mode = "STOPPED"

                        status = {
                            "status_name": sim_mode,
                            "status_code": SIM_STATUS_CODE.get(sim_mode, -1),
                            "current_step": current_step,
                            "episode_index": episode_index,
                        }
                        node.send_output("sim_status", json.dumps(status).encode("utf-8"))

                    case "tick":
                        if sim_mode != "RUNNING":
                            # 完全暂停：不步进、不更新图像，只更新 running_time 由上层按需处理
                            status = {
                                "status_name": sim_mode,
                                "status_code": SIM_STATUS_CODE.get(sim_mode, -1),
                                "current_step": current_step,
                                "episode_index": episode_index,
                            }
                            node.send_output(
                                "sim_status", json.dumps(status).encode("utf-8")
                            )
                            continue

                        # 仿真步进
                        mujoco.mj_step(model, data)
                        current_step += 1

                        # 读取状态并输出 ProprioState
                        robot_states: dict[str, object] = {}
                        for robot in config.robots:
                            sim = simulators[robot.id]
                            sim_state = sim.get_state()
                            robot_states[robot.id] = sim_state_to_robot_state(
                                sim_state, robot
                            )
                        proprio_state = build_proprio_state(robot_states, config)
                        node.send_output(
                            "proprio_state", proprio_state.to_arrow(config.joint_order)
                        )

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

                        status = {
                            "status_name": sim_mode,
                            "status_code": SIM_STATUS_CODE.get(sim_mode, -1),
                            "current_step": current_step,
                            "episode_index": episode_index,
                        }
                        node.send_output("sim_status", json.dumps(status).encode("utf-8"))

                    case "action":
                        action = Action.from_arrow(event["value"], config.joint_order)
                        for robot in config.robots:
                            robot_action = build_robot_action(action, robot)
                            sim = simulators[robot.id]
                            sim.set_action(robot_action)

                    case _:
                        # 处理动态 ID 映射，例如 "robot_0/master_state"
                        if input_id.endswith("/master_state"):
                            target_robot_id = input_id.removesuffix("/master_state")
                            robot_cfg = next(
                                (r for r in config.robots if r.id == target_robot_id),
                                None,
                            )
                            if robot_cfg:
                                state = RobotState.from_arrow(
                                    event["value"], robot_cfg.actuator_order
                                )
                                robot_action = robot_state_to_robot_action(
                                    state, robot_cfg
                                )
                                sim = simulators[target_robot_id]
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
