#!/usr/bin/env python3
"""MuJoCo dora-rs 节点：接收 tick 驱动仿真，接收 Action 控制 ctrl，输出 ProprioState 和相机图像。"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

# 渲染后端：须在 import mujoco 之前设置。默认 egl 便于无显示器/SSH/服务器；
# 若需 GLFW 窗口可设 MUJOCO_GL=glfw；纯软件渲染可设 MUJOCO_GL=osmesa（需 libOSMesa）。
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import mujoco
import numpy as np
from dora import Node
from forge_msgs import Action, Image, Pose2D, Pose2DList, RobotState
from forge_common import get_logger

from config import MuJoCoNodeConfig, load_config
from handlers import (
    build_proprio_state,
    build_robot_action,
    robot_state_to_robot_action,
    sim_state_to_robot_state,
)
from simulator import ActuatorSpec, JointSpec, MuJoCoSimulator

logger = get_logger(__name__)


SIM_STATUS_CODE = {
    "UNINITIALIZED": 0,
    "INITIALIZED": 1,
    "RUNNING": 2,
    "PAUSED": 3,
    "STOPPED": 4,
    "RESETTING": 5,
}

def _resolve_mujoco_body_name(prefix: str, body: str) -> str:
    """将配置的 body 名与 robot prefix 拼成 MuJoCo body name。若 body 已包含 prefix，则不重复拼接。"""
    b = str(body).strip()
    if not b:
        return ""
    p = prefix or ""
    if p and not p.endswith("/"):
        p += "/"
    if p and (b.startswith(p) or b.startswith(p.lstrip("/"))):
        name = b
    else:
        name = p + b
    return name.lstrip("/")


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
    # 位置单位
    if unit == "millimeters":
        return 0.0, 50.0
    if unit == "meters":
        return 0.0, 0.05
    if unit == "radians":
        return -3.14, 3.14
    # 速度单位
    if unit == "millimeters/s":
        return -50.0, 50.0
    if unit == "meters/s":
        return -1.0, 1.0
    if unit in ("radians/s", "radians_s"):
        return -10.0, 10.0
    # 默认按弧度位置
    return -3.14, 3.14


def _build_joints_and_actuators(
    node_config: MuJoCoNodeConfig,
    robot_config,
) -> tuple[list[JointSpec], list[ActuatorSpec]]:
    """从节点配置的 JointConfig 与 RobotConfig 构建 JointSpec 和 ActuatorSpec 列表。"""
    joint_specs = {j.name: j for j in node_config.joints}
    actuator_to_joint = {
        act: joint for joint, act in robot_config.joint_to_actuator.items()
    }

    joints: list[JointSpec] = []
    for name in robot_config.joint_to_actuator.keys():
        spec = joint_specs.get(name)
        mode = spec.mode if spec else "position"
        joints.append(JointSpec(name=name, mode=mode))

    actuators: list[ActuatorSpec] = []
    for i, actuator_name in enumerate(robot_config.actuator_order):
        joint_name = actuator_to_joint.get(actuator_name)
        spec = joint_specs.get(joint_name) if joint_name else None
        unit = spec.unit if spec else "radians"
        mode = spec.mode if spec else "position"
        min_val, max_val = _actuator_limits(unit)
        actuators.append(
            ActuatorSpec(
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
    # 预先构建派生 actuator 状态配置，供 simulator 使用（例如 gripper 由 joint7/joint8 合成）
    derived_state_cfg_for_sim: dict[str, dict[str, object]] = {}
    for name, cfg in config.derived_state_cfg.items():
        derived_state_cfg_for_sim[name] = {
            "joint_names": list(cfg.joint_names),
            "expr": cfg.state_expr,
            "scale": cfg.state_scale,
        }

    simulators: dict[str, MuJoCoSimulator] = {}
    for robot in config.robots:
        joints, actuators = _build_joints_and_actuators(config, robot)
        prefix = robot.prefix
        sim = MuJoCoSimulator(
            model=model,
            data=data,
            joints=joints,
            actuators=actuators,
            derived_state_cfg=derived_state_cfg_for_sim,
            prefix=prefix,
        )
        simulators[robot.id] = sim

    # 解析 pose_2d 的 body id（若配置了 pose_2ds）：缓存 {id -> body_id}
    pose_2d_body_ids: dict[str, int] = {}
    if config.pose_2ds:
        robot_prefix_by_id = {r.id: (r.prefix or "") for r in config.robots}
        targets: dict[str, str] = {}

        # 用 pose_2ds 配置任意 id 的目标（若 id 匹配 robot_id，则使用该 robot 的 prefix 拼接）
        for tid, body_cfg in config.pose_2ds.items():
            prefix = robot_prefix_by_id.get(tid, "")
            body_name = _resolve_mujoco_body_name(prefix, body_cfg)
            if body_name:
                targets[tid] = body_name

        # 最后解析 mj id
        for tid, body_name in targets.items():
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id >= 0:
                pose_2d_body_ids[tid] = body_id
                logger.info(
                    "[mujoco] pose_2d id=%s 使用 body %s (id=%s)",
                    tid,
                    body_name,
                    body_id,
                )
            else:
                logger.warning(
                    "[mujoco] pose_2d body '%s' 在模型中未找到（id=%s），将跳过该 id 的 pose_2d",
                    body_name,
                    tid,
                )

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

    # 相机按仿真时间节流：下次应输出帧的 sim_time
    next_camera_time: dict[str, float] = {name: 0.0 for name in config.camera_names}

    control_period_sec = config.control_period_sec
    physics_dt = float(model.opt.timestep)

    node = Node()

    # 仿真状态机：默认沿用现有行为直接 RUNNING，便于不接 sim_control 时保持兼容
    sim_mode = "RUNNING"
    sim_time = 0.0
    current_step = 0
    episode_index = 0

    for event in node:
        match event["type"]:
            case "INPUT":
                input_id = event["id"]

                match input_id:
                    case "sim_control":
                        # 解析来自 control_gateway 的控制指令（仅支持 Arrow SimControl）
                        action: str | None = None
                        raw = event.get("value")
                        try:
                            from forge_msgs import SimControl  # 局部导入以避免循环依赖

                            ctrl = SimControl.from_arrow(raw)
                        except Exception as e:
                            ctrl = None

                        if ctrl is not None:
                            action = ctrl.action
                        else:
                            print(f"[mujoco] SimControl.from_arrow 解析失败，忽略控制指令，value_type={type(raw)}", file=sys.stderr)
                            action = None

                        if action == "START":
                            if sim_mode in ("UNINITIALIZED", "STOPPED"):
                                current_step = 0
                                sim_time = 0.0
                                episode_index += 1
                                for k in next_camera_time:
                                    next_camera_time[k] = 0.0
                            sim_mode = "RUNNING"
                        elif action == "PAUSE" and sim_mode == "RUNNING":
                            sim_mode = "PAUSED"
                        elif action == "RESUME" and sim_mode == "PAUSED":
                            sim_mode = "RUNNING"
                        elif action == "RESET":
                            mujoco.mj_resetData(model, data)
                            current_step = 0
                            sim_time = 0.0
                            episode_index += 1
                            for k in next_camera_time:
                                next_camera_time[k] = 0.0
                            sim_mode = "RUNNING"
                        elif action == "STOP":
                            sim_mode = "STOPPED"

                        status = {
                            "status_name": sim_mode,
                            "status_code": SIM_STATUS_CODE.get(sim_mode, -1),
                            "current_step": current_step,
                            "episode_index": episode_index,
                            "sim_time": sim_time,
                        }
                        node.send_output("sim_status", json.dumps(status).encode("utf-8"))

                    case "tick":
                        if sim_mode != "RUNNING":
                            # 完全暂停：不步进、不更新图像，只更新 status 由上层按需处理
                            status = {
                                "status_name": sim_mode,
                                "status_code": SIM_STATUS_CODE.get(sim_mode, -1),
                                "current_step": current_step,
                                "episode_index": episode_index,
                                "sim_time": sim_time,
                            }
                            node.send_output(
                                "sim_status", json.dumps(status).encode("utf-8")
                            )
                            continue

                        # 每 tick 执行若干物理子步（控制周期 / 物理步长）
                        n_steps = max(
                            1, round(control_period_sec / physics_dt)
                        )
                        for _ in range(n_steps):
                            mujoco.mj_step(model, data)
                            sim_time += physics_dt
                        current_step += n_steps

                        # 每 tick 输出一次 state
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

                        # 可选：输出世界系 2D 位姿（Pose2DList，按 robot_id keyed）
                        if pose_2d_body_ids:
                            poses: dict[str, Pose2D] = {}
                            for rid, bid in pose_2d_body_ids.items():
                                x, y = float(data.xpos[bid][0]), float(data.xpos[bid][1])
                                q = data.xquat[bid]
                                theta = math.atan2(
                                    2.0 * (q[0] * q[3] + q[1] * q[2]),
                                    1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2),
                                )
                                poses[rid] = Pose2D(x=x, y=y, theta=theta)

                            if poses:
                                node.send_output("pose_2d", Pose2DList(poses=poses).to_arrow())

                        # 相机按 sim_time 节流
                        if renderer is not None:
                            for cam_name, cam_config in config.image_forward.items():
                                output_id = cam_config.output_id
                                if cam_config.fps is not None:
                                    cam_period = 1.0 / cam_config.fps
                                    while sim_time >= next_camera_time[cam_name]:
                                        next_camera_time[cam_name] += cam_period
                                        try:
                                            renderer.update_scene(
                                                data, camera=cam_name
                                            )
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
                                else:
                                    # 无 fps 时随 tick 输出一帧
                                    try:
                                        renderer.update_scene(
                                            data, camera=cam_name
                                        )
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
                            "sim_time": sim_time,
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
