#!/usr/bin/env python3
"""MuJoCo dora-rs node for JointCommand/JointState simulation."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from pathlib import Path

# 渲染后端：须在 import mujoco 之前设置。
# Linux 默认 egl 便于无显示器/SSH/服务器；Windows 不支持 egl，默认 glfw。
# 纯软件渲染可手动设 MUJOCO_GL=osmesa（需 libOSMesa）。
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "glfw" if platform.system() == "Windows" else "egl"

from cv2 import log
import mujoco
import numpy as np
from dora import Node
from forge_msgs import CompressedImage, Image, JointCommand, JointState, Pose, PoseSet
from forge_common import get_logger

from config import CameraConfig, MuJoCoNodeConfig, load_config
from handlers import (
    build_joint_state,
    build_robot_command,
    robot_state_to_robot_command,
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


def _find_root_body_under_prefix(model: mujoco.MjModel, prefix: str) -> int | None:
    """
    在 MuJoCo 模型中查找以 prefix 为前缀的「根 body」：名称形如 prefix/xxx 且 xxx 中无 '/'，
    取 body id 最小的（attach 后该子树根节点）。未找到返回 None。
    """
    p = (prefix or "").strip()
    if not p.endswith("/"):
        p += "/"
    if not p:
        return None

    candidate_id: int | None = None
    for body_id in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if name is None or not name.startswith(p):
            continue
        suffix = name[len(p) :]
        if "/" in suffix:
            continue
        if candidate_id is None or body_id < candidate_id:
            candidate_id = body_id
    return candidate_id


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
    for joint_name, actuator_name in robot_config.joint_to_actuator.items():
        spec = joint_specs.get(joint_name)
        mode = spec.mode if spec else "position"
        # msgs 侧用短名 joint1；MuJoCo XML 常为 prefix/robot_left/joint1
        mj_lookup = joint_name if joint_name != actuator_name else None
        joints.append(JointSpec(name=actuator_name, mode=mode, mj_name=mj_lookup))

    actuators: list[ActuatorSpec] = []
    for i, actuator_name in enumerate(robot_config.actuator_order):
        joint_name = actuator_to_joint.get(actuator_name)
        spec = joint_specs.get(joint_name) if joint_name else None
        unit = spec.unit if spec else "radians"
        mode = spec.mode if spec else "position"
        min_val, max_val = _actuator_limits(unit)
        mj_lookup = (
            joint_name
            if joint_name is not None and joint_name != actuator_name
            else None
        )
        actuators.append(
            ActuatorSpec(
                name=actuator_name,
                id=i,
                control_mode=mode,
                min_value=min_val,
                max_value=max_val,
                unit=unit,
                mj_name=mj_lookup,
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
            height=height,
            width=width,
            encoding="rgb8",
            step=width * 3,
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
        return CompressedImage(format="jpeg", data=encoded.tobytes()).to_arrow()

    ok, encoded = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("PNG 编码失败")
    return CompressedImage(format="png", data=encoded.tobytes()).to_arrow()


def _build_depth_payload(depth_m: np.ndarray):
    """将 MuJoCo 深度图转为 float32 meters 的 Image（encoding=32FC1）。"""
    depth32 = np.ascontiguousarray(depth_m, dtype=np.float32)
    return Image.from_numpy(depth32, encoding="32FC1").to_arrow()


def _render_color_payload(
    renderer,
    data: mujoco.MjData,
    camera: str,
    config: MuJoCoNodeConfig,
    cv2_module: object | None,
):
    """渲染 RGB 图像。Renderer 可能刚渲染过 depth，因此这里显式切回 RGB 模式。"""
    renderer.disable_depth_rendering()
    renderer.update_scene(data, camera=camera)
    pixels = renderer.render()
    return _build_image_payload(pixels, config, cv2_module)


def _render_depth_payload(renderer, data: mujoco.MjData, camera: str):
    """渲染深度图。MuJoCo Renderer depth 输出单位为 meters。"""
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=camera)
    depth_m = renderer.render()
    return _build_depth_payload(depth_m)


def _should_emit_camera(
    sim_time: float,
    next_camera_time: dict[str, float],
    key: str,
    fps: float | None,
) -> bool:
    if fps is None:
        return True
    if sim_time < next_camera_time[key]:
        return False
    next_camera_time[key] += 1.0 / fps
    return True


def _camera_output_key(cam_name: str, cam_config: CameraConfig) -> str:
    """同一 camera 可输出多路图像，因此节流 key 需要包含输出 topic。"""
    return f"{cam_config.image_type}:{cam_name}:{cam_config.output_id}"


def _initial_camera_times(config: MuJoCoNodeConfig) -> dict[str, float]:
    """初始化每路图像输出的下一帧时间。"""
    return {
        _camera_output_key(cam_name, cam_config): 0.0
        for cam_name, cam_config in config.camera_outputs
    }


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

    # 解析 pose_set 的 body id（若配置了 pose_targets）：缓存 {id -> body_id}
    pose_body_ids: dict[str, int] = {}
    if config.pose_targets:
        robot_prefix_by_id = {r.id: (r.prefix or "") for r in config.robots}
        targets: dict[str, str] = {}

        # 用 pose_targets 配置任意 id 的目标（若 id 匹配 robot_id，则使用该 robot 的 prefix 拼接）
        for tid, body_cfg in config.pose_targets.items():
            prefix = robot_prefix_by_id.get(tid, "")
            body_name = _resolve_mujoco_body_name(prefix, body_cfg)
            if body_name:
                targets[tid] = body_name

        # 最后解析 mj id
        for tid, body_name in targets.items():
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id >= 0:
                pose_body_ids[tid] = body_id
                logger.info(
                    "[mujoco] pose_set id=%s 使用 body %s (id=%s)",
                    tid,
                    body_name,
                    body_id,
                )
            else:
                logger.warning(
                    "[mujoco] pose_set body '%s' 在模型中未找到（id=%s），将跳过该 id 的 pose",
                    body_name,
                    tid,
                )

    # 未在 pose_targets 中配置的 robot：按 prefix 自动识别该子树根 body 作为 pose
    for robot in config.robots:
        if robot.id in pose_body_ids:
            continue
        prefix = robot.prefix or ""
        body_id = _find_root_body_under_prefix(model, prefix)
        if body_id is not None:
            pose_body_ids[robot.id] = body_id
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            logger.info(
                "[mujoco] pose_set id=%s 自动识别 body %s (id=%s)",
                robot.id,
                body_name,
                body_id,
            )
        else:
            logger.debug(
                "[mujoco] pose_set id=%s 未配置且 prefix=%r 下未找到根 body，跳过",
                robot.id,
                prefix,
            )

    # 相机渲染器（如有配置）
    renderer = None
    render_height, render_width = 480, 640
    if config.camera_outputs:
        # 如果有相机配置了宽高，则统一使用第一组非空宽高；否则使用默认 480x640
        for _, cam_cfg in config.camera_outputs:
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
    next_camera_time = _initial_camera_times(config)

    control_period_sec = config.control_period_sec
    physics_dt = float(model.opt.timestep)

    node = Node()

    # 默认 RUNNING；高层 runtime 控制通过 task_robot 转为 sim_command。
    sim_mode = "RUNNING"
    sim_time = 0.0
    current_step = 0
    episode_index = 0

    for event in node:
        match event["type"]:
            case "INPUT":
                input_id = event["id"]

                match input_id:
                    case "sim_command":
                        try:
                            payload = json.loads(bytes(event["value"]).decode("utf-8"))
                        except Exception as e:
                            logger.warning("[mujoco] 解析 sim_command 失败: %s", e)
                            continue

                        command = payload.get("command") if isinstance(payload, dict) else None
                        if command == "reset_scene":
                            sim_mode = "RESETTING"
                            mujoco.mj_resetData(model, data)
                            data.ctrl[:] = 0
                            mujoco.mj_forward(model, data)
                            sim_time = 0.0
                            current_step = 0
                            episode_index += 1
                            next_camera_time = _initial_camera_times(config)
                            status = {
                                "status_name": sim_mode,
                                "status_code": SIM_STATUS_CODE.get(sim_mode, -1),
                                "current_step": current_step,
                                "episode_index": episode_index,
                                "sim_time": sim_time,
                            }
                            node.send_output("sim_status", json.dumps(status).encode("utf-8"))
                            sim_mode = "RUNNING"
                        else:
                            logger.debug("[mujoco] 忽略未知 sim_command: %r", command)

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
                        robot_states: dict[str, JointState] = {}
                        for robot in config.robots:
                            sim = simulators[robot.id]
                            sim_state = sim.get_state()
                            robot_states[robot.id] = sim_state_to_robot_state(
                                sim_state, robot
                            )
                        joint_state = build_joint_state(robot_states, config)
                        node.send_output("proprio_state", joint_state.to_arrow())

                        # 可选：输出世界系位姿（PoseSet，按 robot_id keyed）
                        if pose_body_ids:
                            poses: dict[str, Pose] = {}
                            for rid, bid in pose_body_ids.items():
                                x, y = float(data.xpos[bid][0]), float(data.xpos[bid][1])
                                q = data.xquat[bid]
                                theta = math.atan2(
                                    2.0 * (q[0] * q[3] + q[1] * q[2]),
                                    1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2),
                                )
                                poses[rid] = Pose.from_xy_yaw(x=x, y=y, yaw=theta)

                            if poses:
                                node.send_output("pose_set", PoseSet.from_poses(poses).to_arrow())

                        # 相机按 sim_time 节流
                        if renderer is not None:
                            for cam_name, cam_config in config.camera_outputs:
                                camera_key = _camera_output_key(cam_name, cam_config)
                                output_id = cam_config.output_id
                                if _should_emit_camera(
                                    sim_time, next_camera_time, camera_key, cam_config.fps
                                ):
                                    try:
                                        if cam_config.image_type == "depth":
                                            payload = _render_depth_payload(
                                                renderer, data, cam_name
                                            )
                                        else:
                                            payload = _render_color_payload(
                                                renderer,
                                                data,
                                                cam_name,
                                                config,
                                                cv2_module,
                                            )
                                        node.send_output(output_id, payload)
                                    except Exception as e:
                                        image_label = (
                                            "深度相机"
                                            if cam_config.image_type == "depth"
                                            else "相机"
                                        )
                                        print(
                                            f"[mujoco] 渲染{image_label} {cam_name} 失败: {e}",
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
                        command = JointCommand.from_arrow(event["value"])
                        for robot in config.robots:
                            robot_command = build_robot_command(command, robot)
                            sim = simulators[robot.id]
                            sim.set_command(robot_command)

                    case _:
                        # 处理动态 ID 映射，例如 "robot_0/master_state"
                        if input_id.endswith("/master_state"):
                            target_robot_id = input_id.removesuffix("/master_state")
                            robot_cfg = next(
                                (r for r in config.robots if r.id == target_robot_id),
                                None,
                            )
                            if robot_cfg:
                                state = JointState.from_arrow(event["value"])
                                robot_command = robot_state_to_robot_command(
                                    state, robot_cfg
                                )
                                sim = simulators[target_robot_id]
                                sim.set_command(robot_command)

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
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
