"""Dora event lifecycle and output scheduling for the MuJoCo node."""

from __future__ import annotations

import json
import math
import sys
from typing import Any

import mujoco
from dora import Node
from forge_common import get_logger
from forge_msgs import JointCommand, JointState, LocomotionCommand, Pose, PoseSet

from .config import MuJoCoNodeConfig
from .handlers import (
    build_joint_state,
    build_robot_command,
    robot_state_to_robot_command,
    sim_state_to_robot_state,
)
from . import rendering, scene

logger = get_logger(__name__)


SIM_STATUS_CODE = {
    "UNINITIALIZED": 0,
    "INITIALIZED": 1,
    "RUNNING": 2,
    "PAUSED": 3,
    "STOPPED": 4,
    "RESETTING": 5,
}


class _StartupAborted(Exception):
    """Signal an already-reported runtime initialization failure."""


class MuJoCoNodeRuntime:
    """Own initialized MuJoCo state and dispatch Dora runtime events."""

    def __init__(self, config: MuJoCoNodeConfig) -> None:
        self.config = config

        loaded_model = scene.load_model_and_data(config)
        if loaded_model is None:
            raise _StartupAborted
        self.model, self.data = loaded_model
        self.planar_base = scene.initialize_planar_base(
            self.model, self.data, self.config
        )
        self.simulators = scene.initialize_simulators(
            self.model, self.data, self.config
        )
        scene.reset_simulation_state(
            self.model,
            self.data,
            self.config,
            self.simulators,
            self.planar_base,
        )
        self.pose_body_ids = scene.resolve_pose_body_ids(self.model, self.config)

        self.renderer = None
        if self.config.camera_outputs:
            try:
                self.renderer = rendering.create_renderer_for_config(
                    self.model, self.config
                )
            except Exception as error:
                print(
                    f"[mujoco] 创建渲染器失败，将跳过相机输出: {error}",
                    file=sys.stderr,
                )

        try:
            self.cv2_module = rendering.load_cv2_module(self.config)
        except Exception as error:
            print(
                f"[mujoco] image_format={self.config.image_format} "
                f"需要 opencv-python: {error}",
                file=sys.stderr,
            )
            raise _StartupAborted from error

        self.next_camera_time = rendering.initial_camera_times(self.config)
        self.control_period_sec = self.config.control_period_sec
        self.physics_dt = float(self.model.opt.timestep)
        self.node = Node()

        self.sim_mode = "RUNNING"
        self.sim_time = 0.0
        self.current_step = 0
        self.episode_index = 0

    def run(self) -> int:
        """Process Dora events until STOP or ERROR."""
        try:
            for event in self.node:
                match event["type"]:
                    case "INPUT":
                        self._handle_input(event)
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
        finally:
            if self.renderer is not None:
                try:
                    self.renderer.close()
                except Exception:
                    pass

    def _handle_input(self, event: dict[str, Any]) -> None:
        input_id = event["id"]
        match input_id:
            case "sim_command":
                self._handle_sim_command(event)
            case "tick":
                self._handle_tick()
            case "action":
                self._handle_action(event)
            case "locomotion_command":
                self._handle_locomotion(event)
            case _:
                self._handle_master_state(input_id, event)

    def _handle_sim_command(self, event: dict[str, Any]) -> None:
        try:
            payload = json.loads(bytes(event["value"]).decode("utf-8"))
        except Exception as error:
            logger.warning("[mujoco] 解析 sim_command 失败: %s", error)
            return

        command = payload.get("command") if isinstance(payload, dict) else None
        if command != "reset_scene":
            logger.debug("[mujoco] 忽略未知 sim_command: %r", command)
            return

        self.sim_mode = "RESETTING"
        scene.reset_simulation_state(
            self.model,
            self.data,
            self.config,
            self.simulators,
            self.planar_base,
        )
        self.sim_time = 0.0
        self.current_step = 0
        self.episode_index += 1
        self.next_camera_time = rendering.initial_camera_times(self.config)
        self._send_status()
        self.sim_mode = "RUNNING"

    def _handle_tick(self) -> None:
        if self.sim_mode != "RUNNING":
            self._send_status()
            return

        self._step_physics()
        self._send_state_output()
        self._send_pose_output()
        self._send_camera_outputs()
        self._send_status()

    def _step_physics(self) -> None:
        n_steps = max(1, round(self.control_period_sec / self.physics_dt))
        for _ in range(n_steps):
            if self.planar_base is not None:
                self.planar_base.advance(self.physics_dt, self.sim_time)
            mujoco.mj_step(self.model, self.data)
            self.sim_time += self.physics_dt
        self.current_step += n_steps

    def _send_state_output(self) -> None:
        robot_states: dict[str, JointState] = {}
        for robot in self.config.robots:
            simulator = self.simulators[robot.id]
            sim_state = simulator.get_state()
            robot_states[robot.id] = sim_state_to_robot_state(sim_state, robot)
        joint_state = build_joint_state(robot_states, self.config)
        self.node.send_output("proprio_state", joint_state.to_arrow())

    def _send_pose_output(self) -> None:
        if not self.pose_body_ids:
            return

        poses: dict[str, Pose] = {}
        for robot_id, body_id in self.pose_body_ids.items():
            x = float(self.data.xpos[body_id][0])
            y = float(self.data.xpos[body_id][1])
            quaternion = self.data.xquat[body_id]
            theta = math.atan2(
                2.0
                * (
                    quaternion[0] * quaternion[3]
                    + quaternion[1] * quaternion[2]
                ),
                1.0
                - 2.0
                * (quaternion[2] ** 2 + quaternion[3] ** 2),
            )
            poses[robot_id] = Pose.from_xy_yaw(x=x, y=y, yaw=theta)

        if poses:
            self.node.send_output(
                "pose_set", PoseSet.from_poses(poses).to_arrow()
            )

    def _send_camera_outputs(self) -> None:
        if self.renderer is None:
            return

        for camera_name, camera_config in self.config.camera_outputs:
            camera_key = rendering.camera_output_key(camera_name, camera_config)
            if not rendering.should_emit_camera(
                self.sim_time,
                self.next_camera_time,
                camera_key,
                camera_config.fps,
            ):
                continue

            try:
                if camera_config.image_type == "depth":
                    payload = rendering.render_depth_payload(
                        self.renderer, self.data, camera_name
                    )
                else:
                    payload = rendering.render_color_payload(
                        self.renderer,
                        self.data,
                        camera_name,
                        self.config,
                        self.cv2_module,
                    )
                self.node.send_output(camera_config.output_id, payload)
            except Exception as error:
                image_label = (
                    "深度相机" if camera_config.image_type == "depth" else "相机"
                )
                print(
                    f"[mujoco] 渲染{image_label} {camera_name} 失败: {error}",
                    file=sys.stderr,
                )

    def _send_status(self) -> None:
        status = {
            "status_name": self.sim_mode,
            "status_code": SIM_STATUS_CODE.get(self.sim_mode, -1),
            "current_step": self.current_step,
            "episode_index": self.episode_index,
            "sim_time": self.sim_time,
        }
        self.node.send_output(
            "sim_status", json.dumps(status).encode("utf-8")
        )

    def _handle_action(self, event: dict[str, Any]) -> None:
        command = JointCommand.from_arrow(event["value"])
        for robot in self.config.robots:
            robot_command = build_robot_command(command, robot)
            self.simulators[robot.id].set_command(robot_command)

    def _handle_locomotion(self, event: dict[str, Any]) -> None:
        if self.planar_base is None:
            logger.warning(
                "[mujoco] 忽略 locomotion_command：未配置 kinematic_planar_base"
            )
            return
        try:
            command = LocomotionCommand.from_arrow(event["value"])
        except Exception as error:
            logger.error("[mujoco] 忽略无效 locomotion_command: %s", error)
            return
        self.planar_base.set_command(command, self.sim_time)

    def _handle_master_state(
        self,
        input_id: str,
        event: dict[str, Any],
    ) -> None:
        if not input_id.endswith("/master_state"):
            return

        target_robot_id = input_id.removesuffix("/master_state")
        robot_config = next(
            (
                robot
                for robot in self.config.robots
                if robot.id == target_robot_id
            ),
            None,
        )
        if robot_config:
            state = JointState.from_arrow(event["value"])
            robot_command = robot_state_to_robot_command(state, robot_config)
            self.simulators[target_robot_id].set_command(robot_command)


def start_runtime(config: MuJoCoNodeConfig) -> int:
    """Initialize and run the MuJoCo Dora node."""
    try:
        runtime = MuJoCoNodeRuntime(config)
    except _StartupAborted:
        return 1
    return runtime.run()
