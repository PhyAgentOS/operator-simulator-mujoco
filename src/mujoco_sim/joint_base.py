"""Force-driven planar base; MuJoCo contact constraints remain authoritative.

This is a constrained planar velocity servo, not a wheel/terrain simulator.
It never writes qpos during motion and never resets position to bypass contact.
"""

from __future__ import annotations

import math
import time

import mujoco
import numpy as np

from .navigation_config import JointPlanarBaseConfig


class JointPlanarBase:
    def __init__(self, model, data, config: JointPlanarBaseConfig):
        self.model, self.data, self.config = model, data, config
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, config.body)
        if self.body_id < 0:
            raise ValueError(f"joint_planar_base body not found: {config.body}")
        names = (config.x_joint, config.y_joint, config.yaw_joint)
        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names]
        if min(ids) < 0:
            raise ValueError(f"joint_planar_base missing joint among {names}")
        if list(model.jnt_type[ids]) != [mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE]:
            raise ValueError("joint_planar_base requires slide X, slide Y, hinge yaw")
        if not np.allclose(model.jnt_axis[ids], np.eye(3)) or not np.all(model.jnt_bodyid[ids] == self.body_id):
            raise ValueError("joint_planar_base joints must share base body and world-aligned XYZ axes")
        if int(model.body_parentid[self.body_id]) != 0 or not np.allclose(model.body_quat[self.body_id], [1, 0, 0, 0]):
            raise ValueError("joint_planar_base body must be world child with identity initial quaternion; use initial_yaw")
        self.qpos = model.jnt_qposadr[ids]
        self.dofs = model.jnt_dofadr[ids]
        self.mass = np.empty((model.nv, model.nv))
        self.stop()

    def reset(self):
        self.stop()
        self.data.qpos[self.qpos[2]] = self.config.initial_yaw

    def stop(self):
        self.command = np.zeros(3)
        self.last_command_sim = -math.inf
        self.last_command_wall = -math.inf
        self.data.qfrc_applied[self.dofs] = 0

    def set_command(self, command, sim_time):
        values = np.array([command.vx, command.vy, command.wz], dtype=float)
        if not np.isfinite(values).all() or not math.isfinite(sim_time):
            raise ValueError("base command and simulation time must be finite")
        # Zerith's effective differential drive does not implement lateral motion.
        if abs(values[1]) > 1e-8:
            self.stop()
            raise ValueError("joint_planar_base differential drive requires vy=0")
        values[0] = np.clip(values[0], -self.config.max_linear_speed, self.config.max_linear_speed)
        values[2] = np.clip(values[2], -self.config.max_angular_speed, self.config.max_angular_speed)
        self.command = values
        self.last_command_sim = float(sim_time)
        self.last_command_wall = time.monotonic()

    def advance(self, dt, sim_time):
        if not math.isfinite(dt) or dt <= 0 or not math.isfinite(sim_time):
            raise ValueError("invalid physics dt/time")
        cfg = self.config
        if sim_time - self.last_command_sim >= cfg.command_timeout_sec or time.monotonic() - self.last_command_wall >= cfg.command_timeout_sec:
            self.command[:] = 0
        yaw = float(self.data.qpos[self.qpos[2]])
        vx, _, wz = self.command
        target = np.array([vx * math.cos(yaw), vx * math.sin(yaw), wz])
        accel = (target - self.data.qvel[self.dofs]) * cfg.velocity_gain
        linear_norm = float(np.linalg.norm(accel[:2]))
        if linear_norm > cfg.max_acceleration:
            accel[:2] *= cfg.max_acceleration / linear_norm
        accel[2] = np.clip(accel[2], -cfg.max_angular_acceleration, cfg.max_angular_acceleration)
        mujoco.mj_fullM(self.model, self.data, self.mass)
        # Feed-forward gravity/passive forces; contact forces are NOT cancelled.
        # Only the three designated base DOFs receive external servo forces.
        self.data.qfrc_applied[self.dofs] = (
            self.mass[np.ix_(self.dofs, self.dofs)] @ accel
            + self.data.qfrc_bias[self.dofs] - self.data.qfrc_passive[self.dofs]
        )

    def feedback(self):
        yaw = float(self.data.qpos[self.qpos[2]])
        dx, dy, wz = self.data.qvel[self.dofs]
        vx = float(dx * math.cos(yaw) + dy * math.sin(yaw))
        lateral = float(-dx * math.sin(yaw) + dy * math.cos(yaw))
        half_track = self.config.wheel_separation / 2
        return {
            "wheel_speed": {
                "wheel_left": float((vx - half_track * wz) / self.config.wheel_radius),
                "wheel_right": float((vx + half_track * wz) / self.config.wheel_radius),
            },
            "issued_monotonic": time.monotonic(),
            "capture_timestamp_ns": round(float(self.data.time) * 1e9),
            "velocity": {"vx": vx, "vy": lateral, "wz": float(wz)},
            "feedback_source": "actual_planar_joint_velocity_equivalent_wheel_speed",
        }
