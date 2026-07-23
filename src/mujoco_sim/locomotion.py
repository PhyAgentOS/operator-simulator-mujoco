"""Kinematic planar locomotion for MuJoCo mocap bodies."""

from __future__ import annotations

import math
from typing import Any

import mujoco
from forge_msgs import LocomotionCommand


class KinematicPlanarBase:
    """Integrate body-frame planar velocity by moving a MuJoCo mocap body."""

    def __init__(
        self,
        model: Any,
        data: Any,
        body_name: str,
        command_timeout_sec: float | None = None,
    ) -> None:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(
                f"kinematic planar base body '{body_name}' not found in mjModel"
            )

        mocap_id = int(model.body_mocapid[body_id])
        if mocap_id < 0:
            raise ValueError(
                f"kinematic planar base body '{body_name}' must have mocap=\"true\""
            )

        self._data = data
        self._body_name = body_name
        self._mocap_id = mocap_id
        self._command_timeout_sec = command_timeout_sec
        self._command = (0.0, 0.0, 0.0)
        self._last_command_sim_time: float | None = None

    def set_command(
        self, command: LocomotionCommand, sim_time: float
    ) -> None:
        """Latch a finite body-frame velocity command at the current simulation time."""
        if not math.isfinite(sim_time):
            raise ValueError("sim_time must be finite")
        self._command = (float(command.vx), float(command.vy), float(command.wz))
        self._last_command_sim_time = sim_time

    def stop(self) -> None:
        """Clear the latched command without changing the current pose."""
        self._command = (0.0, 0.0, 0.0)
        self._last_command_sim_time = None

    def advance(self, dt: float, sim_time: float) -> None:
        """Advance the mocap pose by one physics substep."""
        if not math.isfinite(dt) or dt < 0.0:
            raise ValueError("dt must be finite and non-negative")
        if not math.isfinite(sim_time):
            raise ValueError("sim_time must be finite")

        integration_dt = dt
        stop_after_advance = False
        if (
            self._command_timeout_sec is not None
            and self._last_command_sim_time is not None
        ):
            command_age = sim_time - self._last_command_sim_time
            remaining_duration = self._command_timeout_sec - command_age
            if remaining_duration <= 0.0:
                self.stop()
                return
            if remaining_duration <= dt:
                integration_dt = remaining_duration
                stop_after_advance = True

        vx, vy, wz = self._command
        if integration_dt == 0.0 or (vx == 0.0 and vy == 0.0 and wz == 0.0):
            if stop_after_advance:
                self.stop()
            return

        quat = self._data.mocap_quat[self._mocap_id]
        yaw = math.atan2(
            2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
            1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2),
        )
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        pos = self._data.mocap_pos[self._mocap_id]
        pos[0] += (cos_yaw * vx - sin_yaw * vy) * integration_dt
        pos[1] += (sin_yaw * vx + cos_yaw * vy) * integration_dt

        half_delta_yaw = wz * integration_dt / 2.0
        delta_w = math.cos(half_delta_yaw)
        delta_z = math.sin(half_delta_yaw)
        current_w, current_x, current_y, current_z = (
            float(value) for value in quat
        )
        self._data.mocap_quat[self._mocap_id] = (
            delta_w * current_w - delta_z * current_z,
            delta_w * current_x - delta_z * current_y,
            delta_w * current_y + delta_z * current_x,
            delta_w * current_z + delta_z * current_w,
        )
        if stop_after_advance:
            self.stop()
