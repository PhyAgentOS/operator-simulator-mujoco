from __future__ import annotations

import math

import mujoco
import pytest
from forge_msgs import LocomotionCommand

from mujoco_sim.config import MuJoCoNodeConfig
from mujoco_sim.locomotion import KinematicPlanarBase
from mujoco_sim.scene import reset_simulation_state


def _base_config() -> dict[str, object]:
    return {
        "model_path": "model.xml",
        "joints": [{"name": "joint", "mode": "position", "unit": "radians"}],
        "robots": [{"id": "robot", "joints": ["joint"]}],
    }


def _make_mocap_model(
    *, yaw: float = 0.0, quat: tuple[float, float, float, float] | None = None
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    if quat is None:
        half_yaw = yaw / 2.0
        quat = (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))
    quat_text = " ".join(str(value) for value in quat)
    xml = f"""
    <mujoco>
      <option gravity="0 0 0" timestep="0.002"/>
      <worldbody>
        <body name="base" mocap="true" pos="1 2 0.5" quat="{quat_text}">
          <body name="payload" pos="0.2 0 0">
            <joint name="joint" type="hinge" axis="0 0 1"/>
            <geom type="sphere" size="0.05" mass="1"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _yaw_from_quat(quat) -> float:
    return math.atan2(
        2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
        1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2),
    )


def test_config_is_backward_compatible_when_planar_base_is_absent() -> None:
    config = MuJoCoNodeConfig.from_dict(_base_config())

    assert config.kinematic_planar_base is None


def test_config_parses_kinematic_planar_base() -> None:
    raw = _base_config()
    raw["kinematic_planar_base"] = {
        "body": " base ",
        "command_timeout_sec": 0.1,
    }

    config = MuJoCoNodeConfig.from_dict(raw)

    assert config.kinematic_planar_base is not None
    assert config.kinematic_planar_base.body == "base"
    assert config.kinematic_planar_base.command_timeout_sec == pytest.approx(0.1)


@pytest.mark.parametrize(
    "planar_base",
    [
        "base",
        {},
        {"body": ""},
        {"body": "base", "command_timeout_sec": True},
        {"body": "base", "command_timeout_sec": 0.0},
        {"body": "base", "command_timeout_sec": -0.1},
        {"body": "base", "command_timeout_sec": math.inf},
        {"body": "base", "command_timeout_sec": math.nan},
    ],
)
def test_config_rejects_invalid_kinematic_planar_base(planar_base: object) -> None:
    raw = _base_config()
    raw["kinematic_planar_base"] = planar_base

    with pytest.raises(ValueError, match="kinematic_planar_base"):
        MuJoCoNodeConfig.from_dict(raw)


def test_planar_base_requires_an_existing_mocap_body() -> None:
    model, data = _make_mocap_model()

    with pytest.raises(ValueError, match="not found"):
        KinematicPlanarBase(model, data, "missing")
    with pytest.raises(ValueError, match='mocap="true"'):
        KinematicPlanarBase(model, data, "payload")


def test_forward_velocity_is_integrated_in_the_current_body_frame() -> None:
    model, data = _make_mocap_model(yaw=math.pi / 2.0)
    base = KinematicPlanarBase(model, data, "base")
    base.set_command(LocomotionCommand(vx=1.0, vy=0.0, wz=0.0), sim_time=0.0)

    base.advance(dt=0.25, sim_time=0.0)
    mujoco.mj_forward(model, data)

    assert data.mocap_pos[0] == pytest.approx([1.0, 2.25, 0.5])


def test_lateral_and_angular_signs_and_child_pose_propagation() -> None:
    model, data = _make_mocap_model()
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    payload_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "payload")
    base = KinematicPlanarBase(model, data, "base")
    base.set_command(LocomotionCommand(vx=0.0, vy=0.4, wz=1.0), sim_time=0.0)

    base.advance(dt=0.5, sim_time=0.0)
    mujoco.mj_forward(model, data)

    assert data.xpos[base_id] == pytest.approx([1.0, 2.2, 0.5])
    assert data.xpos[payload_id] == pytest.approx(
        [1.0 + 0.2 * math.cos(0.5), 2.2 + 0.2 * math.sin(0.5), 0.5]
    )
    assert _yaw_from_quat(data.mocap_quat[0]) == pytest.approx(0.5)
    assert float(sum(data.mocap_quat[0] ** 2)) == pytest.approx(1.0)


def test_yaw_increment_preserves_initial_roll_and_pitch() -> None:
    initial_quat = (0.95154852, 0.23929834, 0.18930786, 0.03813458)
    model, data = _make_mocap_model(quat=initial_quat)
    base = KinematicPlanarBase(model, data, "base")
    base.set_command(LocomotionCommand(vx=0.0, vy=0.0, wz=0.4), sim_time=0.0)

    base.advance(dt=0.5, sim_time=0.0)

    half_delta = 0.1
    delta_w = math.cos(half_delta)
    delta_z = math.sin(half_delta)
    w, x, y, z = initial_quat
    expected = (
        delta_w * w - delta_z * z,
        delta_w * x - delta_z * y,
        delta_w * y + delta_z * x,
        delta_w * z + delta_z * w,
    )
    assert data.mocap_quat[0] == pytest.approx(expected)
    assert abs(float(data.mocap_quat[0, 1])) > 0.0
    assert abs(float(data.mocap_quat[0, 2])) > 0.0


def test_timeout_uses_simulation_time_and_stops_motion() -> None:
    model, data = _make_mocap_model()
    base = KinematicPlanarBase(model, data, "base", command_timeout_sec=0.1)
    command = LocomotionCommand(vx=1.0, vy=0.0, wz=0.0)
    base.set_command(command, sim_time=0.0)

    base.advance(dt=0.05, sim_time=0.0)
    base.advance(dt=0.05, sim_time=0.1)

    assert data.mocap_pos[0, 0] == pytest.approx(1.05)

    base.set_command(command, sim_time=0.11)
    base.advance(dt=0.05, sim_time=0.11)
    assert data.mocap_pos[0, 0] == pytest.approx(1.1)


def test_timeout_clips_motion_when_substep_crosses_deadline() -> None:
    model, data = _make_mocap_model()
    base = KinematicPlanarBase(model, data, "base", command_timeout_sec=0.1)
    base.set_command(
        LocomotionCommand(vx=1.0, vy=0.0, wz=1.0), sim_time=0.0
    )

    base.advance(dt=0.06, sim_time=0.0)
    base.advance(dt=0.06, sim_time=0.06)

    assert data.mocap_pos[0, 0] == pytest.approx(
        1.0 + 0.06 + math.cos(0.06) * 0.04
    )
    assert data.mocap_pos[0, 1] == pytest.approx(
        2.0 + math.sin(0.06) * 0.04
    )
    assert _yaw_from_quat(data.mocap_quat[0]) == pytest.approx(0.1)

    pose_at_timeout = data.mocap_pos[0].copy()
    quat_at_timeout = data.mocap_quat[0].copy()
    base.advance(dt=0.06, sim_time=0.12)
    assert data.mocap_pos[0] == pytest.approx(pose_at_timeout)
    assert data.mocap_quat[0] == pytest.approx(quat_at_timeout)


def test_reset_restores_mocap_pose_and_clears_motion() -> None:
    model, data = _make_mocap_model()
    base = KinematicPlanarBase(model, data, "base")
    base.set_command(LocomotionCommand(vx=1.0, vy=0.0, wz=0.5), sim_time=0.0)
    base.advance(dt=0.2, sim_time=0.0)

    config = MuJoCoNodeConfig.from_dict(
        {"model_path": "model.xml", "joints": [], "robots": []}
    )
    reset_simulation_state(model, data, config, {}, base)

    assert data.mocap_pos[0] == pytest.approx([1.0, 2.0, 0.5])
    assert data.mocap_quat[0] == pytest.approx([1.0, 0.0, 0.0, 0.0])

    base.advance(dt=0.2, sim_time=0.0)
    assert data.mocap_pos[0] == pytest.approx([1.0, 2.0, 0.5])
