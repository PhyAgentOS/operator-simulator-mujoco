from __future__ import annotations

import math


import mujoco
import pytest
from forge_msgs import JointCommand

from mujoco_sim.config import MuJoCoNodeConfig
from mujoco_sim.main import _build_joints_and_actuators, _reset_simulation_state
from mujoco_sim.simulator import MuJoCoSimulator


def _base_config() -> dict[str, object]:
    return {
        "model_path": "model.xml",
        "joints": [{"name": "lift", "mode": "position", "unit": "meters"}],
        "robots": [
            {
                "id": "robot",
                "joints": ["lift"],
                "actuator_order": ["lift_motor"],
                "joint_to_actuator": {"lift": "lift_motor"},
            }
        ],
    }


def _make_lift_simulator(
    gear: float = 1.0,
) -> tuple[mujoco.MjModel, mujoco.MjData, MuJoCoSimulator]:
    xml = f"""
    <mujoco>
      <option gravity="0 0 0"/>
      <worldbody>
        <body>
          <joint name="lift" type="slide" axis="0 0 1" range="0 0.8"/>
          <geom type="sphere" size="0.05" mass="1"/>
        </body>
      </worldbody>
      <actuator>
        <position name="lift_motor" joint="lift" gear="{gear}" kp="1000" ctrlrange="0 0.8"/>
      </actuator>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    config = MuJoCoNodeConfig.from_dict(_base_config())
    joints, actuators = _build_joints_and_actuators(config, config.robots[0])
    simulator = MuJoCoSimulator(
        model=model,
        data=data,
        joints=joints,
        actuators=actuators,
    )
    return model, data, simulator


def test_config_parses_initial_positions() -> None:
    raw = _base_config()
    raw["initial_positions"] = {"lift": 0.2}

    config = MuJoCoNodeConfig.from_dict(raw)

    assert config.initial_positions == {"lift": 0.2}


@pytest.mark.parametrize("value", [True, "0.2", None, math.inf, math.nan])
def test_config_rejects_non_finite_or_non_numeric_initial_position(value: object) -> None:
    raw = _base_config()
    raw["initial_positions"] = {"lift": value}

    with pytest.raises(ValueError, match="initial_positions.lift"):
        MuJoCoNodeConfig.from_dict(raw)


def test_config_rejects_unknown_initial_position_joint() -> None:
    raw = _base_config()
    raw["initial_positions"] = {"unknown": 0.2}

    with pytest.raises(ValueError, match="未知关节 'unknown'"):
        MuJoCoNodeConfig.from_dict(raw)


def test_config_rejects_initial_position_for_velocity_joint() -> None:
    raw = _base_config()
    raw["joints"] = [{"name": "lift", "mode": "velocity", "unit": "meters/s"}]
    raw["initial_positions"] = {"lift": 0.2}

    with pytest.raises(ValueError, match="仅支持 position/prismatic"):
        MuJoCoNodeConfig.from_dict(raw)


def test_initial_position_sets_qpos_and_position_actuator_target() -> None:
    model, data, simulator = _make_lift_simulator()
    raw = _base_config()
    raw["initial_positions"] = {"lift": 0.2}
    config = MuJoCoNodeConfig.from_dict(raw)

    _reset_simulation_state(model, data, config, {"robot": simulator})

    assert data.qpos[0] == pytest.approx(0.2)
    assert data.ctrl[0] == pytest.approx(0.2)

    mujoco.mj_step(model, data)
    assert data.qpos[0] == pytest.approx(0.2)


def test_geared_position_actuator_target_uses_actuator_length() -> None:
    model, data, simulator = _make_lift_simulator(gear=2.0)
    raw = _base_config()
    raw["initial_positions"] = {"lift": 0.2}
    config = MuJoCoNodeConfig.from_dict(raw)

    _reset_simulation_state(model, data, config, {"robot": simulator})

    assert data.qpos[0] == pytest.approx(0.2)
    assert data.ctrl[0] == pytest.approx(0.4)


def test_model_ctrlrange_overrides_generic_unit_fallback_limit() -> None:
    _, data, simulator = _make_lift_simulator()

    simulator.set_command(
        JointCommand(
            name=["lift_motor"],
            position=[0.7],
            velocity=[],
            effort=[],
            kp=[],
            kd=[],
        )
    )

    assert data.ctrl[0] == pytest.approx(0.7)


def test_initial_position_outside_model_ctrlrange_is_rejected() -> None:
    model, data, simulator = _make_lift_simulator()
    raw = _base_config()
    raw["initial_positions"] = {"lift": 0.9}
    config = MuJoCoNodeConfig.from_dict(raw)

    with pytest.raises(ValueError, match="超出模型 ctrlrange"):
        _reset_simulation_state(model, data, config, {"robot": simulator})


def test_derived_initial_position_updates_source_joints_and_tendon_target() -> None:
    xml = """
    <mujoco>
      <option gravity="0 0 0"/>
      <worldbody>
        <body>
          <joint name="finger_left" type="slide" axis="1 0 0" range="-0.05 0.05"/>
          <geom type="sphere" size="0.01" mass="0.1"/>
        </body>
        <body>
          <joint name="finger_right" type="slide" axis="1 0 0" range="-0.05 0.05"/>
          <geom type="sphere" size="0.01" mass="0.1"/>
        </body>
      </worldbody>
      <tendon>
        <fixed name="gripper_tendon">
          <joint joint="finger_left" coef="1"/>
          <joint joint="finger_right" coef="-1"/>
        </fixed>
      </tendon>
      <actuator>
        <position name="gripper" tendon="gripper_tendon" kp="100" ctrlrange="0 0.08"/>
      </actuator>
    </mujoco>
    """
    raw: dict[str, object] = {
        "model_path": "model.xml",
        "joints": [
            {
                "name": "gripper",
                "mode": "prismatic",
                "unit": "millimeters",
                "state_from_joints": ["finger_left", "finger_right"],
                "state_expr": "diff",
                "state_scale": 1000.0,
            }
        ],
        "robots": [
            {
                "id": "robot",
                "joints": ["gripper"],
                "actuator_order": ["gripper"],
                "joint_to_actuator": {"gripper": "gripper"},
            }
        ],
        "initial_positions": {"gripper": 40.0},
    }
    config = MuJoCoNodeConfig.from_dict(raw)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    joints, actuators = _build_joints_and_actuators(config, config.robots[0])
    derived = config.derived_state_cfg["gripper"]
    simulator = MuJoCoSimulator(
        model=model,
        data=data,
        joints=joints,
        actuators=actuators,
        derived_state_cfg={
            "gripper": {
                "joint_names": derived.joint_names,
                "expr": derived.state_expr,
                "scale": derived.state_scale,
            }
        },
    )

    _reset_simulation_state(model, data, config, {"robot": simulator})

    assert data.qpos[0] == pytest.approx(0.02)
    assert data.qpos[1] == pytest.approx(-0.02)
    assert data.ctrl[0] == pytest.approx(0.04)
    assert simulator.get_state().position == pytest.approx([40.0])
