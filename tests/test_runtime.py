from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from forge_msgs import JointCommand, LocomotionCommand, PoseSet
from mujoco_sim import runtime as runtime_module
from mujoco_sim.config import MuJoCoNodeConfig


class FakeNode:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self.events = events
        self.outputs: list[tuple[str, Any]] = []

    def __iter__(self):
        return iter(self.events)

    def send_output(self, output_id: str, value: Any) -> None:
        self.outputs.append((output_id, value))


def _write_model(tmp_path: Path) -> Path:
    model_path = tmp_path / "scene.xml"
    model_path.write_text(
        """
        <mujoco>
          <option gravity="0 0 0" timestep="0.01"/>
          <worldbody>
            <body name="base" mocap="true">
              <geom type="sphere" size="0.05"/>
            </body>
            <body>
              <joint name="joint" type="hinge" axis="0 0 1"/>
              <geom type="sphere" size="0.05" mass="1"/>
            </body>
          </worldbody>
          <actuator>
            <position name="motor" joint="joint" kp="10"/>
          </actuator>
        </mujoco>
        """,
        encoding="utf-8",
    )
    return model_path


def _config(
    model_path: Path,
    *,
    with_camera: bool = False,
    initial_joint_position: float | None = None,
) -> MuJoCoNodeConfig:
    raw: dict[str, Any] = {
        "model_path": str(model_path),
        "joints": [{"name": "joint", "mode": "position", "unit": "radians"}],
        "robots": [
            {
                "id": "robot",
                "joints": ["joint"],
                "actuator_order": ["motor"],
                "joint_to_actuator": {"joint": "motor"},
            }
        ],
        "control_hz": 10.0,
        "pose_targets": {"base": "base"},
        "kinematic_planar_base": {"body": "base"},
    }
    if with_camera:
        raw["image_forward"] = {"unused": "unused_image"}
    if initial_joint_position is not None:
        raw["initial_state"] = {"joint_positions": {"joint": initial_joint_position}}
    return MuJoCoNodeConfig.from_dict(raw)


def _two_actuator_config(tmp_path: Path) -> MuJoCoNodeConfig:
    model_path = tmp_path / "two_actuators.xml"
    model_path.write_text(
        """
        <mujoco>
          <option gravity="0 0 0" timestep="0.01"/>
          <worldbody>
            <body>
              <joint name="arm_joint" type="hinge" axis="0 0 1"/>
              <geom type="sphere" size="0.05" mass="1"/>
            </body>
            <body pos="0.2 0 0">
              <joint name="gripper_joint" type="slide" axis="1 0 0"/>
              <geom type="sphere" size="0.05" mass="1"/>
            </body>
          </worldbody>
          <actuator>
            <position name="arm_motor" joint="arm_joint" kp="10"/>
            <position name="gripper_motor" joint="gripper_joint" kp="10"/>
          </actuator>
        </mujoco>
        """,
        encoding="utf-8",
    )
    return MuJoCoNodeConfig.from_dict(
        {
            "model_path": str(model_path),
            "joints": [
                {"name": "arm", "mode": "position", "unit": "radians"},
                {"name": "gripper", "mode": "position", "unit": "meters"},
            ],
            "robots": [
                {
                    "id": "robot",
                    "joints": ["arm", "gripper"],
                    "actuator_order": ["arm_motor", "gripper_motor"],
                    "joint_to_actuator": {
                        "arm": "arm_motor",
                        "gripper": "gripper_motor",
                    },
                }
            ],
            "control_hz": 50.0,
        }
    )


def test_runtime_applies_initial_joint_position_on_startup_and_reset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = FakeNode([])
    monkeypatch.setattr(runtime_module, "Node", lambda: node)
    config = _config(_write_model(tmp_path), initial_joint_position=0.3)
    runtime = runtime_module.MuJoCoNodeRuntime(config)

    assert float(runtime.data.qpos[0]) == pytest.approx(0.3)
    assert float(runtime.data.ctrl[0]) == pytest.approx(0.3)

    runtime.data.qpos[0] = 0.6
    runtime.data.ctrl[0] = 0.6
    runtime._handle_input(
        {
            "type": "INPUT",
            "id": "sim_command",
            "value": json.dumps({"command": "reset_scene"}).encode("utf-8"),
        }
    )

    assert float(runtime.data.qpos[0]) == pytest.approx(0.3)
    assert float(runtime.data.ctrl[0]) == pytest.approx(0.3)


def test_runtime_accepts_named_sparse_action_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = FakeNode([])
    monkeypatch.setattr(runtime_module, "Node", lambda: node)
    runtime = runtime_module.MuJoCoNodeRuntime(_config(_write_model(tmp_path)))

    runtime._handle_input(
        {
            "type": "INPUT",
            "id": "action/arm",
            "value": JointCommand(name=["joint"], position=[0.4]).to_arrow(),
        }
    )

    assert float(runtime.data.ctrl[0]) == pytest.approx(0.4)


def test_disjoint_sparse_actions_retain_omitted_actuator_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = FakeNode([])
    monkeypatch.setattr(runtime_module, "Node", lambda: node)
    runtime = runtime_module.MuJoCoNodeRuntime(_two_actuator_config(tmp_path))

    runtime._handle_input(
        {
            "type": "INPUT",
            "id": "action/arm",
            "value": JointCommand(name=["arm"], position=[0.4]).to_arrow(),
        }
    )
    runtime._handle_input(
        {
            "type": "INPUT",
            "id": "action/gripper",
            "value": JointCommand(name=["gripper"], position=[0.02]).to_arrow(),
        }
    )

    assert list(runtime.data.ctrl) == pytest.approx([0.4, 0.02])


def test_runtime_locomotion_tick_and_reset_orchestration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = [
        {
            "type": "INPUT",
            "id": "locomotion_command",
            "value": LocomotionCommand(vx=1.0, vy=0.0, wz=0.0).to_arrow(),
        },
        {"type": "INPUT", "id": "tick"},
        {
            "type": "INPUT",
            "id": "sim_command",
            "value": json.dumps({"command": "reset_scene"}).encode("utf-8"),
        },
        {"type": "INPUT", "id": "tick"},
        {"type": "STOP"},
    ]
    node = FakeNode(events)
    monkeypatch.setattr(runtime_module, "Node", lambda: node)
    runtime = runtime_module.MuJoCoNodeRuntime(_config(_write_model(tmp_path)))

    assert runtime.run() == 0
    assert [output_id for output_id, _ in node.outputs] == [
        "proprio_state",
        "pose_set",
        "sim_status",
        "sim_status",
        "proprio_state",
        "pose_set",
        "sim_status",
    ]

    moved_pose = PoseSet.from_arrow(node.outputs[1][1]).to_poses()["base"]
    assert moved_pose.x == pytest.approx(0.1)
    assert moved_pose.y == pytest.approx(0.0)

    first_tick_status = json.loads(node.outputs[2][1])
    assert first_tick_status == {
        "status_name": "RUNNING",
        "status_code": 2,
        "current_step": 10,
        "episode_index": 0,
        "sim_time": pytest.approx(0.1),
    }

    reset_status = json.loads(node.outputs[3][1])
    assert reset_status == {
        "status_name": "RESETTING",
        "status_code": 5,
        "current_step": 0,
        "episode_index": 1,
        "sim_time": 0.0,
    }

    post_reset_pose = PoseSet.from_arrow(node.outputs[5][1]).to_poses()["base"]
    assert post_reset_pose.x == pytest.approx(0.0)
    assert post_reset_pose.y == pytest.approx(0.0)
    post_reset_tick_status = json.loads(node.outputs[6][1])
    assert post_reset_tick_status == {
        "status_name": "RUNNING",
        "status_code": 2,
        "current_step": 10,
        "episode_index": 1,
        "sim_time": pytest.approx(0.1),
    }


def test_runtime_closes_initialized_renderer_on_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRenderer:
        def __init__(self) -> None:
            self.close_count = 0

        def close(self) -> None:
            self.close_count += 1

    node = FakeNode([{"type": "STOP"}])
    renderer = FakeRenderer()
    monkeypatch.setattr(runtime_module, "Node", lambda: node)
    monkeypatch.setattr(
        runtime_module.rendering,
        "create_renderer_for_config",
        lambda model, config: renderer,
    )
    runtime = runtime_module.MuJoCoNodeRuntime(
        _config(_write_model(tmp_path), with_camera=True)
    )

    assert runtime.run() == 0
    assert renderer.close_count == 1


def test_renderer_cleanup_does_not_mask_event_loop_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingNode(FakeNode):
        def __iter__(self):
            raise RuntimeError("event loop failed")

    class FailingRenderer:
        def __init__(self) -> None:
            self.close_count = 0

        def close(self) -> None:
            self.close_count += 1
            raise RuntimeError("renderer cleanup failed")

    node = FailingNode([])
    renderer = FailingRenderer()
    monkeypatch.setattr(runtime_module, "Node", lambda: node)
    monkeypatch.setattr(
        runtime_module.rendering,
        "create_renderer_for_config",
        lambda model, config: renderer,
    )
    runtime = runtime_module.MuJoCoNodeRuntime(
        _config(_write_model(tmp_path), with_camera=True)
    )

    with pytest.raises(RuntimeError, match="event loop failed"):
        runtime.run()
    assert renderer.close_count == 1


def test_renderer_is_closed_when_dora_registration_fails(tmp_path, monkeypatch):
    from unittest.mock import Mock

    renderer = Mock()
    monkeypatch.setattr(
        runtime_module.rendering, "create_renderer_for_config", lambda *_: renderer
    )
    monkeypatch.setattr(
        runtime_module, "Node", Mock(side_effect=RuntimeError("registration failed"))
    )
    with pytest.raises(RuntimeError, match="registration failed"):
        runtime_module.MuJoCoNodeRuntime(
            _config(_write_model(tmp_path), with_camera=True)
        )
    renderer.close.assert_called_once_with()


def test_dora_error_is_a_failed_simulation(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runtime_module,
        "Node",
        lambda: FakeNode([{"type": "ERROR", "error": "transport failed"}]),
    )
    runtime = runtime_module.MuJoCoNodeRuntime(_config(_write_model(tmp_path)))
    assert runtime.run() == 1
