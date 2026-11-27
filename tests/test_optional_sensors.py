"""Public sensor/control contracts on a model independent of the H1 assets."""

import json

import numpy as np
import pytest
from forge_msgs import Imu, JointCommand, JointState, PointCloudBuffer

from mujoco_sim import runtime as runtime_module
from mujoco_sim.config import MuJoCoNodeConfig

MODEL = """
<mujoco model="small-mobile-test">
  <option timestep=".002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="ground-other-name" type="plane" size="10 10 .1"/>
    <geom name="barrier" type="box" pos="1 0 .3" size=".1 2 .3" group="4"/>
    <body name="cart" pos="0 0 .2">
      <joint name="travel" type="slide" axis="1 0 0"/>
      <joint name="turn" type="hinge" axis="0 0 1"/>
      <geom type="box" size=".15 .15 .15" mass="4"/>
      <site name="sensor" pos="0 0 .1"/>
      <site name="rear" pos="-.1 0 .1" quat="0 0 0 1"/>
    </body>
    <body pos="0 2 .5"><joint name="arm"/><geom size=".1"/></body>
  </worldbody>
  <actuator>
    <velocity name="drive" joint="travel" kv="80" ctrlrange="-1 1"/>
    <velocity name="steer" joint="turn" kv="10" ctrlrange="-1 1"/>
    <position name="hold" joint="arm" kp="10"/>
  </actuator>
  <sensor><gyro name="angular" site="sensor"/><accelerometer name="linear" site="sensor"/></sensor>
</mujoco>
"""


class OutputNode:
    def __init__(self):
        self.outputs = []

    def send_output(self, port, value, metadata=None):
        self.outputs.append((port, value, metadata or {}))


def make_runtime(tmp_path, monkeypatch, **options):
    path = tmp_path / "small.xml"
    path.write_text(MODEL)
    node = OutputNode()
    monkeypatch.setattr(runtime_module, "Node", lambda: node)
    raw = dict(
        model_path=str(path),
        control_hz=50,
        joints=[
            dict(name="travel", mode="velocity", unit="meters/s"),
            dict(name="turn", mode="velocity", unit="radians/s"),
            dict(name="arm"),
        ],
        robots=[
            dict(
                id="cart",
                joints=["travel", "turn", "arm"],
                actuator_order=["drive", "steer", "hold"],
                joint_to_actuator=dict(travel="drive", turn="steer", arm="hold"),
            )
        ],
    )
    raw.update(options)
    return runtime_module.MuJoCoNodeRuntime(MuJoCoNodeConfig.from_dict(raw)), node


def command(runtime, **values):
    runtime._handle_action(dict(value=JointCommand(**values).to_arrow()))


def test_opt_in_watchdog_preserves_arm_and_sparse_targets(tmp_path, monkeypatch):
    runtime, node = make_runtime(
        tmp_path,
        monkeypatch,
        joint_velocity_output=True,
        velocity_command_timeouts={"cart": {"drive": 0.1, "steer": 0.1}},
    )
    command(runtime, name=["arm"], position=[0.4])
    command(runtime, name=["travel", "turn"], velocity=[0.4, 0.2])
    for _ in range(3):
        runtime._handle_tick()
    measured = JointState.from_arrow(
        next(v for p, v, _ in reversed(node.outputs) if p == "proprio_state")
    )
    assert measured.velocity[0] > 0
    assert measured.velocity[1] > 0
    assert measured.name == ["travel", "turn", "arm"]
    for _ in range(4):
        command(runtime, name=["arm"], position=[0.4])
        runtime._handle_tick()
    assert runtime.data.ctrl.tolist() == [0.0, 0.0, 0.4]
    command(runtime, name=["travel"], velocity=[0.3])
    assert runtime.data.ctrl.tolist() == [0.3, 0.0, 0.4]
    runtime._handle_sim_command(dict(value=b'{"command":"reset_scene"}'))
    assert runtime.data.ctrl.tolist() == [0.0, 0.0, 0.0]


def test_wall_clock_expiry_and_position_watchdog_rejection(tmp_path, monkeypatch):
    from mujoco_sim import simulator

    now = [20.0]
    monkeypatch.setattr(simulator.time, "monotonic", lambda: now[0])
    runtime, _ = make_runtime(
        tmp_path, monkeypatch, velocity_command_timeouts={"cart": {"drive": 0.1}}
    )
    command(runtime, name=["travel"], velocity=[0.4])
    now[0] += 0.2
    runtime._step_physics()
    assert runtime.data.ctrl[0] == 0
    for _ in range(50):
        runtime._step_physics()
    assert abs(runtime.data.qvel[0]) < 0.01
    with pytest.raises(ValueError, match="velocity actuator"):
        make_runtime(
            tmp_path, monkeypatch, velocity_command_timeouts={"cart": {"hold": 0.1}}
        )


IMU = {"gyro": "angular", "accelerometer": "linear", "frame": "body_imu"}
LIDAR = {"site": "sensor", "frame": "front_scan", "downsample": 24}


@pytest.mark.parametrize(
    "sensors", [{"imu": [IMU]}, {"lidar": [LIDAR]}, {"imu": [IMU], "lidar": [LIDAR]}]
)
def test_independent_sensor_sampling_clock_and_reset(tmp_path, monkeypatch, sensors):
    runtime, node = make_runtime(tmp_path, monkeypatch, sensors=sensors)
    for _ in range(5):
        runtime._handle_tick()
    ports = [p for p, _, _ in node.outputs]
    if "imu" in sensors:
        imus = [(v, m) for p, v, m in node.outputs if p == "imu"]
        assert len(imus) == 21
        assert Imu.from_arrow(imus[0][0]).linear_acceleration.z == pytest.approx(9.81)
    if "lidar" in sensors:
        v, metadata = next((v, m) for p, v, m in node.outputs if p == "point_cloud")
        cloud = PointCloudBuffer.from_arrow(v)
        assert cloud.width > 0
        offsets = np.frombuffer(
            cloud.data,
            dtype=np.dtype(
                {"names": ["time"], "formats": ["<u4"], "offsets": [12], "itemsize": 20}
            ),
        )["time"]
        assert len(set(offsets)) > 1
        assert max(offsets) <= 98_000_000
        assert metadata["frame_id"] == "front_scan"
    first = next(
        m["time_mapping_id"] for _, _, m in node.outputs if "time_mapping_id" in m
    )
    assert ports.index("clock") < min(
        ports.index(p) for p in ("imu", "point_cloud") if p in ports
    )
    runtime._handle_sim_command(dict(value=b'{"command":"reset_scene"}'))
    node.outputs.clear()
    for _ in range(5):
        runtime._handle_tick()
    second = next(
        m["time_mapping_id"] for _, _, m in node.outputs if "time_mapping_id" in m
    )
    assert first != second


def test_multiple_lidar_frames_and_explicit_group_mask(tmp_path, monkeypatch):
    runtime, node = make_runtime(
        tmp_path,
        monkeypatch,
        sensors={
            "lidar": [
                {**LIDAR, "geom_groups": [4]},
                {
                    **LIDAR,
                    "site": "rear",
                    "frame": "rear_scan",
                    "output": "rear_points",
                    "geom_groups": [],
                },
            ]
        },
    )
    for _ in range(5):
        runtime._handle_tick()
    front = next((v, m) for p, v, m in node.outputs if p == "point_cloud")
    rear = next((v, m) for p, v, m in node.outputs if p == "rear_points")
    assert PointCloudBuffer.from_arrow(front[0]).width > 0
    assert PointCloudBuffer.from_arrow(rear[0]).width == 0
    assert rear[1]["frame_id"] == "rear_scan"
    assert front[1]["capture_timestamp_ns"] == rear[1]["capture_timestamp_ns"]


def test_contact_observation_covers_every_step_and_wall_blocks_motion(
    tmp_path, monkeypatch
):
    runtime, node = make_runtime(
        tmp_path, monkeypatch, contact_output=True, state_output=True
    )
    command(runtime, name=["travel"], velocity=[0.6])
    for _ in range(150):
        runtime._handle_tick()
    traces = [json.loads(v) for p, v, _ in node.outputs if p == "contacts"]
    assert sum(len(t["samples"]) for t in traces) == 1500
    assert any(s["contacts"] for t in traces for s in t["samples"])
    assert runtime.data.qpos[0] < 0.77
    state = next(
        json.loads(v) for p, v, _ in reversed(node.outputs) if p == "physics_state"
    )
    assert state["qpos"] == runtime.data.qpos.tolist()
    assert state["qvel"] == runtime.data.qvel.tolist()


@pytest.mark.parametrize(
    "options",
    [
        {"sensors": {"lidar": [{**LIDAR, "pattern": "invented"}]}},
        {"sensors": {"imu": [{**IMU, "output": "sim_status"}]}},
        {"sensors": {"imu": [{**IMU, "hz": float("nan")}]}},
        {"state_output": "true"},
    ],
)
def test_invalid_new_configuration_rejected(options):
    with pytest.raises(ValueError):
        MuJoCoNodeConfig.from_dict(dict(model_path="unused.xml", **options))


@pytest.mark.parametrize(
    "options",
    [{"sensors": {"imu": [IMU]}}, {"state_output": True, "contact_output": True}],
)
def test_observation_without_robot_commands(tmp_path, monkeypatch, options):
    runtime, node = make_runtime(tmp_path, monkeypatch, joints=[], robots=[], **options)
    runtime._handle_tick()
    ports = {port for port, _, _ in node.outputs}
    assert "proprio_state" not in ports
    assert "sim_status" in ports
    assert "imu" in ports if "sensors" in options else "physics_state" in ports


def test_velocity_units_and_names_match_positions(tmp_path, monkeypatch):
    runtime, node = make_runtime(
        tmp_path,
        monkeypatch,
        joint_velocity_output=True,
        joints=[
            dict(name="travel", mode="velocity", unit="millimeters/s"),
            dict(name="turn", mode="velocity", unit="radians/s"),
            dict(name="arm"),
        ],
    )
    command(runtime, name=["travel"], velocity=[400.0])
    for _ in range(30):
        runtime._handle_tick()
    state = JointState.from_arrow(
        next(v for p, v, _ in reversed(node.outputs) if p == "proprio_state")
    )
    assert state.position[0] == pytest.approx(runtime.data.qpos[0] * 1000)
    assert state.velocity[0] == pytest.approx(runtime.data.qvel[0] * 1000)
    assert state.velocity[0] > 390


@pytest.mark.parametrize(
    "port,option",
    [
        ("clock", {"sensors": {"imu": [IMU]}}),
        ("physics_state", {"state_output": True}),
        ("contacts", {"contact_output": True}),
    ],
)
def test_new_outputs_cannot_share_an_existing_camera_port(port, option):
    original = dict(model_path="unused.xml", image_forward={"camera": port})
    MuJoCoNodeConfig.from_dict(original)  # Existing camera configurations stay valid.
    with pytest.raises(ValueError, match="camera output collides"):
        MuJoCoNodeConfig.from_dict({**original, **option})
