"""Physical contact, sensor timing and safety tests (no Dora/ROS daemon)."""

import json
import math
import time

import mujoco
import numpy as np
import pytest
from forge_msgs import Imu, LocomotionCommand, PointCloudBuffer

from mujoco_sim.config import MuJoCoNodeConfig
from mujoco_sim.joint_base import JointPlanarBase
from mujoco_sim.navigation_config import JointPlanarBaseConfig, NavigationSensorsConfig
from mujoco_sim.navigation_sensors import NavigationSensors, POINT_DTYPE


XML = """<mujoco><option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81"/>
<worldbody><geom name="wall" type="box" pos="2 0 1" size=".1 5 1"/>
<body name="robot_root" pos="0 0 .4">
<joint name="nav_base_x" type="slide" axis="1 0 0"/>
<joint name="nav_base_y" type="slide" axis="0 1 0"/>
<joint name="nav_base_yaw" type="hinge" axis="0 0 1"/>
<geom type="box" size=".3 .25 .2" mass="30"/>
<body name="sensor" pos=".31 0 .15"><site name="nav_lidar_site" size=".001"/>
<site name="nav_imu_site" size=".001"/></body></body></worldbody>
<sensor><gyro name="nav_gyro" site="nav_imu_site"/>
<accelerometer name="nav_accelerometer" site="nav_imu_site"/></sensor></mujoco>"""


def make_base(**kwargs):
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    base = JointPlanarBase(model, data, JointPlanarBaseConfig(**kwargs))
    base.reset()
    mujoco.mj_forward(model, data)
    return model, data, base


def step(model, data, base, count, command=None):
    for _ in range(count):
        if command is not None:
            base.set_command(command, data.time)
        base.advance(model.opt.timestep, data.time)
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)


def test_force_driven_base_hits_wall_without_penetrating_or_teleporting():
    model, data, base = make_base()
    step(model, data, base, 3000, LocomotionCommand(vx=0.6, vy=0, wz=0))
    # Wall near surface 1.9 m; robot half-length 0.3 m. Contact solver
    # permits sub-mm penetration, but cannot pass through to the far side.
    assert 1.595 < data.qpos[0] < 1.602
    assert abs(data.qvel[0]) < 0.003
    assert data.ncon > 0
    assert abs(base.feedback()["wheel_speed"]["wheel_left"]) < 0.05


def test_body_frame_velocity_and_native_imu_turn_sign():
    model, data, base = make_base(initial_yaw=math.pi / 2)
    step(model, data, base, 300, LocomotionCommand(vx=0.3, vy=0, wz=0))
    assert data.qpos[1] > 0.1
    assert abs(data.qpos[0]) < 0.001
    step(model, data, base, 300, LocomotionCommand(vx=0, vy=0, wz=0.4))
    gyro = data.sensor("nav_gyro").data
    assert gyro[2] == pytest.approx(0.4, abs=0.01)
    assert data.sensor("nav_accelerometer").data[2] == pytest.approx(9.81, abs=0.01)


def test_deadman_and_explicit_stop_use_measured_feedback():
    model, data, base = make_base(command_timeout_sec=0.2)
    step(model, data, base, 200, LocomotionCommand(vx=0.4, vy=0, wz=0))
    assert base.feedback()["wheel_speed"]["wheel_left"] > 1
    before = data.qpos[0]
    step(model, data, base, 600)
    assert data.qpos[0] - before < 0.2
    assert abs(base.feedback()["wheel_speed"]["wheel_left"]) < 0.005
    base.set_command(LocomotionCommand(vx=0.4, vy=0, wz=0), data.time)
    base.last_command_wall = time.monotonic() - 1
    step(model, data, base, 1)
    assert base.command.tolist() == [0, 0, 0]
    with pytest.raises(ValueError, match="vy=0"):
        base.set_command(LocomotionCommand(vx=0, vy=0.1, wz=0), data.time)


def test_scan_batches_record_actual_moving_ranges_and_sampling_times():
    model, data, base = make_base()
    sensors = NavigationSensors(model, data, NavigationSensorsConfig(downsample=4))
    # Controlled known directions make geometric range/time relation testable.
    # The real Mid360 loader and CPU raycaster still run; only the pattern in
    # this wall test is replaced. A separate test verifies real Mid360 coverage.
    sensors.pattern.sample_ray_angles = lambda downsample: (np.zeros(100), np.zeros(100))
    outputs = []
    origins = {}
    for _ in range(50):
        step(model, data, base, 1, LocomotionCommand(vx=0.3, vy=0, wz=0))
        origins[round(data.time * 1e9)] = data.site("nav_lidar_site").xpos[0]
        outputs.extend(sensors.sample())
    clouds = [(p, m) for name, p, m in outputs if name == "point_cloud"]
    assert len(clouds) == 1
    payload, metadata = clouds[0]
    cloud = PointCloudBuffer.from_arrow(payload)
    points = np.frombuffer(cloud.data, dtype=POINT_DTYPE)
    assert len(points) == 100
    times = metadata["capture_timestamp_ns"] + points["time_offset_ns"].astype(np.int64)
    assert len(np.unique(times)) == 50
    assert times.min() == 2_000_000 and times.max() == 100_000_000
    for p, stamp in zip(points, times):
        assert p["x"] == pytest.approx(1.9 - origins[int(stamp)], abs=1e-6)
    assert points["x"][0] > points["x"][-1]
    imus = [(Imu.from_arrow(p), m) for name, p, m in outputs if name == "imu"]
    assert 20 <= len(imus) <= 21
    assert len({m["time_mapping_id"] for _, m in imus} | {metadata["time_mapping_id"]}) == 1
    assert all(m["clock_domain"] == "simulation" and m["ros_time_offset_ns"] == 0 for _, m in imus)
    assert max(imu.linear_acceleration.x for imu, _ in imus) > 0.1
    old_mapping = sensors.mapping_id
    mujoco.mj_resetData(model, data)
    sensors.reset()
    assert sensors.mapping_id != old_mapping


def test_real_mid360_pattern_produces_finite_three_dimensional_returns():
    model, data, base = make_base()
    sensors = NavigationSensors(model, data, NavigationSensorsConfig())
    outputs = []
    for _ in range(50):
        step(model, data, base, 1)
        outputs.extend(sensors.sample())
    cloud = PointCloudBuffer.from_arrow(next(p for name, p, _ in outputs if name == "point_cloud"))
    points = np.frombuffer(cloud.data, dtype=POINT_DTYPE)
    assert len(points) > 100
    assert np.ptp(points["y"]) > 1 and np.ptp(points["z"]) > 0.3
    assert all(np.isfinite(points[a]).all() for a in "xyz")


@pytest.mark.parametrize('source_labels',[False,True])
def test_rigid_auxiliary_returns_use_primary_calibration_and_actual_batch_times(source_labels):
    xml = XML.replace('</body></worldbody>',
        '<body name="rear_sensor" pos="-.31 0 .35"><site name="rear_lidar" size=".001"/></body></body></worldbody>')
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    base = JointPlanarBase(model, data, JointPlanarBaseConfig())
    base.reset();mujoco.mj_forward(model, data)
    sensors = NavigationSensors(model, data, NavigationSensorsConfig(auxiliary_lidar_site='rear_lidar',sensor_identity_reflectivity=source_labels))
    assert sensors.lidars[1][2][:3, 3] == pytest.approx([-.62, 0, .2])
    sensors.pattern.sample_ray_angles = lambda downsample: (np.zeros(100), np.zeros(100))
    outputs = []
    origins = {}
    for _ in range(50):
        step(model, data, base, 1, LocomotionCommand(vx=.3, vy=0, wz=0))
        origins[round(data.time*1e9)] = data.site('nav_lidar_site').xpos[0]
        outputs.extend(sensors.sample())
    cloud, metadata = next((p, m) for name, p, m in outputs if name == 'point_cloud')
    points = np.frombuffer(PointCloudBuffer.from_arrow(cloud).data, dtype=POINT_DTYPE)
    front = points[points['line'] == 0]
    rear = points[points['line'] == 1]
    assert len(front) == 100
    assert len(rear) == 100
    for point in points:
        stamp = metadata['capture_timestamp_ns'] + int(point['time_offset_ns'])
        assert point['x'] == pytest.approx(1.9-origins[stamp],abs=1e-6)
    assert rear['z'] == pytest.approx(np.full(100,.2),abs=1e-6)
    assert metadata['sensor_sites'] == ['nav_lidar_site', 'rear_lidar']
    assert metadata['line_model'] == 'sensor_stream_index'
    assert np.allclose(json.loads(metadata['auxiliary_to_primary_json'])[0],sensors.lidars[1][2])
    assert set(points['line']) == {0, 1}
    assert set(front['reflectivity'])=={100}
    assert set(rear['reflectivity'])==({101} if source_labels else {100})
    assert metadata['reflectivity_model']==('stream_identity_100_plus_index' if source_labels else 'constant_100')
    assert points['time_offset_ns'].max() == 98_000_000


def test_auxiliary_lidar_on_articulated_mount_is_rejected():
    xml = XML.replace('</body></worldbody>',
        '<body name="moving_sensor" pos="-.5 0 .15"><joint type="hinge" axis="0 0 1"/>'
        '<geom size=".01" mass=".1"/><site name="moving_lidar" size=".001"/></body></body></worldbody>')
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    with pytest.raises(ValueError, match='rigidly mounted'):
        NavigationSensors(model, data, NavigationSensorsConfig(auxiliary_lidar_site='moving_lidar'))


def test_auxiliary_self_occlusion_never_becomes_a_distant_free_return():
    xml = XML.replace('</body></worldbody>',
        '<body name="rear_sensor" pos="-.31 0 .15"><site name="rear_lidar" size=".001"/></body></body></worldbody>')
    model = mujoco.MjModel.from_xml_string(xml);data = mujoco.MjData(model)
    base = JointPlanarBase(model,data,JointPlanarBaseConfig());base.reset();mujoco.mj_forward(model,data)
    sensors = NavigationSensors(model,data,NavigationSensorsConfig(auxiliary_lidar_site='rear_lidar'))
    sensors.pattern.sample_ray_angles = lambda downsample:(np.zeros(100),np.zeros(100))
    outputs=[]
    for _ in range(50):
        step(model,data,base,1);outputs.extend(sensors.sample())
    cloud=PointCloudBuffer.from_arrow(next(p for name,p,_ in outputs if name=='point_cloud'))
    points=np.frombuffer(cloud.data,dtype=POINT_DTYPE)
    # Rear rays first hit the real body at 0.01m, below the sensor's min range.
    # There must be no invented return to the far wall behind that occluder.
    assert len(points)==100
    assert set(points['line'])=={0}


@pytest.mark.parametrize("bad", [{"downsample": 1.5}, {"imu_hz": float("nan")}, {"min_range": 40}, {"typo": 1}])
def test_invalid_sensor_configuration_fails_early(bad):
    with pytest.raises(ValueError):
        MuJoCoNodeConfig.from_dict({"model_path": "unused.xml", "navigation_sensors": bad})


def test_runtime_outputs_clock_before_sensors_and_requires_fresh_command(tmp_path, monkeypatch):
    from mujoco_sim import runtime
    path = tmp_path / "scene.xml"
    path.write_text(XML)
    emitted = []
    class FakeNode:
        def send_output(self, name, payload, metadata=None):
            emitted.append((name, payload, metadata))
    monkeypatch.setattr(runtime, "Node", FakeNode)
    config = MuJoCoNodeConfig.from_dict({"model_path": str(path), "joint_planar_base": {}, "navigation_sensors": {}})
    sim = runtime.MuJoCoNodeRuntime(config)
    command = LocomotionCommand(vx=0.4, vy=0, wz=0).to_arrow()
    sim._handle_locomotion({"value": command, "metadata": {}})
    assert sim.planar_base.command[0] == 0
    sim._handle_locomotion({"value": command, "metadata": {"navigation_authorized": True, "issued_monotonic_ns": time.monotonic_ns()}})
    assert sim.planar_base.command[0] == 0.4
    for _ in range(5):
        sim._handle_tick()
    latest_clock = 0
    seen = set()
    for name, payload, metadata in emitted:
        seen.add(name)
        if name == "clock":
            latest_clock = payload[0].as_py()
        if name in ("imu", "point_cloud"):
            assert 0 < metadata["capture_timestamp_ns"] <= latest_clock
        if name == "base_feedback":
            feedback = json.loads(payload[0].as_py())
            assert set(feedback["wheel_speed"]) == {"wheel_left", "wheel_right"}
            assert feedback["feedback_source"].startswith("actual_planar")
    assert {"clock", "imu", "point_cloud", "base_feedback"} <= seen
    clocks = [payload[0].as_py() for name, payload, _ in emitted if name == "clock"]
    assert clocks == [20_000_000, 40_000_000, 60_000_000, 80_000_000, 100_000_000]
    # Lower clock publication rate must not lower physical sampling rates.
    imu_stamps = [meta["capture_timestamp_ns"] for name, _, meta in emitted if name == "imu"]
    assert len(imu_stamps) == 21
    assert imu_stamps[0] == 2_000_000 and imu_stamps[-1] == 100_000_000
