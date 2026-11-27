"""Optional native IMU and time-resolved Mid360 ray scans.

Sampling runs in the owning physics instance. No estimator or navigation policy
is involved. Each configured sensor has its own frame and output port.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, fields

import mujoco
import numpy as np
from forge_msgs import Imu, PointCloudBuffer, PointField


@dataclass(frozen=True)
class ImuConfig:
    gyro: str
    accelerometer: str
    frame: str
    output: str = "imu"
    hz: float = 200.0


@dataclass(frozen=True)
class LidarConfig:
    site: str
    frame: str
    output: str = "point_cloud"
    pattern: str = "mid360"
    hz: float = 10.0
    downsample: int = 4
    min_range: float = 0.2
    max_range: float = 30.0
    geom_groups: tuple[int, ...] = (0, 1, 2, 3, 4, 5)


def parse_sensors(raw):
    """Validate the small, explicit list of supported sensor configurations."""
    if not isinstance(raw, dict) or set(raw) - {"imu", "lidar"}:
        raise ValueError("sensors must contain only imu/lidar lists")
    result = {}
    outputs = set()
    for kind, cls in (("imu", ImuConfig), ("lidar", LidarConfig)):
        entries = raw.get(kind, [])
        if not isinstance(entries, list):
            raise ValueError(f"sensors.{kind} must be a list")
        result[kind] = []
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) - {
                f.name for f in fields(cls)
            }:
                raise ValueError(f"invalid sensors.{kind} fields")
            try:
                config = cls(**entry)
            except TypeError as error:
                raise ValueError(f"missing sensors.{kind} fields") from error
            for field in fields(cls):
                value = getattr(config, field.name)
                if field.name in {"hz", "min_range", "max_range"}:
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        or value <= 0
                    ):
                        raise ValueError(
                            f"sensors.{kind}.{field.name} must be finite and positive"
                        )
                elif field.name not in {"downsample", "geom_groups"}:
                    if not isinstance(value, str) or not value.strip():
                        raise ValueError(
                            f"sensors.{kind}.{field.name} must be nonempty text"
                        )
            if config.output in outputs:
                raise ValueError("sensor output ports must be unique")
            outputs.add(config.output)
            if kind == "lidar":
                if config.pattern != "mid360":
                    raise ValueError("only the mid360 scan pattern is supported")
                if (
                    type(config.downsample) is not int
                    or not 1 <= config.downsample <= 24000
                ):
                    raise ValueError(
                        "lidar downsample must be an integer in [1, 24000]"
                    )
                if config.min_range >= config.max_range:
                    raise ValueError("lidar min_range must be below max_range")
                if not isinstance(config.geom_groups, (tuple, list)) or any(
                    type(g) is not int or g not in range(6) for g in config.geom_groups
                ):
                    raise ValueError(
                        "lidar geom_groups must contain integers from 0 to 5"
                    )
            result[kind].append(config)
    return result


POINT_DTYPE = np.dtype(
    {
        "names": ["x", "y", "z", "time_offset_ns", "reflectivity", "line", "tag"],
        "formats": ["<f4", "<f4", "<f4", "<u4", "u1", "u1", "u1"],
        "offsets": [0, 4, 8, 12, 16, 17, 18],
        "itemsize": 20,
    }
)


class Sensors:
    def __init__(self, model, data, config):
        self.model, self.data = model, data
        self.imus = []
        self.lidars = []
        for cfg in config["imu"]:
            sensor_ids = []
            slices = []
            for name, kind in (
                (cfg.gyro, mujoco.mjtSensor.mjSENS_GYRO),
                (cfg.accelerometer, mujoco.mjtSensor.mjSENS_ACCELEROMETER),
            ):
                sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
                if (
                    sid < 0
                    or model.sensor_type[sid] != kind
                    or model.sensor_dim[sid] != 3
                ):
                    raise ValueError(f"invalid native IMU sensor: {name}")
                sensor_ids.append(sid)
                start = int(model.sensor_adr[sid])
                slices.append(slice(start, start + 3))
            if model.sensor_objid[sensor_ids[0]] != model.sensor_objid[sensor_ids[1]]:
                raise ValueError("IMU gyro and accelerometer must share a site")
            if cfg.hz > 1 / model.opt.timestep:
                raise ValueError("IMU rate exceeds physics rate")
            self.imus.append((cfg, slices))
        if config["lidar"]:
            from ._vendor.lidar.mjlidar_cpu import MjLidarCPU
            from ._vendor.lidar.scan_gen import LivoxGenerator

            for cfg in config["lidar"]:
                site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, cfg.site)
                if site < 0:
                    raise ValueError(f"lidar site not found: {cfg.site}")
                steps = 1 / (float(model.opt.timestep) * cfg.hz)
                if (
                    steps < 1
                    or abs(steps - round(steps)) > 1e-8
                    or 1e9 / cfg.hz >= 2**32
                ):
                    raise ValueError(
                        "lidar period must fit uint32 ns and an integer number of physics steps"
                    )
                caster = MjLidarCPU(
                    model,
                    cutoff_dist=cfg.max_range,
                    geomgroup=np.array(
                        [int(g in cfg.geom_groups) for g in range(6)], dtype=np.uint8
                    ),
                    bodyexclude=int(model.site_bodyid[site]),
                )
                self.lidars.append(
                    dict(
                        config=cfg,
                        site=site,
                        caster=caster,
                        pattern=LivoxGenerator("mid360"),
                        steps=round(steps),
                    )
                )
        self.reset()

    def reset(self):
        self.mapping_id = uuid.uuid4().hex
        self.next_imu = [float(self.data.time)] * len(self.imus)
        for lidar in self.lidars:
            lidar.update(step=0, chunks=[])
            lidar["pattern"].currStartIndex = 0

    def metadata(self, stamp, frame):
        return dict(
            capture_timestamp_ns=int(stamp),
            frame_id=frame,
            clock_domain="simulation",
            time_mapping_id=self.mapping_id,
            ros_time_offset_ns=0,
        )

    def sample(self):
        t = float(self.data.time)
        stamp = round(t * 1e9)
        output = []
        for index, (cfg, slices) in enumerate(self.imus):
            if t + 1e-10 >= self.next_imu[index]:
                gyro, accel = (self.data.sensordata[s] for s in slices)
                imu = Imu(
                    angular_velocity=dict(zip("xyz", map(float, gyro))),
                    linear_acceleration=dict(zip("xyz", map(float, accel))),
                )
                output.append(
                    (cfg.output, imu.to_arrow(), self.metadata(stamp, cfg.frame))
                )
                while self.next_imu[index] <= t + 1e-10:
                    self.next_imu[index] += 1 / cfg.hz
        for sensor_id, lidar in enumerate(self.lidars):
            cfg = lidar["config"]
            step, steps = lidar["step"], lidar["steps"]
            if step == 0:
                theta, phi = lidar["pattern"].sample_ray_angles(
                    downsample=cfg.downsample
                )
                lidar.update(theta=theta, phi=phi, stamp=stamp, chunks=[])
            begin = step * len(lidar["theta"]) // steps
            end = (step + 1) * len(lidar["theta"]) // steps
            if end > begin:
                pose = np.eye(4)
                pose[:3, :3] = self.data.site_xmat[lidar["site"]].reshape(3, 3)
                pose[:3, 3] = self.data.site_xpos[lidar["site"]]
                caster = lidar["caster"]
                caster.update(self.data)
                caster.trace_rays(
                    pose, lidar["theta"][begin:end], lidar["phi"][begin:end]
                )
                distances = caster.get_distances()
                valid = (
                    np.isfinite(distances)
                    & (distances >= cfg.min_range)
                    & (distances <= cfg.max_range)
                )
                xyz = caster.get_hit_points()[valid]
                points = np.zeros(len(xyz), dtype=POINT_DTYPE)
                for axis, name in enumerate("xyz"):
                    points[name] = xyz[:, axis]
                points["time_offset_ns"] = stamp - lidar["stamp"]
                points["reflectivity"] = (
                    100  # Synthetic constant, not material photometry.
                )
                lidar["chunks"].append(points)
            lidar["step"] = (step + 1) % steps
            if lidar["step"] == 0:
                points = (
                    np.concatenate(lidar["chunks"], dtype=POINT_DTYPE)
                    if lidar["chunks"]
                    else np.empty(0, dtype=POINT_DTYPE)
                )
                metadata = self.metadata(lidar["stamp"], cfg.frame)
                metadata.update(
                    lidar_id=sensor_id,
                    scan_end_timestamp_ns=stamp,
                    sampling_quantization_ns=round(
                        float(self.model.opt.timestep) * 1e9
                    ),
                    scan_pattern=cfg.pattern,
                    reflectivity_model="constant_100",
                    line_model="single_stream_0",
                )
                output.append((cfg.output, cloud_message(points).to_arrow(), metadata))
        return output


def cloud_message(points):
    types = ["float32", "float32", "float32", "uint32", "uint8", "uint8", "uint8"]
    fields = [
        PointField(
            name=name, offset=POINT_DTYPE.fields[name][1], datatype=kind, count=1
        )
        for name, kind in zip(POINT_DTYPE.names, types)
    ]
    return PointCloudBuffer(
        width=len(points),
        height=1,
        is_dense=True,
        byte_order="little_endian",
        point_stride=POINT_DTYPE.itemsize,
        row_stride=len(points) * POINT_DTYPE.itemsize,
        fields=fields,
        data=points.tobytes(),
    )
