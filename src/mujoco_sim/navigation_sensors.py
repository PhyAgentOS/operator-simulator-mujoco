"""Time-resolved Mid360-pattern CPU scans and native MuJoCo IMU.

One batch is physically raycast after each physics step. Points in that batch
share its real sampling timestamp (2 ms quantization at 500 Hz), rather than
inventing unique acquisition times for a single instantaneous whole-frame scan.
"""

from __future__ import annotations

import uuid
import json

import mujoco
import numpy as np
from forge_msgs import Imu, PointCloudBuffer, PointField

from .navigation_config import NavigationSensorsConfig


POINT_DTYPE = np.dtype({
    "names": ["x", "y", "z", "time_offset_ns", "reflectivity", "line", "tag"],
    "formats": ["<f4", "<f4", "<f4", "<u4", "u1", "u1", "u1"],
    "offsets": [0, 4, 8, 12, 16, 17, 18], "itemsize": 20,
})


class NavigationSensors:
    def __init__(self, model, data, config: NavigationSensorsConfig):
        # Optional capability: arm-only profiles never load the licensed LiDAR support.
        from ._vendor.lidar.mjlidar_cpu import MjLidarCPU
        from ._vendor.lidar.scan_gen import LivoxGenerator

        self.model, self.data, self.config = model, data, config
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, config.lidar_site)
        self.imu_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, config.imu_site)
        if min(self.site_id, self.imu_site_id) < 0:
            raise ValueError("navigation_sensors lidar/IMU sites must exist")
        self.gyro = self._sensor(config.gyro_sensor, mujoco.mjtSensor.mjSENS_GYRO)
        self.accel = self._sensor(config.accelerometer_sensor, mujoco.mjtSensor.mjSENS_ACCELEROMETER)
        dt = float(model.opt.timestep)
        steps = 1.0 / (dt * config.scan_hz)
        if abs(steps - round(steps)) > 1e-8 or steps < 1:
            raise ValueError("scan period must be an integer number of physics timesteps")
        if config.imu_hz > 1.0 / dt:
            raise ValueError("IMU rate cannot exceed physics rate")
        self.steps_per_scan = round(steps)
        # Group 4 is presentation-only decals; group 5 is invisible collision proxies.
        # Sensor housing is excluded, while robot arms remain real occluders.
        self.raycaster = MjLidarCPU(model, cutoff_dist=config.max_range,
            geomgroup=np.array([1, 1, 1, 1, 0, 0], dtype=np.uint8),
            bodyexclude=int(model.site_bodyid[self.site_id]))
        self.lidars = [(self.site_id, self.raycaster, np.eye(4))]
        if config.auxiliary_lidar_site:
            auxiliary = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, config.auxiliary_lidar_site)
            if auxiliary < 0 or auxiliary == self.site_id:
                raise ValueError("auxiliary lidar must be a distinct existing site")
            root, primary_mount = self._rigid_mount(self.site_id)
            other_root, auxiliary_mount = self._rigid_mount(auxiliary)
            if root != other_root:
                raise ValueError("auxiliary lidar must be rigidly mounted with primary lidar")
            # Asset calibration only: no robot/world pose is used for this TF.
            calibration = np.linalg.inv(primary_mount) @ auxiliary_mount
            caster = MjLidarCPU(model, cutoff_dist=config.max_range,
                geomgroup=np.array([1, 1, 1, 1, 0, 0], dtype=np.uint8),
                bodyexclude=int(model.site_bodyid[auxiliary]))
            self.lidars.append((auxiliary, caster, calibration))
        self.pattern = LivoxGenerator("mid360")
        self.reset()

    def _rigid_mount(self, site):
        """Return fixed calibration to the first joint-bearing ancestor."""
        def transform(position, quaternion):
            matrix = np.eye(4)
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, quaternion)
            matrix[:3, :3] = rotation.reshape(3, 3)
            matrix[:3, 3] = position
            return matrix
        body = int(self.model.site_bodyid[site])
        matrix = transform(self.model.site_pos[site], self.model.site_quat[site])
        while body and not self.model.body_jntnum[body]:
            matrix = transform(self.model.body_pos[body], self.model.body_quat[body]) @ matrix
            body = int(self.model.body_parentid[body])
        return body, matrix

    def _sensor(self, name, kind):
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sid < 0 or self.model.sensor_type[sid] != kind or self.model.sensor_dim[sid] != 3:
            raise ValueError(f"navigation_sensors invalid native sensor: {name}")
        if self.model.sensor_objid[sid] != self.imu_site_id:
            raise ValueError(f"sensor {name} must be attached to configured imu_site")
        address = int(self.model.sensor_adr[sid])
        return slice(address, address + 3)

    def reset(self):
        self.mapping_id = f"{self.config.time_mapping_id}-{uuid.uuid4().hex}"
        self.pattern.currStartIndex = 0
        self.frame_step = 0
        self.chunks = []
        self.next_imu_time = float(self.data.time)
        self.last_sample_time = float(self.data.time)

    def metadata(self, stamp_ns, frame):
        return {
            "capture_timestamp_ns": int(stamp_ns), "frame_id": frame,
            "clock_domain": "simulation", "time_mapping_id": self.mapping_id,
            "ros_time_offset_ns": 0,
        }

    def sample(self):
        """Return (output_id, Forge Arrow batch, Dora metadata) after mj_forward."""
        t = float(self.data.time)
        if t <= self.last_sample_time:
            raise ValueError("sensor acquisition time did not advance; reset sensor session explicitly")
        self.last_sample_time = t
        stamp = round(t * 1e9)
        outputs = []
        if t + 1e-10 >= self.next_imu_time:
            gyro, accel = self.data.sensordata[self.gyro], self.data.sensordata[self.accel]
            imu = Imu(angular_velocity=dict(zip("xyz", map(float, gyro))),
                      linear_acceleration=dict(zip("xyz", map(float, accel))))
            outputs.append(("imu", imu.to_arrow(), self.metadata(stamp, self.config.imu_frame)))
            while self.next_imu_time <= t + 1e-10:
                self.next_imu_time += 1.0 / self.config.imu_hz
        if self.frame_step == 0:
            self.theta, self.phi = self.pattern.sample_ray_angles(downsample=self.config.downsample)
            self.frame_stamp = stamp
            self.chunks = []
        begin = self.frame_step * len(self.theta) // self.steps_per_scan
        end = (self.frame_step + 1) * len(self.theta) // self.steps_per_scan
        active_lidars = self.lidars if end > begin else []
        for stream, (site_id, caster, calibration) in enumerate(active_lidars):
            pose = np.eye(4)
            pose[:3, :3] = self.data.site_xmat[site_id].reshape(3, 3)
            pose[:3, 3] = self.data.site_xpos[site_id]
            caster.update(self.data)
            caster.trace_rays(pose, self.theta[begin:end], self.phi[begin:end])
            distances = caster.get_distances()
            xyz = caster.get_hit_points()
            if stream:
                xyz = xyz @ calibration[:3, :3].T + calibration[:3, 3]
            valid = np.isfinite(distances) & (distances >= self.config.min_range) & (distances <= self.config.max_range)
            points = np.zeros(int(valid.sum()), dtype=POINT_DTYPE)
            for axis, name in enumerate("xyz"):
                points[name] = xyz[valid, axis]
            points["time_offset_ns"] = stamp - self.frame_stamp
            # Reflectivity is synthetic, not material photometry. The upstream
            # pattern has no physical channel IDs; stream is a sensor index,
            # rather than a claim to reproduce hardware firing order.
            # Reflectivity was already synthetic. Opt-in stream identity lets
            # the mapping adapter split the real deskewed returns downstream;
            # it is explicitly not simulated material photometry.
            points["reflectivity"] = 100+stream if self.config.sensor_identity_reflectivity else 100
            points["line"] = stream
            self.chunks.append(points)
        self.frame_step += 1
        if self.frame_step == self.steps_per_scan:
            self.frame_step = 0
            # np.concatenate otherwise repacks structured records and drops
            # trailing padding, invalidating the declared Arrow point stride.
            points = np.concatenate(self.chunks, dtype=POINT_DTYPE) if self.chunks else np.empty(0, dtype=POINT_DTYPE)
            if len(points):
                cloud = self._cloud(points)
                metadata = self.metadata(self.frame_stamp, self.config.lidar_frame)
                metadata.update(lidar_id=0, scan_end_timestamp_ns=stamp,
                    sampling_quantization_ns=round(float(self.model.opt.timestep) * 1e9),
                    scan_pattern="mid360", reflectivity_model=("stream_identity_100_plus_index"
                        if self.config.sensor_identity_reflectivity else "constant_100"),
                    line_model="sensor_stream_index" if len(self.lidars) > 1 else "single_stream_0",
                    sensor_sites=[self.model.site(site).name for site, _, _ in self.lidars],
                    # Dora coerces nested metadata arrays to strings. Declare
                    # this as explicit JSON text so wire type is unambiguous.
                    auxiliary_to_primary_json=json.dumps(
                        [calibration.tolist() for _, _, calibration in self.lidars[1:]],
                        separators=(",", ":")))
                outputs.append(("point_cloud", cloud.to_arrow(), metadata))
        return outputs

    @staticmethod
    def _cloud(points):
        types = ["float32", "float32", "float32", "uint32", "uint8", "uint8", "uint8"]
        fields = [PointField(name=name, offset=POINT_DTYPE.fields[name][1], datatype=dtype, count=1)
                  for name, dtype in zip(POINT_DTYPE.names, types)]
        return PointCloudBuffer(width=len(points), height=1, is_dense=True,
            byte_order="little_endian", point_stride=POINT_DTYPE.itemsize,
            row_stride=len(points) * POINT_DTYPE.itemsize, fields=fields, data=points.tobytes())
