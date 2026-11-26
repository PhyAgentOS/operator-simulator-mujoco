"""Opt-in navigation simulation configuration, independent of arm profiles."""

from dataclasses import dataclass, fields
import math


@dataclass(frozen=True)
class JointPlanarBaseConfig:
    body: str = "robot_root"
    x_joint: str = "nav_base_x"
    y_joint: str = "nav_base_y"
    yaw_joint: str = "nav_base_yaw"
    initial_yaw: float = 0.0
    command_timeout_sec: float = 0.5
    max_linear_speed: float = 0.6
    max_angular_speed: float = 0.8
    max_acceleration: float = 0.8
    max_angular_acceleration: float = 1.5
    velocity_gain: float = 20.0
    wheel_radius: float = 0.1
    wheel_separation: float = 0.4


@dataclass(frozen=True)
class NavigationSensorsConfig:
    lidar_site: str = "nav_lidar_site"
    # Optional synchronized rigid sensor; all returns use the primary frame.
    auxiliary_lidar_site: str = ""
    # Simulation-only provenance label, retained by FAST-LIO intensity output.
    sensor_identity_reflectivity: bool = False
    imu_site: str = "nav_imu_site"
    gyro_sensor: str = "nav_gyro"
    accelerometer_sensor: str = "nav_accelerometer"
    lidar_frame: str = "livox_frame"
    imu_frame: str = "imu_frame"
    scan_hz: float = 10.0
    imu_hz: float = 200.0
    downsample: int = 4
    min_range: float = 0.2
    max_range: float = 30.0
    # New reset creates a new mapping ID. Algorithms must restart on that change.
    time_mapping_id: str = "mujoco"


def parse_navigation_config(raw, cls):
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"{cls.__name__} must be a mapping")
    unknown = set(raw) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown fields {sorted(unknown)}")
    result = cls(**raw)
    for f in fields(cls):
        value = getattr(result, f.name)
        if isinstance(f.default, bool):
            if not isinstance(value,bool):raise ValueError(f"{cls.__name__}.{f.name} must be boolean")
            continue
        if f.name == "auxiliary_lidar_site" and value == "":
            continue
        if isinstance(f.default, str):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{cls.__name__}.{f.name} must be nonempty")
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{cls.__name__}.{f.name} must be finite numeric")
        elif f.name != "initial_yaw" and value <= 0:
            raise ValueError(f"{cls.__name__}.{f.name} must be positive")
    if cls is NavigationSensorsConfig:
        if not isinstance(result.downsample, int) or result.downsample > 24000:
            raise ValueError("navigation_sensors.downsample must be an integer in [1,24000]")
        if result.min_range >= result.max_range:
            raise ValueError("navigation_sensors.min_range must be below max_range")
    return result
