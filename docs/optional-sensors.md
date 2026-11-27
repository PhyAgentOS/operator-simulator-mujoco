[English](optional-sensors.md) | [简体中文](optional-sensors_zh.md)

# Optional simulation capabilities

All options below are disabled by default. No navigation controller or additional
physics instance is introduced. Robot and task adaptation belong to consumers.

```yaml
sensors:
  imu:
    - gyro: gyro_sensor
      accelerometer: acceleration_sensor
      frame: imu_link
      output: imu
      hz: 200
  lidar:
    - site: lidar_mount
      frame: lidar_link
      output: point_cloud
      pattern: mid360
      hz: 10
      downsample: 4
      min_range: 0.2
      max_range: 30
      geom_groups: [0, 1, 2, 3, 4, 5]
joint_velocity_output: true
velocity_command_timeouts:
  mobile_robot:
    left_motor: 0.5
    right_motor: 0.5
state_output: false
contact_output: false
```

## Sensors

IMU and LiDAR lists are independent. Native gyro and accelerometer sensors must
share a MuJoCo site. IMU uses rad/s and m/s². Every sensor has a unique output port
and explicit frame. Calibration and fusion remain downstream responsibilities.

Mid360 rays are sampled in batches after each physics step. A scan period must be
an integer number of steps. `PointCloudBuffer` contains metre x/y/z, uint32
`time_offset_ns` from the first sample, constant synthetic reflectivity 100 and
zero line/tag. Material reflectance and firing electronics are not simulated.
`geom_groups` is the explicit ray mask; all six groups are visible by default.
Empty scans are emitted with width zero. The sensor's own body is excluded.

Metadata contains `capture_timestamp_ns`, `frame_id`, `clock_domain=simulation`,
`time_mapping_id`, `ros_time_offset_ns=0`; LiDAR also includes zero-based configured
`lidar_id`, `scan_end_timestamp_ns`, `sampling_quantization_ns`, `scan_pattern`,
`reflectivity_model` and `line_model`. `clock` emits int64 nanoseconds before the
sensor batches for that control period. Reset creates a new time mapping ID;
estimators must start a new session. Rates cannot exceed the physics rate.
Nonintegral IMU intervals are quantized to physics steps.

CPU rays and the Mid360 pattern derive from MIT-licensed MuJoCo-LiDAR. The fixed
source and compatibility changes are in `src/mujoco_sim/_vendor/lidar/SOURCE.json`.

## Velocity feedback and expiry

`joint_velocity_output` adds measured velocity to `proprio_state` for directly
mapped scalar joints. Derived and multi-DOF joints are rejected when enabled.
Position/velocity use consistent radians, metres or millimetres and per-second
units. Metadata adds `capture_timestamp_ns`, `capture_monotonic_ns` and
`episode_index`. The original empty velocity field is retained when disabled.

`velocity_command_timeouts` maps robot IDs and logical velocity actuator names to
seconds. Each updated actuator renews its own lease. Simulation-time or host-
monotonic expiry sets only that actuator's target to zero. Sparse updates to other
actuators do not renew it. Reset clears leases. This is target expiry, not an
instantaneous physical stop; braking depends on the configured model/actuator.

## Read-only observations

`state_output` emits JSON bytes on `physics_state` with `schema_version: 1`,
`episode`, `monotonic_ns`, `simulation_time`, `nq`, `nv`, `qpos`, `qvel`,
`mocap_pos` and `mocap_quat` in MuJoCo order. Consumers own recording and rendering.

`contact_output` emits JSON bytes on `contacts`: `schema_version: 1`, `episode`
and `samples`, one per physics step including empty samples. Each sample contains
`capture_timestamp_ns` and `contacts`; each contact has `geom1`, `geom2` (IDs in
the same model), signed `distance` in metres and world-space `normal`. Positive-
distance candidates are included. No floor exclusion or task verdict is applied.
Consumers must check sample continuity and the final timestamp before claiming
complete collision coverage. Configure bounded queues for the selected rates.
