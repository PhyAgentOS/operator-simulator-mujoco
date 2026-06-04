# MuJoCo 仿真节点

MuJoCo 节点属于 robot+sensor 层的仿真实现：接收 `JointCommand`，输出 `JointState`、图像和可选 `PoseSet`。它不接收 `PolicyCommand`，也不接收 `sim_control`；仿真是否运行由上游 policy/dataflow 决定是否持续发送 `tick` 和 `action`。

## 时间与频率

- `tick` 表示一个控制周期，例如 `dora/timer/millis/20` 对应 50Hz。
- 每个 `tick` 内执行 `round(control_period_sec / model.opt.timestep)` 个 MuJoCo 物理子步。
- 每个 `tick` 输出一次 `proprio_state`。
- 相机按仿真时间节流，`image_forward` 中可为每个输出配置 `fps`。

## 输入 / 输出

输入：

- `tick`: 控制周期定时器。
- `action`: `forge_msgs.JointCommand`，按配置映射到 MuJoCo actuator。
- `{robot_id}/master_state`: `forge_msgs.JointState`，可直接驱动对应机器人。

输出：

- `proprio_state`: `forge_msgs.JointState`，全局关节状态。
- `pose_set`: `forge_msgs.PoseSet`，可选，平面位姿通过 `Pose.from_xy_yaw` 表示。
- `image/*`: raw 图像使用 `forge_msgs.Image`，jpeg/png 使用 `forge_msgs.CompressedImage`。
- `depth/*`: 深度图使用 `forge_msgs.Image`，固定 `encoding=32FC1`，数据类型 `float32`，单位 meters。
- `camera_info/*`: JSON bytes，包含内参、深度单位和当前世界系外参。
- `sim_status`: 仿真进度 JSON，仅反映本节点内部 step/time 状态，不作为外部控制接口。

## 配置

复制 `config.example.yaml` 为 `config.yaml` 并修改。配置来源：

- `--config /path/to/config.yaml`
- 环境变量 `MUJOCO_NODE_CONFIG`
- dataflow 节点 `config` 字段

### 派生 actuator 状态

例如 gripper 由多个 MuJoCo joint 的 qpos 合成：

```yaml
joints:
  - name: gripper
    mode: prismatic
    unit: millimeters
    state_from_joints: ["joint7", "joint8"]
    state_expr: diff
    state_scale: 1000.0
```

### 位姿输出

配置 `pose_targets` 后，节点输出单行 `PoseSet`：

```yaml
pose_targets:
  robot_0: "base"
  goal: "nav_goal_marker"
```

如果不写 `pose_targets`，节点会尝试按 `robots[].prefix` 自动识别每个 robot 的根 body。

### 图像与深度输出

`image_forward` 的 key 是 MuJoCo XML 中的 camera 名称，value 可以是单个输出配置，也可以是输出列表。`type` 缺省为普通 `image`；深度图写 `type: depth`，用于 mask 区域反投影 3D 点，因此输出契约固定为米制 float32：

```yaml
image_forward:
  top:
    - { output: "image/top", fps: 30, width: 640, height: 480 }
    - { output: "depth/top", fps: 30, width: 640, height: 480, type: depth }
```

`type: depth` 的输出 payload 为 `forge_msgs.Image`：

- `encoding`: `32FC1`
- `dtype`: `float32`
- `unit`: meters
- `shape`: `(height, width)`

真机 RealSense 节点建议把原始 `z16` 深度在 camera 节点内根据 `depth_scale` 转成 meters 后，也输出同样的 `32FC1` 契约，避免 policy 侧再关心相机厂商和 scale。

### 相机信息输出

配置 `camera_info_forward` 后，节点会按 MuJoCo camera 名称输出 JSON bytes。内参优先使用 XML 中的 `focal` / `sensorsize` / `resolution` 计算；如果模型只配置了 `fovy`，则按 pinhole camera 和输出分辨率推导。动态 wrist camera 的外参会随 `tick` 更新：

```yaml
camera_info_forward:
  top:
    output: "camera_info/top"
    fps: 30
    width: 640
    height: 480
    depth_min_m: 0.02
    depth_max_m: 1.5
    depth_unit_scale: 1.0
```

payload 同时提供通用字段和兼容 `sam3_policy.yaml` 的字段名：

- `camera_width` / `camera_height`
- `camera_fx` / `camera_fy` / `camera_cx` / `camera_cy`
- `depth_min_m` / `depth_max_m` / `depth_unit_scale`
- `K` / `D` / `distortion_model`
- `position_world` / `quaternion_wxyz_world` / `rotation_matrix_world` / `T_world_camera`

MuJoCo 当前深度图已经是 `32FC1` meters，因此 `depth_unit_scale` 默认应为 `1.0`。只有真机节点输出原始 `z16`/mm 时才应使用 `0.001`。

## 运行

```yaml
- id: mujoco
  path: ./packages/nodes/simulators/mujoco/main.py
  inputs:
    tick: dora/timer/millis/20
    action: <上游 JointCommand 源>
  outputs:
    - proprio_state
    - pose_set
    - image/main
    - depth/top
    - camera_info/top
  config:
    config_path: ./packages/nodes/simulators/mujoco/config.yaml
```

运行时 topic 名保持旧约定 `action` / `proprio_state`，payload 使用 `JointCommand` / `JointState`。
