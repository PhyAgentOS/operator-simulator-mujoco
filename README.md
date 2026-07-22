# mujoco_sim

独立的 MuJoCo Dora 仿真节点：接收 `JointCommand`，输出 `JointState`、图像和可选 `PoseSet`。它不接收 `PolicyCommand`，也不接收 `sim_control`；仿真是否运行由上游 policy/dataflow 决定是否持续发送 `tick` 和 `action`。

完整机器人模型和 Forge 端到端 dataflow 由 `forge_runtime` 仓库维护；本仓库只包含节点实现与一个不依赖外部 assets 的最小示例。

## 安装与开发

需要 Python 3.12 和 [uv](https://docs.astral.sh/uv/)：

```bash
uv sync --frozen --all-groups
uv run pytest -q
uv run mujoco_sim --help
uv run python -m mujoco_sim --help
```

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
- `sim_status`: 仿真进度 JSON，仅反映本节点内部 step/time 状态，不作为外部控制接口。

## 配置

复制 `config.example.yaml` 为 `config.yaml` 并修改。配置来源：

- `--config /path/to/config.yaml`
- 环境变量 `MUJOCO_NODE_CONFIG`
- dataflow 节点 `config` 字段

### 初始关节位置

`initial_positions` 可覆盖 MuJoCo 模型的 `qpos0`，并在节点启动和收到 `reset_scene` 时应用：

```yaml
initial_positions:
  lift: 0.2
```

- key 必须是顶层 `joints` 中声明的全局逻辑关节名；进入每个 robot 时按 `joint_to_actuator` 映射。
- value 单位与对应 `joints[].unit` 一致，例如 `radians`、`meters` 或 `millimeters`；仅支持 `position` / `prismatic` 关节。
- 配置可以是稀疏的；未列出的关节保留模型 `qpos0`。
- `state_from_joints` 派生关节支持 `diff` / `sum` 反向初始化；调整底层 qpos 时会保留其均值或总偏移。
- position actuator 的 `ctrl` 会在 `mj_forward` 后同步到 actuator length，避免首个物理步把非零初始位拉回默认目标，并兼容 gear/tendon transmission。
- 初始化后的 actuator target 必须落在 MuJoCo `ctrlrange` 内。运行时命令也优先使用模型 `ctrlrange`，只有模型未声明限位时才使用按单位推断的兼容限位。

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

### 深度 Policy 的相机参数

相机参数写在 policy 配置中，不通过 Dora 高频透传静态标定信息。深度图本身继续通过 `depth/*` 高频发送，policy 使用静态内参完成反投影：

```yaml
# policy config 示例
camera_width: 640
camera_height: 480
camera_fx: 296.375
camera_fy: 296.375
camera_cx: 320.0
camera_cy: 240.0
depth_min_m: 0.02
depth_max_m: 1.5
depth_unit_scale: 1.0
```

这些参数的来源：

- `camera_fx/fy/cx/cy`：由 MuJoCo XML 中的 camera 光学参数计算。若配置了 `focal` / `sensorsize` / `resolution`，按这些值折算到输出分辨率；若只配置了 `fovy`，按 pinhole 模型从 `fovy` 和输出 `width/height` 推导。
- `depth_min_m` / `depth_max_m`：policy 侧有效深度范围，按任务工作空间设置，用于过滤太近、太远或异常的 depth。
- `depth_unit_scale`：深度原始值转 meters 的比例。MuJoCo 输出 `32FC1` meters，因此为 `1.0`；真机 `16UC1` 毫米深度才通常是 `0.001`。

## 运行

安装后可将 console script 直接用于 Dora dataflow：

```yaml
- id: mujoco
  path: mujoco_sim
  inputs:
    tick: dora/timer/millis/20
    action: <上游 JointCommand 源>
  outputs:
    - proprio_state
    - pose_set
    - image/main
    - depth/top
  args: --config simulator.yaml
```

最小示例位于 `examples/minimal`：

```bash
cd examples/minimal
uv run dora run dataflow.yaml
```

运行时 topic 名保持旧约定 `action` / `proprio_state`，payload 使用 `JointCommand` / `JointState`。

## 构建独立二进制

```bash
./scripts/build_pyinstaller.sh
./dist/mujoco_sim --help
```

构建产物固定为 `dist/mujoco_sim`。Forge 部署可以将该 artifact 安装到 `${FORGE_RUNTIME_BIN}/mujoco_sim`，从而保持现有 dataflow 的二进制路径约定。
