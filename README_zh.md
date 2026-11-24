[English](README.md) | 中文

[English](README.md) | [简体中文](README_zh.md)

# mujoco_sim

## BREAKING CHANGE：Forge 2.x / Dora 1.x 迁移

本次从 Forge 1.x / Dora 0.4.x 升级需要协同部署。节点对接前必须对齐 Forge 包、Dora CLI/coordinator/daemon、本节点及上下游节点的版本；不支持在 Forge 1.x 部署中只替换单个节点。可执行产物也需要使用匹配环境重新构建和部署。仍使用旧技术栈的部署请保留在 `stable/1.x`。

本轮验证的 Python 基线为 `forge-msgs==2.1.0`、`Pillow==13.0.0`、`forge-common==2.1.0`、按需使用的 `forge-robot` / `forge-policy` / `forge-kinematics` / `forge-tool` 为 `2.0.0`，以及 `dora-rs==1.0.1`。同时遵守本仓库的 Python 和 SDK 平台要求。Forge 各包族独立发布，无需强行使用同一个版本号。Rust 节点继续使用已发布的 `forge_msgs` crate `2.0.0` 和 Dora `1.0.1`；Python Msgs `2.1.0` 不要求升级到尚未发布的 Rust `2.1.0`。消息命名空间 `forge_msgs.v1` 保持不变。

这里的 breaking change 指旧技术栈的部署迁移，不代表真机、真实模型或长时间运行已完成验收。实际测试和未完成项见[本轮验证报告](migration/msgs-2.1.0-report.json)。

独立的 MuJoCo Dora 仿真节点：接收 `JointCommand` 和可选的 `LocomotionCommand`，输出 `JointState`、图像和可选 `PoseSet`。它不接收 `PolicyCommand`，也不接收 `sim_control`；仿真是否运行由上游 policy/dataflow 决定是否持续发送 `tick` 和控制输入。

完整机器人模型和 Forge 端到端 dataflow 由 `forge_runtime` 仓库维护；本仓库只包含节点实现与一个不依赖外部 assets 的最小示例。

本项目由 X-ERA 维护，以 Apache License 2.0 开源。参与开发前请阅读 [`CONTRIBUTING.md`](CONTRIBUTING.md)。

## 安装与开发

Forge 2.x 节点默认用 MuJoCo Filament 渲染相机流，依赖同版本的 `mujoco-core` 和 `mujoco-studio` 4.0.1。无显示器的 Linux 环境使用 Vulkan，Ubuntu 需安装 `libvulkan1` 和显卡对应的 Vulkan 驱动；软件渲染可安装 `mesa-vulkan-drivers`。macOS 使用 OpenGL。已有 `image_forward` 配置继续输出 RGB 或沿相机光轴的米制 float32 深度（`32FC1`）。运行 `uv run --frozen python tests/camera_smoke.py` 可验证像素对齐和深度读数。

在节点 YAML 中设置 `renderer_backend: filament`（默认）或 `renderer_backend: classic`，统一选择 RGB 和深度渲染器。无显示器 Linux 使用 classic 时，启动前设置 `MUJOCO_GL=egl` 或 `MUJOCO_GL=osmesa` 并安装对应运行库。

使用 Filament 时，`filament_graphics_api` 可取 `auto`（默认）、`vulkan` 或 `opengl`，统一作用于已配置的 RGB 和深度流。无显示器的 Linux 可用 `auto` 或 `vulkan`。Linux 官方 4.0.1 wheel 显式选择 OpenGL 时需要可用的 X 显示。

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
- 配置 `kinematic_planar_base` 后，每个物理子步按当前 `LocomotionCommand` 更新一次 mocap body；命令超时也按仿真时间判断。
- 每个 `tick` 输出一次 `proprio_state`。
- 相机按仿真时间节流，`image_forward` 中可为每个输出配置 `fps`。

## 输入 / 输出

输入：

- `tick`: 控制周期定时器。
- `action` / `action/<source>`: `forge_msgs.JointCommand`，按配置映射到 MuJoCo actuator。命令是 sparse update：只写 `name` 中列出的 actuator，未列出的 actuator 保持原控制目标。
- `locomotion_command`: `forge_msgs.LocomotionCommand`；仅在配置 `kinematic_planar_base` 时生效。
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

### Kinematic planar base

固定根或不需要轮地接触动力学的模型可以把一个顶层 wrapper body 声明为 MuJoCo mocap body：

```xml
<body name="mobile_base" mocap="true">
  <!-- articulated robot subtree -->
</body>
```

再配置：

```yaml
kinematic_planar_base:
  body: mobile_base
  command_timeout_sec: 0.1
```

`locomotion_command` 使用标准机体系语义：`vx` 向前为正、`vy` 向左为正，单位 m/s；`wz` 逆时针为正，单位 rad/s。节点把机体系线速度旋转到世界系，并在每个物理子步直接更新 mocap body 的 X/Y 和绕世界 Z 轴的 yaw；Z 以及初始 roll/pitch 保持不变。

`command_timeout_sec` 可选并按仿真时间计算。超时后速度自动清零；省略时最后一条命令保持到下一条命令或 `reset_scene`。`reset_scene` 始终恢复 MJCF 中的 mocap 初始位姿并清除当前移动命令。未配置此节时，现有 joint-only 行为不变。

### 初始关节位置

`initial_state.joint_positions` 可覆盖 MuJoCo 模型的 `qpos0`，并在节点启动和收到 `reset_scene` 时应用：

```yaml
initial_state:
  joint_positions:
    lift: 0.2
```

`initial_state` 是场景级扩展点。当前仅支持 `joint_positions`；未来可在同一层级增加物体位姿、速度或 MuJoCo keyframe，而无需改变关节位置配置路径。未实现的字段会明确报错，不会被静默忽略。

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
    locomotion_command: <上游 LocomotionCommand 源>  # 仅配置 kinematic_planar_base 时需要
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

运行时 joint topic 名保持旧约定 `action` / `proprio_state`，payload 使用 `JointCommand` / `JointState`；可选移动输入固定为 `locomotion_command`，payload 使用 `LocomotionCommand`。

## 构建独立二进制

```bash
./scripts/build_pyinstaller.sh
./dist/mujoco_sim --help
```

构建产物固定为 `dist/mujoco_sim`。Forge 部署可以将该 artifact 安装到 `${FORGE_RUNTIME_BIN}/mujoco_sim`，从而保持现有 dataflow 的二进制路径约定。

公开 Release 仅额外发布包含 `mujoco_sim` 的 Linux x86_64 归档及 SHA-256 文件；具体流程见 [`RELEASING.md`](RELEASING.md)。Ubuntu 20.04 二进制工作流在 focal 容器内构建。运行相机输出需要可用的 Vulkan 驱动。

## License

本项目使用 [Apache License 2.0](LICENSE)。


## Forge 2 / Dora 1 迁移

目标版本：`2.0.2`，当前为待验收改动。首轮验证组合使用 Dora 1.0.1、Forge Msgs 2.1.0、Common 2.1.0；依赖与构建工具以提交的锁文件为准。原有消息规范继续使用 `forge_msgs.v1`。

本轮要求：检查所有示例最终 YAML，使用 `dora validate <dataflow.yaml>` 验证，并实际执行消息往返。相机取流、资源释放和异常退出须分别验收；高吞吐图像链预热后连续测试至少 4 小时，覆盖慢消费者、扇出、退出与重启，记录实际内存占用。无硬件单测不能替代设备验收。

最终发布归档须先在声明支持的平台完成验收，记录源码与归档、可执行文件校验值；通过后发布同一份包，重建后重新验收。Linux 发布包仍需 Linux 验证，其他平台的开发测试不代表其已通过。

渲染器创建后的初始化失败也会执行清理；Dora 错误返回失败。

## 分版本发布

**发布规则（强制）：** 合并版本/锁文件修改 → 在已审查提交上创建 annotated 版本标签 → 从该标签精确构建 → 校验产物 → 发布。禁止使用移动分支、未打标签的提交构建 Release，禁止覆盖已发布标签。

| 维护线 | 源码 | 节点/包版本 | 运行时 |
| --- | --- | --- | --- |
| Forge 1.x | `stable/1.x` | `1.x`，标签 `v1.X.Y` | Forge 1.x / Dora 0.4.x |
| Forge 2.x | `master` | `2.x`，标签 `v2.X.Y` | Forge 2.x / Dora 1.x |

**旧版版本号统一：** `stable/1.x` 原为 0.x 的节点以 **1.0.0** 发布，保留 Forge 1.x / Dora 0.4.x 依赖与运行行为。已有 1.x 版本保留。节点/包主版本与 Forge 兼容线一致，不再接受 0.x 发布标签。

两条线均从 `master` 运行 **Build versioned release**。先将版本和锁文件修改通过 PR 合并至对应源码分支，再在已审查提交上创建未占用的 **annotated** 标签，标签必须匹配包元数据。禁止移动已有标签或复用已发布版本。工作流使用自身已审查的发布工具与精确标签源码，旧标签无需包含新发布脚本。

```bash
# 替换为已存在、已审查、未发布且与源码元数据一致的标签。
TAG=v1.X.Y
gh workflow run release.yml --ref master -f release_line=1.x -f tag="$TAG"
TAG=v2.X.Y
gh workflow run release.yml --ref master -f release_line=2.x -f tag="$TAG"
```

流水线校验标签类型、源码提交、分支归属、源码版本及锁定的 Forge/Dora 兼容性；构建源码归档及项目支持的 wheel/sdist，检查 Python 分发元数据与 SHA-256，并创建或附加至 **GitHub Release 草稿**。不覆盖已发布 Release 或产物；发布前检查全部构建结果与验收依据。该流程不发布节点包至 PyPI。

设置 `build_binaries=true` 复用已有 Ubuntu 20.04 二进制工作流，保留原架构、CLI 和归档校验。也可直接运行 `build-ubuntu20.yml`，传入 `ref`、`version`、`release_line`；`ref` 必须为 annotated 标签，不能为移动分支。仅在所有选定任务通过后附加二进制产物。

源码归档保留标签中的文件；外部 SDK/submodule、模型下载与现场配置仍是独立前置条件。依赖或 SDK 构建失败不能视为发布验收通过。构建成功不代表单节点或串联 `dora run`、真机、真实模型、长稳态或回滚验收通过；每条验收链必须包含相同兼容线的 `adapter-forge-gateway`。Forge 2.x 属于 **breaking change**，整个部署的 Forge 库、节点版本与 Dora 必须对齐；1.x 保留在维护分支。
