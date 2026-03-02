# MuJoCo 仿真节点

基于 dora-rs 的 MuJoCo 仿真节点：接收 tick 驱动仿真步进、接收 forge-msg.Action 控制内部 ctrl、聚合 joint 输出 ProprioState、按配置输出相机图像。

## 时间与频率

节点采用**控制周期为唯一驱动**的模型，与常见机器人仿真实践（Isaac Sim / Gazebo 子步模式）一致：

- **控制周期**：dataflow 的 `tick` 表示「一个控制周期」到点（如 20ms → 50Hz）。配置项 `control_hz`（默认 50）需与 dataflow 中 mujoco 的 tick 间隔一致，例如 `tick: dora/timer/millis/20` 对应 `control_hz: 50`。
- **物理子步**：每收到一次 tick，节点执行 `n_steps = round(control_period_sec / model.opt.timestep)` 次 `mj_step`（如 0.02s / 0.002s = 10 步），使仿真时间与真实时间 1:1。物理步长由 MuJoCo 模型 XML 的 `timestep` 决定（常见 0.002）。
- **State 输出**：每 tick 仅输出一次 `proprio_state` 与 `sim_status`，频率等于控制频率（control_hz）。
- **相机输出**：按**仿真时间**节流。每个相机在配置中可指定 `fps`；当仿真时间 `sim_time >= next_camera_time[cam]` 时输出一帧并更新 `next_camera_time`，与控制周期不必成整倍数。
- **Action**：收到即 `set_action`，在本 tick 及后续子步中生效（零阶保持）。

推荐配置示例：physics 500Hz（XML `timestep="0.002"`）、control 50Hz（`control_hz: 50`，tick 20ms）、camera 30Hz（`fps: 30`）。`sim_status` 输出中包含 `sim_time`（秒），供下游需要仿真时间戳时使用。

## 配置

复制 `config.example.yaml` 为 `config.yaml` 并修改。配置来源（优先级从高到低）：

- `--config /path/to/config.yaml`
- 环境变量 `MUJOCO_NODE_CONFIG`
- dataflow 节点 `config` 字段

### 派生 actuator 状态（例如 gripper）

有些模型中，某个逻辑执行器的状态需要由多个 MuJoCo joint 的 qpos 合成，例如 Piper 的夹爪由 `joint7` / `joint8` 两个 finger joint 驱动，而对外只暴露一个 `gripper`。

在 `joints` 配置中可以为该逻辑关节增加派生字段：

```yaml
joints:
  - name: joint1
    mode: position
    unit: radians
  # ...
  - name: gripper
    mode: prismatic
    unit: millimeters
    # 由 joint7 / joint8 的开度合成：
    #   value_raw = qpos[joint7] - qpos[joint8]  （joint8 = -joint7 时约等于 2 * joint7）
    #   value = value_raw * state_scale          （米 → 毫米）
    state_from_joints: ["joint7", "joint8"]
    state_expr: diff        # 或 sum
    state_scale: 1000.0     # 缺省为 1.0
```

节点在内部会使用这些配置，从 MuJoCo 的多个 joint qpos 合成一个 `gripper` 的状态值，并仅在 `RobotState.actuators['gripper']` / `ProprioState.joints['gripper']` 中对外暴露，与硬件侧 `get_joint_positions` 中「joint7/joint8 → gripper」的旧逻辑保持一致。

## 输入 / 输出

| 输入 | 说明 |
|------|------|
| `tick` | 控制周期定时器（如 20ms），每 tick 执行若干物理子步后输出一次 state、按 sim_time 节流相机 |
| `action` | forge-msg.Action，按 joint 配置转为 ctrl |
| `sim_control` | （可选）来自 `control_gateway` 的仿真控制指令 JSON，形如 `{ "action": "PAUSE" }` |

| 输出 | 说明 |
|------|------|
| `proprio_state` | forge-msg.ProprioState |
| `image/*` | 相机图像（按 image_forward 配置） |
| `sim_status` | 当前仿真状态 JSON（含 `status_name`、`current_step`、`episode_index`、`sim_time`），供 `control_gateway` / 前端查询 |

## 运行

在 dora dataflow 中声明：

```yaml
- id: mujoco
  path: ./packages/nodes/simulators/mujoco/main.py
  inputs:
    tick: dora/timer/millis/20
    action: <上游 Action 源>
  outputs:
    - proprio_state
    - image/main
  config:
    config_path: ./packages/nodes/simulators/mujoco/config.yaml
```

或单独运行（需 `dora start` 已启动 dataflow）：

```bash
cd packages/nodes/simulators/mujoco
uv run python main.py --config config.yaml
```

## 状态机（sim_control / sim_status）

当 dataflow 中接入 `control_gateway` 时，`mujoco` 内部维护一个简单状态机，并通过 `sim_status` 对外暴露：

- **状态枚举**（实现中主要使用以下子集）：
  - `UNINITIALIZED`：尚未加载/初始化场景（预留，不一定在所有场景下显式使用）。
  - `INITIALIZED`：已初始化场景但尚未开始运行（预留）。
  - `RUNNING`：正常按 `tick` 步进仿真、输出状态与图像。
  - `PAUSED`：暂停；收到 `tick` 时**不会调用 `mj_step`、不会更新图像/状态**。
  - `STOPPED`：停止；具体是否退出进程由 dataflow 策略决定。
  - `RESETTING`：预留，用于未来支持更细粒度的 reset 流程。

- **状态迁移**（由 `sim_control` 触发）：
  - `START`：
    - 若在 `UNINITIALIZED` / `STOPPED`，重置步数并进入 `RUNNING`；
    - 其它状态下视为「强制运行」，进入 `RUNNING`。
  - `PAUSE`：仅当 `RUNNING` 时进入 `PAUSED`。
  - `RESUME`：仅当 `PAUSED` 时恢复为 `RUNNING`。
  - `RESET`：清空内部 step 计数、开启新 episode，并进入 `RUNNING`。
  - `STOP`：进入 `STOPPED`。

- **与 topic / HTTP 的关系**：
  - 输入 `sim_control` 由 `control_gateway` 根据 `/sim/control` HTTP 请求生成；
  - 每次处理控制指令或 `tick` 后，`mujoco` 会通过 `sim_status` 输出当前：
    - `status_name`（上述字符串枚举）
    - `status_code`（简单整数映射，供前端/监控使用）
    - `current_step`（当前仿真物理步数累计）
    - `episode_index`（第几次 reset / start）
    - `sim_time`（仿真时间，秒）
  - `control_gateway` 订阅 `sim_status` 并通过 `GET /sim/status` 暴露给前端。当前端调用 `POST /sim/control` 改变状态时，新的状态会体现在后续的 `sim_status` 中。 

## 渲染后端

离屏渲染默认使用 **EGL**（不依赖 GLFW/显示器），适合无头环境、SSH、服务器。可通过环境变量覆盖：

- `MUJOCO_GL=egl`（默认）：无头 GPU 渲染
- `MUJOCO_GL=osmesa`：软件渲染（需安装 libOSMesa）
- `MUJOCO_GL=glfw`：需要显示器/窗口时使用

## 依赖

- `mujoco`：由本包直接依赖
- `forge-simulators-mujoco`：MuJoCo 仿真封装
- `forge-msgs`：Action、ProprioState 等消息格式
