# MuJoCo 仿真节点

基于 dora-rs 的 MuJoCo 仿真节点：接收 tick 驱动仿真步进、接收 forge-msg.Action 控制内部 ctrl、聚合 joint 输出 ProprioState、按配置输出相机图像。

## 配置

复制 `config.example.yaml` 为 `config.yaml` 并修改。配置来源（优先级从高到低）：

- `--config /path/to/config.yaml`
- 环境变量 `MUJOCO_NODE_CONFIG`
- dataflow 节点 `config` 字段

## 输入 / 输出

| 输入 | 说明 |
|------|------|
| `tick` | 定时器驱动，触发仿真步进、ProprioState 与相机输出 |
| `action` | forge-msg.Action，按 joint 配置转为 ctrl |
| `sim_control` | （可选）来自 `control_gateway` 的仿真控制指令 JSON，形如 `{ "action": "PAUSE" }` |

| 输出 | 说明 |
|------|------|
| `proprio_state` | forge-msg.ProprioState |
| `image/*` | 相机图像（按 image_forward 配置） |
| `sim_status` | 当前仿真状态 JSON，供 `control_gateway` / 前端查询 |

## 运行

在 dora dataflow 中声明：

```yaml
- id: mujoco
  path: ./packages/nodes/simulators/mujoco/main.py
  inputs:
    tick: dora/timer/millis/50
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
    - `current_step`（当前仿真步数）
    - `episode_index`（第几次 reset / start）
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
