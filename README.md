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

| 输出 | 说明 |
|------|------|
| `proprio_state` | forge-msg.ProprioState |
| `image/*` | 相机图像（按 image_forward 配置） |

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

## 依赖

- `mujoco`：由本包直接依赖
- `forge-simulators-mujoco`：MuJoCo 仿真封装
- `forge-msgs`：Action、ProprioState 等消息格式
