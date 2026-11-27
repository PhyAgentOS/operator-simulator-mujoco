[English](optional-sensors.md) | [简体中文](optional-sensors_zh.md)

# 可选仿真能力

所有新增能力默认关闭，不改变原配置的命令、mocap、相机及状态行为。
不新增导航控制器或第二个物理仿真实例。完整 YAML 示例见[英文配置](optional-sensors.md)。

## 传感器

`sensors.imu` 和 `sensors.lidar` 是可分别启用的列表，每个条目配置独立的
`output` 和 `frame`。IMU 的 `gyro`、`accelerometer` 指向同一 site 上的原生
陀螺仪和加速度计，单位 rad/s、m/s²，`hz` 默认 200。

雷达配置 `site`、`pattern: mid360`、`hz`（默认 10）、`downsample`（默认 4）、
`min_range`/`max_range`（默认 0.2/30 米）、`geom_groups`（默认包含 0–5）。
射线排除传感器所属 body，其他可见几何由显式分组决定，没有场景专用排除规则。
每个物理步采样一批射线；扫描周期必须包含整数个物理步。输出 `PointCloudBuffer`
包含米制 x/y/z、相对首批采样的 uint32 `time_offset_ns`、合成常数反射率 100、
零值 line/tag，不模拟材质反射率或硬件发射电子学。空扫描仍输出 width=0。
多传感器安装标定、融合及下游字段转换由消费者负责。

元数据包含 `capture_timestamp_ns`、`frame_id`、`clock_domain=simulation`、
`time_mapping_id`、`ros_time_offset_ns=0`；雷达另有配置顺序从零开始的
`lidar_id`、扫描结束时间、采样量化间隔和扫描/反射率/line 模型标识。
`clock` 在同控制周期传感器批次之前输出 int64 纳秒。重置产生新的时间代际，
下游估计器必须建立新会话。采样率不可超过物理频率；非整数 IMU 间隔量化到物理步。
CPU 射线与 Mid360 模式来自 MIT 许可的 MuJoCo-LiDAR，来源及兼容修改记录在
`src/mujoco_sim/_vendor/lidar/SOURCE.json`。

## 速度反馈与命令时效

`joint_velocity_output: true` 为直接映射的标量关节输出实际 velocity；启用时
拒绝派生关节和多自由度关节。位置/速度保持一致的弧度、米、毫米及每秒单位。
元数据增加仿真采样时间、宿主单调时钟时间和 episode。关闭时保留原空 velocity。

`velocity_command_timeouts` 按 robot ID、逻辑速度执行器名称配置超时秒数。
每个被更新的执行器单独续期；仿真时间或墙钟任一过期，仅将其目标归零。
其他关节的稀疏命令不能替它续期，重置清空时效记录。归零目标不等于瞬间停稳，
制动效果仍由模型和执行器决定；原位置控制目标不受影响。

## 只读观察

`state_output: true` 在 `physics_state` 输出 JSON bytes，固定 schema_version=1，
包含 episode、monotonic_ns、simulation_time、nq/nv、MuJoCo 顺序的 qpos/qvel
和 mocap_pos/mocap_quat。消费者负责渲染、降帧和存盘，节点不写录制文件。

`contact_output: true` 在 `contacts` 输出 JSON bytes，包含 schema_version=1、
episode 和每个物理步的 samples（包括空样本）。样本有采样时间及 contacts，
每项有 geom1/geom2（同一模型的几何 ID）、米制有符号 distance 和世界系 normal。
保留正距离候选，不过滤地面、不判断任务是否通过。消费者需核验样本连续性及
终点时间才能声称完整碰撞覆盖，并为所选频率配置有界队列。
