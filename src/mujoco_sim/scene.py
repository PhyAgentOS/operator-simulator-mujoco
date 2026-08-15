"""MuJoCo model, body, simulator, and reset setup."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
from forge_common import get_logger

from .config import MuJoCoNodeConfig, RobotConfig
from .locomotion import KinematicPlanarBase
from .simulator import ActuatorSpec, JointSpec, MuJoCoSimulator

logger = get_logger(__name__)


def _resolve_mujoco_body_name(prefix: str, body: str) -> str:
    """Combine a configured body name with its robot prefix without duplication."""
    b = str(body).strip()
    if not b:
        return ""
    p = prefix or ""
    if p and not p.endswith("/"):
        p += "/"
    if p and (b.startswith(p) or b.startswith(p.lstrip("/"))):
        name = b
    else:
        name = p + b
    return name.lstrip("/")


def _find_root_body_under_prefix(model: mujoco.MjModel, prefix: str) -> int | None:
    """Find the lowest-id direct child body under a model name prefix."""
    p = (prefix or "").strip()
    if not p.endswith("/"):
        p += "/"
    if not p:
        return None

    candidate_id: int | None = None
    for body_id in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if name is None or not name.startswith(p):
            continue
        suffix = name[len(p) :]
        if "/" in suffix:
            continue
        if candidate_id is None or body_id < candidate_id:
            candidate_id = body_id
    return candidate_id


def resolve_pose_body_ids(
    model: mujoco.MjModel,
    config: MuJoCoNodeConfig,
) -> dict[str, int]:
    """Resolve configured and prefix-derived pose output IDs to MuJoCo body IDs."""
    pose_body_ids: dict[str, int] = {}
    if config.pose_targets:
        robot_prefix_by_id = {r.id: (r.prefix or "") for r in config.robots}
        targets: dict[str, str] = {}

        for target_id, body_config in config.pose_targets.items():
            prefix = robot_prefix_by_id.get(target_id, "")
            body_name = _resolve_mujoco_body_name(prefix, body_config)
            if body_name:
                targets[target_id] = body_name

        for target_id, body_name in targets.items():
            body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, body_name
            )
            if body_id >= 0:
                pose_body_ids[target_id] = body_id
                logger.info(
                    "[mujoco] pose_set id=%s 使用 body %s (id=%s)",
                    target_id,
                    body_name,
                    body_id,
                )
            else:
                logger.warning(
                    "[mujoco] pose_set body '%s' 在模型中未找到（id=%s），将跳过该 id 的 pose",
                    body_name,
                    target_id,
                )

    for robot in config.robots:
        if robot.id in pose_body_ids:
            continue
        prefix = robot.prefix or ""
        body_id = _find_root_body_under_prefix(model, prefix)
        if body_id is not None:
            pose_body_ids[robot.id] = body_id
            body_name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            )
            logger.info(
                "[mujoco] pose_set id=%s 自动识别 body %s (id=%s)",
                robot.id,
                body_name,
                body_id,
            )
        else:
            logger.debug(
                "[mujoco] pose_set id=%s 未配置且 prefix=%r 下未找到根 body，跳过",
                robot.id,
                prefix,
            )

    return pose_body_ids


def _actuator_limits(unit: str) -> tuple[float, float]:
    """Return fallback clipping limits for an actuator unit."""
    if unit == "millimeters":
        return 0.0, 50.0
    if unit == "meters":
        return 0.0, 0.05
    if unit == "radians":
        return -3.14, 3.14
    if unit == "millimeters/s":
        return -50.0, 50.0
    if unit == "meters/s":
        return -1.0, 1.0
    if unit in ("radians/s", "radians_s"):
        return -10.0, 10.0
    return -3.14, 3.14


def build_joints_and_actuators(
    node_config: MuJoCoNodeConfig,
    robot_config: RobotConfig,
) -> tuple[list[JointSpec], list[ActuatorSpec]]:
    """Build simulator joint and actuator specs for one configured robot."""
    joint_specs = {joint.name: joint for joint in node_config.joints}
    actuator_to_joint = {
        actuator: joint
        for joint, actuator in robot_config.joint_to_actuator.items()
    }

    joints: list[JointSpec] = []
    for joint_name, actuator_name in robot_config.joint_to_actuator.items():
        spec = joint_specs.get(joint_name)
        mode = spec.mode if spec else "position"
        mj_lookup = joint_name if joint_name != actuator_name else None
        joints.append(JointSpec(name=actuator_name, mode=mode, mj_name=mj_lookup))

    actuators: list[ActuatorSpec] = []
    for index, actuator_name in enumerate(robot_config.actuator_order):
        joint_name = actuator_to_joint.get(actuator_name)
        spec = joint_specs.get(joint_name) if joint_name else None
        unit = spec.unit if spec else "radians"
        mode = spec.mode if spec else "position"
        min_value, max_value = _actuator_limits(unit)
        actuators.append(
            ActuatorSpec(
                name=actuator_name,
                id=index,
                control_mode=mode,
                min_value=min_value,
                max_value=max_value,
                unit=unit,
            )
        )
    return joints, actuators


def initialize_simulators(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: MuJoCoNodeConfig,
) -> dict[str, MuJoCoSimulator]:
    """Initialize one MuJoCoSimulator per configured robot."""
    simulators: dict[str, MuJoCoSimulator] = {}
    for robot in config.robots:
        joints, actuators = build_joints_and_actuators(config, robot)
        derived_state_config: dict[str, dict[str, object]] = {}
        for joint_name, actuator_name in robot.joint_to_actuator.items():
            derived = config.derived_state_cfg.get(joint_name)
            if derived is not None:
                derived_state_config[actuator_name] = {
                    "joint_names": list(derived.joint_names),
                    "expr": derived.state_expr,
                    "scale": derived.state_scale,
                }
        simulators[robot.id] = MuJoCoSimulator(
            model=model,
            data=data,
            joints=joints,
            actuators=actuators,
            derived_state_cfg=derived_state_config,
            prefix=robot.prefix,
        )
    return simulators


def initialize_planar_base(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: MuJoCoNodeConfig,
) -> KinematicPlanarBase | None:
    """Initialize the optional kinematic planar base."""
    planar_config = config.kinematic_planar_base
    if planar_config is None:
        return None
    return KinematicPlanarBase(
        model=model,
        data=data,
        body_name=planar_config.body,
        command_timeout_sec=planar_config.command_timeout_sec,
    )


def load_model_and_data(
    config: MuJoCoNodeConfig,
) -> tuple[mujoco.MjModel, mujoco.MjData] | None:
    """Resolve and load the configured MuJoCo model, reporting a missing path."""
    model_path = Path(config.model_path)
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path
    if not model_path.exists():
        print(f"[mujoco] 模型文件不存在: {model_path}", file=sys.stderr)
        return None

    model = mujoco.MjModel.from_xml_path(str(model_path))
    return model, mujoco.MjData(model)


def reset_simulation_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: MuJoCoNodeConfig,
    simulators: dict[str, MuJoCoSimulator],
    planar_base: KinematicPlanarBase | None = None,
) -> None:
    """Restore model state, apply initial positions, and sync actuator targets."""
    mujoco.mj_resetData(model, data)
    if planar_base is not None:
        planar_base.stop()
    joint_positions = config.initial_state.joint_positions
    applied_joint_names: set[str] = set()

    for robot in config.robots:
        actuator_positions: dict[str, float] = {}
        for joint_name, actuator_name in robot.joint_to_actuator.items():
            if joint_name not in joint_positions:
                continue
            actuator_positions[actuator_name] = joint_positions[joint_name]
            applied_joint_names.add(joint_name)
        simulators[robot.id].apply_initial_positions(actuator_positions)

    unapplied = sorted(set(joint_positions) - applied_joint_names)
    if unapplied:
        raise ValueError(
            "initial_state.joint_positions 关节未分配给任何 robot: "
            f"{unapplied}"
        )

    mujoco.mj_forward(model, data)
    for simulator in simulators.values():
        simulator.sync_actuator_targets()
