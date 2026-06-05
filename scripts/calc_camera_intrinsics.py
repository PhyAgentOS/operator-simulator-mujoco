#!/usr/bin/env python3
"""Calculate MuJoCo camera intrinsics for policy depth consumption.

Examples:
  python calc_camera_intrinsics.py scene.xml --camera item_1/top
  python calc_camera_intrinsics.py scene.xml --camera item_1/hand --width 640 --height 480
  python calc_camera_intrinsics.py scene.xml --list
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import mujoco


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="根据 MJCF camera 参数计算 camera_fx/fy/cx/cy。"
    )
    parser.add_argument("mjcf_path", help="MuJoCo XML / MJCF 路径")
    parser.add_argument(
        "--camera",
        "-c",
        help="MuJoCo camera 名称，例如 item_1/top 或 item_1/hand",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="policy 实际消费的图像宽度，默认 640",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="policy 实际消费的图像高度，默认 480",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出 MJCF 中所有 camera 并退出",
    )
    return parser.parse_args()


def _camera_name(model: mujoco.MjModel, cam_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_id) or f"<camera:{cam_id}>"


def _list_cameras(model: mujoco.MjModel) -> None:
    if model.ncam == 0:
        print("No cameras found.")
        return
    print("Cameras:")
    for cam_id in range(model.ncam):
        name = _camera_name(model, cam_id)
        sensorsize = model.cam_sensorsize[cam_id]
        intrinsic = model.cam_intrinsic[cam_id]
        mode = (
            "focal"
            if float(sensorsize[0]) > 0.0
            and float(sensorsize[1]) > 0.0
            and float(intrinsic[0]) > 0.0
            and float(intrinsic[1]) > 0.0
            else "fovy"
        )
        print(f"  - {name} ({mode}, fovy={float(model.cam_fovy[cam_id]):.6g})")


def _calc_intrinsics(
    model: mujoco.MjModel,
    cam_id: int,
    width: int,
    height: int,
) -> tuple[str, float, float, float, float]:
    intrinsic = model.cam_intrinsic[cam_id]
    sensorsize = model.cam_sensorsize[cam_id]
    sensor_w = float(sensorsize[0])
    sensor_h = float(sensorsize[1])
    focal_x = float(intrinsic[0])
    focal_y = float(intrinsic[1])
    principal_x = float(intrinsic[2])
    principal_y = float(intrinsic[3])

    if sensor_w > 0.0 and sensor_h > 0.0 and focal_x > 0.0 and focal_y > 0.0:
        fx = focal_x / sensor_w * width
        fy = focal_y / sensor_h * height
        cx = width * (0.5 + principal_x / sensor_w)
        cy = height * (0.5 + principal_y / sensor_h)
        return "focal", float(fx), float(fy), float(cx), float(cy)

    fovy = math.radians(float(model.cam_fovy[cam_id]))
    fy = height / (2.0 * math.tan(fovy / 2.0))
    fx = fy
    return "fovy", float(fx), float(fy), float(width / 2.0), float(height / 2.0)


def main() -> int:
    args = _parse_args()
    mjcf_path = Path(args.mjcf_path).expanduser()
    if not mjcf_path.exists():
        raise FileNotFoundError(f"MJCF 文件不存在: {mjcf_path}")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width/--height 必须大于 0")

    model = mujoco.MjModel.from_xml_path(str(mjcf_path.resolve()))
    if args.list:
        _list_cameras(model)
        return 0

    if not args.camera:
        raise ValueError("请通过 --camera 指定 camera 名称，或使用 --list 查看可用 camera。")

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera)
    if cam_id < 0:
        available = ", ".join(_camera_name(model, i) for i in range(model.ncam))
        raise ValueError(f"camera 不存在: {args.camera}. 可用 camera: {available}")

    source, fx, fy, cx, cy = _calc_intrinsics(model, cam_id, args.width, args.height)
    intrinsic = model.cam_intrinsic[cam_id]
    sensorsize = model.cam_sensorsize[cam_id]
    resolution = model.cam_resolution[cam_id]
    fovy = float(model.cam_fovy[cam_id])

    print(f"mjcf_path: {mjcf_path.resolve()}")
    print(f"camera: {args.camera}")
    print(f"source: {source}")
    print(f"output_size: {args.width}x{args.height}")
    print("")
    print("raw_mujoco_camera:")
    print(f"  fovy: {fovy:.10g}")
    print(f"  intrinsic: [{', '.join(f'{float(v):.10g}' for v in intrinsic)}]")
    print(f"  sensorsize: [{', '.join(f'{float(v):.10g}' for v in sensorsize)}]")
    print(f"  resolution: [{int(resolution[0])}, {int(resolution[1])}]")
    print("")
    print("policy_yaml:")
    print(f"  camera_width: {args.width}")
    print(f"  camera_height: {args.height}")
    print(f"  camera_fx: {fx:.6f}")
    print(f"  camera_fy: {fy:.6f}")
    print(f"  camera_cx: {cx:.6f}")
    print(f"  camera_cy: {cy:.6f}")
    print("  depth_min_m: 0.02      # 按任务工作空间调整")
    print("  depth_max_m: 1.5       # 按任务工作空间调整")
    print("  depth_unit_scale: 1.0  # MuJoCo 32FC1 depth 已是 meters")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
