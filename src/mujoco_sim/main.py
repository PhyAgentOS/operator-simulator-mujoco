#!/usr/bin/env python3
"""Console entry point for the MuJoCo Dora node."""

from __future__ import annotations

import argparse
import os
import platform
import sys

# 渲染后端：须在 import mujoco 之前设置。
# Linux 默认 egl 便于无显示器/SSH/服务器；Windows 不支持 egl，默认 glfw。
# 纯软件渲染可手动设 MUJOCO_GL=osmesa（需 libOSMesa）。
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "glfw" if platform.system() == "Windows" else "egl"

from .config import load_config
from .runtime import start_runtime


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MuJoCo dora 节点")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML 配置文件路径",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = load_config(config_path=args.config)
    return start_runtime(config)


def cli() -> int:
    """Console-script entry point."""
    try:
        return main()
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(cli())
