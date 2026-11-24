#!/usr/bin/env python3
"""Console entry point for the MuJoCo Dora node."""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .config import load_config
from .runtime import start_runtime


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MuJoCo dora 节点")
    parser.add_argument(
        "--version",
        action="version",
        version=f"mujoco-sim {__version__}",
    )
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
