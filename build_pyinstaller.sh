#!/usr/bin/env bash

set -euo pipefail

# 专用于打包 MuJoCo dora 节点的脚本。
# 需要：
#   - 已安装 mujoco、PyOpenGL、forge 相关依赖
#   - 已安装 PyInstaller（pip install pyinstaller）

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${ROOT_DIR}"

BIN_DIR="${ROOT_DIR}/bin"
mkdir -p "${BIN_DIR}"

echo "==> Building mujoco_sim with PyInstaller (using local mujoco_sim.spec)"

pyinstaller \
  --noconfirm \
  --clean \
  packages/nodes/simulators/mujoco/mujoco_sim.spec

if [[ -f "dist/mujoco_sim" ]]; then
  cp "dist/mujoco_sim" "${BIN_DIR}/mujoco_sim"
  echo "mujoco_sim built at ${BIN_DIR}/mujoco_sim"
else
  echo "WARNING: dist/mujoco_sim 未找到，请检查 PyInstaller 输出。" >&2
fi

