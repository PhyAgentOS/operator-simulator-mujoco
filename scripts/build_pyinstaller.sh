#!/usr/bin/env bash
# 专用于局部虚拟环境隔离打包的统一构建脚本。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 寻找 forge_runtime 根目录 (含有 pyproject.toml 且名字为 forge-runtime)
FIND_DIR="${SCRIPT_DIR}"
ROOT_DIR=""
while [[ "${FIND_DIR}" != "/" ]]; do
  if [[ -f "${FIND_DIR}/pyproject.toml" ]] && grep -q 'name = "forge-runtime"' "${FIND_DIR}/pyproject.toml"; then
    ROOT_DIR="${FIND_DIR}"
    break
  fi
  FIND_DIR="$(dirname "${FIND_DIR}")"
done

if [[ -z "${ROOT_DIR:-}" ]]; then
  echo "ERROR: 无法定位 forge_runtime 根目录" >&2
  exit 1
fi

echo "==> [mujoco_sim] 正在初始化隔离的局部构建虚拟环境..."
VENV_DIR="${NODE_DIR}/.venv_build"
rm -rf "${VENV_DIR}"

# 1. 建立纯净隔离虚拟环境
uv venv --no-workspace "${VENV_DIR}" --python 3.12

# 2. 安装本地包依赖和打包器 pyinstaller
uv pip install --python "${VENV_DIR}/bin/python" -r "${NODE_DIR}/pyproject.toml" pyinstaller

echo "==> [mujoco_sim] 开始使用 PyInstaller 进行打包..."
# 3. 运行 pyinstaller 打包
"${VENV_DIR}/bin/pyinstaller" \
  --noconfirm \
  --clean \
  "${SCRIPT_DIR}/mujoco_sim.spec"

# 4. 清理编译虚拟环境
rm -rf "${VENV_DIR}"

DIST_FILE="${ROOT_DIR}/dist/mujoco_sim"

if [[ -f "${DIST_FILE}" ]]; then
  echo "==> [mujoco_sim] 构建成功：${DIST_FILE}"
else
  echo "WARNING: dist/mujoco_sim 未找到，请检查 PyInstaller 输出。" >&2
  exit 1
fi
