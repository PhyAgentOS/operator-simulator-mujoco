#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv_build"

cleanup() {
  rm -rf "${VENV_DIR}"
}
trap cleanup EXIT

rm -rf "${VENV_DIR}"
echo "==> [mujoco_sim] Installing locked build environment..."
UV_PROJECT_ENVIRONMENT="${VENV_DIR}" uv sync \
  --project "${ROOT_DIR}" \
  --frozen \
  --no-dev \
  --group build

mkdir -p "${ROOT_DIR}/dist" "${ROOT_DIR}/build/pyinstaller/mujoco_sim"

echo "==> [mujoco_sim] Building standalone executable..."
(
  cd "${ROOT_DIR}"
  "${VENV_DIR}/bin/pyinstaller" \
    --noconfirm \
    --clean \
    --distpath "${ROOT_DIR}/dist" \
    --workpath "${ROOT_DIR}/build/pyinstaller/mujoco_sim" \
    "${SCRIPT_DIR}/mujoco_sim.spec"
)

DIST_FILE="${ROOT_DIR}/dist/mujoco_sim"
if [[ ! -f "${DIST_FILE}" ]]; then
  echo "ERROR: dist/mujoco_sim was not produced" >&2
  exit 1
fi

EXPECTED_VERSION="$("${VENV_DIR}/bin/python" -c 'import importlib.metadata; print("mujoco-sim " + importlib.metadata.version("mujoco-sim"))')"
ACTUAL_VERSION="$("${DIST_FILE}" --version)"
if [[ "${ACTUAL_VERSION}" != "${EXPECTED_VERSION}" ]]; then
  echo "ERROR: version mismatch: expected '${EXPECTED_VERSION}', got '${ACTUAL_VERSION}'" >&2
  exit 1
fi

echo "==> [mujoco_sim] Built ${DIST_FILE} (${ACTUAL_VERSION})"
