#!/usr/bin/env bash
set -euo pipefail

# 1) Create conda environment (skip if exists)
CONDA_ENV_PATH="/workspace/output/node_1/conda_env"
conda create -p "${CONDA_ENV_PATH}" python=3.11 -y 2>/dev/null || true

# 2) Reference environment binaries by full absolute path
PY="${CONDA_ENV_PATH}/bin/python"
PIP="${CONDA_ENV_PATH}/bin/pip"
UV="${CONDA_ENV_PATH}/bin/uv"

REPO_ROOT="/workspace/output/node_1"
REQUIREMENTS_TOOL="${REPO_ROOT}/requirements_tool.txt"
REQUIREMENTS_COMMON="${REPO_ROOT}/requirements_common.txt"

# 3) Install uv
"${PIP}" install uv

# 4) Install required packages (always pass --python "$PY" to uv pip install)
"${UV}" pip install --python "${PY}" -r "${REQUIREMENTS_TOOL}" --prerelease=allow -r "${REQUIREMENTS_COMMON}"

# 5) Run the Python script using the environment's Python
SCRIPT_PATH="/workspace/output/node_1/generated_code.py"
"${PY}" "${SCRIPT_PATH}"