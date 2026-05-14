#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="/workspace/output/node_0/conda_env"

# Create the conda environment if it doesn't exist
if [ ! -d "$CONDA_ENV" ]; then
  echo "Creating conda environment at $CONDA_ENV with Python 3.11..."
  conda create -p "$CONDA_ENV" python=3.11 -y 2>/dev/null || true
else
  echo "Conda environment already exists at $CONDA_ENV"
fi

# Reference environment binaries by absolute path
PY="$CONDA_ENV/bin/python"
PIP="$CONDA_ENV/bin/pip"
UV="$CONDA_ENV/bin/uv"

if [ ! -x "$PY" ]; then
  echo "Error: Python executable not found in the conda environment at $PY"
  exit 1
fi

echo "Installing uv..."
"$PIP" install uv

echo "Installing required packages via uv (using Python: $PY)..."
"$UV" pip install --python "$PY" -r "/workspace/output/node_0/requirements_tool.txt" --prerelease=allow -r "/workspace/output/node_0/requirements_common.txt"

echo "Installing additional packages needed for the Python script..."
"$PIP" install pillow numpy scikit-learn joblib

echo "Running the Python script with the environment's Python..."
"$PY" "/workspace/output/node_0/generated_code.py"