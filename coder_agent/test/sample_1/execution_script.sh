#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="/home/gem/workspace/.venv"
PYTHON_SCRIPT="/home/gem/workspace/iteration_0/generated_code.py"

# Ensure virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
  uv venv "$VENV_DIR"
fi

# Activate virtual environment
if [ -f "$VENV_DIR/bin/activate" ]; then
  source "$VENV_DIR/bin/activate"
else
  echo "Activation script not found: $VENV_DIR/bin/activate" >&2
  exit 1
fi

# Install required packages
uv pip install --prerelease=allow pandas scikit-learn numpy joblib

# Run the Python script
if [ -f "$PYTHON_SCRIPT" ]; then
  python "$PYTHON_SCRIPT"
else
  echo "Python script not found: $PYTHON_SCRIPT" >&2
  exit 1
fi