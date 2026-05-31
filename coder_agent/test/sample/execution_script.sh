#!/usr/bin/env bash
set -euo pipefail

# Ensure virtual environment exists
if [ ! -d "/home/gem/workspace/.venv" ]; then
  uv venv /home/gem/workspace/.venv
fi

# Activate environment
source /home/gem/workspace/.venv/bin/activate

# Install required packages
uv pip install --prerelease=allow pandas scikit-learn joblib numpy

# Run the Python script
python /home/gem/workspace/iteration_0/generated_code.py