#!/bin/bash
set -e

# Ensure we are in the local/ directory
cd "$(dirname "$0")"

echo "===================================================="
echo "1. Building and starting local Docker stack..."
echo "===================================================="
docker compose build
docker compose up -d
echo "Waiting for services to initialize (giving sandbox and agent servers 30s to be fully healthy)..."
sleep 30

echo ""
echo "===================================================="
echo "2. Checking container status..."
echo "===================================================="
docker compose ps

USER_PROMPT="${1:-Solve this ML problem with an appropiate choice of Deep Learning Model}"

echo ""
echo "===================================================="
echo "3. Starting 5 batch runs for each dataset..."
echo "===================================================="
cd ..

DATASETS="tabular-playground-series-may-2022,dog-breed-identification,mlsp-2013-birds"

for run in {1..5}; do
  echo ""
  echo "----------------------------------------------------"
  echo "Starting Run Batch $run of 5..."
  echo "----------------------------------------------------"
  python local/run_benchmark.py \
    --datasets "$DATASETS" \
    --config-file local/config.json \
    --max-iterations 40 \
    --user-prompt "$USER_PROMPT"
done

echo ""
echo "===================================================="
echo "All done! Monitor runs/ for telemetry and output logs."
echo "===================================================="
