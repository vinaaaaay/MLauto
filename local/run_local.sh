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

USER_PROMPT="${1:-}"

echo ""
echo "===================================================="
echo "3. Starting benchmark run on tabular dataset..."
echo "===================================================="
cd ..
python local/run_benchmark.py --datasets tabular-playground-series-may-2022 --config-file local/config.json --max-iterations 10 --user-prompt "$USER_PROMPT"

echo ""
echo "===================================================="
echo "All done! Monitor runs/ for telemetry and output logs."
echo "===================================================="
