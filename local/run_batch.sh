#!/bin/bash
set -e

# Ensure we are in the local/ directory
cd "$(dirname "$0")"

echo "===================================================="
echo "1. Building and starting local Docker stack..."
echo "===================================================="

# Only rebuild if images are missing or FORCE_BUILD=1 is set
if [ "${FORCE_BUILD:-0}" = "1" ] || ! docker image inspect local-orchestrator > /dev/null 2>&1; then
  echo "Building images (first run or FORCE_BUILD=1)..."
  docker compose build
else
  echo "Images already exist, skipping build. Set FORCE_BUILD=1 to force rebuild."
fi

restart_and_wait_services() {
  echo "Stopping containers (fresh reset)..."
  docker compose down
  
  echo "Starting containers..."
  docker compose up -d

  echo "Waiting for all services to be healthy..."
  MAX_WAIT=120
  ELAPSED=0
  INTERVAL=5
  while true; do
    STATUS=$(docker inspect --format='{{.State.Health.Status}}' local-sandbox-1 2>/dev/null || echo "missing")
    
    # Check if services respond to health check
    curl -s -f --max-time 2 http://localhost:8000/health >/dev/null && ORCH_UP="up" || ORCH_UP="down"
    curl -s -f --max-time 2 http://localhost:8001/health >/dev/null && MCTS_UP="up" || MCTS_UP="down"
    curl -s -f --max-time 2 http://localhost:8020/health >/dev/null && PERC_UP="up" || PERC_UP="down"
    curl -s -f --max-time 2 http://localhost:8088/health >/dev/null && SEMA_UP="up" || SEMA_UP="down"
    curl -s -f --max-time 2 http://localhost:8089/health >/dev/null && CODER_UP="up" || CODER_UP="down"
    
    if [ "$STATUS" = "healthy" ] && [ "$ORCH_UP" = "up" ] && [ "$MCTS_UP" = "up" ] && [ "$PERC_UP" = "up" ] && [ "$SEMA_UP" = "up" ] && [ "$CODER_UP" = "up" ]; then
      echo "All services are up and healthy! (${ELAPSED}s)"
      break
    fi
    
    if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
      echo "Warning: Timed out waiting for service health after ${MAX_WAIT}s. Proceeding anyway."
      break
    fi
    
    echo "  sandbox: $STATUS, orch: $ORCH_UP, mcts: $MCTS_UP, perc: $PERC_UP, sema: $SEMA_UP, coder: $CODER_UP — waiting... (${ELAPSED}s)"
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
  done

  echo ""
  echo "Checking container status..."
  docker compose ps
}

restart_and_wait_services

USER_PROMPT="${1:-Solve this ML problem with an appropiate choice of Deep Learning Model. Also use lower presets when training with auto gluon. This system has 2 vcpus and 8GB RAM}"

echo ""
echo "===================================================="
echo "3. Starting 2 batch runs for each dataset..."
echo "===================================================="
cd ..

DATASETS="tabular-playground-series-may-2022,the-icml-2013-whale-challenge-right-whale-redux"

for run in {1..2}; do
  for dataset in ${DATASETS//,/ }; do
    echo ""
    echo "----------------------------------------------------"
    echo "Starting Run Batch $run of 2 for dataset: $dataset..."
    echo "----------------------------------------------------"
    python local/run_benchmark.py \
      --datasets "$dataset" \
      --config-file local/config.json \
      --max-iterations 40 \
      --max-runtime-seconds 14400 \
      --user-prompt "$USER_PROMPT"

    echo ""
    echo "===================================================="
    echo "Stopping and restarting containers to reset state..."
    echo "===================================================="
    cd local
    restart_and_wait_services
    cd ..
  done
done

echo ""
echo "===================================================="
echo "All done! Monitor runs/ for telemetry and output logs."
echo "===================================================="
