#!/bin/bash
set -e

# Ensure we are in the local/ directory
cd "$(dirname "$0")"

echo "===================================================="
echo "1. Building and starting local Docker stack..."
echo "===================================================="

# Only rebuild if images are missing or FORCE_BUILD=1 is set
if [ "${FORCE_BUILD:-0}" = "1" ] || ! docker image inspect local-orchestrator >/dev/null 2>&1; then
  echo "Building images (first run or FORCE_BUILD=1)..."
  docker compose build
else
  echo "Images already exist, skipping build. Set FORCE_BUILD=1 to force rebuild."
fi

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
python local/run_benchmark.py --datasets tabular-playground-series-may-2022 --config-file local/config.json --max-iterations 40 --user-prompt "$USER_PROMPT"

echo ""
echo "===================================================="
echo "All done! Monitor runs/ for telemetry and output logs."
echo "===================================================="
