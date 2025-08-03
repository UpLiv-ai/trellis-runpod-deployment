#!/usr/bin/env bash
set -e
echo "[MODE=${MODE_TO_RUN}] starting at $(date)"

if [ "$MODE_TO_RUN" = "pod" ]; then
  echo "Running in POD mode (test mode)..."
  python handler.py
  echo "Pod mode test run completed. Container sleeping for debugging..."
  tail -f /dev/null

elif [ "$MODE_TO_RUN" = "serverless" ]; then
  echo "Starting Runpod Serverless handler..."
  python -u handler.py
else
  echo "ERROR: MODE_TO_RUN must be 'pod' or 'serverless'."
  exit 1
fi
