#!/bin/bash
set -euo pipefail

# Shared log file on your mounted volume
LOG_FILE="/storage/output/save_bert.log"
mkdir -p "$(dirname "$LOG_FILE")"

PYTHON_BIN=$(which python)
echo "Using Python: $PYTHON_BIN" | tee -a "$LOG_FILE"

job_id="best_hp"

echo "[$(date)] Running job $job_id..." | tee -a "$LOG_FILE"
python save_BERT.py \
  >> "$LOG_FILE" 2>&1

echo "[$(date)] Finished job $job_id" | tee -a "$LOG_FILE"
echo "----------------------------------------" >> "$LOG_FILE"
