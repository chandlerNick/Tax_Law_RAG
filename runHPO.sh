#!/bin/bash

# Shared log file on your mounted volume
LOG_FILE="/DL-Data/output/grid_search.log"
mkdir -p "$(dirname "$LOG_FILE")"

# Grid values
lrs=("2e-5" "3e-5")
batch_sizes=(8 16)
epochs=(3 4)

# Loop over combinations
for lr in "${lrs[@]}"; do
  for bs in "${batch_sizes[@]}"; do
    for ep in "${epochs[@]}"; do
      job_id="lr${lr}_bs${bs}_ep${ep}"

      echo "[$(date)] Running job $job_id..." | tee -a "$LOG_FILE"
      python FineTuneBERT.py \
        --job_id "$job_id" \
        --lrs "$lr" \
        --batch_sizes "$bs" \
        --epochs "$ep" \
        >> "$LOG_FILE" 2>&1

      echo "[$(date)] Finished job $job_id" | tee -a "$LOG_FILE"
      echo "----------------------------------------" >> "$LOG_FILE"
    done
  done
done