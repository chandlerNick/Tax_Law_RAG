#!/bin/bash
set -euo pipefail

# Shared log file on your mounted volume
LOG_FILE="/DL-data/output/grid_search.log"
mkdir -p "$(dirname "$LOG_FILE")"

PYTHON_BIN=$(which python)
echo "Using Python: $PYTHON_BIN" | tee -a "$LOG_FILE"


# Grid values
lrs_class=("5e-5" "1e-3")
lrs_bert=("1e-5" "2e-5")
batch_sizes=(8 16)
epochs=(3 4)

# Loop over combinations
for lr_class in "${lrs_class[@]}"; do
  for lr_bert in "${lrs_bert[@]}"; do
    for bs in "${batch_sizes[@]}"; do
      for ep in "${epochs[@]}"; do
        job_id="lr${lr}_bs${bs}_ep${ep}"

        echo "[$(date)] Running job $job_id..." | tee -a "$LOG_FILE"
        python FineTuneBERT_german.py \
          --job_id "$job_id" \
          --lrs_bert "$lr_bert" \
          --lrs_class "$lr_class" \
          --batch_sizes "$bs" \
          --epochs "$ep" \
          >> "$LOG_FILE" 2>&1

        echo "[$(date)] Finished job $job_id" | tee -a "$LOG_FILE"
        echo "----------------------------------------" >> "$LOG_FILE"
      done
    done
  done
done
