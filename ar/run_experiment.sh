#!/bin/bash
# FROZEN autoresearch harness. Usage: run_experiment.sh <run_id> [model_name] [config_json]
set -u
N=$1; MODEL=${2:-best}; CONFIG=${3:-}
AR=$(cd "$(dirname "$0")" && pwd)
cd "$AR/../bla_gpt"
mkdir -p "$AR/runs/$N"
CFGARG=""
if [ -n "$CONFIG" ]; then CFGARG="--config $CONFIG"; fi
AR_TIME_BUDGET=${AR_TIME_BUDGET:-600} timeout 1500 torchrun --standalone --nproc_per_node=8 \
  train_ar.py --run_name "ar_$N" --model_name "$MODEL" $CFGARG \
  > "$AR/runs/$N/run.log" 2>&1
grep "^final_val_loss:" "$AR/runs/$N/run.log"
