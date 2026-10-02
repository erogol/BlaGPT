#!/bin/bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT/bla_gpt"
RUN_NAME=${RUN_NAME:-combined_keeps_full_5100}
exec torchrun --standalone --nproc_per_node=8 train.py \
  --model_name best \
  --config "$ROOT/ar/full_train/combined_keeps.json" \
  --run_name "$RUN_NAME"
