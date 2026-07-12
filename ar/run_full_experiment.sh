#!/bin/bash
# Full-training autoresearch runner. Usage: run_full_experiment.sh <id> <config.json>
set -euo pipefail
ID=$1
CONFIG=$2
ROOT=/nvme/BlaGPT
RUN_DIR="$ROOT/ar/full_runs/$ID"
RUN_NAME="ar_full_${ID}"
mkdir -p "$RUN_DIR"
cp "$CONFIG" "$RUN_DIR/config.json"
cd "$ROOT"
{
  echo "started_utc=$(date -u +%FT%TZ)"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "run_name=$RUN_NAME"
  echo "config=$RUN_DIR/config.json"
} > "$RUN_DIR/manifest.txt"
cd "$ROOT/bla_gpt"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set +e
torchrun --standalone --nproc_per_node=8 train.py \
  --model_name best --config "$RUN_DIR/config.json" --run_name "$RUN_NAME" \
  > "$RUN_DIR/run.log" 2>&1
rc=$?
set -e
echo "exit_code=$rc" >> "$RUN_DIR/manifest.txt"
if [ "$rc" -eq 0 ]; then
  LOGDIR=$(find logs -maxdepth 1 -type d -name "${RUN_NAME}_*" | sort | tail -1)
  echo "checkpoint_dir=$ROOT/bla_gpt/$LOGDIR" >> "$RUN_DIR/manifest.txt"
  grep -E "step:5100/5100 val_loss:" "$RUN_DIR/run.log" | tail -1 | tee "$RUN_DIR/result.txt"
fi
exit "$rc"
