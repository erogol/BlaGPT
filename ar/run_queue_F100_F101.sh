#!/bin/bash
# Sequential experiment queue: F100a/b/c (Aurora rms_match LR sweep) + F101 (Muon true wd=0)
set -u
ROOT=/nvme/BlaGPT
cd "$ROOT"
for SPEC in \
  "F100a:ar/full_train/configs/F100a_aurora_rms_lr0.003.json" \
  "F100b:ar/full_train/configs/F100b_aurora_rms_lr0.005.json" \
  "F100c:ar/full_train/configs/F100c_aurora_rms_lr0.01.json" \
  "F101:ar/full_train/configs/F101_muon_true_wd0.json"; do
  ID="${SPEC%%:*}"
  CFG="${SPEC#*:}"
  echo "=== QUEUE: starting $ID ($CFG) at $(date -u +%FT%TZ) ==="
  bash ar/run_full_experiment.sh "$ID" "$CFG"
  rc=$?
  echo "=== QUEUE: $ID finished rc=$rc at $(date -u +%FT%TZ) ==="
done
echo "=== QUEUE_DONE ==="
