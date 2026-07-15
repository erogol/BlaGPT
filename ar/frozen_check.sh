#!/bin/bash
# frozen_check.sh — verifies the FROZEN harness invariants after an experiment.
# Exit 0 = OK, 1 = VIOLATION (run is void regardless of val_loss).
set -u
R=/nvme/BlaGPT/bla_gpt
fail=0
chk() { grep -qF "$1" "$2" || { echo "FROZEN VIOLATION: missing '$1' in $2"; fail=1; }; }

# 1. time-budget machinery
chk 'AR_TIME_BUDGET = float(os.environ.get("AR_TIME_BUDGET", "600"))' "$R/train_ar.py"
chk 'dist.all_reduce(_stop, op=dist.ReduceOp.MAX)' "$R/train_ar.py"
chk 'FATAL: NaN train loss' "$R/train_ar.py"

# 2. final eval + summary block format
chk 'print(f"final_val_loss:   {val_loss:.6f}")' "$R/train_ar.py"
chk 'print(f"training_seconds: {training_time_ms / 1000:.1f}")' "$R/train_ar.py"

# 2b. fixed val data (pattern + val token budget in train_ar.py)
chk 'fineweb_val_*.bin' "$R/train_ar.py"
chk 'val_tokens: int = 10485760' "$R/train_ar.py"

# 2c. no val_tokens override via experiment config (pass config.json as $1)
if [ "${1:-}" != "" ] && [ -f "$1" ]; then
    python3 - "$1" <<'PYEOF' || fail=1
import json, sys
cfg = json.load(open(sys.argv[1]))
vt = cfg.get("val_tokens", 10485760)
if vt != 10485760:
    print(f"FROZEN VIOLATION: config overrides val_tokens={vt}")
    sys.exit(1)
PYEOF
fi

# 2d. no pretrained weights / external artifacts in experiment diffs
# (scan lines ADDED since base commit for banned loading patterns)
BANNED='torch\.load|from_pretrained|hf_hub|safetensors|urlretrieve|urlopen|requests\.get'
if (cd /nvme/BlaGPT && git diff 9740ded -- bla_gpt/ | grep '^+' | grep -Ev '^\+\+\+' | grep -Eq "$BANNED"); then
    echo "FROZEN VIOLATION: banned weight-loading/download pattern added since base commit"
    (cd /nvme/BlaGPT && git diff 9740ded -- bla_gpt/ | grep '^+' | grep -E "$BANNED" | head -5)
    fail=1
fi

# 2e. clock-integrity tripwire (non-fatal): flag diffs touching time accounting
if (cd /nvme/BlaGPT && git diff f32b8a3 -- bla_gpt/train_ar.py | grep '^+' | grep -Ev '^\+\+\+' | grep -Eq 'training_time_ms|t0 = time'); then
    echo "WARNING: diff touches clock accounting (training_time_ms/t0) — driver must review for free-compute windows"
fi

# 3. runner untouched
if [ -f /nvme/ar/run_experiment.sh.sha256 ]; then
    (cd /nvme/ar && sha256sum -c --quiet run_experiment.sh.sha256) \
        || { echo "FROZEN VIOLATION: run_experiment.sh modified"; fail=1; }
fi

[ "$fail" -eq 0 ] && echo "frozen_check: OK"
exit "$fail"
