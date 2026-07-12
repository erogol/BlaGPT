#!/usr/bin/env python3
"""Post-run ledger recorder. Usage: record_run.py <id> <parent> <mech_name> <best_val> <best_label>

Appends a row to ar/full_results.tsv from ar/full_runs/<id>/run.log and commits,
so a completed run is never left unrecorded even if the supervisor dies.
Dedup-guarded: exits if the id already has a ledger row.
"""
import re, subprocess, sys

ID, PARENT, NAME, BEST, BEST_LABEL = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5]
ROOT = "/nvme/BlaGPT"
tsv = f"{ROOT}/ar/full_results.tsv"
if any(l.startswith(ID + "\t") for l in open(tsv)):
    sys.exit(f"{ID} already recorded")
log = open(f"{ROOT}/ar/full_runs/{ID}/run.log").read()
m = re.findall(r"step:5100/5100 val_loss:([0-9.]+)", log)
if not m:
    sys.exit(f"no final val_loss for {ID}; not recording")
val = float(m[-1])
commit = [l.split("=")[1][:7] for l in open(f"{ROOT}/ar/full_runs/{ID}/manifest.txt") if l.startswith("git_commit=")][0]
delta = val - BEST
if delta < 0:
    verdict = "keep" if -delta >= 0.003 else "keep_pending_confirmation"
    note = f"Full 5100-step run; {delta:+.4f} vs best {BEST_LABEL}={BEST}; " + (
        "clean keep, no confirmation required because improvement >=0.003" if verdict == "keep"
        else "improvement <0.003, requires one full confirmation from random init")
else:
    verdict = "discard"
    note = f"Full 5100-step run; {delta:+.4f} vs best {BEST_LABEL}={BEST}; no confirmation because not lower"
ckpt = f"bla_gpt/logs/ar_full_{ID}_0/state_step005100.pt"
with open(tsv, "a") as f:
    f.write(f"{ID}\t{commit}\t{PARENT}\t{val:.4f}\t{verdict}\t{NAME}\t{ckpt}\t{note}\n")
subprocess.run(["git", "add", "ar/full_results.tsv", f"ar/full_runs/{ID}"], cwd=ROOT, check=True)
subprocess.run(["git", "commit", "-q", "-m", f"autoresearch: record {ID} full run ({verdict}, {val:.4f} vs best {BEST})"], cwd=ROOT, check=True)
print(f"{ID} recorded: {val:.4f} {verdict}")
