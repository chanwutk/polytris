#!/usr/bin/env bash
# Rescore Fig. 11's CalDoT1/CalDoT2 panels with the canonical (padded) scoring window; BYTETrack only.
set -euo pipefail
LOG=/work/cwkt/projects/tetris/measurements-results/scoring_window/fig11_padded.log
exec > >(tee -a "$LOG") 2>&1
py() { docker exec -i -w /polyis polyis taskset -c 0-39 python "$@"; }
T=bytetrackcython
echo "== $(date -Is) snapshot"
py measurements/snapshot_outputs.py --suffix .pre-padding $(for d in caldot1-y05 caldot2-y05; do echo /polyis-cache/$d/ablation/mistrack-rate/3x3/$T/test; done)
for d in caldot1-y05 caldot2-y05; do docker exec polyis rm /polyis-cache/$d/ablation/mistrack-rate/3x3/$T/test/results.csv; done
echo "== $(date -Is) evaluate"
py ablation/mistrack-rate/run.py --datasets caldot1-y05,caldot2-y05 --tracker $T --num-workers 40
echo "== $(date -Is) combine"
py ablation/mistrack-rate/run.py --combine-visualize --tracker $T
echo "== $(date -Is) fig11 padded complete"
