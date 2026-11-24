#!/usr/bin/env bash
# Rescore the mistrack-rate ablation (Fig. 11) for B3D on human ground truth; BYTETrack only.
set -euo pipefail
LOG=/work/cwkt/projects/tetris/measurements-results/b3d_human_gt/fig11.log
exec > >(tee -a "$LOG") 2>&1
py() { docker exec -i -w /polyis polyis taskset -c 0-39 python "$@"; }
T=bytetrackcython
echo "== $(date -Is) snapshot"
py measurements/snapshot_outputs.py $(for d in jnc0 jnc2 jnc6 jnc7; do echo /polyis-cache/$d/ablation/mistrack-rate/3x3/$T/test; done) \
  /polytris-paper/figures/generated/pruning_vs_hota.png /polytris-paper/figures/generated/mistrack_rate_heuristic_hota_loss_macros_$T.tex
for d in jnc0 jnc2 jnc6 jnc7; do docker exec polyis rm /polyis-cache/$d/ablation/mistrack-rate/3x3/$T/test/results.csv; done
echo "== $(date -Is) evaluate"
py ablation/mistrack-rate/run.py --datasets jnc0,jnc2,jnc6,jnc7 --tracker $T --num-workers 40
echo "== $(date -Is) combine"
py ablation/mistrack-rate/run.py --combine-visualize --tracker $T
echo "== $(date -Is) fig11 complete"
