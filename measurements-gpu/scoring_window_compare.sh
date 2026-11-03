#!/usr/bin/env bash
# Item 1: rescore every dataset with the TrackEval scoring window padded, to compare with the current results.
#
# Run on ace (host), inside tmux, after the B3D human-GT phases:  measurements-gpu/scoring_window_compare.sh
# The current (human ground truth, unpadded) outputs are kept as <path>.human-gt; the padded results replace them in place.
# Stops at the first failure; each stage writes a done marker, so a rerun resumes after the last finished stage.
set -euo pipefail

ALL=(caldot1-y05 caldot2-y05 ams-y05 jnc0 jnc2 jnc6 jnc7)
OUT=/work/cwkt/projects/tetris/measurements-results/scoring_window
COUT=/polyis/measurements-results/scoring_window
CACHE=/polyis-cache
PAPER=/polytris-paper
mkdir -p "$OUT/done"
exec > >(tee -a "$OUT/run.log") 2>&1

# Python in the polyis container on 40 cores, with padding on or off.
padded() { docker exec -i -w /polyis -e POLYIS_PAD_SCORING_WINDOW=1 polyis taskset -c 0-39 python "$@"; }
plain() { docker exec -i -w /polyis polyis taskset -c 0-39 python "$@"; }
stage() {
  local name=$1; shift
  if [[ -e "$OUT/done/$name" ]]; then echo "== skip $name (done)"; return; fi
  echo "== $(date -Is) start $name"
  "$@"
  touch "$OUT/done/$name"
  echo "== $(date -Is) done $name"
}

SNAPSHOT=("$PAPER/build/figure8-alternative" "$PAPER/figures/generated")
for d in "${ALL[@]}"; do
  SNAPSHOT+=("$CACHE/$d/evaluation/070_accuracy" "$CACHE/$d/evaluation/090_tradeoff")
  for s in otif leap; do SNAPSHOT+=("$CACHE/SOTA/$s/$d/accuracy" "$CACHE/SOTA/$s/$d/tradeoff.csv"); done
done

stage C01_snapshot plain measurements/snapshot_outputs.py --suffix .human-gt "${SNAPSHOT[@]}"
stage C02_p110 padded evaluation/p110_accuracy_compute.py --test
stage C03_p111 padded evaluation/p111_accuracy_aggregate.py --test
stage C04_p130 padded evaluation/p130_tradeoff_compute.py --test
stage C05_p141 padded evaluation/p141_otif_accuracy.py --test
stage C06_p142 padded evaluation/p142_otif_tradeoff.py --test
# Gates and per-dataset report against the unpadded snapshot (no tables are expected to stay byte-identical).
echo '{}' > "$OUT/no_untouched.json"
stage C07_report plain measurements/b3d_human_gt_report.py --datasets "${ALL[@]}" --suffix .human-gt \
  --untouched-checksums "$COUT/no_untouched.json" --output "$COUT/report.json"
# Matched comparisons (5%/10% points, prior matches, headline macros) on the padded tables, in ace's paper build only.
stage C08_render plain "$PAPER/scripts/render_revised_hota.py" --alternative
echo "== $(date -Is) scoring-window comparison complete"
