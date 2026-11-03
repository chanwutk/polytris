#!/usr/bin/env bash
# Phase B of re-scoring B3D on human ground truth: cross-dataset statistics, measurements, evidence, paper outputs.
#
# Run on ace (host), inside tmux, after phase A:  measurements-gpu/b3d_human_gt_phase_b.sh
# Writes into ace's paper checkout (/polytris-paper in the container); nothing is pushed anywhere.
# Stops at the first failure; each stage writes a done marker, so a rerun resumes after the last finished stage.
set -euo pipefail

B3D=(jnc0 jnc2 jnc6 jnc7)
OUT=/work/cwkt/projects/tetris/measurements-results/b3d_human_gt
COUT=/polyis/measurements-results/b3d_human_gt
RESULTS=/polyis/measurements-results
PAPER=/polytris-paper
EVIDENCE=$PAPER/scripts/data/revision_evidence
mkdir -p "$OUT/done_b"
exec > >(tee -a "$OUT/phase_b.log") 2>&1

# Run a Python entry point in the polyis container from /polyis on 40 cores, over every configured dataset.
py() { docker exec -i -w /polyis polyis taskset -c 0-39 python "$@"; }
# Same, from another working directory inside the container.
py_in() { local dir=$1; shift; docker exec -i -w "$dir" polyis taskset -c 0-39 python "$@"; }
# Run a stage once: skip it when its done marker exists, otherwise run and then mark it.
stage() {
  local name=$1; shift
  if [[ -e "$OUT/done_b/$name" ]]; then echo "== skip $name (done)"; return; fi
  echo "== $(date -Is) start $name"
  "$@"
  touch "$OUT/done_b/$name"
  echo "== $(date -Is) done $name"
}

# Everything this phase overwrites, as container paths.
SNAPSHOT=(
  "$PAPER/figures" "$PAPER/scripts/data"
  /polyis-cache/SUMMARY/101_compare_pareto /polyis-cache/SUMMARY/205_compare_classifiers
  /polyis-cache/SUMMARY/206_classifier_and_packing
)
for name in r7o1_hota_matched_speedup r1o4_packing_detection_accuracy r1o2_count_query r1o2_leap_native_queries \
            r1o2_leap_sampling_diagnostics r1o2_leap_sampling_unmasked r1o2_leap_original_evaluator \
            r9o1_classifier_cost r9o3_ilp_timings r9o3_ilp_instance_sizes r9o2_preparation_cost; do
  SNAPSHOT+=("$RESULTS/$name")
done

score_queries() {
  for d in "${B3D[@]}"; do
    py "$PAPER/scripts/score_leap_queries.py" --evaluator /polyis/baselines/tetris-leap/evaluate.py \
      --native-intervals "$EVIDENCE/native_intervals.json" --output "$RESULTS/r1o2_leap_original_evaluator/$d.json" --datasets "$d"
  done
}
evidence() {
  py measurements/b3d_human_gt_evidence.py --evidence "$EVIDENCE" --results "$RESULTS" \
    --leap-rerun "$COUT/leap_rerun/results-jnc0-jnc2-jnc6-jnc7.json" \
    --sampling "$RESULTS/r1o2_leap_sampling_diagnostics/results.json"
}

stage B01_snapshot py measurements/snapshot_outputs.py "${SNAPSHOT[@]}"
# Cross-dataset statistics and figures (all datasets; they read the rescored B3D tables).
stage B02_p201 py evaluation/p201_compare_pareto.py --test
stage B03_p203 py evaluation/p203_compare_stats.py --test
stage B04_p204 py evaluation/p204_ablation_stats.py --test
stage B05_p205 py evaluation/p205_compare_classifiers.py --test
stage B06_p206 py evaluation/p206_classifier_and_packing.py --test
# Measurements that select operating points or score against ground truth.
stage B07_r7o1 py measurements/r7o1_hota_matched_speedup.py
stage B08_count_query py measurements/r1o2_count_query.py
stage B09_native_queries py measurements/r1o2_leap_native_queries.py
stage B10_leap_rerun py measurements/r1o2_leap_otif_rerun.py --datasets "${B3D[@]}" --output-dir "$COUT/leap_rerun"
stage B11_sampling py measurements/r1o2_leap_sampling_diagnostics.py --datasets "${B3D[@]}"
stage B12_score_queries score_queries
stage B13_refresh_leap_matched py "$PAPER/scripts/refresh_leap_matched.py" --tetris-root /polyis
stage B14_r9o1_cost py measurements/r9o1_classifier_cost.py
stage B15_r9o3_timings py measurements/r9o3_ilp_timings.py
stage B16_r9o3_sizes py measurements/r9o3_ilp_instance_sizes.py
stage B17_r9o2_cost py measurements/r9o2_preparation_cost.py
stage B18_r1o4 py measurements/r1o4_packing_detection_accuracy.py
stage B19_r1o4_map py "$RESULTS/r1o4_packing_detection_accuracy/map.py"
# Evidence: splice B3D, rerun the duration statistics on the spliced file, then re-copy.
stage B20_evidence evidence
stage B21_duration py_in "$RESULTS/r1o2_leap_sampling_unmasked" duration_variability.py
stage B22_evidence_again evidence
# Paper outputs, written into ace's paper checkout.
stage B23_render_hota py "$PAPER/scripts/render_revised_hota.py" --alternative
stage B24_query_pareto py "$PAPER/scripts/generate_query_pareto.py"
stage B25_revision_figures py "$PAPER/scripts/generate_revision_figures.py"
echo "== $(date -Is) phase B complete"
