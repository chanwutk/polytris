#!/usr/bin/env bash
# Phase A of re-scoring B3D on human ground truth: snapshot, install, rescore, gate.
#
# Run on ace (host), inside tmux:  measurements-gpu/b3d_human_gt_phase_a.sh <human-annotation commit>
# Stops at the first failure. Each stage writes a done marker, so a rerun resumes after the last
# finished stage. Logs and the report go to measurements-results/b3d_human_gt/.
set -euo pipefail

COMMIT=${1:?usage: $0 <human-annotation commit of the staged labels>}
B3D=(jnc0 jnc2 jnc6 jnc7)
OTHERS=(caldot1-y05 caldot2-y05 ams-y05)
OUT=/work/cwkt/projects/tetris/measurements-results/b3d_human_gt
# Container-side paths of the same directory and of the cache.
COUT=/polyis/measurements-results/b3d_human_gt
CACHE=/polyis-cache
mkdir -p "$OUT/done"
exec > >(tee -a "$OUT/phase_a.log") 2>&1

# Run a Python entry point in the polyis container on 40 cores, optionally limited to some datasets.
py() { docker exec -i -w /polyis -e POLYIS_DATASETS="${DATASETS:?set DATASETS}" polyis taskset -c 0-39 python "$@"; }
py_all() { docker exec -i -w /polyis polyis taskset -c 0-39 python "$@"; }
# Run a stage once: skip it when its done marker exists, otherwise run and then mark it.
stage() {
  local name=$1; shift
  if [[ -e "$OUT/done/$name" ]]; then echo "== skip $name (done)"; return; fi
  echo "== $(date -Is) start $name"
  "$@"
  touch "$OUT/done/$name"
  echo "== $(date -Is) done $name"
}

# Outputs this phase overwrites, as container paths.
snapshot_paths() {
  for d in "${B3D[@]}"; do
    echo "$CACHE/$d/evaluation/070_accuracy" "$CACHE/$d/evaluation/090_tradeoff"
    for s in otif leap; do echo "$CACHE/SOTA/$s/$d/accuracy" "$CACHE/SOTA/$s/$d/tradeoff.csv"; done
  done
}

# Checksums of non-B3D tables that must stay byte-identical.
record_untouched() {
  py_all - "$COUT/untouched_checksums.json" "${OTHERS[@]}" <<'EOF'
import hashlib, json, sys
out, datasets = sys.argv[1], sys.argv[2:]
paths = [f'/polyis-cache/{d}/evaluation/{p}' for d in datasets for p in ('090_tradeoff/tradeoff.csv', '070_accuracy/accuracy.csv')]
paths += [f'/polyis-cache/SOTA/{s}/{d}/tradeoff.csv' for s in ('otif', 'leap') for d in datasets]
json.dump({p: hashlib.sha256(open(p, 'rb').read()).hexdigest() for p in paths}, open(out, 'w'), indent=2)
print(f'recorded {len(paths)} checksums')
EOF
}

stage 00_untouched_checksums record_untouched
# shellcheck disable=SC2046
stage 01_snapshot py_all measurements/snapshot_outputs.py $(snapshot_paths)
stage 02_install py_all preprocess/p004_install_human_groundtruth.py --source-commit "$COMMIT" --datasets "${B3D[@]}"
export DATASETS=$(IFS=,; echo "${B3D[*]}")
stage 03_p110 py evaluation/p110_accuracy_compute.py --test
stage 04_p111 py evaluation/p111_accuracy_aggregate.py --test
stage 05_p130 py evaluation/p130_tradeoff_compute.py --test
stage 06_p141 py evaluation/p141_otif_accuracy.py --test
stage 07_p142 py evaluation/p142_otif_tradeoff.py --test
stage 08_report py_all measurements/b3d_human_gt_report.py --datasets "${B3D[@]}" \
  --untouched-checksums "$COUT/untouched_checksums.json" --output "$COUT/report.json"
echo "== $(date -Is) phase A complete"
