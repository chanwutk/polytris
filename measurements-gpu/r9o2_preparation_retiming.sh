#!/bin/bash
# R9.O2 preparation re-timing on bombe: measurements-gpu/r9o2_preparation_cost.py --run inside the shared launcher
# (measurements-gpu/bombe_container.sh), one container per GPU, each timing its own datasets in sequence.
# Usage, from the checkout /work/cwkt/projects/tetris (the same files on ace and bombe):
#   ace:   measurements-gpu/r9o2_preparation_retiming.sh stage <GPU> <dataset...>
#            Push every tracking.jsonl under the datasets' execution/va*/ in ace's cache (read only) into the GPU's
#            staging cache on bombe, $TRACK/g<GPU>/polyis-cache, which the launcher mounts at /polyis-cache. The plan
#            copies them from there into its scratch cache (copy-on-write clones on bombe's ZFS).
#   bombe: measurements-gpu/r9o2_preparation_retiming.sh run <GPU> <dataset...>
#            Time STEPS for the datasets on that GPU, pinned to its NUMA node; the plan's scratch is $TRACK/g<GPU>/retime
#            (/scratch/retime in the container). Run it in a detached tmux session; it prints the plan, then one line
#            per step.
#   bombe: measurements-gpu/r9o2_preparation_retiming.sh collect
#            Merge every GPU's records into measurements-results/r9o2_preparation_retiming/ (timings.jsonl,
#            detector_load.jsonl, files.jsonl.gz, launcher_runs.jsonl, logs/), which measurements/r9o2_preparation_cost.py
#            reads on ace.
# Optional environment: R9O2_STEPS replaces the steps that run times (for example 'p110_density', the deployed-density
# cross-check); R9O2_TRACK and R9O2_TETRIS move the track directory and the checkout (tests run collect on a fake one).
set -euo pipefail
USAGE="usage: $0 stage <GPU> <dataset...> | run <GPU> <dataset...> | collect"
TRACK=${R9O2_TRACK:-/data/cwkt/data/bombe-revision/scratch/r9o2-retiming}
TETRIS=${R9O2_TETRIS:-/work/cwkt/projects/tetris}
BOMBE=bombe.millennium.berkeley.edu
# Every polyis step of the plan by default; p011 runs before p012 and p016, so they read its fresh detections.
STEPS=${R9O2_STEPS:-p110 p011 p012 p013 detector_load p016}
CMD=${1:?$USAGE}
shift
case "$CMD" in
  stage)
    G=${1:?$USAGE}
    shift
    [ $# -gt 0 ] || { echo "$USAGE" >&2; exit 2; }
    # Paths relative to ace's cache root, so rsync recreates <dataset>/execution/<video>/... on bombe.
    cd /data/cwkt/data/polyis-cache
    LIST=$(mktemp)
    trap 'rm -f "$LIST"' EXIT
    for ds in "$@"; do find "$ds"/execution/va* -name tracking.jsonl; done > "$LIST"
    ssh "$BOMBE" mkdir -p "$TRACK/g$G/polyis-cache"
    rsync -a --files-from="$LIST" . "$BOMBE:$TRACK/g$G/polyis-cache/"
    echo "staged $(wc -l < "$LIST") tracking files of $* into $BOMBE:$TRACK/g$G/polyis-cache"
    ;;
  run)
    G=${1:?$USAGE}
    shift
    [ $# -gt 0 ] || { echo "$USAGE" >&2; exit 2; }
    cd "$TETRIS"
    # The launcher shows the container only its GPU, as cuda:0.
    exec measurements-gpu/bombe_container.sh "$G" "$TRACK/g$G" \
      "python measurements-gpu/r9o2_preparation_cost.py --run --gpu 0 --scratch /scratch/retime --steps $STEPS --datasets $*"
    ;;
  collect)
    OUT=$TETRIS/measurements-results/r9o2_preparation_retiming
    mkdir -p "$OUT/logs"
    # One file per record kind across both GPUs (each GPU's plan appended to its own scratch copy).
    cat "$TRACK"/g*/retime/timings.jsonl > "$OUT/timings.jsonl"
    cat "$TRACK"/g*/retime/detector_load.jsonl > "$OUT/detector_load.jsonl"
    cat "$TRACK"/g*/retime/files.jsonl | gzip > "$OUT/files.jsonl.gz"
    # The launcher's run summaries (GPU, NUMA node, CPU set, image, contaminated) of every container started here.
    cat "$TRACK"/g*/samples/*.json > "$OUT/launcher_runs.jsonl"
    # Step logs and each GPU's driver log, prefixed with the GPU (gitignored).
    for g in "$TRACK"/g*; do
      for f in "$g"/retime/logs/*.log "$g"/*.log; do
        if [ -e "$f" ]; then cp "$f" "$OUT/logs/$(basename "$g")_$(basename "$f")"; fi
      done
    done
    echo "collected $(wc -l < "$OUT/timings.jsonl") step records into $OUT"
    ;;
  *)
    echo "$USAGE" >&2
    exit 2
    ;;
esac
