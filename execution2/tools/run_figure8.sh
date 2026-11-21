#!/usr/bin/env bash
# Run execution2 on every Figure 8 full-Tetris config, one at a time on one GPU, then summarize.
# Meant for a tmux session on ace (survives the client disconnecting):
#   tmux new-session -d -s execution2-figure8 \
#     'bash /work/cwkt/projects/tetris/execution2/tools/run_figure8.sh 2 2>&1 | tee -a /work/cwkt/projects/tetris/measurements-results/execution2-figure8/run.log'
# Arguments: <gpu index> [passes, default 8] [minutes between passes, default 10].
# Each pass runs every config that has no result.json yet, so configs that failed (e.g. pruning configs while
# the Gurobi web license has no free session) are retried in later passes.  A finished run is never redone;
# a failed run's directory is kept as <dir>.failed-<time>.  Never writes to the real cache: each run's
# /polyis-cache is its own scratch directory, and the real cache is mounted read-only.
set -u
GPU=${1:?usage: run_figure8.sh <gpu index> [passes] [minutes between passes]}
PASSES=${2:-8}
WAIT_MIN=${3:-10}
REPO=/work/cwkt/projects/tetris
IMAGE=polyis-execution2-bench
OUT=$REPO/measurements-results/execution2-figure8/runs
mkdir -p "$OUT"

gurobi_status() {
  # Whether a Gurobi session can be opened right now (pruning configs need one).
  docker run --rm --network host -v "$HOME"/.gurobi/gurobi.lic:/opt/gurobi/gurobi.lic:ro "$IMAGE" \
    python -c "import gurobipy; gurobipy.Model(); print('available')" 2>&1 | grep -E 'available|sessions|Error' | tail -1
}

for pass in $(seq 1 "$PASSES"); do
  remaining=0
  for cfg in $(ls "$REPO"/execution2/configs/figure8/*/*.yaml | sort); do
    [ -f "$OUT/$(basename "$(dirname "$cfg")")/$(basename "$cfg" .yaml)/result.json" ] || remaining=$((remaining + 1))
  done
  echo "$(date -Is) pass $pass/$PASSES: $remaining configs without a result; gurobi: $(gurobi_status)"
  [ "$remaining" -eq 0 ] && break

  for cfg in $(ls "$REPO"/execution2/configs/figure8/*/*.yaml | sort); do
    ds=$(basename "$(dirname "$cfg")")
    name=$(basename "$cfg" .yaml)
    run=$OUT/$ds/$name
    [ -f "$run/result.json" ] && continue
    if [ -e "$run" ]; then mv "$run" "$run.failed-$(date +%Y%m%dT%H%M%S)"; fi
    mkdir -p "$run/cache"
    echo "$(date -Is) start $ds/$name gpu[$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader -i "$GPU")] load[$(cut -d' ' -f1-3 /proc/loadavg)]"
    docker run --rm --gpus "\"device=$GPU\"" --shm-size 128g --network host -w /polyis \
      -v "$REPO":/polyis -v /data/cwkt/data/polyis-data:/polyis-data:ro \
      -v /work/cwkt/data/otif-dataset:/otif-dataset:ro -v "$HOME"/.gurobi/gurobi.lic:/opt/gurobi/gurobi.lic:ro \
      -v /data/cwkt/data/polyis-cache:/polyis-cache-ro:ro -v "$run/cache":/polyis-cache -v "$run":/bench \
      "$IMAGE" python -m execution2.tools.benchmark --dataset "$ds" --config "execution2/configs/figure8/$ds/$name.yaml" \
      --bench /bench --skip-scripts > "$run/benchmark.log" 2>&1
    status=$?
    reason=$(grep -hoE 'Too many sessions[^"]*|GRBstartenv failed[^"]*' "$run"/*.log 2>/dev/null | head -1)
    echo "$(date -Is) end $ds/$name exit $status $(grep -h '^configured' "$run/benchmark.log" | cut -c1-160) ${reason}"
  done
  [ "$pass" -lt "$PASSES" ] && { echo "$(date -Is) pass $pass done; waiting $WAIT_MIN min"; sleep $((WAIT_MIN * 60)); }
done

echo "$(date -Is) summarizing"
docker run --rm --network host -w /polyis -v "$REPO":/polyis "$IMAGE" python measurements/execution2_figure8_throughput.py
echo "$(date -Is) done"
