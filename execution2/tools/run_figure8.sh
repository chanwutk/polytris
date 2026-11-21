#!/usr/bin/env bash
# Run execution2 on every Figure 8 full-Tetris config, one at a time on one GPU, then summarize.
# Meant for a tmux session on ace (survives the client disconnecting):
#   tmux new-session -d -s execution2-figure8 \
#     'bash /work/cwkt/projects/tetris/execution2/tools/run_figure8.sh 2 2>&1 | tee -a /work/cwkt/projects/tetris/measurements-results/execution2-figure8/run.log'
# Resumable: a config with a result.json is skipped; a partial run directory is kept as <dir>.failed-<time>.
# Never writes to the real cache: each run's /polyis-cache is its own scratch directory, the real one is read-only.
set -u
GPU=${1:?usage: run_figure8.sh <gpu index>}
REPO=/work/cwkt/projects/tetris
IMAGE=polyis-execution2-bench
OUT=$REPO/measurements-results/execution2-figure8/runs
mkdir -p "$OUT"

for cfg in $(ls "$REPO"/execution2/configs/figure8/*/*.yaml | sort); do
  ds=$(basename "$(dirname "$cfg")")
  name=$(basename "$cfg" .yaml)
  run=$OUT/$ds/$name
  if [ -f "$run/result.json" ]; then echo "$(date -Is) skip $ds/$name (done)"; continue; fi
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
  echo "$(date -Is) end $ds/$name exit $status $(grep -h '^configured' "$run/benchmark.log" | cut -c1-160)"
done

echo "$(date -Is) all configs attempted; summarizing"
docker run --rm --network host -w /polyis -v "$REPO":/polyis "$IMAGE" python measurements/execution2_figure8_throughput.py
echo "$(date -Is) done"
