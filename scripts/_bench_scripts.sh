#!/bin/bash
# Time each stage of the sequential scripts/ pipeline end-to-end.
# Run via: docker exec polyis bash -c "cd /polyis && CONFIG=bench.yaml bash scripts/_bench_scripts.sh"

set -e
cd /polyis
export CONFIG=bench.yaml

echo "=== SCRIPTS PIPELINE TIMING (CONFIG=$CONFIG) ==="

# Python helper for sub-second timing.
now() { python -c 'import time; print(time.time())'; }
elapsed() { python -c "print(f'{$1 - $2:.2f}')"; }

total_start=$(now)

stages=(
  scripts/p020_exec_classify.py
  scripts/p022_exec_prune_polyominoes.py
  scripts/p030_exec_compress.py
  scripts/p040_exec_detect.py
  scripts/p050_exec_uncompress.py
  scripts/p060_exec_track.py
)

for stage in "${stages[@]}"; do
  echo ""
  echo "--- $stage ---"
  start=$(now)
  python "$stage" --valid > /tmp/bench_${stage##*/}.log 2>&1
  rc=$?
  end=$(now)
  d=$(elapsed "$end" "$start")
  echo "${stage}: ${d}s (rc=$rc)"
  if [ "$rc" -ne 0 ]; then
    echo "=== stage failed ==="
    tail -50 /tmp/bench_${stage##*/}.log
    exit 1
  fi
done

total_end=$(now)
t=$(elapsed "$total_end" "$total_start")
echo ""
echo "TOTAL: ${t}s"
