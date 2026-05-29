#!/bin/bash
# Time execution/main.py end-to-end on the configured videos.
# Run via: docker exec polyis bash -c "cd /polyis && bash scripts/_bench_exec.sh"

cd /polyis

echo "=== EXECUTION PIPELINE TIMING ==="

now() { python -c 'import time; print(time.time())'; }
elapsed() { python -c "print(f'{$1 - $2:.2f}')"; }

start=$(now)
python execution/main.py \
  --dataset caldot2-y05 --videoset valid \
  --classifier ShuffleNet05 --tile-size 60 --sample-rate 16 \
  --tilepadding bl --canvas-scale 1 \
  --tracker sortcython --tracking-accuracy-threshold 0.4 \
  --relevance-threshold 0.5 \
  --classify-gpu 0 --detect-gpu 1 \
  --prune-workers 4 --compress-workers 8 \
  --max-videos-in-flight 2 \
  --no-warmup \
  > /tmp/bench_exec.log 2>&1
rc=$?
end=$(now)
d=$(elapsed "$end" "$start")
echo "execution/main.py: ${d}s (rc=$rc)"
echo ""
echo "=== Output from execution/main.py ==="
tail -15 /tmp/bench_exec.log
