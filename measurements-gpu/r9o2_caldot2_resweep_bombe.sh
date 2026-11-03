#!/bin/bash
# R9.O2: CalDoT2's validation sweep on va00-va09, whose outputs on ace were cleared on 2026-05-29, re-run on bombe
# because ace's GPUs are busy. Runs inside measurements-gpu/bombe_container.sh, whose scratch cache (/polyis-cache)
# holds CalDoT2's mirrored indexing outputs and whose CONFIG lists only caldot2-y05, so it writes only to scratch.
# Each stage runs on the cleared clips only (--videos), then the serial packing measurement adds their packing units.
# Usage on bombe (in tmux):
#   measurements-gpu/bombe_container.sh 4 /data/cwkt/data/bombe-revision/scratch/r9o2-caldot2-resweep \
#     'bash measurements-gpu/r9o2_caldot2_resweep_bombe.sh' >> <scratch>/run.log 2>&1
set -euo pipefail
VIDEOS=(va00.mp4 va01.mp4 va02.mp4 va03.mp4 va04.mp4 va05.mp4 va06.mp4 va07.mp4 va08.mp4 va09.mp4)
STAGES=(p020_exec_classify p022_exec_prune_polyominoes p030_exec_compress p040_exec_detect p050_exec_uncompress p060_exec_track)
for stage in "${STAGES[@]}"; do
    echo "$(date -Is) start $stage"
    python "scripts/$stage.py" --valid --videos "${VIDEOS[@]}"
    echo "$(date -Is) end $stage"
done
# The serial packing measurement, pinned to 4 CPUs of GPU 4's NUMA node, as each dataset's run on ace was.
echo "$(date -Is) start packing_serial"
python measurements/r9o2_packing_serial.py --dataset caldot2-y05 --cpus 20,21,22,23
echo "$(date -Is) done"
