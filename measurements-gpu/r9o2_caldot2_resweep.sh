#!/usr/bin/env bash
# R9.O2: restore CalDoT2's validation sweep on va00-va09, whose sweep outputs were cleared on 2026-05-29 (only one
# configuration was rerun), so the preparation cost measures all 60 validation clips instead of scaling 50.
#
# Runs on ace's host (not in the container). It waits until measurements/r9o2_packing_serial.py has finished and every
# GPU is idle, then runs p020, p022, p030, p040, p050, and p060 on the validation split with --videos va00-va09 and a
# config listing only caldot2-y05, so no other clip or dataset is deleted or rerun. The trade-off and Pareto tables
# (p130, p135) are not rerun: CalDoT2's were built on 2026-04-20, before the clearing, from all 60 clips.
# It then checks each clip has as many outputs per stage as a complete clip (va10) and resumes the serial packing
# measurement, which adds the 4,200 new packing units.
#
# Usage on ace: nohup measurements-gpu/r9o2_caldot2_resweep.sh > measurements-results/r9o2_caldot2_resweep.log 2>&1 &
set -euo pipefail

DATASET=caldot2-y05
VIDEOS=(va00.mp4 va01.mp4 va02.mp4 va03.mp4 va04.mp4 va05.mp4 va06.mp4 va07.mp4 va08.mp4 va09.mp4)
CONFIG_NAME=resweep-${DATASET}.yaml
# A GPU counts as idle below this utilization (%) and memory (MiB), sampled every minute for IDLE_MINUTES in a row.
IDLE_UTIL=10
IDLE_MEM=1000
IDLE_MINUTES=5
STAGES=(p020_exec_classify p022_exec_prune_polyominoes p030_exec_compress p040_exec_detect p050_exec_uncompress p060_exec_track)
STAGE_DIRS=(020_relevancy 022_pruned_polyominoes 033_compressed_frames 040_compressed_detections 050_uncompressed_detections 060_uncompressed_tracks)

log() { echo "$(date '+%F %T') $*"; }
in_container() { docker exec -w /polyis -e CONFIG="$CONFIG_NAME" polyis "$@"; }

# Wait for the serial packing measurement, which shares the host's CPUs.
while docker exec polyis pgrep -f r9o2_packing_serial.py > /dev/null; do
    log "waiting for r9o2_packing_serial.py"; sleep 600
done

# Wait until every GPU has been idle for IDLE_MINUTES consecutive minutes.
idle=0
while [ "$idle" -lt "$IDLE_MINUTES" ]; do
    busy=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits |
           awk -F', ' -v u="$IDLE_UTIL" -v m="$IDLE_MEM" '$1 >= u || $2 >= m' | wc -l)
    if [ "$busy" -eq 0 ]; then idle=$((idle + 1)); else idle=0; log "waiting: $busy GPUs busy"; fi
    sleep 60
done
log "GPUs idle for $IDLE_MINUTES minutes"

# The global config with only this dataset, so each stage's task list covers no other dataset.
in_container python -c "
import yaml
config = yaml.safe_load(open('configs/global.yaml'))
config['EXEC']['DATASETS'] = ['$DATASET']
yaml.safe_dump(config, open('configs/$CONFIG_NAME', 'w'), sort_keys=False)
"

# Each stage in pipeline order, on the cleared clips only.
for stage in "${STAGES[@]}"; do
    log "start $stage"
    in_container python "scripts/$stage.py" --valid --videos "${VIDEOS[@]}"
    log "end $stage"
done

# Every cleared clip must now have as many outputs per stage as a complete clip.
for i in "${!STAGE_DIRS[@]}"; do
    expected=$(in_container ls "/polyis-cache/$DATASET/execution/va10.mp4/${STAGE_DIRS[$i]}" | wc -l)
    for video in "${VIDEOS[@]}"; do
        got=$(in_container ls "/polyis-cache/$DATASET/execution/$video/${STAGE_DIRS[$i]}" | wc -l)
        [ "$got" -eq "$expected" ] || { log "FAIL $video ${STAGE_DIRS[$i]}: $got outputs, expected $expected"; exit 1; }
    done
    log "ok ${STAGE_DIRS[$i]}: $expected outputs on every clip"
done

# Add the new packing units to the serial measurement (it skips the units it already has).
in_container python measurements/r9o2_packing_serial.py --dataset "$DATASET" --cpus 4,6,9,10
log "done"
