#!/bin/bash
# Run one command in a throwaway polyis container on one idle bombe GPU (see measurements-gpu/BOMBE.md).
# Usage: measurements-gpu/bombe_container.sh <physical GPU 0-6> <scratch dir> '<command>'
#   <scratch dir> must be under /data/cwkt/data/bombe-revision/scratch/. It is mounted at /scratch; <scratch>/polyis-cache is
#   /polyis-cache and <scratch>/global.yaml (a copy of configs/global.yaml, created when missing) is CONFIG.
#   Inside, the GPU is the only one visible (cuda:0). The command runs with bash -c under taskset on the CPUs of the
#   GPU's NUMA node, because rootless docker on bombe ignores --cpuset-cpus (no cpuset cgroup delegation).
# Host-side samplers write <scratch>/samples/<run>.gpus.csv (every GPU's utilization and memory each second) and
# <scratch>/samples/<run>.jsonl (each second: this GPU's utilization and memory, its compute processes and how many
# belong to other containers or users, load average); <scratch>/samples/<run>.json summarizes the run and sets
# contaminated=true when any other process used the GPU. The exit code is the command's.
# Optional environment: BOMBE_IMAGE (default polyis-polyis:latest) runs another image with the same mounts, sampling, and
# pinning; BOMBE_DOCKER_ARGS (whitespace-separated) adds docker run arguments, such as extra mounts, after the defaults;
# BOMBE_CPU_SUBSET pins to part of the GPU's node instead of all of it.
set -u
USAGE="usage: $0 <physical GPU 0-6> <scratch dir> '<command>'"
G=${1:?$USAGE}
S=$(realpath -m "${2:?$USAGE}")
CMD=${3:?$USAGE}
ROOT=/data/cwkt/data/bombe-revision
TETRIS=/work/cwkt/projects/tetris
# Refuse a scratch directory outside bombe-revision/scratch/ (the mirror and everything else on /data are read-only).
case "$S" in "$ROOT"/scratch/?*) ;; *) echo "scratch must be a directory under $ROOT/scratch/, not $S" >&2; exit 2;; esac
mkdir -p "$S/polyis-cache" "$S/samples"
# The scratch config starts as the checkout's global.yaml (its CACHE_DIR is already /polyis-cache) and is kept.
[ -e "$S/global.yaml" ] || cp "$TETRIS/configs/global.yaml" "$S/global.yaml"
# Without linger, the rootless daemon restarts with an SSH login; wait up to 30 s for it.
i=0; until docker info > /dev/null 2>&1; do i=$((i + 1)); [ $i -le 30 ] || { echo "docker daemon not reachable" >&2; exit 1; }; sleep 1; done
# The GPU's identity, and its NUMA node and CPUs from sysfs (nvidia-smi writes 00000000:B2:00.0, sysfs 0000:b2:00.0).
UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$G") || { echo "no GPU $G: $UUID" >&2; exit 2; }
BUS=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i "$G")
SYSFS=/sys/bus/pci/devices/$(printf '%04x' $((16#${BUS%%:*}))):$(echo "${BUS#*:}" | tr 'A-F' 'a-f')
NODE=$(cat "$SYSFS/numa_node")
CPUS=$(cat "$SYSFS/local_cpulist")
# Optional BOMBE_CPU_SUBSET (a cpulist such as 20-29,60-69) narrows the pinning to part of the node, so two jobs on
# one node's GPUs do not share cores; every CPU in it must be on the GPU's node. Unset, the whole node is used.
if [ -n "${BOMBE_CPU_SUBSET:-}" ]; then
  expand() { python3 -c 'import sys; print(" ".join(str(c) for r in sys.argv[1].split(",") for a, _, b in [r.partition("-")] for c in range(int(a), int(b or a) + 1)))' "$1"; }
  NODE_CPUS=" $(expand "$CPUS") "
  for c in $(expand "$BOMBE_CPU_SUBSET"); do
    case "$NODE_CPUS" in *" $c "*) ;; *) echo "CPU $c of BOMBE_CPU_SUBSET is not on GPU $G's NUMA node ($CPUS)" >&2; exit 2;; esac
  done
  CPUS=$BOMBE_CPU_SUBSET
fi
IMAGE_NAME=${BOMBE_IMAGE:-polyis-polyis:latest}
read -r -a EXTRA_ARGS <<< "${BOMBE_DOCKER_ARGS:-}"
IMAGE=$(docker image inspect --format '{{.Id}}' "$IMAGE_NAME") || { echo "no image $IMAGE_NAME" >&2; exit 2; }
RUN=g$G-$(date +%Y%m%dT%H%M%S)
NAME=bombe-revision-$RUN
# Idle check: three samples 1 s apart must show 0% utilization, under 100 MiB, and no compute process on the GPU (a
# failed query counts as busy).
for k in 1 2 3; do
  read -r UTIL MEM <<<"$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -i "$G" | tr -d ,)"
  if [ "$UTIL" != 0 ] || [ "${MEM:-100}" -ge 100 ] || nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$UUID"; then
    echo "GPU $G busy (utilization ${UTIL}%, ${MEM} MiB, or a compute process); not starting" >&2; exit 3
  fi
  sleep 1
done
echo "run $RUN $(date -Is): GPU $G $UUID bus $BUS, NUMA node $NODE, CPUs $CPUS, image $IMAGE, load $(cut -d' ' -f1-3 /proc/loadavg)"
echo "command: $CMD"
SAMPLES=$S/samples/$RUN.jsonl
GPUS=$S/samples/$RUN.gpus.csv
CIDFILE=$S/samples/$RUN.cid
touch "$S/samples/$RUN.running"
START=$(date -Is)
# Host-side samplers. nvidia-smi's own loop records every GPU's utilization and memory once per second, and keeps the
# driver initialized: persistence mode is off on bombe, so a cold nvidia-smi call takes about 5 s, a warm one 0.4 s.
nice nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader,nounits -lms 1000 > "$GPUS" &
SMI=$!
# On any exit, including a killed tmux session, stop both samplers and remove the container if it outlived the client.
trap 'rm -f "$S/samples/$RUN.running"; kill "$SMI" 2> /dev/null; docker rm -f "$NAME" > /dev/null 2>&1' EXIT
# Once per second: the compute processes on this GPU, how many are foreign (a live process whose cgroup is not this
# container's; nvidia-smi inside the container cannot see other containers' processes, and bombe's /proc shows every
# user's processes, so a pid missing from /proc has exited), this GPU's latest utilization and memory, and the load.
( CID=
  while [ -e "$S/samples/$RUN.running" ]; do
    [ -n "$CID" ] || CID=$(cat "$CIDFILE" 2>/dev/null)
    PIDS=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep "$UUID" | cut -d, -f2 | tr -d ' ')
    FOREIGN=0
    for p in $PIDS; do
      [ -e "/proc/$p" ] && ! grep -qs "${CID:-no-container-yet}" "/proc/$p/cgroup" && FOREIGN=$((FOREIGN + 1))
    done
    read -r UTIL MEM <<<"$(tail -n 7 "$GPUS" | awk -F', ' -v g="$G" '$2 == g {u = $3; m = $4} END {print u + 0, m + 0}')"
    printf '{"t": "%s", "util": %s, "mem_mib": %s, "pids": [%s], "foreign": %s, "load_1m": %s}\n' "$(date -Is)" "$UTIL" \
      "$MEM" "$(echo $PIDS | tr ' ' ,)" "$FOREIGN" "$(cut -d' ' -f1 /proc/loadavg)"
    sleep 1
  done > "$SAMPLES" ) &
SAMPLER=$!
# Mounts and environment mirror ace's docker-compose.base.yml; the inputs are bombe's read-only mirror of ace's data.
# duckdb (pip-installed in ace's container, absent from the image) comes from $ROOT/pylib on PYTHONPATH; the trailing
# empty entry is the image's own (the working directory, /polyis), through which scripts import polyis and measurements.
docker run --rm --name "$NAME" --cidfile "$CIDFILE" --gpus "device=$G" --network host --hostname "$(hostname)" \
  --shm-size 8g --tmpfs /tmp -w /polyis \
  -e TERM=xterm-256color -e POLYIS_DATA=/polyis-data -e POLYIS_CACHE=/polyis-cache \
  -e PYTHONWARNINGS=ignore::UserWarning:numpy._core.getlimits -e PYTHONPATH=/polyis/python:/opt/bombe-pylib: \
  -e CONFIG=/scratch/global.yaml -e BOMBE_GPU="$G" -e BOMBE_CPUS="$CPUS" -e BOMBE_RUN="$RUN" \
  -v "$TETRIS:/polyis" -v "$S:/scratch" -v "$S/polyis-cache:/polyis-cache" \
  -v "$ROOT/mirror/polyis-data:/polyis-data:ro" -v "$ROOT/mirror/otif-dataset:/otif-dataset:ro" \
  -v "$ROOT/mirror/sota-results:/sota-results:ro" -v "$ROOT/pylib:/opt/bombe-pylib:ro" \
  -v "$HOME/.gurobi/gurobi.lic:/opt/gurobi/gurobi.lic:ro" "${EXTRA_ARGS[@]}" \
  "$IMAGE_NAME" taskset -c "$CPUS" bash -c "$CMD"
STATUS=$?
rm -f "$S/samples/$RUN.running"
wait "$SAMPLER"
# Summary: contaminated when any sample saw a foreign compute process on the GPU.
N=$(grep -c . "$SAMPLES")
BAD=$(grep -c '"foreign": [1-9]' "$SAMPLES")
[ "$BAD" -eq 0 ] && CONTAMINATED=false || CONTAMINATED=true
printf '{"run": "%s", "host": "%s", "gpu": %s, "gpu_uuid": "%s", "numa_node": %s, "cpus": "%s", "image": "%s", "start": "%s", "end": "%s", "exit": %s, "samples": %s, "foreign_samples": %s, "contaminated": %s, "command": %s}\n' \
  "$RUN" "$(hostname)" "$G" "$UUID" "$NODE" "$CPUS" "$IMAGE" "$START" "$(date -Is)" "$STATUS" "$N" "$BAD" "$CONTAMINATED" \
  "$(python3 -c 'import json, sys; print(json.dumps(sys.argv[1]))' "$CMD")" > "$S/samples/$RUN.json"
echo "run $RUN exit $STATUS, $N samples, $BAD with a foreign process on GPU $G (contaminated=$CONTAMINATED)"
exit "$STATUS"
