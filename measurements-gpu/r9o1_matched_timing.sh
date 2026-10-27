#!/bin/bash
# Launch measurements-gpu/r9o1_matched_timing.py on one GPU of a host without the polyis container (bombe used GPU 6).
# Usage: measurements-gpu/r9o1_matched_timing.sh <staging dir S> <physical GPU index> [script args...]
# S holds polyis/ (this checkout), polyis-cache/ and polyis-data/ (the copied inputs); the polyis-polyis image is loaded.
# The job runs pinned to the GPU's NUMA node with numactl, since rootless docker discards --cpuset-cpus/--cpuset-mems.
# A host-side sampler records compute processes and the load average once per second, because nvidia-smi inside the
# container does not list other containers' processes. Writes S/run.log via the caller's redirection and S/finished.
set -u
S=$1
G=$2
shift 2
RAW=$S/polyis/measurements-results/r9o1_matched_timing/raw
mkdir -p "$RAW"
rm -f "$S/finished"
# The GPU's identity, and its NUMA node from sysfs (nvidia-smi writes 00000000:B2:00.0, sysfs names it 0000:b2:00.0).
UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$G")
BUS=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i "$G")
SYSFS=/sys/bus/pci/devices/$(printf '%04x' $((16#${BUS%%:*}))):$(echo "${BUS#*:}" | tr 'A-F' 'a-f')
NODE=$(cat "$SYSFS/numa_node")
IMAGE=$(docker image inspect --format '{{.Id}}' polyis-polyis)
echo "host check $(date -Is): $(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader -i "$G");" \
  "NUMA node $NODE, local CPUs $(cat "$SYSFS/local_cpulist"); image $IMAGE; load $(cat /proc/loadavg)"
APPS=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader)
echo "compute apps on host: $(tr '\n' ';' <<<"$APPS")"
# Refuse a GPU another process holds (the job itself also requires 0% utilization and 0 MiB).
if grep -q "$UUID" <<<"$APPS"; then
  echo "GPU $G has compute processes; not starting"
  echo 1 > "$S/finished"
  exit 1
fi
touch "$S/running"
START=$(date +%s%3N)
# Host-side samples: compute processes on this GPU and on all GPUs, and the 1-minute load average.
( while [ -e "$S/running" ]; do
    NOW=$(date +%s%3N)
    ALL=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
    printf '{"t_ms": %s, "apps": %s, "apps_all": %s, "load_1m": %s}\n' "$NOW" "$(grep -c "$UUID" <<<"$ALL")" \
      "$(grep -c . <<<"$ALL")" "$(cut -d' ' -f1 /proc/loadavg)"
    sleep 1
  done | gzip > "$RAW/host_compute_apps.$START.jsonl.gz" ) &
SAMPLER=$!
# SYS_NICE lets numactl set the memory policy under docker's default seccomp profile.
docker run --rm --name r9o1-matched-timing --gpus all --cap-add SYS_NICE --hostname "$(hostname)" --shm-size 8g -w /polyis \
  -e CUDA_VISIBLE_DEVICES="$G" -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e POLYIS_IMAGE="polyis-polyis@$IMAGE" \
  -e PYTHONWARNINGS=ignore::UserWarning:numpy._core.getlimits -e POLYIS_DATA=/polyis-data -e POLYIS_CACHE=/polyis-cache \
  -v "$S/polyis:/polyis" -v "$S/polyis-cache:/polyis-cache:ro" -v "$S/polyis-data:/polyis-data:ro" \
  polyis-polyis bash -c "set -e; (apt-get update -qq && apt-get install -y -qq numactl) > /dev/null; pip install -q duckdb==2.0.0;
    numactl --cpunodebind=$NODE --membind=$NODE python measurements-gpu/r9o1_matched_timing.py --gpu $G $*"
STATUS=$?
rm -f "$S/running"
wait "$SAMPLER"
echo "exit $STATUS $(date -Is)"
echo "$STATUS" > "$S/finished"
