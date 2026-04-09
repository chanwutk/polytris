#!/usr/bin/env bash

# Exit on errors, unset variables, and failed pipelines.
set -euo pipefail

# Print the wrapper usage text.
print_usage() {
  cat <<'EOF'
Usage:
  bash ablation/mistrack-rate/run.sh [wrapper options] [run.py all options]

Wrapper options:
  --tracker TRACKER     Run one tracker only.
  --all-trackers        Run every configured tracker.
  --datasets CSV        Run a comma-separated subset of datasets.
  -h, --help            Show this help message.

Examples:
  bash ablation/mistrack-rate/run.sh
  bash ablation/mistrack-rate/run.sh --all-trackers
  bash ablation/mistrack-rate/run.sh --datasets jnc0,jnc2 --no-hota --limit-configs 64
  bash ablation/mistrack-rate/run.sh --video-fraction-divisor 3

All remaining arguments are forwarded to:
  python ablation/mistrack-rate/run.py ...
EOF
}

# Forward execution into the project container when the script is launched outside Docker.
forward_to_container_if_needed() {
  if [[ -f /.dockerenv ]]; then
    return
  fi

  local -a quoted_args
  local arg
  for arg in "$@"; do
    quoted_args+=("$(printf '%q' "$arg")")
  done

  exec docker exec polyis sh -lc "cd /polyis && bash /polyis/ablation/mistrack-rate/run.sh ${quoted_args[*]:-}"
}

# Resolve the absolute ablation directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve the Python entrypoint used for each dataset invocation.
RUN_PY="${SCRIPT_DIR}/run.py"

# Forward into the container before reading project config.
forward_to_container_if_needed "$@"

# Read the configured datasets from the project config.
mapfile -t CONFIGURED_DATASETS < <(
  python - <<'PY'
from polyis.utilities import get_config

for dataset in get_config()['EXEC']['DATASETS']:
    print(dataset)
PY
)

# Read the configured trackers from the project config.
mapfile -t CONFIGURED_TRACKERS < <(
  python - <<'PY'
from polyis.utilities import get_config

for tracker in get_config()['EXEC']['TRACKERS']:
    print(tracker)
PY
)

# Use the first configured tracker as the default tracker.
DEFAULT_TRACKER="${CONFIGURED_TRACKERS[0]}"

# Initialize wrapper-level defaults.
RUN_ALL_TRACKERS=0
TRACKER_OVERRIDE=""
DATASETS_OVERRIDE=""
FORWARD_ARGS=()

# Parse wrapper-specific arguments and preserve the rest for run.py.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tracker)
      TRACKER_OVERRIDE="$2"
      shift 2
      ;;
    --all-trackers)
      RUN_ALL_TRACKERS=1
      shift
      ;;
    --datasets)
      DATASETS_OVERRIDE="$2"
      shift 2
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    --dataset)
      echo "Do not pass --dataset to run.sh; use --datasets instead." >&2
      exit 1
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift
      ;;
  esac
done

# Reject conflicting tracker-selection modes.
if [[ ${RUN_ALL_TRACKERS} -eq 1 && -n "${TRACKER_OVERRIDE}" ]]; then
  echo "Choose either --tracker or --all-trackers, not both." >&2
  exit 1
fi

# Select the tracker list for this wrapper invocation.
TRACKERS_TO_RUN=()
if [[ ${RUN_ALL_TRACKERS} -eq 1 ]]; then
  TRACKERS_TO_RUN=("${CONFIGURED_TRACKERS[@]}")
elif [[ -n "${TRACKER_OVERRIDE}" ]]; then
  TRACKERS_TO_RUN=("${TRACKER_OVERRIDE}")
else
  TRACKERS_TO_RUN=("${DEFAULT_TRACKER}")
fi

# Select the dataset list for this wrapper invocation.
DATASETS_TO_RUN=()
if [[ -n "${DATASETS_OVERRIDE}" ]]; then
  IFS=',' read -r -a DATASETS_TO_RUN <<< "${DATASETS_OVERRIDE}"
else
  DATASETS_TO_RUN=("${CONFIGURED_DATASETS[@]}")
fi

# Run the ablation for every requested tracker and dataset.
for tracker in "${TRACKERS_TO_RUN[@]}"; do
  for dataset in "${DATASETS_TO_RUN[@]}"; do
    # Print a stable header before each long-running command.
    printf '\n########################################\n'
    printf '>>> %s | %s\n' "${tracker}" "${dataset}"
    printf '########################################\n\n'

    # Build per-dataset arguments.
    # Non-jnc datasets (e.g. caldot, ams) use a 1/3 video subsample by default.
    DATASET_ARGS=()
    if [[ "${dataset}" != jnc* ]]; then
      DATASET_ARGS+=(--video-fraction-divisor 3)
    fi

    # Execute the full ablation pipeline for one dataset/tracker pair.
    python "${RUN_PY}" --dataset "${dataset}" --tracker "${tracker}" "${DATASET_ARGS[@]}" "${FORWARD_ARGS[@]}"
  done
done
