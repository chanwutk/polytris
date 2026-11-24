#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: bash reproduce/test.sh DATASET VALIDATION.json [VARIANT_ID]" >&2
    exit 2
fi

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
use_dataset "$1"
selection=$(mktemp)
trap 'rm -f "$selection"' EXIT
python reproduce/selection.py select "$1" "$2" "$selection" "${@:3}"
export POLYIS_TEST_PARETO_PARAMS="$selection"

# The reference pipeline (full-frame detection and tracking) that HOTA and speedups are measured against.
python preprocess/p002_preprocess_naive_detection.py --test
python preprocess/p002_preprocess_naive_tracking.py --test
# Each selected configuration through execution on one GPU: tracks for p110 and execution's runtime.
python -m execution.tools.run_variants run "$1" test "$selection"
python evaluation/p110_accuracy_compute.py --test
python evaluation/p111_accuracy_aggregate.py --test
python evaluation/p120_throughput_gather.py --test
python evaluation/p121_throughput_compute.py
python -m execution.tools.run_variants record "$1" test
python evaluation/p130_tradeoff_compute.py --test
