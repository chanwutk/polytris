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

python preprocess/p002_preprocess_naive_detection.py --test
python preprocess/p002_preprocess_naive_tracking.py --test
extract_tracks --test
evaluate_split --test
