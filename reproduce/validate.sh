#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 2 ]]; then
    echo "Usage: bash reproduce/validate.sh DATASET VALIDATION.json" >&2
    exit 2
fi

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
use_dataset "$1"
unset POLYIS_TEST_PARETO_PARAMS
mkdir -p -- "$(dirname -- "$2")"

python preprocess/p002_preprocess_naive_detection.py --valid
python preprocess/p002_preprocess_naive_tracking.py --valid
extract_tracks --valid
evaluate_split --valid
python evaluation/p135_pareto_extract.py --valid
python reproduce/selection.py export "$1" "$2"
