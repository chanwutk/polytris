#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 ]]; then
    echo "Usage: bash reproduce/train.sh DATASET" >&2
    exit 2
fi

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
use_dataset "$1"

python scripts/p011_tune_detect.py
python scripts/p012_tune_create_training_data.py
python scripts/p013_tune_optimize_training_data.py
python scripts/p014_tune_train_classifier.py --no_visualize
python scripts/p015_tune_optimize_classifier.py
python scripts/p016_tune_track_rate.py
