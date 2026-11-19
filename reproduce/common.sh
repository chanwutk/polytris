cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export POLYIS_PAD_SCORING_WINDOW=1

use_dataset() {
    if [[ -z "$1" || "$1" == *,* ]]; then
        echo "Pass exactly one dataset name" >&2
        exit 2
    fi
    export POLYIS_DATASETS="$1"
}

extract_tracks() {
    python scripts/p020_exec_classify.py "$1"
    python scripts/p022_exec_prune_polyominoes.py "$1"
    python scripts/p030_exec_compress.py "$1"
    python scripts/p040_exec_detect.py "$1"
    python scripts/p050_exec_uncompress.py "$1"
    python scripts/p060_exec_track.py "$1"
}

evaluate_split() {
    python evaluation/p110_accuracy_compute.py "$1"
    python evaluation/p111_accuracy_aggregate.py "$1"
    python evaluation/p120_throughput_gather.py "$1"
    python evaluation/p121_throughput_compute.py
    python evaluation/p130_tradeoff_compute.py "$1"
}
