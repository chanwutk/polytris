#!/usr/bin/env bash
set -euo pipefail
if [[ $# == 0 ]]; then
    echo "Usage: bash reproduce/render.sh {fig2|fig7|fig8|fig9|fig10|fig11|fig12|table2|response-tables} [...]" >&2
    exit 2
fi

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
unset POLYIS_DATASETS POLYIS_TEST_PARETO_PARAMS
render_cache=$(mktemp -d)
trap 'rm -rf "$render_cache"' EXIT
tar -xzf reproduce/chart-inputs.tar.gz -C "$render_cache"
python - "$render_cache" <<'PY'
import os
import sys
from pathlib import Path

import yaml

cache_dir = Path(sys.argv[1])
config_path = Path('configs') / os.environ.get('CONFIG', 'global.yaml')
config = yaml.safe_load(config_path.read_text())
experiment_cache = Path(config['DATA']['CACHE_DIR']).resolve()
for dataset in ('caldot1', 'caldot2', 'ams', 'jnc0', 'jnc2', 'jnc6', 'jnc7'):
    labels = cache_dir / dataset / 'execution'
    labels.parent.mkdir(exist_ok=True)
    labels.symlink_to(experiment_cache / dataset / 'execution', target_is_directory=True)
config['DATA']['CACHE_DIR'] = str(cache_dir)
(cache_dir / 'config.yaml').write_text(yaml.safe_dump(config))
PY
export CONFIG="$render_cache/config.yaml"
mkdir -p paper/figures/generated
revision_rendered=0

for item in "$@"; do
    case "$item" in
        fig2)
            python evaluation/p003_tile_irrelevance_visualize.py --split test
            python evaluation/p016_track_rate_visualize.py
            python evaluation/p004_polyomino_vs_roi_visualize.py --split test
            python - <<'PY'
from pathlib import Path
from shutil import copyfile

from polyis.io import cache

source = cache.eval('jnc7', 'track-rate-vis', 'bytetrackcython_60', 'jnc7_mistrack_rate_004.png')
copyfile(source, Path('paper/figures/generated') / source.name)
PY
            ;;
        fig7)
            evidence=paper/scripts/data/missrate_figure_cache
            frame=paper/figures/generated/jnc7_frame_crop.png
            ffmpeg -y -i "$evidence/full_frame.png" -vf crop=180:120:600:540 -frames:v 1 "$frame"
            python paper/scripts/generate_missrate_mapping_figure.py --artifact-dir "$evidence" --frame-image "$frame"
            python paper/scripts/generate_missrate_curves.py --width 205 --height 55
            ;;
        fig8)
            python paper/scripts/render_revised_hota.py --alternative
            ;;
        fig9|table2|response-tables)
            if [[ $revision_rendered == 0 ]]; then
                python paper/scripts/generate_revision_figures.py
                revision_rendered=1
            fi
            if [[ $item == response-tables ]]; then
                python paper/scripts/generate_resolution_scaling.py
                python paper/scripts/generate_padding_tradeoff.py
                python paper/scripts/generate_preparation_breakdown.py
            fi
            ;;
        fig10)
            python evaluation/p206_classifier_and_packing.py --test
            ;;
        fig11)
            python ablation/mistrack-rate/run.py --combine-visualize --tracker bytetrackcython
            ;;
        fig12)
            python paper/scripts/render_runtime_breakdown.py
            ;;
        *)
            echo "Unknown export: $item" >&2
            exit 2
            ;;
    esac
done
