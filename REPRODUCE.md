# Reproduce the Tetris evaluation

Tetris prepares and extracts reusable object tracks from stored stationary-camera
video on one server. Start with the [Docker and Gurobi setup](README.md#setup).
Run the commands below from `/polyis` inside the container.

## Inputs and splits

| Datasets | Source layout in the container | Training / validation / test |
| --- | --- | --- |
| CalDoT1, CalDoT2, Amsterdam (`caldot1-y05`, `caldot2-y05`, `ams-y05`) | Unpack the [OTIF dataset](https://github.com/favyen2/otif) at `/otif-dataset/dataset`. Each source dataset has `train/video`, `valid/video`, and `test/video`. | 60 / 60 / 60 per dataset. Preserve the source split membership. |
| B3D1–4 (`jnc0`, `jnc2`, `jnc6`, `jnc7`) | Put each junction's raw MP4 files in `/polyis-data/sources/<junction>` and the domain polygons in `/polyis-data/sources/b3d/annotations.xml`. | 8 / 5 / 5 per junction. After segmentation, the split script sorts clip names and shuffles them with seed 0. |

In a fresh data volume, `python preprocess/p000_preprocess_dataset.py` converts
videos to 15 FPS, segments B3D into 18 clips per junction, and applies the B3D
domain masks. Follow it with `python preprocess/p001_preprocess_split.py --split 8:5:5`.
Preprocessing replaces the CalDoT and Amsterdam dataset output directories.
Use fresh volumes when reproducing the workflow. Training learns the classifier
and per-tile sampling limits, validation selects configurations, and test
evaluates the selected configurations on held-out clips.

`python scripts/weights.py download` fetches the three YOLOv5 fine-tunes and the
B3D RetinaNet checkpoint and checks their SHA-256 hashes.
For the B3D test evaluation, use the
[human-reviewed labels and annotation workflow](https://anonymous.4open.science/r/tetris-b3d-human-annotation)
and [Faster R-CNN pre-labeling code and weights](https://anonymous.4open.science/r/tetris-b3d-fasterrcnn).
`preprocess/p004_install_human_groundtruth.py` installs the labels into the
evaluation cache. Use its `--source` and `--source-commit` options to identify the
released label snapshot.

## From preparation to extracted tracks

This one-camera example uses the CalDoT1 splits and the default configuration
grid. It runs the validation sweep and can take several hours. Use fresh data
and cache volumes and initialize the submodules before building the container.

```sh
set -e
export PYTHONPATH=/polyis
export POLYIS_DATASETS=caldot1-y05
export POLYIS_PAD_SCORING_WINDOW=1

python scripts/weights.py download
python preprocess/p000_preprocess_dataset.py
python preprocess/p001_preprocess_split.py

python scripts/p011_tune_detect.py
python scripts/p012_tune_create_training_data.py
python scripts/p013_tune_optimize_training_data.py
python scripts/p014_tune_train_classifier.py --no_visualize
python scripts/p015_tune_optimize_classifier.py
python scripts/p016_tune_track_rate.py

run_stages() {
    python scripts/p020_exec_classify.py "$1"
    python scripts/p022_exec_prune_polyominoes.py "$1"
    python scripts/p030_exec_compress.py "$1"
    python scripts/p040_exec_detect.py "$1"
    python scripts/p050_exec_uncompress.py "$1"
    python scripts/p060_exec_track.py "$1"
}

python preprocess/p002_preprocess_naive_detection.py --valid
python preprocess/p002_preprocess_naive_tracking.py --valid
run_stages --valid
python evaluation/p100_evaluation_val.py
run_stages --test
```

The tracking stage writes per-clip tracking JSONL under the dataset's execution
cache in `/polyis-cache`. The staged commands also write the operator timings
used by the paper's validation evaluator. The test HOTA and query evaluation
additionally require the canonical test annotations. Use all seven datasets
and the same validation-to-test selection for the paper evaluation.

## Figures, tables, and reported numbers

Use a writable copy of the matching paper checkout as `PAPER_DIR` and keep the
paper's `scripts/data/` evidence with it. For a separate plotting container,
start the shell below from the artifact root. The extra mount gives the
paper-side scripts that resolve a sibling named `tetris` access to this checkout.

```sh
docker compose run --rm --volume "$PWD:/tetris" polyis bash
```

Inside that shell, run `cd /polyis`, `export PYTHONPATH=/polyis`, and
`export POLYIS_PAD_SCORING_WINDOW=1`. In the table, run each listed Python script
in order with `python` and the shown arguments. Prefix paper paths with
`paper/`. The renderers use saved profiles, scores, and timings where listed.
The paper checkout's `GENERATE_FIGURE.md` and `GENERATE_NUMBERS.md` cover the
upstream measurements and numeric macro provenance in more detail.

| Paper item | Scripts or source | Required inputs |
| --- | --- | --- |
| Figures 1, 3–5 | Paper assets `figures/Pipeline-Naive.pdf`, `figures/Tetris-System.pdf`, `figures/pack.pdf`, and `figures/Classifier.pdf` | Static diagrams included by LaTeX. |
| Figure 2 and observation numbers | `evaluation/p003_tile_irrelevance_visualize.py --split test`, `evaluation/p016_track_rate_visualize.py`, `evaluation/p004_polyomino_vs_roi_visualize.py --split test`, `evaluation/p017_mistrack_variance.py` | Processed videos, canonical detections, and training association profile. Copy the B3D4 gap-4 BYTETrack PNG from the cache as described in `GENERATE_FIGURE.md`. |
| Figure 6 and appendix Figures 13–15 | Paper TikZ sources `figures/prunable_polyominoes_option2.tex`, `figures/np_layout_example.tex`, `figures/np_reduction_example.tex`, `figures/np_universal_example.tex` | LaTeX. |
| Figure 7 | Paper `scripts/generate_missrate_mapping_figure.py --artifact-dir paper/scripts/data/missrate_figure_cache --frame-image paper/figures/generated/jnc7_frame_crop.png`, then paper `scripts/generate_missrate_curves.py --width 205 --height 55` | B3D4 profile and frame in the paper's `scripts/data/missrate_figure_cache/`. Create the crop with `ffmpeg -y -i paper/scripts/data/missrate_figure_cache/full_frame.png -vf crop=180:120:600:540 -frames:v 1 paper/figures/generated/jnc7_frame_crop.png`. |
| Figure 8 and end-to-end comparison numbers | Paper `scripts/render_revised_hota.py` | Test tradeoffs and the scored LEAP reconstruction and timing snapshots. |
| Figure 9, Table 2, and revision numbers | Paper `scripts/generate_revision_figures.py` | Saved revision evidence for preparation, amortization, and query scores. To refresh query inputs from scored runs, first run paper `scripts/refresh_leap_matched.py --tetris-root /polyis`. |
| Figure 10 and classifier/packing numbers | `evaluation/p205_compare_classifiers.py --test --threshold 0.5`, `evaluation/p038_packing_efficacy.py --test`, `evaluation/p206_classifier_and_packing.py --test` | Test classifier predictions, the no-pruning packing control, and paper `scripts/data/revision_evidence/packing.json`. |
| Figure 11 and sampling ablation numbers | `ablation/mistrack-rate/run.py --combine-visualize --tracker bytetrackcython` | Cached exhaustive and heuristic sweep results. The same runner without `--combine-visualize` runs the experiment. |
| Figure 12 and fixed operator/ILP numbers | Paper `scripts/generate_fixed_operator_stats.py` and `scripts/render_runtime_breakdown.py` | Saved fixed-test and ILP rerun measurements in this checkout. |
| Table 1 | Paper `content/03_overview.tex` | LaTeX table of settings. The sweep is defined in `configs/global.yaml`. |
| Pipeline ablation numbers | `evaluation/p204_ablation_stats.py --test` | Test tradeoffs. |
| Reviewer response tables | Paper `scripts/generate_revision_figures.py`, `scripts/generate_resolution_scaling.py`, `scripts/generate_padding_tradeoff.py`, `scripts/generate_preparation_breakdown.py` | LEAP reconstruction and packing/preparation snapshots in the paper, plus `measurements-results/r1o3_resolution_scaling/summary.json`. |

The release still needs a clean-checkout run. It also needs a pinned paper and
evidence bundle and a verified delivery path for external datasets, submodules,
baseline caches, and any raw inputs missing from Git. Preserve the ongoing
Seiden and NoScope results until the paper's comparison and inputs are fixed.
