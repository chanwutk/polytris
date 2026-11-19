# Reproduce the Tetris evaluation

Tetris prepares and extracts reusable object tracks from stored stationary-camera
video on one server. Start with the [Docker and Gurobi setup](README.md#setup).
Run the commands below from `/polyis` inside the container.

## Inputs and splits

| Datasets | Source layout in the container | Training / validation / test |
| --- | --- | --- |
| CalDoT1, CalDoT2, Amsterdam (`caldot1-y05`, `caldot2-y05`, `ams-y05`) | Unpack the [OTIF dataset archive](https://favyen.com/files/otif-dataset.zip) so its `dataset/` directory appears at `/otif-dataset/dataset`. Each source dataset has `train/video`, `valid/video`, and `test/video`. | 60 / 60 / 60 per dataset. Preserve the source split membership. |
| B3D1–4 (`jnc0`, `jnc2`, `jnc6`, `jnc7`) | Get the [upstream B3D videos](https://github.com/b3d-project/b3d). Put each junction's raw MP4 files in `/polyis-data/sources/<junction>` and the domain polygons in `/polyis-data/sources/b3d/annotations.xml`. | 8 / 5 / 5 per junction. After segmentation, the split script sorts clip names and shuffles them with seed 0. |

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

## Run one dataset end to end

This example uses CalDoT1 and the paper's configuration grid. Use fresh data
and cache volumes and initialize the submodules before building the container.

```sh
set -e
export PYTHONPATH=/polyis
export POLYIS_DATASETS=caldot1-y05

python scripts/weights.py download
python preprocess/p000_preprocess_dataset.py
python preprocess/p001_preprocess_split.py
python preprocess/p003_preprocess_groundtruth_detection.py --test
python preprocess/p003_preprocess_groundtruth_tracking.py --test

bash reproduce/train.sh caldot1-y05
bash reproduce/validate.sh caldot1-y05 validation.json
bash reproduce/test.sh caldot1-y05 validation.json
```

- `train.sh` generates training detections, trains the relevance classifier,
  and profiles the per-tile sampling limits.
- `validate.sh` runs the configured sweep on validation clips and writes JSON
  with the config, throughput in FPS, HOTA, and Pareto membership for each
  variant. The Pareto set includes the configurations needed for the paper's
  ablations.
- `test.sh` takes that JSON and runs the marked Pareto configurations on test
  clips. Pass a third argument, a `variant_id` from the JSON, to test just that
  configuration. The script writes per-clip tracking JSONL and the dataset's
  evaluation `090_tradeoff/tradeoff.csv` under `/polyis-cache`.

Paths passed to the wrappers are relative to the checkout or absolute. Reuse
the trained artifacts and the same sweep config across validation and test.
Each dataset's cache should have one active writer. Run a new dataset by preparing its splits and test
annotations, then invoking the same three scripts with its configured name.
Install the released human labels before testing B3D.

## Figures, tables, and reported numbers

Use a writable copy of the matching paper checkout as `PAPER_DIR` and keep the
paper's `scripts/data/` evidence with it. For a separate plotting container,
start the shell below from the artifact root. The extra mount gives the
paper-side scripts that resolve a sibling named `tetris` access to this checkout.

```sh
docker compose run --rm --volume "$PWD:/tetris" polyis bash
```

Inside that shell, run the commands below from `/polyis`.

```sh
bash reproduce/render.sh fig2 fig7 fig8 fig9 fig10 fig11 fig12 table2
```

Pass only the targets you want to regenerate. The exporters use the saved
profiles, scores, and timings below. Some existing renderers also emit
companion LaTeX macros as part of export. The paper checkout's
`GENERATE_FIGURE.md` has the detailed input and output locations.

| Target | Figure or table | Required inputs |
| --- | --- | --- |
| `fig2` | Stationary video observations | Processed clips, canonical detections, and B3D4 training association profile. |
| `fig7` | Mistrack tolerance and sampling gap | The paper's `scripts/data/missrate_figure_cache/` profile and frame. |
| `fig8` | End-to-end HOTA and throughput | Test tradeoff CSVs and the paper's scored LEAP reconstruction and timing snapshots. |
| `fig9`, `table2` | Preparation, amortization, and native query comparison | The paper's `scripts/data/revision_evidence/` snapshots. Both targets use the same exporter. |
| `fig10` | Classifier and packing | Saved summaries `p205_compare_classifiers_test.jsonl`, `p038_packing_efficacy_test.jsonl`, and the paper's `packing.json`. |
| `fig11` | Sampling ablation | Saved exhaustive and heuristic sweep results. |
| `fig12` | Extraction runtime breakdown | Saved results in `measurements-results/fixed_test_s1_square_m02/results/`. |
| `response-tables` | LEAP reconstruction, resolution scaling, padding, and preparation | The paper's saved revision evidence and `measurements-results/r1o3_resolution_scaling/summary.json`. |

Figures 1 and 3–5 use the static PDFs shipped with the paper. Figure 6 and
appendix Figures 13–15 are LaTeX diagrams. Table 1 is a LaTeX table of sweep
settings defined in `configs/global.yaml`.

The release still needs a clean-checkout run, a pinned paper and evidence
bundle, and a verified delivery path for the external inputs. A B3D release
bundle would contain all 72 processed clips in their existing split directories,
the human-verified test labels, and a manifest with clip names, frame counts,
and checksums. Preserve the clip filenames used by the labels.
