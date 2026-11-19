# Reproduce the Tetris evaluation

Tetris prepares and extracts reusable object tracks from stored stationary-camera
video on one server. Start with the [Docker and Gurobi setup](README.md#setup).
Run the Python commands from `/polyis` inside the container.

## Inputs and splits

| Datasets | Source layout in the container | Training / validation / test |
| --- | --- | --- |
| CalDoT1, CalDoT2, Amsterdam (`caldot1-y05`, `caldot2-y05`, `ams-y05`) | Unpack the [OTIF dataset archive](https://favyen.com/files/otif-dataset.zip) so its `dataset/` directory appears at `/otif-dataset/dataset`. Each source dataset has `train/video`, `valid/video`, and `test/video`. | 60 / 60 / 60 per dataset. Preserve the source split membership. |
| B3D1–4 (`jnc0`, `jnc2`, `jnc6`, `jnc7`) | The [processed B3D bundle](https://huggingface.co/datasets/tetris-db/b3d-processed) provides the processed clips and human-verified test labels used by Tetris. The original footage is available from the [upstream B3D project](https://github.com/b3d-project/b3d). | 8 / 5 / 5 per junction, with filenames and split membership preserved in the bundle. |

For CalDoT1, CalDoT2, or Amsterdam, set `POLYIS_DATASETS` to the dataset being
prepared. Run `python preprocess/p000_preprocess_dataset.py` and
`python preprocess/p001_preprocess_split.py` to convert its OTIF clips to 15 FPS
and normalize their filenames. The first command replaces the dataset output
directory. Use fresh volumes. Prepare the test annotations with the
`p003_preprocess_groundtruth_detection.py --test` and
`p003_preprocess_groundtruth_tracking.py --test` scripts in `preprocess/`.

`python scripts/weights.py download` fetches the three YOLOv5 fine-tunes and the
B3D RetinaNet checkpoint and checks their SHA-256 hashes. For details about how
the B3D test labels were created, see the
[human-reviewed labels and annotation workflow](https://anonymous.4open.science/r/tetris-b3d-human-annotation)
and [Faster R-CNN pre-labeling code and weights](https://anonymous.4open.science/r/tetris-b3d-fasterrcnn).

## Run one dataset end to end

This example uses B3D1 (`jnc0`) and the paper's configuration grid. On the
Docker host, set `DATA` to the same absolute path as in `.env`. The dataset is
public and can be downloaded without signing in to Hugging Face. Copy the
clips and scoring inputs into fresh data and cache volumes.

```sh
set -e
: "${DATA:?Set DATA to the data parent from .env}"
bundle=$(mktemp -d)
hf download tetris-db/b3d-processed --repo-type dataset \
    --revision 8b169c08266c8c78515112735051067f2a4abd4c --local-dir "$bundle"
mkdir -p "$DATA/polyis-data/datasets" "$DATA/polyis-cache"
cp -a "$bundle/datasets/." "$DATA/polyis-data/datasets/"
cp -a "$bundle/groundtruth/." "$DATA/polyis-cache/"
```

After [setting up Docker and building the native extensions](README.md#setup),
run these commands inside the container from `/polyis`:

```sh
set -e
python scripts/weights.py download --b3d-only
bash reproduce/train.sh jnc0
bash reproduce/validate.sh jnc0 validation.json
bash reproduce/test.sh jnc0 validation.json
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
Each dataset's cache should have one active writer. For another dataset, stage
its splits and test annotations and use its name in the same three scripts.

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

The release still needs a clean-checkout GPU run and a pinned paper and
figure evidence bundle.
