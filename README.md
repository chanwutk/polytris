# Reproduce the Tetris evaluation

Tetris materializes object tracks from stored stationary-camera video on a single
GPU server. The artifact includes the preparation, validation, and extraction
pipeline, the datasets and reference detectors needed to run it, and the inputs
and exporters for the paper's charts and tables.

## Docker setup

Use a Linux host with an NVIDIA GPU, Docker Compose, and the NVIDIA Container
Toolkit. Install the Hugging Face CLI on the host to download the datasets.
Get a [Gurobi license](https://www.gurobi.com/academia/) for pruning and place
it at `~/.gurobi/gurobi.lic`. The image supplies Gurobi 13.0.1 and the native
library environment.

From the repository root on the Docker host:

```sh
git submodule update --init --recursive modules/TrackEval modules/lap
```

Set `DATA` to an absolute directory on the host and put that same assignment
in `.env` in the repository root. Docker mounts
`$DATA/polyis-data` at `/polyis-data` and `$DATA/polyis-cache` at
`/polyis-cache`. The data download commands below create these directories.

```sh
docker compose up --detach --build
docker compose exec polyis bash
```

From `/polyis` in that shell, build the native extensions before running the
experiments:

```sh
python setup.py build_ext
```

## Inputs and splits

| Datasets | Source layout in the container | Training / validation / test |
| --- | --- | --- |
| CalDoT1, CalDoT2, Amsterdam (`caldot1-y05`, `caldot2-y05`, `ams-y05`) | The [processed OTIF bundle](https://huggingface.co/datasets/tetris-db/otif-processed) provides the processed clips and canonical test annotations used by Tetris. The original inputs are available from the [OTIF project](https://favyen.com/otif/). | 60 / 60 / 60 per dataset, with filenames and split membership preserved in the bundle. |
| B3D1–4 (`jnc0`, `jnc2`, `jnc6`, `jnc7`) | The [processed B3D bundle](https://huggingface.co/datasets/tetris-db/b3d-processed) provides the processed clips and human-verified test labels used by Tetris. The original footage is available from the [upstream B3D project](https://github.com/b3d-project/b3d). | 8 / 5 / 5 per junction, with filenames and split membership preserved in the bundle. |

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

After [setting up Docker and building the native extensions](#docker-setup),
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

For CalDoT1, CalDoT2, or Amsterdam, substitute this command for the B3D
download above and use the same `datasets/` and `groundtruth/` copy steps.

```sh
hf download tetris-db/otif-processed --repo-type dataset \
    --revision 98d1809eb4d367032e878f53bdd584103776e34a --local-dir "$bundle"
```

On the Docker host, create the configuration names as relative symlinks in
the data volume:

```sh
cd "$DATA/polyis-data/datasets"
ln -s caldot1 caldot1-y05
ln -s caldot2 caldot2-y05
ln -s ams ams-y05
```

Inside Docker, run `python scripts/weights.py download` for the YOLOv5
checkpoints and pass the corresponding configuration name to the runners.

## Generate figures and tables

From `/polyis` inside the Docker container, run the chart and table exporters:

```sh
bash reproduce/render.sh fig2 fig7 fig8 fig9 fig10 fig11 fig12 table2 response-tables
```

Pass individual targets to export only the outputs you need. Figure 2 uses
the clips and test annotations from both public dataset bundles above. The
other targets use the saved benchmark results shipped with the artifact.
The render command uses a temporary cache for the saved results and writes
the exported figures and table rows to `paper/figures/generated/`.

| Target | Paper output |
| --- | --- |
| `fig2` | Stationary-camera observations |
| `fig7` | Mistrack rate and sampling gap |
| `fig8` | End-to-end HOTA and throughput |
| `fig9` | Preparation cost and amortization |
| `fig10` | Classifier and packing results |
| `fig11` | Sampling ablation |
| `fig12` | Extraction runtime breakdown |
| `table2` | Native query comparison |
| `response-tables` | LEAP reconstruction, resolution scaling, padding, and preparation |

Figures 1 and 3–5 are static PDFs in `paper/figures/`. Figure 6 and appendix
Figures 13–15 use the TikZ sources in the same directory. Table 1 is the
experiment configuration table. Its sweep settings are in `configs/global.yaml`.

The chart scripts and paper-side evidence are from manuscript revision
`0532af685e3f6f46c97120673be1804d7401269d`. The artifact carries the saved
Tetris, OTIF, and LEAP results used for the chart and table exports.
