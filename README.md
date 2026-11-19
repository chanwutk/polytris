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
`/polyis-cache`. The download script creates these directories.

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

On the Docker host, download and stage all seven datasets and their test labels
with one command. The script uses the pinned public bundles and creates the
`caldot1-y05`, `caldot2-y05`, and `ams-y05` symlinks automatically.

```sh
bash reproduce/download.sh "$DATA"
```

Pass `b3d` or `otif` after `"$DATA"` to download just that bundle.
The script skips identical files and stops if an existing file differs.

Inside Docker, `python scripts/weights.py download` fetches and verifies all
four detector checkpoints. For details about how the B3D test labels were
created, see the
[human-reviewed labels and annotation workflow](https://anonymous.4open.science/r/tetris-b3d-human-annotation)
and [Faster R-CNN pre-labeling code and weights](https://anonymous.4open.science/r/tetris-b3d-fasterrcnn).

## Run one dataset end to end

This example uses B3D1 (`jnc0`) and the paper's configuration grid. Download
the B3D bundle on the Docker host if you have yet to stage it.

```sh
bash reproduce/download.sh "$DATA" b3d
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
Each dataset's cache should have one active writer.

### Commands by dataset

Start with the [Docker setup](#docker-setup). Skip the host download if you
already staged that bundle.

<details>
<summary>B3D1 (jnc0)</summary>

On the Docker host:

```sh
bash reproduce/download.sh "$DATA" b3d
```

Inside Docker from `/polyis`:

```sh
set -e
python scripts/weights.py download --b3d-only
bash reproduce/train.sh jnc0
bash reproduce/validate.sh jnc0 validation-jnc0.json
bash reproduce/test.sh jnc0 validation-jnc0.json
```

</details>

<details>
<summary>B3D2 (jnc2)</summary>

On the Docker host:

```sh
bash reproduce/download.sh "$DATA" b3d
```

Inside Docker from `/polyis`:

```sh
set -e
python scripts/weights.py download --b3d-only
bash reproduce/train.sh jnc2
bash reproduce/validate.sh jnc2 validation-jnc2.json
bash reproduce/test.sh jnc2 validation-jnc2.json
```

</details>

<details>
<summary>B3D3 (jnc6)</summary>

On the Docker host:

```sh
bash reproduce/download.sh "$DATA" b3d
```

Inside Docker from `/polyis`:

```sh
set -e
python scripts/weights.py download --b3d-only
bash reproduce/train.sh jnc6
bash reproduce/validate.sh jnc6 validation-jnc6.json
bash reproduce/test.sh jnc6 validation-jnc6.json
```

</details>

<details>
<summary>B3D4 (jnc7)</summary>

On the Docker host:

```sh
bash reproduce/download.sh "$DATA" b3d
```

Inside Docker from `/polyis`:

```sh
set -e
python scripts/weights.py download --b3d-only
bash reproduce/train.sh jnc7
bash reproduce/validate.sh jnc7 validation-jnc7.json
bash reproduce/test.sh jnc7 validation-jnc7.json
```

</details>

<details>
<summary>CalDoT1 (caldot1-y05)</summary>

On the Docker host:

```sh
bash reproduce/download.sh "$DATA" otif
```

Inside Docker from `/polyis`:

```sh
set -e
python scripts/weights.py download
bash reproduce/train.sh caldot1-y05
bash reproduce/validate.sh caldot1-y05 validation-caldot1-y05.json
bash reproduce/test.sh caldot1-y05 validation-caldot1-y05.json
```

</details>

<details>
<summary>CalDoT2 (caldot2-y05)</summary>

On the Docker host:

```sh
bash reproduce/download.sh "$DATA" otif
```

Inside Docker from `/polyis`:

```sh
set -e
python scripts/weights.py download
bash reproduce/train.sh caldot2-y05
bash reproduce/validate.sh caldot2-y05 validation-caldot2-y05.json
bash reproduce/test.sh caldot2-y05 validation-caldot2-y05.json
```

</details>

<details>
<summary>Amsterdam (ams-y05)</summary>

On the Docker host:

```sh
bash reproduce/download.sh "$DATA" otif
```

Inside Docker from `/polyis`:

```sh
set -e
python scripts/weights.py download
bash reproduce/train.sh ams-y05
bash reproduce/validate.sh ams-y05 validation-ams-y05.json
bash reproduce/test.sh ams-y05 validation-ams-y05.json
```

</details>

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
