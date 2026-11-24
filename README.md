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

Download and extract the ZIP from the anonymous artifact page. The ZIP
includes the pinned public dependencies TrackEval and LAP. Before building,
check that it contains `modules/TrackEval/trackeval/__init__.py` and
`modules/lap/_lapjv_cpp/lapjv.cpp`.

Set `DATA` to an absolute directory on the host and put that same assignment
in `.env` in the repository root. Docker mounts
`$DATA/polyis-data` at `/polyis-data` and `$DATA/polyis-cache` at
`/polyis-cache`. The download script creates these directories. Optionally set
`SHM_SIZE` in `.env` too (default `128gb`); see
[Run one configuration with execution2](#run-one-configuration-with-execution2).

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
| CalDoT1, CalDoT2, Amsterdam (`caldot1-y05`, `caldot2-y05`, `ams-y05`) | Processed OTIF clips and canonical test annotations. | 60 / 60 / 60 per dataset. |
| B3D1–4 (`jnc0`, `jnc2`, `jnc6`, `jnc7`) | Processed B3D clips and the final human-verified test labels. | 8 / 5 / 5 per junction. |

On the Docker host, download and stage all seven datasets and their test labels
with one command. The script uses the pinned public bundles and creates the
`caldot1-y05`, `caldot2-y05`, and `ams-y05` symlinks automatically.

```sh
bash reproduce/download.sh "$DATA"
```

Pass `b3d` or `otif` after `"$DATA"` to download just that bundle.
The script skips identical files and stops if an existing file differs.

Inside Docker, `python scripts/weights.py download` fetches and verifies all
four detector checkpoints.

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
- `validate.sh` runs the configured sweep on validation clips with the
  extraction scripts (`scripts/p020`–`p060`) and writes JSON
  with the config, throughput in FPS, HOTA, and Pareto membership for each
  variant. The Pareto set includes the configurations needed for the paper's
  ablations.
- `test.sh` takes that JSON and runs the marked Pareto configurations on test
  clips with `execution2`, the pipeline-parallel extraction engine, one
  configuration at a time on one GPU. Pass a third argument, a `variant_id`
  from the JSON, to test just that configuration. The script writes per-clip
  tracking JSONL and the dataset's evaluation `090_tradeoff/tradeoff.csv`
  under `/polyis-cache`.

The test results are the paper's end-to-end numbers: HOTA and throughput
(§7.1, Figure 8), the extraction throughput behind the amortization analysis
(§7.2, Figure 9), and the speedups of the operator ablation (§7.4.4). Each
configuration's throughput is the test clips' frame count divided by
`execution2`'s runtime, measured from the first finished clip to the last
after a warm-up clip, so process start-up and model loading are excluded.
The reference pipeline that HOTA and speedups are measured against is the
full-frame detector and tracker (`preprocess/p002_*`).
<br/>
**Note:** The figures shipped with the artifact come from the older implementation (in `./scripts`) instead
([Please see Generate figures and tables](#generate-figures-and-tables)).

The wrappers run from the repository root, so a relative path like
`validation.json` means `/polyis/validation.json`; pass an absolute path to
put it elsewhere. Run `validate.sh` and `test.sh` with the same trained
artifacts and the same `configs/global.yaml`: `test.sh` refuses
configurations that are not in the current sweep grid. Run one command at a
time per dataset, since each one rewrites that dataset's folders under
`/polyis-cache`; different datasets can run in parallel.

### Run one configuration with execution2

`execution2` runs one configuration over a directory of clips and writes one
tracking JSONL per clip. After `train.sh` has trained the dataset's artifacts,
write the config for any `variant_id` of the sweep grid and run it from
`/polyis` inside the container:

```sh
python -m execution2.tools.run_variants config jnc0 \
    ShuffleNet05_60_4_r025_none_s100_sortcython > jnc0.yaml
python -m execution2.main --config jnc0.yaml \
    --videos /polyis-data/datasets/jnc0/test --out tracks-jnc0
```

`execution2.main` prints one JSON line with the number of clips, their total
frame count, and the runtime in seconds. The YAML that `run_variants config`
writes lists the dataset's trained classifier, sampling limits, and detector,
the configuration's parameters, and how many processes each pipeline stage
runs on which GPU.

`execution2` passes decoded frames and packed canvases between its processes
through shared memory (`/dev/shm`). Before it runs, it estimates the most it
can need and stops if that does not fit. The worst case for the test clips is
about 15 GB for CalDoT1 and CalDoT2 and about 40 GB for Amsterdam and B3D
(sample rate 1, pruning on). The container's shared memory is capped at
`SHM_SIZE` from `.env` (default `128gb`). Please keep `SHM_SIZE` within your free RAM and
at least the worst case of the datasets you run.

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

**Note:** Figures 8 and 9 and the operator ablation (§7.4.4) are drawn from saved
results of the implementation published in the paper, the extraction scripts
(`scripts/p020`–`p060`). `reproduce/test.sh` runs the same configurations with
the new
`execution2`, which produces the same tracks as the scripts but runs faster.
The `tradeoff.csv` it writes therefore shows the same HOTA as these figures
and higher throughput. The pruning solver stops at 0.1 s, so a configuration
whose solve reaches that limit can select different polyominoes from run to
run, in either implementation.

| Target | Paper output |
| --- | --- |
| `fig2` | Stationary-camera observations |
| `fig7` | Mistrack rate and sampling gap |
| `fig8` | End-to-end HOTA and throughput |
| `fig9` | Preparation cost and amortization |
| `fig10` | Classifier and packing results |
| `fig11` | Sampling ablation |
| `fig12` | Extraction runtime breakdown |
| `table2` | Native query comparison with OTIF, LEAP, Seiden, and NoScope |
| `response-tables` | LEAP reconstruction, resolution scaling, padding, and preparation |