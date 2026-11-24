<p align="center">
  <img src="assets/tetris-logo.svg" alt="Tetris" width="360">
</p>

<h1 align="center">Tile-level Sampling for Efficient and High-Fidelity Video Object Tracking</h1>

> The Python package is `tetris`. The Docker setup still uses the legacy name
> `polyis` for the container, the in-container paths (`/polyis`, `/polyis-data`,
> `/polyis-cache`), and the environment variables (`POLYIS_DATA`, `POLYIS_CACHE`).

## Overview

Tetris is a video object-tracking system designed for **stationary cameras**.
It targets *track materialization*: extracting reusable object tracks from
raw video so that downstream queries do not have to rerun tracking. Detector
calls dominate the cost of this workload, and stationary video gives Tetris a
structural lever — large regions of every frame contain no objects of
interest, and the regions that do tolerate different sampling rates.

Tetris decomposes each video into a tile-level **polyomino** data model.
Three operators run upstream of the user-provided detector:

- **Classify** — a learned relevance classifier identifies tiles with
  potential objects and groups them into polyominoes.
- **Prune** — an integer linear program (ILP) drops polyominoes that are
  redundant under a user-specified accuracy constraint.
- **Pack** — the survivors are assembled into dense canvases that minimize
  detector calls.

Across 7 stationary-video datasets, Tetris stays within a 5% tracking-accuracy
loss of a full-frame reference pipeline while delivering substantial
throughput improvements over prior systems.

## Paper

Paper is currently under review. A link will be added on publication.

## Citation

Citation will be added on publication.

## Prerequisites

- **Docker** with the **NVIDIA Container Toolkit** (the pipeline depends on
  CUDA inside the container).
- **Gurobi license** mounted at `~/.gurobi/gurobi.lic`. The ILP-based pruning
  stage and several Cython extensions link against Gurobi at build time.
  Academic licenses are available at <https://www.gurobi.com/academia/>.
- **OTIF dataset** (see [Datasets](#datasets)).

## Setup

The development environment is Docker-first. The repo ships a
`docker-compose.yml` with GPU reservations. You should not need to edit it for
a default install — just set the host paths via a `.env` file.

### 1. Create a `.env` file at the repo root

`docker-compose.yml` reads `${DATA}` and `${WORK}` from the environment
to locate your dataset and work-area parents. Create `.env` next to
`docker-compose.yml`:

```bash
# .env
DATA=/absolute/path/to/your/data-parent
WORK=/absolute/path/to/your/work-parent
```

The resulting mounts will be:

| Host path                                | Container path     | Purpose                                  |
| ---------------------------------------- | ------------------ | ---------------------------------------- |
| `${DATA}/polyis-data`                    | `/polyis-data`     | Datasets, intermediate results, models   |
| `${DATA}/polyis-cache`                   | `/polyis-cache`    | Per-stage caches and partial outputs     |
| `${WORK}/../data/otif-dataset`           | `/otif-dataset`    | Raw OTIF videos                          |
| `${HOME}/.gurobi/gurobi.lic`             | Gurobi license     | ILP solver license                       |

`docker-compose.yml` also includes a few mounts that are specific to the
author's environment (`/sota-results`, `/polytris-paper`). If you do not have
those paths, comment out the corresponding lines. They are only required for
SOTA comparisons and paper-figure rendering.

### 2. Build and start the container

```bash
docker compose up --detach --build
```

### 3. Enter the container

```bash
./dock
```

This runs `docker exec -it polyis bash`. From inside the container the repo
is mounted at `/polyis`.

### 4. Build native extensions

The pack/group/ILP modules are written in Cython and link against the Gurobi
C library. Build them once after entering the container:

```bash
cd /polyis
python setup.py build_ext
```

To rebuild from scratch:

```bash
python setup.py clean
python setup.py build_ext
```

Set `GUROBI_HOME` to your Gurobi install directory if `setup.py` cannot find
it automatically (for example,
`export GUROBI_HOME=/opt/gurobi1100/linux64`).

### 5. Run the test suite

```bash
pytest tests/ -v
```

A focused subset for the packing extension:

```bash
pytest tests/pack -v
```

## Datasets

Tetris is evaluated on 7 stationary-camera splits drawn from the public
[OTIF dataset](https://github.com/favyen2/otif):

```
caldot1-y05  caldot2-y05  jnc0  jnc2  jnc6  jnc7  ams-y05
```

Place the OTIF tarballs under `${DATA}/../data/otif-dataset` so that they
appear at `/otif-dataset` inside the container. Then materialize the
tile-aligned splits and segmentations Tetris consumes:

```bash
./run experiments/preprocess/p000_preprocess_dataset.py --help
./run experiments/preprocess/p001_preprocess_split.py --help
```

The active dataset list lives in `configs/global.yaml` under `EXEC.DATASETS`.

## Pipeline Overview

The online tracking pipeline has five stages, mirroring the architecture in
the paper:

```
Decode -> Classify -> [Prune] -> Compress -> Detect+Uncompress -> Track
```

`execution/` runs the stages as one streaming pipeline. The original
per-stage scripts, which the paper's numbers come from, are kept under
`experiments/scripts/` as the reference the pipeline is checked against.
Pick the entrypoint that matches your use case:

| Mode               | Command                                  | Notes                                                                          |
| ------------------ | ---------------------------------------- | ------------------------------------------------------------------------------ |
| Pipeline-parallel  | `python -m execution.main --config <yaml> --videos <dir> --out <dir>` | One configuration over a directory of clips; write the YAML with `profiling` or `experiments/tools/execution/make_*_configs.py`. |
| Per-stage scripts  | `./run experiments/scripts/p020_exec_classify.py`    | Useful for debugging a single stage. See each script's `--help`.               |
| Phase orchestrator | `bash experiments/scripts/run.sh`                    | Legacy two-pass (`--valid` then `--test`) driver. Most phases are commented out; uncomment what you need. |

The full reproduction recipe also involves three other phases that are
orthogonal to the online pipeline:

- **Tuning** (`scripts/p011`–`p017_tune_*.py`): trains the relevance
  classifier, learns per-tile sampling thresholds, and selects baselines.
- **Preprocessing** (`experiments/preprocess/p000`–`p008_preprocess_*.py`): builds
  ground-truth detections, naive baselines, and torchvision/ultralytics
  training sets.
- **Evaluation** (`experiments/evaluation/p1**`–`p2**`): computes accuracy
  and throughput, runs comparisons against OTIF and other SOTA trackers,
  and produces the figures in the paper.

Every entrypoint takes `--help`. Use it.

## Repository Layout

```
tetris/         Core library: Cython modules, models, trackers, IO helpers
execution/      Pipeline-parallel runtime (`python -m execution.main`)
profiling/      Pipeline-parallel validation sweep
scripts/        Offline tuning (p011-p017): classifier training, sampling limits
configs/        Global config, detector/tracker registry, sync config
tests/          Pytest suite for the system (Cython, ILP, trackers, pipelines)
experiments/    Everything that produces the paper's numbers:
  scripts/          the original per-stage pipeline (p020-p060, run.sh)
  preprocess/       dataset prep, ground truth, baseline detectors and tracking
  evaluation/       metric computation, comparisons, paper-figure generators
  measurements/     one script per revision measurement
  measurements_gpu/ long-running GPU jobs behind those measurements
  measurements-results/  committed outputs of the measurements
  ablation/         the mistrack-rate ablation (Fig. 11)
  tools/            scripts-vs-pipeline comparisons, benchmarks, paper configs
  configs/          configs for those tools
  tests/            tests for the code above, incl. the equivalence tests
modules/        Vendored third-party code (Detectron2, TrackEval, Darknet)
demo/           Interactive visualization assets
paper/          Manuscript source (LaTeX, anonymized for review)
```

In-container conventions:

- Project root: `/polyis`
- Data: `/polyis-data` (env var `POLYIS_DATA`)
- Cache: `/polyis-cache` (env var `POLYIS_CACHE`)

## Development

See [`AGENTS.md`](./AGENTS.md) for coding conventions, the dev workflow,
commit guidelines, and tips for working with Docker on a shared host.
