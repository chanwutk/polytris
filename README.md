<p align="center">
  <img src="assets/tetris-logo.svg" alt="Tetris" width="360">
</p>

<h1 align="center">Tile-level Sampling for Efficient and High-Fidelity Video Object Tracking</h1>

> The codebase still uses the legacy identifier `polyis` for directory names,
> container names, and environment variables (`POLYIS_DATA`, `POLYIS_CACHE`).
> Treat `polyis` and `Tetris` as the same project; only the public-facing
> name has changed.

## Overview

Tetris materializes tracks from stored video captured by stationary cameras on
a single GPU server.
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
- **Git submodules** for the pinned third-party code. From the checkout, run
  `git submodule update --init --recursive`.
- **Gurobi license** mounted at `~/.gurobi/gurobi.lic`. The ILP-based pruning
  stage and several Cython extensions link against Gurobi at build time.
  Academic licenses are available at <https://www.gurobi.com/academia/>.
- **OTIF dataset** (see [Datasets](#datasets)).

## Setup

The development environment is Docker-first. The repo ships a
`docker-compose.yml` that extends `docker-compose.base.yml` and adds GPU
reservations. You should not need to edit either file for a default install —
just set the host paths via a `.env` file.

### 1. Create a `.env` file at the repo root

`docker-compose.base.yml` reads `${DATA}` and `${WORK}` from the environment
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

`PAPER_DIR` and `SOTA_RESULTS` can be set in `.env` to mount a writable copy of
the paper and the baseline result directory for figure rendering and
comparisons. They map to `/polytris-paper` and `/sota-results`, respectively.

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

The Docker image installs the Gurobi 13.0.1 native library and sets
`GUROBI_HOME=/opt/gurobi1301/linux64`. Mount your own Gurobi license before running
pruning.

### 5. Run the test suite

```bash
pytest tests/ -v
```

A focused subset for the packing extension:

```bash
pytest tests/pack -v
```

## Datasets

The evaluation covers seven stationary-camera datasets:

```
caldot1-y05  caldot2-y05  jnc0  jnc2  jnc6  jnc7  ams-y05
```

See [Reproduction](REPRODUCE.md) for source layouts, split rules, model
weights, an end-to-end example, and the figure and table commands.

## Pipeline Overview

The extraction pipeline has five stages, mirroring the architecture in
the paper:

```
Decode -> Classify -> [Prune] -> Compress -> Detect+Uncompress -> Track
```

Each stage has a standalone script under `scripts/` and a streaming
implementation under `execution/`. Pick the entrypoint that matches your use
case:

| Mode               | Command                                  | Notes                                                                          |
| ------------------ | ---------------------------------------- | ------------------------------------------------------------------------------ |
| Pipeline-parallel  | `./run execution/main.py --test`         | One pipeline per GPU. |
| Per-stage scripts  | `./run scripts/p020_exec_classify.py`    | Useful for debugging a single stage. See each script's `--help`.               |
| Paper reproduction | [Train, validate, and test](REPRODUCE.md#run-one-dataset-end-to-end) | One dataset at a time, with a validation JSON for test selection. |

The full reproduction recipe also involves three other phases that are
orthogonal to the online pipeline:

- **Tuning** (`scripts/p011`–`p017_tune_*.py`): trains the relevance
  classifier, learns per-tile sampling thresholds, and selects baselines.
- **Preprocessing** (`preprocess/p000`–`p008_preprocess_*.py`): builds
  ground-truth detections, naive baselines, and torchvision/ultralytics
  training sets.
- **Evaluation** (`evaluation/p1**`–`evaluation/p2**`): computes accuracy
  and throughput, runs comparisons against OTIF and other SOTA trackers,
  and produces the figures in the paper.

Every entrypoint takes `--help`. Use it.

## Repository Layout

```
polyis/         Core library: Cython modules, models, trackers, IO helpers
scripts/        Per-stage offline tuning + online execution scripts
reproduce/      Per-dataset experiment and figure/table entry points
preprocess/     Dataset prep, ground truth, baseline tracking
execution/      Pipeline-parallel runtime (execution/main.py is the entrypoint)
evaluation/     Metric computation, comparisons, paper-figure generators
configs/        Global config, detector/tracker registry, sync config
tests/          Pytest suite (Cython, ILP, trackers, evaluation)
modules/        Pinned third-party submodules, including TrackEval
measurements/   Run aggregation and analysis
measurements-gpu/ GPU launchers for revision experiments
measurements-results/ Saved evidence used by aggregators and paper exporters
paper           Symlink to the paper checkout mounted at /polytris-paper
```

In-container conventions:

- Project root: `/polyis`
- Data: `/polyis-data` (env var `POLYIS_DATA`)
- Cache: `/polyis-cache` (env var `POLYIS_CACHE`)

## Development

See [`AGENTS.md`](./AGENTS.md) for coding conventions, the dev workflow,
commit guidelines, and tips for working with Docker on a shared host.
