# Repository Guidelines

## Executing commands
- Execute commands inside the container:
  - This project is developed inside a docker container.
  - Any command should be executed using `docker exec polyis` if this environment is not already inside a container.

## Execution Pipeline (scripts/)
- Primary entrypoint: scripts in `scripts/` run the full pipeline; numeric prefixes define order.
- Typical flow: `p000_preprocess_dataset.py` → `p020_exec_classify.py` → `p030_exec_compress.py` → `p040_exec_detect.py` → `p050_exec_uncompress.py` → `p060_exec_track.py` → `p070_results_statistics_acc.py`/`p090_results_track_visualize.py`.
- Tuning stage: optional `p010–p015` scripts for segmentation, training, and parameter selection.
- Usage pattern: run inside the container with `./run scripts/<script>.py --help` to see options.

## Environment & Dependencies
- Runtime: Docker-first workflow. Start services with `docker compose up --detach --build`, enter with `./dock`.
- Python: version 3.13. Use `conda` for packages that do not build reliably with `pip`.
- Dependencies: `requirements.txt` is the authoritative list; use `pip install -r requirements.txt` inside the env. Seed the env with `conda env update -f environment.yml` when needed.
- Poetry: not used. Ignore `poetry.lock`; `pyproject.toml` configures tools (e.g., `pyright`) only.

## In-Container Paths
- Root directory: `/polyis` (project root when inside Docker).
- Data directory: `/polyis-data` (mounted via `POLYIS_DATA`).
- Cache directory: `/polyis-cache` (mounted via `POLYIS_CACHE`).

## Project Structure & Module Organization
- `scripts/`: End-to-end pipeline steps, ordered by numeric prefixes.
- `modules/`: Submodule dependencies vendored for reproducibility (e.g., Detectron2, TrackEval, Darknet).
- `lib/`: Cython implementations of performance‑critical algorithms (`pack_append.pyx`, `group_tiles.pyx`) plus tests under `lib/tests/`.
- `polyis/`: Core system components and utilities used by scripts (`images.py`, `utils.py`).
- `assets/`, `output/`, `pipeline-stages/`, `detection_experiments/`: Data, results, and experiment artifacts.

## Build, Test, and Development Commands
- Run a script: `./run scripts/p040_exec_detect.py --help` (from outside the container).
- Build Cython (from `lib/`): `./build.sh` or `python setup.py build_ext --inplace`.
- Tests: `pytest lib/tests -v` for Cython and fast unit tests.
- Local (non‑Docker) setup: `conda env update -f environment.yml` then `pip install -r requirements.txt`.

## Coding Style & Naming Conventions
- Python 3.13, 4‑space indentation, PEP 8. Prefer type hints and concise docstrings.
- Filenames/modules: `snake_case.py`; classes: `PascalCase`; constants: `UPPER_SNAKE_CASE`.
- Scripts: keep numeric prefixes (`p0xx_`, `p0yy_exec_*`, `p0zz_tune_*`).
- Type checking: `pyright` configured in `pyproject.toml`.
- Always add comments to every operations of the code.
- Do not add any comment or any doc-string to the parse_args function and its contents.

## Testing Guidelines
- Framework: `pytest`. Place new tests alongside targets or under `lib/tests/` as `test_*.py`.
- Prioritize fast unit tests for Cython boundaries; mock external dependencies.
- Example: `pytest lib/tests/test_pack_append.py -q`.

## Commit & Pull Request Guidelines
- Conventional Commits: `feat(scripts): ...`, `refactor(lib): ...`, `perf: ...`, `chore: ...`.
- PRs: include description, linked issues, relevant logs or small screenshots, and performance/accuracy notes.
- Keep Docker configs consistent (names in `docker-compose.yml`, `dock`, and `run`).

## Security & Configuration Tips
- Do not commit datasets, cache, or large models. Use volumes: `POLYIS_DATA=/polyis-data`, `POLYIS_CACHE=/polyis-cache` (see `docker-compose.yml`).
- Keep credentials out of the repo. Verify paths before running scripts that write to `/polyis-data` or `/polyis-cache`.
