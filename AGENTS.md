Review this plan thoroughly before making any code changes. For every issue or recommendation, explain the concrete tradeoffs, give me an opinionated recommendation, and ask for my input before assuming a direction.
My engineering preferences (use these to guide your recommendations):
* DRY is important-flag repetition aggressively.
* Well-tested code is non-negotiable, I'd rather have too many tests than too few.
* I want code that's "engineered enough"--not under-engineered (fragile, hacky) and not over-engineered (premature abstraction, unnecessary complexity).
* l err on the side of handling more edge cases, not fewer; thoughtfulness > speed.
* Bias toward explicit over cleverness.

1. Architecture review
Evaluate:
  * Overall system design and component boundaries.
  * Dependency graph and coupling concerns.
  * Data flow patterns and potential bottlenecks.
  * Scaling characteristics and single points of failure.
  * Security architecture (auth, data access, API boundaries)
2. Code quality review
Evaluate:
  * Code organization and module structure.
  * DRY violations–be aggressive here.
  * Error handling patterns and missing edge cases (call these out explicitly).
  * Technical debt hotspots.
  * Areas that are over-engineered or under-engineered relative to my preferences.
3. Test review
Evaluate:
  * Test coverage gaps (unit, integration, e2e). test quay and assertion strength
  * Missing edge case coverage–be thorough.
  * Untested failure modes and error paths.
4. Performance review
Evaluate:
  * N+1 queries and database access patterns.
  * Memory-usage concerns.
  * Caching opportunities.
  * Slow or high-complexity code paths.

For each issue you find
For every specific issue (bug, smell, design concern, or risk):
* Describe the problem concretely, with file and line references.
* Presence-s options, including do nothing where that's reasonable.
* For each option, specify: implementation effort, risk, impact on other code, and maintenance burden.
* Give me your recommended option and why, mapped to my preferences above.
* Then explicitly ask whether i agree or want to choose a different direction before proceeding
Workflow and interaction
* Do not assume my priorities on timeline or scale.
* After each section, pause and ask for my feedback before moving on.

BEFORE YOU START:
Ask it I want one of two options:
1/ BIG CHANGE: Work through this interactively, one section at a time (Architecture → Code Quality → Tests → Performance) with at most 4 top issues in each section.
SMALL CHANGE: work through interactively oNe question per review section

FOR EACH STAGE OF REVIEW: output the explanation and pros and cons of each stage's questions AND your opinionated recommendation and why, and then use AskUserQuestion. Also NUMBER issues and then give LETTERS for options and when using AskUserQuestion make sure each option clearly labels the issue NUMBER and option LETTER so the user doesn't get confused. Make the recommended option always the 1st option.


# Repository Guidelines

## Executing commands
- Execute commands inside the container:
  - This project is developed inside a docker container.
  - Any scripts in the repository should be executed using `docker exec polyis` if this environment is not already inside a container.
    - Note: read and write command may be executed directly without `docker exec polyis`.

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
- If a Python file requires argument parsing, define the `parse_args` function near the top of the file, below imports, global variable definitions, and type definitions.
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
