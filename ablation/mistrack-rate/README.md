# Mistrack-Rate Ablation

This directory contains a separate ablation-only pipeline for the paper experiment described in `paper/content/07_e2e.tex`.

The goal of the experiment is to show that the learned mistrack-rate constraints are a good approximation of the global `3 x 3` sampling-rate design space when the system is evaluated with oracle boxes rather than detector outputs.

This ablation is intentionally isolated from the main execution pipeline in `scripts/`.

## Agreed Plan

The experiment uses the following fixed design decisions.

- Use a rectangular `3 x 3` tile grid over the standard `1080 x 720` preprocessed video frames.
- Restrict tile-level sampling-rate choices to `{1, 2, 4}` only.
- Do not rerun detection.
- Use the oracle-like box cache from `002_naive/tracking.jsonl` as the source of boxes for both train and test.
- Learn the heuristic on `train`.
- Evaluate both the learned heuristic and the exhaustive global grid search on `test`.
- Keep the paper semantics:
  - build relevancy from cached oracle boxes,
  - group those relevant tiles into polyominoes,
  - prune with the ILP under a per-cell max-rate grid,
  - load cached boxes from `002_naive/tracking.jsonl`,
  - retain only boxes whose center falls inside a kept tile,
  - rerun tracking on the filtered boxes.
- Report:
  - mistrack rate,
  - HOTA,
  - retention rate.
- Use retention rate rather than raw retained-tile count as the main efficiency metric.
- Plot:
  - mistrack rate vs HOTA,
  - mistrack rate vs retention rate,
  - retention rate vs HOTA.
- In each plot:
  - show all exhaustive points in gray,
  - highlight the exhaustive Pareto frontier,
  - highlight the heuristic points.

## Search Space

With a `3 x 3` grid and per-cell rates `{1, 2, 4}`, the exhaustive search space is:

- `3^9 = 19683` global grids.

Each grid is treated as one global configuration over the full test split.

## Implementation Summary

The ablation is implemented as a standalone package under `ablation/mistrack-rate/`.

- [run.py](/work/cwkt/projects/polyis/ablation/mistrack-rate/run.py)
  - CLI entrypoint with `heuristic`, `evaluate`, `plot`, and `all` subcommands.
- [mistrack_rate/common.py](/work/cwkt/projects/polyis/ablation/mistrack-rate/mistrack_rate/common.py)
  - shared constants and cache paths,
  - `3 x 3` rectangular-grid geometry helpers,
  - rate-grid encoding and enumeration,
  - naive-source loading,
  - box filtering and retention helpers.
- [mistrack_rate/heuristic.py](/work/cwkt/projects/polyis/ablation/mistrack-rate/mistrack_rate/heuristic.py)
  - adapts the `p016` mistrack-counting logic to the rectangular `3 x 3` grid,
  - reruns the tracker at rates `{1, 2, 4}` on train,
  - aggregates per-cell correct and incorrect counts,
  - emits one thresholded max-rate grid for each threshold in `30,40,50,60,70,80,90,95,100`.
- [mistrack_rate/evaluate.py](/work/cwkt/projects/polyis/ablation/mistrack-rate/mistrack_rate/evaluate.py)
  - prepares test videos from `002_naive/tracking.jsonl`,
  - builds relevancy bitmaps,
  - groups tiles into polyominoes,
  - prunes with the ILP using a candidate `3 x 3` max-rate grid,
  - filters cached boxes by center-in-kept-cell,
  - reruns tracking,
  - computes mistrack rate,
  - optionally computes HOTA.
- [mistrack_rate/analysis.py](/work/cwkt/projects/polyis/ablation/mistrack-rate/mistrack_rate/analysis.py)
  - writes the canonical CSV,
  - annotates Pareto-front membership,
  - produces the requested plots.
- [tests/test_mistrack_rate_ablation.py](/work/cwkt/projects/polyis/tests/test_mistrack_rate_ablation.py)
  - covers the new geometry, enumeration, heuristic, mistrack, and Pareto logic.

## Data Flow

The ablation has two stages.

### 1. Train-Side Heuristic Construction

For each train video:

1. Load `002_naive/tracking.jsonl`.
2. Convert cached tracks `[track_id, x1, y1, x2, y2]` into detector-like rows `[x1, y1, x2, y2, 1.0]`.
3. Rerun the tracker at rates `1`, `2`, and `4`.
4. Use the `p016` temporal-consistency logic to count correct and incorrect associations.
5. Map each groundtruth box to overlapping cells in the rectangular `3 x 3` grid.
6. Aggregate per-cell counts across the full train split.
7. Convert those counts into Laplace-smoothed per-cell accuracies.
8. Build one thresholded `C^M` max-rate grid per threshold.

Train outputs are written under:

- `/polyis-cache/<dataset>/ablation/mistrack-rate/3x3/<tracker>/train/`

Key files:

- `counts.npy`
- `accuracy.npy`
- `max_rate_table.npy`
- `metadata.json`

### 2. Test-Side Evaluation

For each test video:

1. Load `002_naive/tracking.jsonl`.
2. Build `3 x 3` per-frame relevancy bitmaps from box overlap.
3. Group tiles into polyominoes.
4. For each evaluated rate grid:
   - solve the pruning ILP,
   - keep only the selected polyominoes,
   - filter cached boxes by whether the box center lies inside a retained tile,
   - rerun the tracker on those filtered boxes,
   - compute a scalar mistrack rate,
   - compute retention rate,
   - optionally compute HOTA.

Test outputs are written under:

- `/polyis-cache/<dataset>/ablation/mistrack-rate/3x3/<tracker>/test/`

Key files:

- `results.csv`
- `plots/mistrack_vs_hota.png`
- `plots/mistrack_vs_retention.png`
- `plots/retention_vs_hota.png`

## Metric Definitions

### Mistrack Rate

The heuristic-learning stage uses the same `p016` temporal-consistency idea at sampled frames.

The test-side scalar mistrack rate is schedule-aware:

- anchor frames are the retained frames where a groundtruth track center lies inside a kept tile,
- each anchor requires a valid GT-to-prediction match on that frame,
- the same association must hold at the next and previous anchor for that GT track whenever those anchors exist.

The reported mistrack rate is:

- `incorrect_anchors / total_anchors`

### Retention Rate

The reported retention rate is:

- `retained_active_cells / original_active_cells`

aggregated over all test videos and frames.

### HOTA

HOTA is computed against the standard test groundtruth tracking files after the filtered boxes are re-tracked.

## CSV Schema

The canonical CSV contains one row per evaluated configuration.

Important columns:

- `dataset`
- `split`
- `tracker`
- `method`
  - `exhaustive` or `heuristic`
- `heuristic_threshold`
- `variant_id`
- `grid_key`
- `grid_rates_json`
- `mistrack_rate`
- `HOTA_HOTA`
- `retention_rate`
- `anchor_correct`
- `anchor_incorrect`
- `anchor_total`
- `is_pareto_mistrack_vs_hota`
- `is_pareto_mistrack_vs_retention`
- `is_pareto_retention_vs_hota`

## Commands

Run the ablation through:

```bash
python ablation/mistrack-rate/run.py heuristic --dataset jnc0
python ablation/mistrack-rate/run.py evaluate --dataset jnc0 --no-hota --limit-configs 64
python ablation/mistrack-rate/run.py evaluate --dataset jnc0 --no-hota --limit-configs 64 --num-workers 8
python ablation/mistrack-rate/run.py evaluate --dataset jnc0 --video-fraction-divisor 3
python ablation/mistrack-rate/run.py plot --dataset jnc0
python ablation/mistrack-rate/run.py all --dataset jnc0
bash ablation/mistrack-rate/run.sh
bash ablation/mistrack-rate/run.sh --all-trackers
```

Notes:

- `--no-hota` is useful for cheap smoke runs.
- `--limit-configs` is useful for partial exhaustive runs while debugging.
- `--num-workers` parallelizes configuration evaluation within one dataset/tracker pair.
- `--video-fraction-divisor 3` keeps one deterministic third of the sorted train and test videos.
- The full exhaustive search is intentionally expensive.
- `run.sh` loops over all configured datasets and forwards the remaining flags to `run.py all`.

## Verification Performed

The current implementation has been checked with:

- Python compilation of the ablation modules and tests,
- targeted unit tests in [tests/test_mistrack_rate_ablation.py](/work/cwkt/projects/polyis/tests/test_mistrack_rate_ablation.py:26),
- a real-data smoke run on `jnc0` test video `te01.mp4` with the all-ones grid and `compute_hota=False`.

The smoke run produced:

- `variant_id = exhaustive_1-1-1-1-1-1-1-1-1`
- `mistrack_rate = 0.0`
- `retention_rate = 1.0`
- `anchor_total = 15421`

## Practical Notes

- The ablation reads from the existing `002_naive` cache and does not mutate the main execution outputs.
- TrackEval imports are loaded only when HOTA is actually requested.
- Temporary HOTA tracking files preserve the true video frame count so that sparse predictions do not truncate evaluation inputs.
- The code currently defaults to the first configured tracker from `CONFIG['EXEC']['TRACKERS']`.
