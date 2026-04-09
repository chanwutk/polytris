# Mistrack-Rate Ablation

Ablation for `paper/content/07_e2e.tex`. Shows that learned mistrack-rate constraints approximate the full `3x3` sampling-rate design space using oracle boxes from `002_naive/tracking.jsonl`.

Isolated from the main pipeline in `scripts/`.

## Design

- `3x3` rectangular grid over `1080x720` frames, per-cell rates `{1, 2, 4}`.
- Exhaustive search: `3^9 = 19,683` global grids.
- Heuristic: one grid per threshold `{30, 40, ..., 95, 100}%`, learned on train.
- Both evaluated on test. Reports mistrack rate, HOTA, and retention rate.

## Pipeline

**Stage 1 -- Heuristic (train).** For each video, rerun the tracker at rates 1/2/4 using oracle boxes. Count correct vs incorrect associations per cell (p016 temporal-consistency logic). Aggregate across train, Laplace-smooth, threshold into max-rate grids.

**Stage 2 -- Evaluate (test).** For each candidate grid (exhaustive + heuristic): prune polyominoes via ILP, filter boxes by center-in-kept-tile, rerun tracker, score mistrack rate and retention rate, optionally compute HOTA.

**Stage 3 -- Plot.** Scatter plots of mistrack vs HOTA, mistrack vs retention, retention vs HOTA. Exhaustive Pareto frontier highlighted; heuristic points overlaid.

## Files

| File | Role |
|------|------|
| `run.py` | CLI entrypoint (runs all three stages) |
| `run.sh` | Multi-dataset/tracker wrapper |
| `mistrack_rate/common.py` | Constants, geometry, grid encoding, I/O helpers |
| `mistrack_rate/heuristic.py` | Train-side heuristic construction |
| `mistrack_rate/evaluate.py` | Test-side evaluation (parallel via `multiprocessing.Pool`) |
| `mistrack_rate/analysis.py` | Pareto annotation and plotting |
| `tests/test_mistrack_rate_ablation.py` | Unit tests |

## Cache Layout

```
/polyis-cache/<dataset>/ablation/mistrack-rate/3x3/<tracker>/
  train/  counts.npy, accuracy.npy, max_rate_table.npy, metadata.json
  test/   results.csv, plots/*.png
```

## Metrics

- **Mistrack rate** = `incorrect_anchors / total_anchors`. Anchor = retained frame where GT track center is in a kept tile. Association must hold at adjacent anchors.
- **Retention rate** = `retained_active_cells / original_active_cells` (aggregated over all frames and videos).
- **HOTA** = standard TrackEval HOTA against groundtruth after re-tracking filtered boxes.

## Usage

```bash
# Single dataset
python ablation/mistrack-rate/run.py --dataset jnc0
python ablation/mistrack-rate/run.py --dataset jnc0 --no-hota --limit-configs 64

# Multi-dataset via wrapper
bash ablation/mistrack-rate/run.sh
bash ablation/mistrack-rate/run.sh --all-trackers
bash ablation/mistrack-rate/run.sh --datasets jnc0,jnc2 --no-hota --limit-configs 64
```

Key flags:
- `--no-hota` -- skip HOTA for fast smoke runs.
- `--limit-configs N` -- cap exhaustive configs; heuristic configs always run.
- `--num-workers N` -- parallelism for evaluation, video prep, and heuristic training (default: 75).
- `--video-fraction-divisor D` -- keep every D-th video. `run.sh` defaults to 3 for non-jnc datasets.
- `--force` -- recompute even if cached.

## Notes

- Reads from `002_naive` cache; does not mutate main pipeline outputs.
- TrackEval is imported only when HOTA is requested.
- Parallelism is scoped per dataset/tracker pair; `run.sh` loops over pairs.
