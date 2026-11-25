# profiling: choose operating points on a validation split

Profiling runs the execution pipeline once per configuration of a parameter grid, over every video of a dataset's validation split, and picks the Pareto-optimal configurations (HOTA vs throughput).
Each configuration is an ordinary `python -m execution.main` run, so its runtime is execution's own end-to-end measurement (timing dummy included), not a sum of overlapping stage times.

The earlier fan-out sweep, which ran every configuration at once and shared intermediates, is kept as an experiment in `experiments/fanout_profiling/`.

## Steps

```sh
python -m profiling.main --config experiments/configs/profiling/<dataset>.yaml [--save]
python -m profiling.evaluate --config experiments/configs/profiling/<dataset>.yaml [--procs N] [--force]
python -m profiling.aggregate --config experiments/configs/profiling/<dataset>.yaml
```

1. `profiling.main` runs the configurations one at a time.
   A rerun resumes: configurations with `result.json` are skipped, failed or interrupted ones are cleaned and retried.
   A different sweep document is refused in an output that holds another sweep (its hash is in `sweep.json`).
   Exit code 1 when any configuration failed (`runs/<id>/error.txt` has the exit code and the end of stderr).
2. `profiling.evaluate` scores every finished configuration's tracks against the reference tracks with p110's settings (TrackEval: HOTA, Count), in parallel over configurations.
   It never overlaps a timed run, and can be rerun alone (`--force`).
3. `profiling.aggregate` combines each configuration's per-video results into the dataset HOTA (TrackEval's `combine_sequences`, as p110 and p111), and writes the Pareto fronts of every ablation condition in `tetris.ablation` (full, no sampling, no sampling or pruning): `configurations.json`, `pareto.json`, and `pareto.html` (click a point to get its execution YAML).

## Test split

The Pareto configurations chosen on the validation split are rerun on the test split by the same three steps.
`python -m experiments.tools.profiling.make_test_configs` writes `experiments/configs/profiling-test/<dataset>.yaml` from each aggregated validation sweep: the same document with `videoset: test`, results under `/polyis-data/profiling/test`, and `configurations` listing the ids in its `pareto.json` (every ablation condition's front).
A sweep config's optional `configurations` restricts any sweep to those ids of its grid (in grid order; dedup applies among them).
The test sweep's `configurations.json` holds each configuration's test HOTA and end-to-end throughput, and its `pareto.json` the test-split fronts.

## Identical pruning runs once

With `params.dedup_pruning` (default true), tolerances whose per-tile sampling gaps are identical on every tile the never-relevant mask allows give the same pruning ILP, so the same outputs and the same work (`profiling/reduction.py`).
Only the first tolerance of each group runs; the others get a copy of its `result.json` with `duplicate_of` set, and are scored through it.

## Saving stage outputs

Off by default.
`--save` passes `--save <output_root>/<dataset>` to execution, which writes every stage's non-pixel outputs (scores, pruned grids with each solve's record, index maps and offsets, detections, tracks) in the scripts' formats at the cache's paths (`execution/save.py`); frames and canvases are never saved.
Saving shares the machine with the timed run, so such runs are marked `saved` and the aggregator warns about them.

## Config

One YAML per dataset, written by `python -m experiments.tools.profiling.make_configs` (see `profiling/config.py` for every key):
`dataset`, `videoset`, `output_root`; `params` with a list per swept parameter (`sample_rates`, `relevance_thresholds`, `mistrack_tolerances` with `null` for no pruning, `tile_paddings`, `canvas_scales`, `trackers`) plus the fixed `classifier`, `tile_size`, `canvas_format`, and `dedup_pruning`; explicit `artifacts`; `evaluation` (each video's reference tracks); and execution's `system` section, used for every configuration.
Every configuration's execution config is validated by `execution.config.parse_config` when the sweep loads.

## Output

Under `<output_root>/<dataset>/profiling/`:

| Path | What |
| --- | --- |
| `sweep.json` | the sweep document and its hash |
| `runs/<id>/config.yaml` | the configuration's execution config (`<id>` is the scripts' 060 directory name) |
| `runs/<id>/tracks/` | execution's `--out`: `<video stem>.jsonl` per video and `summary.json` |
| `runs/<id>/stdout.log`, `stderr.log` | execution's output |
| `runs/<id>/result.json` | written last: parameters, `runtime_s`, frames, video files, `saved`, `duplicate_of`, process `wall_s` |
| `runs/<id>/evaluation.json` | per-video TrackEval results (configurations that ran) |
| `configurations.json`, `pareto.json`, `pareto.html` | the aggregator's outputs |
