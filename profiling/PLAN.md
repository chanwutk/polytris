# profiling: pipeline-parallel validation sweep

Replaces the validation pass of `experiments/scripts/run.sh` (p020 -> p060 over the full parameter grid on the valid
videoset, then p110 accuracy and p120/p121 throughput, then Pareto selection) with one pipeline that
runs every configuration at once, reusing each intermediate result for every configuration that shares it.
Decisions were agreed in review on 2026-11-21 (issue numbers in brackets).

## Shape

```
main ─→ decode ─→ classify ─→ prune ─→ pack ─→ detect ─→ track ─→ evaluate ─→ main
        (1/video) (1→5 by s) (→ T_r × (tol×tracker | none)) (→ P × scale) (1:1) (→ tracker if unpruned) (1:1)
```

- Same stages as `execution`, but each one runs its operator for **every value of its own parameters**,
  given the parameters chosen upstream, and emits one message per resulting configuration branch.
  With the current grid, per video: classify 5, prune 105, pack 420, detect 420, track 480, evaluate 480.
- Each message carries its data, a frozen `Branch` (the parameters chosen so far, ordered; builds the
  cache directory name with `build_param_str`), and `StageTiming(stage, start, end)` records [10A].
- The pipeline's output is one record per (configuration, video): the configuration, the video's raw
  TrackEval counts, and its stage timings [5A]. An aggregator turns these into results per configuration.

## Stages

- decode: once per video, every frame, into one shared block (s = 1 needs all frames).
- classify: once per video on every frame [5]; emits one message per s with that s's slice of the
  scores (sampled frames plus the last, as p020). Timing: each batch's time is split evenly across its
  frames, and an s branch is charged for the frames it samples [1A]. Decode's time is recorded on every
  branch but kept separate (whether it counts toward runtime is decided later) [1].
- prune: per (T_r, tolerance, tracker) with the tracker's max-rate table; the "no pruning" branch passes
  scores through. Exactly one prune worker (Gurobi web license: 2 concurrent sessions, shared) [16B].
- pack: per (P, canvas scale), with p030's `> T_r*255` when unpruned; canvases go into one shared block
  per branch. Frees the video's frames block when the last of that video's pack inputs is done, through a
  shared map video -> processed count guarded by a lock that every pack worker shares [3'A].
- detect: 1:1, canvases freed after use; workers spread over a GPU list [4A].
- track: per tracker on unpruned branches (pruned branches already fixed their tracker at prune).
- evaluate: TrackEval on one video, p110's settings and metrics, reference = that video's
  `002_naive/tracking.jsonl` (validation split; no ground-truth mapping needed) [11A, 9].

## Timing and saving

- Each stage records when `process()` starts and ends; no per-operation breakdown [1].
- Every non-pixel result (scores, pruned grids, index maps and offsets, detections, tracks) is saved in
  the scripts' formats at cache-shaped paths under an output root (never the real cache by default) [2A, 18B].
- Saving happens in one saver process per stage, fed after the worker's timing stops, so it is not counted
  and does not compete for the worker's GIL [6A].

## Engine changes (shared with execution)

- `process()` may return a list of outputs (0..N).
- Per-stage GPU lists, assigned to workers in turn; each worker still sees one GPU [4A].
- Main admits at most `system.max_videos_in_flight` videos; a startup check refuses to start if
  videos-in-flight x largest video's frames block + in-flight canvases blocks exceed free /dev/shm [3'A, 17A].

## Config [9A]

One YAML per dataset: `params` with a list per parameter (s, T_r, tolerance incl. null, P, canvas scale,
tracker), explicit `artifacts` (max-rate table per tracker), `system` (per-stage procs and gpus,
queue_maxsize, max_videos_in_flight, savers), `evaluation` (where each video's reference tracks are),
and an output root. Strict checks shared with `execution.config`.

Default system split on ace [19A]: classify 1 worker on GPU 0; detect 7 workers on GPUs 1-7; decode 2,
prune 1, pack 4, track 4, evaluate 4; one saver per stage.

## Aggregator [5A, 7A]

- Combines each configuration's per-video TrackEval results with the metrics' `combine_sequences`
  (dataset HOTA as p110 computes it) and sums stage times (decode reported separately).
- Writes the Pareto configurations (HOTA vs throughput) as JSON, and a self-contained interactive HTML page
  (Vega-Lite): clicking a point shows that configuration's `execution` YAML, generated in Python and embedded.

## Code reuse [8A]

`execution/operators.py`: the stage computations as plain functions with explicit parameters;
`execution/stages.py` becomes thin wrappers over them, and `profiling` calls them directly.

## Tests

- [12A] The refactor leaves execution unchanged: `tests/execution`, the golden tests, and
  `compare_scripts` on caldot1-y05 te00-te02 (both configs) still identical / bit-identical.
- [13A] Engine: 0/1/N outputs, per-worker GPUs, saver timing and crashes, frames freed exactly at a video's
  last pack input with interleaved videos and 1 or N pack workers, admission cap, no leaks after crash/SIGKILL.
- [14A] Equivalence with the scripts' cached validation outputs on a small grid over 2-3 valid videos:
  configurations without pruning must match exactly at every stage (detector output compared in p040's
  order); with pruning, the solve limit is raised for the test, (config, video) pairs where either our solve
  or the cached p022 solve stopped at its limit are dropped, and remaining pairs must have equal ILP
  objectives (differing selections reported). Classify slicing may not reproduce p020's per-s batches bit
  for bit; the tool reports it, and exactness vs the 5x saving is decided then.
- [15A] Aggregator HOTA equals p110's `DATASET.json` for cached validation configurations; unit tests for
  classify charging, decode recorded uncharged, Pareto selection; every embedded YAML passes
  `execution.config.load_config`, and the page's data matches the Pareto JSON.

## Where it runs

Local worktree `.claude/worktrees/profiling` (branch `profiling`). On ace a separate checkout,
`/work/cwkt/projects/tetris-profiling` (`./sync --config` with that path; Cython built once there), tested in
throwaway containers, so the Figure 8 job running from `/work/cwkt/projects/tetris` is never touched.

## Build order

1. ~~`execution/operators.py` refactor; verify [12A] on the separate ace checkout~~: done (f588aed).
   On the ace checkout, 1435 of 1437 tests pass (1 skipped; the 1 failure needs a `measurements-results/` export the checkout does not have), and `compare_scripts` on caldot1-y05 te00-te02 is unchanged for both configs: classify, prune, pack identical; bit-identical in p040's order; end-to-end tracks equal up to id renaming.
   The ace checkout needs, besides `./sync --config <copy with path /work/cwkt/projects/tetris-profiling>`: the compiled `.so` files (copied from the main checkout by exact list), `modules/` (darknet, TrackEval, lap), and `baselines/` (a submodule a fresh worktree does not check out).
2. ~~Engine: list outputs, per-worker GPUs, saver processes, shared per-video counter; tests [13A]~~: done.
   `Stage(fan_out=True, gpu=(...), save=fn, savers=n)` in `execution/pipeline.py`; `profiling/completion.py` (`VideoCompletion`); 155 tests pass (`tests/execution`, `tests/profiling`, golden).
3. ~~Sweep config, `Branch`, messages~~: done (`profiling/config.py`, `profiling/messages.py`; tests in `tests/profiling`).
4. ~~Profiling stages (decode, classify-once with charging, prune, pack with freeing, detect, track, evaluate)~~: done (`profiling/stages.py`, `profiling/save.py`; p110's `run_trackeval`).
5. ~~`profiling/main.py`: admission control, memory check, savers writing the scripts' formats~~: done (`python -m profiling.main --config ...`; tests in `tests/profiling/test_main.py`).
6. ~~Aggregator: dataset HOTA, runtime sums, Pareto JSON, interactive page; tests [15A]~~: done (`python -m profiling.aggregate --config ...`).
7. Equivalence tool [14A] on validation videos; then a first full-dataset run timed against the scripts.
   - Equivalence (2026-11-21, `measurements-results/profiling-check/caldot1-y05.json`): caldot1-y05 va00-va02, grid
     s {1,4} x T_r {0.25,0.5} x (none, {0.4,0.8} x 2 trackers) x P {none,tr}, solve limit 10 s.
     Classify scores equal p020's for every sample rate (classifying every frame once and slicing is exact).
     Pruned grids: 41 of 41 compared pairs identical; 7 pairs dropped because p022's cached solve ran past its
     0.1 s limit (103-114 ms).  Without pruning every stage matches (detections: same box counts on every frame,
     coordinates within 6.1e-5 px; tracks and per-video HOTA equal).  All 6 downstream mismatches (canvases,
     detections, tracks, HOTA) descend from those 7 dropped pairs; on them the optimal solve's HOTA is slightly
     lower (0.949-0.963) than the time-limited cached one's (0.956-0.964).
