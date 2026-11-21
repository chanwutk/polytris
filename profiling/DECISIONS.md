# profiling: decisions made and questions for the author

The author asked (2026-11-21, overnight) to keep working without asking, guess where needed, and record
every open question and decision here for audit.  Each entry: what was decided, why, and what to check.

## Decided in review (2026-11-21, with the author)

See `PLAN.md`: issues 1-19 (charging, outputs, frame lifetime, GPUs, HOTA aggregation, savers, counter,
Pareto page, operators refactor, config, messages, TrackEval reuse, tests, Gurobi, memory, storage, procs).

## Made overnight without the author (please audit)

### D1. Engine: fan-out is an explicit stage flag
`Stage(fan_out=True)` makes `process()` return a list (0..N outputs); otherwise exactly one output, as
before.  Why: a message could itself be a list, so guessing from the return type would be fragile.
Check: nothing; execution2 is unchanged (all its stages keep `fan_out=False`).

### D2. Engine: savers are per stage, fed after `process()` returns, with a bounded queue (64)
Each output goes downstream first, then to the stage's saver queue.  A full saver queue blocks the worker
(after its timing), which slows the pipeline rather than growing memory without bound.
Question: is backpressure from slow saving acceptable, or should saving never slow the pipeline (unbounded
queue, memory risk)?  Guess: backpressure is the safer default.

### D3. The per-video counter lives in `profiling/completion.py`, not in the engine
It is profiling-specific (pack freeing frames).  A Manager dict + Manager lock, created by main.

### D4. One classifier and one tile size per sweep
`params.classifier` and `params.tile_size` are single values; the artifacts (weights, never-relevant bitmap,
max-rate tables) belong to one classifier and tile size.  The current grid has one of each.
Question: should one sweep cover several classifiers?  Guess: no; run one sweep per classifier.

### D5. A max-rate table per tracker whenever any tolerance is swept
Every tracker can pair with every tolerance, so the config requires `max_rate_tables.<tracker>` for all trackers.

### D6. output_root and the reference tracks may not contain one another
Saved results go to `<output_root>/<dataset>/execution/<video>/<stage dir>/<param str>/...` (the cache's
layout), so output_root must not be (or contain, or sit inside) the directory the reference tracks are read
from.  The run additionally mounts the real cache read-only.

### D7. TrackEval metrics fixed to p110's default (HOTA)
p110's default `--metrics` is HOTA; the evaluate stage uses the same, plus Count, as p110 stores.
Question: add CLEAR / Identity?  Guess: not needed for the Pareto selection.

### D8. Message modules avoid importing the detector stack
`profiling/messages.py` imports execution2.operators' types only under TYPE_CHECKING, so tests and tools that
only need messages do not load ultralytics/darknet (which need a GPU driver to import).

### D9. p040's per-canvas detections are not saved
Detect returns per-frame detections (p050's output) only, so the saver writes `050_uncompressed_detections`
but not `040_compressed_detections`.  Question: needed for any analysis?  Guess: no; p110 reads only 060.

### D10. The unpruned branch's prune record is empty, and no pruned file is written for it
Its prune StageTiming has start == end and charged 0; its scores are classify's, already saved under 020.

### D11. Decode and evaluate are recorded but charged 0
Decode: the author decides later whether it counts (PLAN 1).  Evaluate: measurement, not system work.

### D12. p110's TrackEval call is now `run_trackeval()`, shared with the evaluate stage
`evaluation/p110_accuracy_compute.py` behaves as before (full suite: 1484 passed; the 1 failure is a
`measurements-results/` export missing on the profiling checkout, unrelated).

### D13. classify charges with a GPU synchronize after each batch
`classify_frames(time_batches=True)` synchronizes after each batch so the batch's GPU time is attributed to
it.  This adds a sync per 16 frames to classify; execution2 does not use it (time_batches defaults to False).

### D14. A sweep refuses to start if `<output_root>/<dataset>` exists
"Do not override old results": rather than merging into or overwriting an earlier sweep, main stops with an
error.  Question: should a crashed sweep be resumable (skip finished videos)?  Guess: not yet; rerun into a
fresh output_root.

### D15. A video's admission slot is freed when all its expected results arrive
`max_videos_in_flight` bounds frames blocks (each video's all-frames block lives until its last pack input).
Slots return at the last result, slightly later than the frames are freed, so the bound is conservative.

### D16. Records are written by main as they arrive (records.jsonl), not by a saver
They are the pipeline's output, small, and written outside every stage's timing.

### D17. The Pareto front follows p135: the union of every ablation condition's front, time minimized
`time` = summed charged seconds over the split's videos (decode excluded, PLAN 1); each configuration also
reports `throughput_fps_with_decode`.  Points carry the conditions whose front they are on ('full',
'no_sampling', 'no_both').  Question: should the test pass use only the full front?  Guess: keep p135's union
so the ablation curves in Figure 8 can still be drawn from it.

### D18. The generated execution2 YAML uses a one-GPU default system
decode 2, classify 1 on GPU 0, prune 1, pack 2, detect 1 on GPU 0, track 2, queue_maxsize 2; artifacts come
from the sweep config (the tracker's max-rate table only for pruned configurations).

### D19. The page loads Vega from jsdelivr (pinned: vega 5.30.0, vega-lite 5.21.0, vega-embed 6.26.0)
It needs internet in the browser that opens it; everything else (data, YAML) is embedded.

### D20. Dataset HOTA is TrackEval's combination, checked against TrackEval itself
The aggregator combines each configuration's per-video HOTA results with HOTA.combine_sequences; a test scores
two videos both ways (per video then combined, vs. TrackEval on both at once) and requires equal HOTA arrays.

### D21. No cached validation HOTA exists, so the equivalence check computes it
`/polyis-cache/<ds>/evaluation/070_accuracy` holds only the test split.  The tool runs p110's `run_trackeval`
on the scripts' cached validation tracks (060) for the same video and configuration and compares HOTA arrays.

### D22. `params.prune_time_limit_s` (default 0.1) exists only so the equivalence check can raise it
The tool runs the small grid with 10 s; (config, video) pairs are compared only when our solve is optimal
(status 2) and p022's logged solve time is below 95 ms (its limit is 100 ms).  Each pruned result saves an
extra `ilp.json` (status, solve_ms, kept tiles), which is not a script output.

### D23. Heavy profiling runs wait for the Figure 8 job
The Figure 8 serial benchmark (tmux `execution2-figure8` on ace) measures throughput; a concurrent profiling run
would compete for CPU and memory bandwidth.  Only light unit tests run alongside it; the equivalence check and
the first full sweep start after it finishes.

## The author's search-space reduction (proposed 2026-11-21, analysis only so far)

Idea (author): for tolerance X at sample rate s, if every possibly-relevant tile's allowed gap is 1, the ILP
keeps every polyomino, so every stricter tolerance gives the same pruned result and need not run.

### A1. Generalized: tolerances with identical gap arrays give identical ILPs
The ILP's inputs are the scores (same), p022's binarization (same T_r), and the per-tile gap
`max(table[:, :, tol] // s, 1)`.  Tiles the never-relevant mask excludes always score 0 and are never in a
polyomino, so only gaps on mask tiles matter.  So two tolerances whose gaps agree on mask tiles give the same
ILP, hence the same result; this needs no monotonicity assumption.  (It does not make the all-1s case a
keep-everything case; see A4.)

### A2. Measured on the real tables (7 datasets x 2 trackers x 5 sample rates x tolerances 0.4/0.6/0.8)
- Merges happen only at sparse sampling: at s=16 all three tolerances are all-1s (identical) on every
  dataset and tracker; at s=8 ByteTrack's 0.6 and 0.8 are all-1s on caldot2-y05 and jnc6.  Nowhere else.
- ILP branches: 210 -> 180 unique (14% fewer).  (An earlier note said 50 all-1s branches need no solve; A4
  shows that is wrong: they still solve, once per group.)
- Per video (paper grid) that removes about 12-15 of 105 pack inputs, with their pack/detect/track/evaluate.

### A3. A pruned branch that keeps everything is not the no-pruning branch
p022 binarizes with `>= floor(T_r*255)`, p030 with `> T_r*255`, so they differ on tiles scoring exactly
63 / 127 / 191 (for T_r 0.25 / 0.5 / 0.75).  Not merged.

### A4. The author's premise does not hold: all-1s gaps do not keep every polyomino
Found by a test (Gurobi status 2, yet tiles dropped).  In `polyis/sample/ilp/gurobi.py` a tile covered in at
most one sampled frame (`len(pos) <= 1`) gets no temporal constraint; only consecutive appearances are linked
(and first/last appearances forced).  With every gap 1, the optimum keeps exactly the polyominoes containing
a tile that appears in >= 2 sampled frames and drops the rest, since the objective minimizes kept cells.
So the solve-skip was removed: every group, all-1s included, runs its ILP once.  The identical-gap merge (A1)
is unaffected.  Test: `test_all_ones_gaps_drop_polyominoes_whose_tiles_appear_only_once` checks this rule
against Gurobi.  Question for the author: is dropping single-appearance polyominoes intended in p022?  (It
is what the scripts do, so the sweep matches them either way.)

### Questions for the author (to decide before implementing)
- Q1. Which tolerance labels a merged branch?  Guess: run it once and record the result under every
  equivalent tolerance (the aggregator expands it), so each configuration still has a row; the Pareto front
  then has identical duplicates, and p135's dedupe keeps one.
- Q2. Charged time for the duplicates: the representative's measured time for all of them?  Guess: yes.

### D24. Search-space reduction implemented (pending the author's answers to Q1/Q2), on by default
`params.reduce_search_space` (default true): prune runs one representative tolerance per group of identical
gaps (per tracker and sample rate, on mask tiles); every group solves once (A4); the aggregator copies
each representative's row to every tolerance of its group (`ran_as` names the one that ran; same HOTA and
time).  Main's expected results and pack's expected inputs count groups.  Set false to run every tolerance
(the equivalence check uses the grid as-is either way).  Commit e9c4161 still skipped all-1s solves; fixed
after A4.

### D25. The scripts' validation-pass wall clock cannot be recovered from the cache
p120/p121 ran only for the test split (080_throughput has no validation rows), and output-directory
modification times are unusable (020 spans 88 h across reruns; 022 was written at once).  So the first sweep's
wall clock is reported on its own, with its per-stage busy times.  Question: is there a log of how long the
scripts' validation pass took?  If so, it is the comparison to make.

### D26. run.json records `reduce_search_space`; the aggregator expands groups only for reduced runs
The first caldot1-y05 sweep ran every tolerance before the flag existed; expanding it by the config's
current flag would look up non-representative tolerances and fail.  A run.json without the flag counts as
unreduced.  Result of that sweep (480 configurations, 2,405 s wall): 56 Pareto configurations (44 on the full
front, 20 no-sampling, 10 no-both), HOTA 0.633-0.971, 142-4,599 fps; committed under
`measurements-results/profiling-sweep/caldot1-y05/`.  At s=16 the three tolerances give identical HOTA, as A1
predicts (identical gaps), with different measured times (noise).

### D27. Measured: the reduction saves 6% of wall clock on caldot1-y05
Same config with `reduce_search_space: true`: 432 of 480 configurations run (the 48 merged are s=16 tolerance
duplicates), 2,251 s wall vs 2,405 s unreduced (6.4% less).  Against the unreduced run, all 120 unpruned
configurations have identical HOTA; 328 of 360 pruned do, and the other 32 differ by up to 0.048 because the
0.1 s ILP limit makes pruned results load-dependent (as in the Figure 8 rerun), not because of the merge.
Pareto: 58 configurations (vs 56).  Results under `measurements-results/profiling-sweep/caldot1-y05-reduced/`.

### D28. Scope of the reduction settled with the author: same-gap merge only (1A)
Intent (author): drop (sample rate, T_r, tolerance) prefixes provably giving the same polyominoes.  Kept
the static same-gap merge.  Not pursued: merging across trackers (measured 180 -> 171 ILP branches on all 7
datasets) and per-video ILP hashing (unmeasured; constraint building lives in gurobi.pyx).

### D29. B3D validation sweeps (reduction on, 480-configuration grid, one run each, ace, 2026-11-21)
| dataset | videos | frames | configurations run | wall clock | Pareto configurations |
| --- | --- | --- | --- | --- | --- |
| jnc0 | 5 | 5,375 | 432 | 2,365 s | 78 |
| jnc2 | 5 | 4,691 | 432 | 2,890 s | 74 |
| jnc6 | 5 | 5,194 | 420 | 2,256 s | 77 |
| jnc7 | 5 | 5,226 | 432 | 1,426 s | 72 |
Wall clock is profiling.main's (`run.json` `wall_s`), excluding the aggregator.  The detector weights are
mounted from the main checkout (`weights/` is not synced to the profiling checkout).  Results under
`measurements-results/profiling-sweep/<dataset>-reduced/`.

## TODO (author, 2026-11-21)
- Keep polyominoes whose tiles appear in only one sampled frame.  The ILP's early exit `len(pos) <= 1` skips
  its own "always choose the first and last appearance" rule for such tiles.  Change it to skip only
  `len(pos) == 0` in all three backends (`polyis/sample/ilp/gurobi.py:80`, `polyis/sample/ilp/c/gurobi.pyx:231`,
  and likely `pulp.py`), with a test that a single-appearance polyomino is kept.  Not yet: it changes every
  cached p022 result and everything downstream.  Afterwards all-1s gaps really do keep everything, so the
  solve-skip (removed in 9691538) could return, binarizing as p022 does.

### D30. Equivalence rechecked on the current code (caldot1-y05 and jnc0, 3 videos each, 2026-11-21)
`profiling.tools.compare_scripts` with its small grid and a 10 s solve limit.
- caldot1-y05 (va00-02): identical totals to the first check (7ae340d).
- jnc0 (va05-07): every mismatch is explained.
  - Classifier scores: s=1 equal on all 3 videos; s=4 differs by at most 1 (out of 255) on 3-7 values per
    video, likely GPU float nondeterminism (classify runs once on every frame, p020 batched only the sampled
    frames).  No downstream result changes: every s=4 output matches.
  - Pruned grids: 23 of 23 compared are equal.  All 24 s=1 pruned pairs (plus 1 at s=4) are excluded because
    the cached p022 solve ran 107-167 ms, past its 0.1 s limit, so it is not the optimum.
  - Canvases, detections, tracks and HOTA differ only on those excluded s=1 pruned branches (48 of 144
    branches; HOTA within 0.030).  Unpruned and s=4 detections agree within 1.2e-4 px.
Reports: `measurements-results/profiling-check/{caldot1-y05-rerun,jnc0}.json`.

### D31. The reduction is off by default (author, 2026-11-21)
To remove a potential confound against the scripts, `params.reduce_search_space` now defaults to false, so a
sweep runs every tolerance as the scripts' validation did.  The all-1s solve-skip was already removed
(9691538); only the identical-gap merge remains, behind the flag.  The aggregator expands groups whenever
run.json says the run merged them, whatever the config says now.  The B3D sweeps in D29 and the
caldot1-y05-reduced sweep ran with the merge on.
