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
