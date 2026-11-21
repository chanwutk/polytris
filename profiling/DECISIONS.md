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
