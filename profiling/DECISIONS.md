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
