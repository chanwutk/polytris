# Revision measurements (SIGMOD '27 #965, major revision)

Data collected for the revision lives in three top-level directories.

| Directory | Contents | Runs |
| --- | --- | --- |
| `measurements-gpu/` | One Python script per long-running GPU job the revision needs (retraining for timing, rerunning detection on packed canvases, ...). It writes into the cache, like the pipeline stages. | On ace, in a tmux session |
| `measurements/` | One Python script per measurement. It reads existing data (cache, tradeoff tables, outputs of `measurements-gpu/`) and aggregates it. | On ace, minutes at most |
| `measurements-results/` | Outputs of `measurements/`, committed to git. | Pulled back from ace |

## Conventions

- `measurements/X.py` writes only into `measurements-results/X/`, a directory holding one or more JSON files. `measurements.common.results_dir(__file__)` returns that directory, so the name always matches the script.
- `X` starts with the review item it answers, as in the response letter and the checklists: `r1o2_count_query.py`, `r9o2_preparation_cost.py`. Shared helpers live in `measurements/common.py`.
- A `measurements/X.py` that depends on a `measurements-gpu/Y.py` says so in its docstring and in the table below, and fails with a clear error when Y's outputs are missing. For example, R1.O4's accuracy script reads the canvas detections that a GPU script reruns, and R9.O2's preparation-cost script reads the training times that a GPU script records.
- Every `measurements/X.py` has tests in `tests/test_measurements_X.py` covering its rules and edge cases.
- Scripts follow `AGENTS.md`: comments on every operation, `parse_args` near the top without comments.

## Running

```sh
# Mac: push code (measurements/ and measurements-gpu/ are in configs/sync.yaml).
./sync
# ace: run tests, then a measurement, inside the polyis container (/polyis is this checkout).
ssh ace 'docker exec polyis bash -c "cd /polyis && pytest tests/test_measurements_r1o2_count_query.py -q"'
ssh ace 'docker exec polyis bash -c "cd /polyis && python measurements/r1o2_count_query.py"'
# Mac: pull results back. measurements-results/ is not in sync.yaml, so ./sync never overwrites newer results on ace.
rsync -az ace:/work/cwkt/projects/tetris/measurements-results/ measurements-results/
```

Run long `measurements-gpu/` jobs in tmux on ace, since a dropped SSH session kills a foreground job.

## Status

| Item | `measurements/` | `measurements-gpu/` | Status |
| --- | --- | --- | --- |
| R1.O2 counting query | `r1o2_count_query.py` | none | Done: bounds 5%, 10%, and LEAP's own accuracy per dataset (`leap_fit`) |
| R1.O2 LEAP native queries | to write | none | Not started |
| R1.O4 packing precision/recall | to write | to decide whether canvases need rerunning | Blocked on R1.O1 for B3D |
| R1.O5 tolerance-to-gap | to write | none | From cached mistrack profiles |
| R7.O1 matched HOTA speedup | port `matched_accuracy_speedup.py --metric hota` | none | Recompute after the B3D re-scoring |
| R9.O1 classifier cost | port `classifier_vs_detector.py`, `operator_breakdown.py` | none | Feedback-repo numbers exist |
| R9.O2 preparation cost | to write | re-time p011 (training-split reference detection), p016 (mistrack profiling); OTIF/LEAP preparation | Validation sweep and classifier training times are in the cache |
| R9.O3 ILP | port `ilp_timings.py`; rewrite `ilp_instance_sizes.py` | none | Instance sizes must count input polyominoes, not pruned output |

The feedback-repo originals are in `tetris-paper-feedback/measurements/`; its README records how each number was derived and corrected.

## Fix log

Problems found while porting or running measurements, and what was changed. Numbers quoted from the feedback repo before a fix should be replaced by the measurement results.

| Date | Measurement | Problem | Fix | Effect |
| --- | --- | --- | --- | --- |
| 2026-10-25 | `r1o2_count_query` | OTIF's Amsterdam rule needs tracks of at least 10 frames of OTIF's 30 FPS video; the feedback script applied 10 frames to our 15 FPS clips, twice as strict. | Threshold converted per video from the two videos' frame rates (5 clip frames for Amsterdam). | Amsterdam at the 5% bound: Tetris 19.5x to 22.0x over the reference, 10.3x to 11.6x over OTIF. 10% numbers unchanged. CalDoT unchanged. |
| 2026-10-25 | `r1o2_count_query` | OTIF's CalDoT test videos are not all 15 FPS (19 of 60 CalDoT1 and 2 of 30 CalDoT2 videos are 30 FPS), so one threshold per dataset is wrong in general. | Duration threshold is per video; `configs.json` records the frame-rate groups. | None on CalDoT, whose polygon rule has no duration. |
