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
| R1.O2 LEAP native queries | `r1o2_leap_native_queries.py` | none | In progress (selection F1, aggregation MAPE, top-10 precision; merged vehicle class only) |
| R1.O4 packing precision/recall | `r1o4_packing_detection_accuracy.py` | `r1o4_packing_detection_accuracy.py` (JPEG control, ran on bombe GPU 4) | Done for CalDoT/Amsterdam; B3D provisional until R1.O1 |
| R1.O5 tolerance-to-gap | `r1o5_gap_tolerance.py` | none | Done; see the tolerance-label finding below |
| R7.O1 matched HOTA speedup | `r7o1_hota_matched_speedup.py` | none | Done; reproduces all p203/p204 macros; rerun after R1.O1 |
| R9.O1 classifier cost | `r9o1_classifier_cost.py` | none | Done |
| R9.O2 preparation cost | `r9o2_preparation_cost.py` | `r9o2_preparation_cost.py` (dry-run plan: re-time p011, p016, OTIF/LEAP preparation) | Recorded costs and break-even done; OTIF/LEAP preparation not recorded |
| R9.O3 ILP | `r9o3_ilp_timings.py`, `r9o3_ilp_instance_sizes.py` | none | Done |

Key numbers (B3D uses automatic ground truth until R1.O1):
- R1.O2 counting, at LEAP's own accuracy: Tetris is 3.5x (CalDoT1), 1.8x (CalDoT2), and 5.1x (Amsterdam) faster than LEAP, and 1.2-2.7x faster than OTIF. At 5%, OTIF is faster than Tetris on CalDoT1/2 (0.7x, 0.3x) and slower on Amsterdam (11.6x); LEAP never meets 5% or 10%.
- R7.O1 HOTA: Tetris 4.7-68.8x over the reference at 5% and 10.4-91.2x at 10%; over OTIF 2.9-16.5x on the four datasets where OTIF meets 5%, and 3.4-25.9x at 10%. Ablation means 6.7x, 11.1x, 26.7x (5%).
- R9.O1: classification is 20-45x cheaper than detection per frame and 12-33% (median 17.6%) of extraction at the operating points; detection 55-78%, packing 5.5-9.4%.
- R9.O2: break-even video duration against the reference at p121's timing boundary is 19 h (Amsterdam), 25 h (CalDoT1), 37 h (CalDoT2), and 5-8 h (B3D); at a process (wall-clock) boundary 56 h, 58 h, 85 h, and 8-11 h.
- R9.O3: ILP construction and solve medians 6.8 ms and 10.1 ms per clip; 7.9-20.7% of runs reach the 0.1 s budget (Gurobi status is not logged); variables per clip up to 1,835-9,455 per dataset.

## Fix log

Problems found while porting or running measurements, and what was changed. Numbers quoted from the feedback repo before a fix should be replaced by the measurement results.

| Date | Measurement | Problem | Fix | Effect |
| --- | --- | --- | --- | --- |
| 2026-10-25 | `r1o2_count_query` | OTIF's Amsterdam rule needs tracks of at least 10 frames of OTIF's 30 FPS video; the feedback script applied 10 frames to our 15 FPS clips, twice as strict. | Threshold converted per video from the two videos' frame rates (5 clip frames for Amsterdam). | Amsterdam at the 5% bound: Tetris 19.5x to 22.0x over the reference, 10.3x to 11.6x over OTIF. 10% numbers unchanged. CalDoT unchanged. |
| 2026-10-25 | `r1o2_count_query` | OTIF's CalDoT test videos are not all 15 FPS (19 of 60 CalDoT1 and 2 of 30 CalDoT2 videos are 30 FPS), so one threshold per dataset is wrong in general. | Duration threshold is per video; `configs.json` records the frame-rate groups. | None on CalDoT, whose polygon rule has no duration. |
| 2026-10-26 | LEAP baseline (`evaluation/p140_otif_transform.py`) | LEAP's exporter (`export-leap-outputs.py`) writes boxes for intervals starting before the clip at negative list indices, i.e. at the clip's end; p140's gap interpolation then filled every frame between, inflating LEAP's cached box-frames by 36-86%. | p140 drops the wrapped tail, clips to the clip length, and writes `unwrap.json` per video. Cache regenerated with `p140/p141/p142 --test --systems leap`; the old cache is in `/polyis-cache/SOTA/leap-wrapped-backup-20261026`. | LEAP HOTA rises slightly everywhere (CalDoT1 0.070 to 0.077, CalDoT2 0.093 to 0.107, Amsterdam 0.049 to 0.056, B3D 0.012-0.036 to 0.015-0.043). The paper's LEAP points need regenerating (p200-p205). |
| 2026-10-26 | LEAP baseline (`p140`) | The CalDoT1/2 LEAP runtime CSVs hold two runs of every video and their `total` row sums both, halving LEAP's throughput. | Runtime is each video's mean run, summed. | LEAP CalDoT1 373.5 to 747.0 fps, CalDoT2 506.4 to 1012.8 fps. |
| 2026-10-26 | `measurements/common.py` | The ground truth of the detector variants (caldot1-y05, ams-y05, ...) was read from their own stale `003_groundtruth` copies, not the root dataset's tracks that p110/p141 score against. | `output_path` maps to the root dataset. | Only R1.O2's reference-only ground-truth row changed. |
| 2026-10-26 | `measurements/common.py` | Speedup ratios divided throughputs already rounded to 0.1 fps. | Ratios use unrounded throughput. | B3D4 Tetris over OTIF at 10%: 25.8x to 25.9x (matches p203). |
| 2026-10-26 | `r9o1_classifier_cost` | The feedback script read p020's per-batch (16-frame) log lines as frames and used its own I/O exclusions instead of p121's. | Per-frame cost from batches; p121's exclusions. | Classifier 11-47 ms to 0.7-2.6 ms per frame; share 6.7% to 17.6% median; packing 26.8% to 6.7%. Replace the response's feedback numbers. |
| 2026-10-26 | `r9o3_ilp_instance_sizes` | The feedback script counted the pruned output (kept polyominoes), not the ILP's input. | Variables counted from cached 020 scores with p022's grouping; checked against all 21,770 saved runs. | Variables are 1.45-5.41x the old counts. |

## Findings that need an author decision

Found while measuring; not changed, because each touches the manuscript or the pipeline.

1. **Tolerance labels (Table 2, Fig. 7, R1.O5).** The values listed as the mistrack-rate tolerance M (none, 0.4, 0.6, 0.8) are the code's accuracy threshold t; p016 keeps a gap when accuracy >= t, i.e. MissRate <= 1 - t. The validated tolerances are M = 0.2, 0.4, 0.6, and Fig. 7's "M = 0.8" panel is the strictest setting (MissRate <= 0.2). See `measurements-results/r1o5_gap_tolerance/`.
2. **"Up to 17.4x higher throughput than prior systems" (abstract, introduction, Section 7, conclusion).** p203's accuracy-matched rule compares Tetris with OTIF points that miss the 5% bound on 6 of 7 datasets (all 7 at 10%). Under one shared bound, Tetris is 2.9-16.5x faster than OTIF on the 4 datasets where OTIF qualifies. See `r7o1_hota_matched_speedup`.
3. **Timing boundary (R1.O3).** p121's throughput counts canvas reads (disk I/O) for Tetris and video decoding for the reference, but not p030's per-configuration video decode; the response says extraction excludes disk I/O. At a wall-clock process boundary the speedups are much smaller (for example Amsterdam 5.9x instead of 68.8x). Being verified independently; see `r9o2_preparation_cost`, `r9o1_classifier_cost`.
4. **R9.O1 "same videos and hardware".** The classifier and detector timings come from logs six weeks apart under concurrent multi-GPU load; no matched re-timing exists, so quote the per-frame ratio as a range (about 20-45x).
5. **CalDoT2 validation cache.** A 2026-05-29 single-configuration rerun deleted the 022-060 outputs of 89 of 90 configurations for va00-va09 (890 missing pruning runs), so CalDoT2's Pareto selection cannot be re-derived from the cache.
6. **p022 binarization.** The April 2026 cache was produced by an uncommitted p022 that keeps tiles with score >= floor(r x 255); the current code keeps score > r x 255 (median -0.1% variables, up to -5.4%). p022's `--time-limit` help says 0.5 s; the effective budget is 0.1 s.
7. **Section 6.2 TotalA.** The text says associations are counted at the native rate; p016 counts them at the frames each gap samples.
8. **02_background.tex lines 168-169** pair the `\mistrackStdCaldottwo...` macro with the name CalDoT1.
9. **LEAP native queries.** No output (ours, OTIF's, LEAP's export, or the automatic ground truth) carries car/truck/bus classes, so LEAP's per-class workload cannot be run; only a merged vehicle class. LEAP's native results and pseudo-labels for our runs are not on ace (possibly on host freddie).
10. **p012** removes `always_relevant/` for each video while 20 workers write to it (pipeline bug, not fixed).
