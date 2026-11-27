# execution: pipeline-parallel test-pass engine

Replaces `execution/`. Decisions below were agreed in review on 2026-11-20 (issue numbers in brackets).

## Interface

```
python -m execution.main --config <yaml> --videos <dir> --out <dir>
```

- Writes one `<out>/<video stem>.jsonl` per video, in p060's format (`save_tracking_results`).
- `main` writes the outputs. Stages never touch `--out`.

## Topology

```
main ─q→ decode ─q→ classify ─q→ [prune] ─q→ pack ─q→ detect ─q→ track ─q→ main
```

- Each stage runs `system.stages.<name>.procs` processes, and they all read from the one queue that feeds the stage.
- Every queue has `maxsize = system.queue_maxsize` [2B].
- One message per video. Stages scale by working on different videos at once.
- Prune is started only when `params.mistrack_tolerance` is set, and must have `procs == 1` until the ILP is redesigned [17].
- Stop markers: each process sends one marker when it finishes. A stage passes markers on after receiving one from every upstream process.

## Large data: shared memory [1A, 16A]

- `frames.py` is the one module that creates, opens and frees named shared-memory blocks. Pixels are written and read through numpy views; only handles (name, shape, dtype) and small metadata travel through queues. Nothing large is pickled.
- Frame block, one per video:
  - Decode takes it.
  - Classify reads it and copies frames to its GPU through a pinned staging buffer it reuses for every batch.
  - Prune never sees it; the handle just passes through.
  - Pack reads it and frees it.
- Canvas block, one per video: pack takes it (and clears it), detect reads it and frees it.
- Detect gets CPU numpy BGR from the block directly, so there is no GPU-to-CPU canvas copy.
- Startup check [15A]:
  - The worst-case number of videos in flight is the queue sizes plus the process counts from decode to detect.
  - Worst-case bytes are that count × the largest video's frame-block plus canvas-block size, using OpenCV metadata.
  - `main` refuses to start if this exceeds free `/dev/shm`.
  - Container `shm_size` is to be raised in `docker-compose.yml`. Restarting the container is the author's call.
- Pools (decided 2026-11-26), so the timed run allocates no shared memory:
  - Before the timing dummy, `main` creates a frames pool and a canvases pool. Each has one slot per video the startup check counts for that block, sized for the largest video, with every page allocated then (`MADV_POPULATE_WRITE`). The pools are exactly the startup check's worst case.
  - Decode and pack take a slot from a pool instead of creating a block. Pack and detect give it back instead of unlinking. Each process maps a slot once and keeps the mapping, so no page faults hit the timed run.
  - Why: on ace (ZFS cache holding most of RAM), fresh blocks and per-batch arrays sent the stages into kernel reclaim and compaction. ams runs (1280x720) took 2-15x their normal time, varying run to run.
  - A canvases block larger than its slot is an error. The bound (one collage per sampled frame) holds at `canvas_scale` >= 1, since a frame's tiles always fit one canvas at their own positions.
  - After a clean run every slot must be free exactly once (`frames.check_returned`). The pools are unlinked whether or not the run succeeds.
  - Operators take a store, either a run prefix (a new block) or a pool (a slot), so `experiments/` code that passes a prefix still gets fresh blocks.

## Decode [4A, 18A]

- OpenCV for now, wrapped in one function so swapping the decoder is local.
  Decided 2026-11-20 after the step-1 smoke test:
  - On one caldot1 video, decord matched OpenCV exactly: 900/900 frames identical, and random access returned the same frames.
  - Uncontended, 1 thread each, decord was faster: 1.6 s against 3.1 s for all frames.
  - The author chose to keep OpenCV unless decoding turns out to be the bottleneck. `experiments/tools/execution/compare_decoders.py` (modes `pixels` and `timings`) re-checks this before any switch.
- Frames needed, matching p020 ([p020:277-281](../scripts/p020_exec_classify.py)): sampled `idx % s == 0`, plus the last frame when it isn't already sampled, plus for each sampled frame `idx-1`, or `idx+1` for frame 0 (diff channel).
- p020 asserts that frames read == `CAP_PROP_FRAME_COUNT`; execution keeps that assert.
- A video with fewer than 2 frames is rejected at startup (p020 would raise IndexError).
- With OpenCV, frames are read in order and only the needed ones are kept, as p020 does.

## Stage contract [6A]

- Each stage module defines `setup(config, worker) -> state` and `process(state, msg) -> msg`.
- `pipeline.py` owns the single worker loop: queue reads, stop-marker counting, error reporting, and freeing shared memory on failure.
- `process` functions are unit-testable without processes, and can later be driven by a loader/saver harness for per-operator timing (not built now).

## Failure handling [3A]

- Workers report `(stage, video, traceback)` on an error queue.
- `main` also watches for processes that die without reporting (out-of-memory kill, segfault).
- On any failure, `main` stops all workers, frees every block, and exits non-zero.

## Timing (main only) [8, 9A, 10B]

- The pipeline records no timing.
- `main` copies the last video to a temp dir under a new name and feeds it first as the dummy. The pipeline doesn't know it is a dummy.
- `main` records each result's arrival time. Runtime = last arrival − first arrival (whichever video arrives first).
- The dummy's result is not written, and the temp copy is deleted at exit.
- Per-operator timing (each operator run alone with a loader and a saver) is future work.

## Config (`--config`) [7B]

The YAML has three sections:
- `params`: Table 1 parameters (s, T_r, M̄, P, A) plus tile_size, canvas_scale, and canvas_format (`jpeg` | `raw`).
- `artifacts`: explicit paths, all checked before any process starts.
  - classifier name, weights and compilation benchmark
  - never-relevant bitmap
  - max-rate table (only when M̄ is set)
  - detector type and its fields
- `system`: `queue_maxsize`, plus `stages.<name>.procs` and `gpu`.

Loaded into frozen dataclasses. Unknown keys, missing keys and wrong types are errors. `mistrack_tolerance` must be in p022's threshold list.

## Code reuse [5A]

Move these pure functions from the scripts into `tetris/` with no logic change. Both the scripts and `execution` import them.
- p020: `load_model`, `classify_batch`
- p022: the ILP core of `process_video`, split from its file I/O
- p030: CPU canvas drawing helpers (so canvases match p030 by construction)
- p050: `unpack_detections`
- p060: the per-video tracking loop

## Build order

1. ~~Decoder comparison [14A, 18A]~~: done as a smoke test; OpenCV kept (see Decode).
2. Before/after tests [11A], then the move to `tetris/` [5A].
3. ~~`config.py`, `frames.py`, `pipeline.py` with fake-stage tests [12A]~~: done (`tests/execution/`).
   - numpy views over shared memory hold no buffer export, so `SharedMemory.close()` would unmap memory under a surviving view. `frames.py` detects escaped views by the mmap's reference count and refuses to close.
   - The video edge cases below (empty dir, non-video file, unreadable or 1-frame video, zero collages, duplicate stems) belong to steps 4–5, where main and the stages exist.

   Originally planned:
   - process counts 1×1, 1×N, N×1, N×M
   - an exception and a SIGKILL in each stage, with no leftover `/dev/shm` blocks
   - backpressure
   - empty dir, non-video file, unreadable video, 1-frame video, zero collages, duplicate stems
4. ~~Stages: decode, classify, prune, pack, detect, track~~: done (`execution/stages.py`, `tests/execution/test_stages.py`).
   - Decisions: `params.canvas_format` chooses the detector's input: `jpeg` passes each canvas through `cv2.imencode('.jpg')`/`cv2.imdecode`, exactly the pixels p040 reads back from p030's files; `raw` passes the drawn canvas (first chosen as [20B], made a mode on 2026-11-20); [21A] `tetris.models.detector.build_detector` is shared with `get_detector`; p020's model loading moved to `tetris.classify.load_classifier`.
   - Checked against the scripts themselves (loaders redirected to temp files): prune equals p022's `process_video` exactly; pack's index maps and offset lookups equal p030's `compress` exactly, and every drawn tile equals its source tile; track equals p060's `track` file for file; classify equals p020's batching with a fake model.
   - With `canvas_format: jpeg`, pack's canvases equal p030's decoded JPEGs pixel for pixel.
   - Known differences from the scripts: with `canvas_format: raw`, canvases are not JPEG-compressed; within a frame, boxes are merged in collage order (p050 follows p040's `os.listdir` order); classify picks its model optimization from the first video it receives.
   - A padded polyomino larger than the canvas (e.g. `canvas_scale` < 1) aborts inside the C `pack()`, as it would in p030; the engine reports it as a worker death.
5. ~~`main.py`: dummy-video timing and writing outputs~~: done (`python -m execution.main`, `tests/execution/test_main.py`).
   - Startup checks before any worker starts: video files only (other files ignored), no empty dir, no two videos with the same stem, every video openable with >= 2 frames, worst-case `/dev/shm` use within free space.
   - Results are kept in memory and written after the last arrival, so writing cannot stall the timed run. Prints `{"videos", "runtime_s", "frames"}` as one JSON line.
6. ~~Stage-by-stage diff against p020→p060 on 2–3 test videos of one dataset [13A]~~: done (`experiments/tools/execution/compare_scripts.py`, reports in `experiments/measurements-results/execution-compare/`).
   - caldot1-y05 te00–te02, real classifier and detector, two configs: s=4 without pruning, and s=8 with mistrack tolerance 0.8 (`experiments/configs/execution/`), both `canvas_format: jpeg`, compared with the cached April 2026 script outputs (read only).
   - Each stage fed the script's previous output: classify, prune, and pack (index maps, offset lookups, JPEG canvases) are identical to p020, p022, and p030; detect has the same box count in every frame, with coordinates within 1.2e-4 px of p050 (batching order); track fed p050's detections is identical to p060.
   - Bit-identity (2026-11-20): our detector on our JPEG canvases, batched exactly as p040 batched its files and merged in p050's order (both read from p040's `detections.jsonl`), reproduces p040's raw detections, p050's per-frame detections, and p060's `tracking.jsonl` byte for byte on all six cases. The only differences in normal runs come from that order, which is p040's `os.listdir` order of files execution never writes.
   - End to end through `main` (multi-process): every output equals p060's tracks up to a consistent renaming of track ids, with every box matched within 9.2e-5 px. The renaming comes from box order within a frame (collage order here, p040's `os.listdir` order there), which sets SORT's id numbering.
   - The first `/dev/shm` estimate refused `queue_maxsize: 2` on ace's then 8 GiB (it charged every in-flight video for both blocks and assumed one canvas per stored frame).
     It now counts the two block lifetimes separately (frames: decode through pack; canvases: pack through detect), with exact block sizes and one canvas per sampled frame: 5.0 GiB instead of 10.5 GiB for the s=4 config over all 60 caldot1-y05 test videos.
   - ace's `polyis` container was recreated with `shm_size: 64gb` on 2026-11-20, from `/work/cwkt/projects/tetris` (the old compose directory `/work/cwkt/projects/polyis` is empty); `duckdb` was pip-installed again because the image predates it.

## Benchmark (2026-11-20)

`experiments/tools/execution/benchmark.py`, results in `experiments/measurements-results/execution-benchmark/`. All 60 caldot1-y05 test videos (54,000 frames) on ace GPUs 2-3, in throwaway containers whose `/polyis-cache` is a scratch directory and whose real cache is mounted read-only, so no old result can be overwritten. Videos were read once beforehand. ace's load average was 12-17 (other users) during the runs.

| config | p020-p060 (sum of stage wall clocks) | execution, 2 GPUs: process / first-to-last result | execution, 1 GPU: process / first-to-last result |
|---|---|---|---|
| s=4, no pruning | 547.7 s | 96.7 s / 66.2 s | 93.3 s / 59.8 s |
| s=8, tolerance 0.8 | 430.5 s | 74.1 s / 35.9 s | 72.3 s / 30.9 s |

- The scripts' stage times include each stage's process start and model load; execution's "process" includes them too (about 30-40 s), and its first-to-last time excludes them (dummy-video timing).
- p030 alone took 274 s and 168 s: drawing and writing JPEGs serially, video by video.
- One GPU was slightly faster than two here, so this workload is not limited by the classifier and detector sharing a GPU.
- Outputs vs the scripts' fresh outputs from the same run: 119 of 120 equal up to track-id renaming. The exception (te06, s=4) is the canvas-order effect: its detector boxes differ by up to 0.057 px in collage order, and in p040's order te06 is bit-identical (raw detections, per-frame detections, tracks).
- Single runs; repeatability and the best process counts per stage are not measured yet.

## Paper operating points (2026-11-20)

The 14 operating points the paper reports (per dataset, the fastest configuration within 5% and within 10% HOTA loss; R1.O3's selection) as execution configs in `experiments/configs/execution/paper/` (`experiments/tools/execution/make_paper_configs.py`), run with `benchmark.py --skip-scripts` on all test videos on one ace GPU (classify and detect sharing it, JPEG canvases), outputs checked against the cached script tracks. Table: `experiments/measurements/execution_paper_throughput.py` -> `experiments/measurements-results/execution-paper/summary.json`.

- execution's first-to-last throughput beats the matched-boundary paper number (`one_decode`, logged ops plus one decode) on 13 of 14 points (1.31x-2.54x) and the clean staged-scripts wall clock without startup on all 14 (1.47x-10.8x). The exception is CalDoT2 at 10% (s=1, pruning, ByteTrack): 222 fps vs 263 fps.
- Against the published p121 number, which does not count Tetris's decode, it is faster on 10 of 14 (1.05x-1.52x) and slower on Amsterdam (0.57x, 0.84x), CalDoT1 10% (0.95x), and CalDoT2 10% (0.69x).
- Outputs equal the cached script tracks up to track-id renaming on 309 of 310 videos. The exception, CalDoT2 10% te32, comes from the ILP's 0.1 s time limit: alone, its solve takes about 69 ms and reaches the optimum the cache holds (2,807 kept tiles); with 40 CPU-busy processes alongside it stops at the limit (status 9) with 2,808 tiles, which changes the canvases and everything after them. In isolation every te32 stage, prune included, matches the scripts, and in p040's order its output is bit-identical. p022 has the same limit, so the paper's pruned results can also depend on machine load.
- B3D test sets are 5 videos (about 5,000 frames), so their first-to-last times are 5-16 s and the process times are dominated by about 35-40 s of startup.
- Single runs on a shared host (load average about 15), untuned process counts.
- These 14 runs used the classify code at 5b40884. At s=1 that classify took 3.7 s per CalDoT2 video, which made CalDoT2 10% the slow point (243 s for 60 videos). The later classify commits (9b291e8, 2345899: one GPU transfer per batch, scores copied once per video) take 0.75 s per video with identical scores. With them, CalDoT2 10% runs at 529 fps on one GPU and 596 fps with detect on a second GPU (`experiments/measurements-results/execution-paper/caldot2-y05-0.10-1gpu-current`, `-2gpu`), against the paper's 263 fps matched and 321.5 fps published. The other 13 points have not been rerun with the current classify.

### Rerun with the current code (2026-11-21)

All 14 points again on one ace GPU with the code at f1c7581 plus the classify commits (`experiments/measurements-results/execution-paper-current`; `python experiments/measurements/execution_paper_throughput.py --runs experiments/measurements-results/execution-paper-current --baseline experiments/measurements-results/execution-paper`).

- execution is now faster than the published p121 throughput at all 14 points (1.07x-1.72x), than the matched-boundary number (`one_decode`) at all 14 (1.41x-3.25x), and than the clean staged-scripts wall clock at all 14 (2.0x-13.9x).
- Against the first run it is 1.00x-2.48x faster; the largest gains are where classify handles the most frames (Amsterdam 5%: 603 -> 1495 fps; CalDoT2 10%: 222 -> 527 fps).
- CalDoT2 5% (s=1, no pruning) stays at 186 fps: it is detector-bound (about 200 canvases per video).
- Tracks equal the cached script tracks up to track-id renaming on 309 of 310 videos (te32 in CalDoT2 10%, the ILP time limit).


## Figure 8 rerun (2026-11-21)

Every point of Figure 8's full-Tetris curve (183 configurations, 7 datasets), run one at a time on one ace GPU
(GPU 2, 128 GB /dev/shm, JPEG canvases) by `experiments/tools/execution/run_figure8.sh` in tmux; results in
`experiments/measurements-results/execution-figure8/` (`experiments/measurements/execution_figure8_throughput.py`).

- execution's first-to-last throughput beats the throughput Figure 8 reports on 162 of 183 points; per-dataset
  medians: Amsterdam 1.36x, CalDoT1 1.25x, CalDoT2 1.47x, B3D1 1.52x, B3D2 1.24x, B3D3 1.72x, B3D4 1.71x.
  Figure 8's numbers use the published p121 boundary (Tetris decode not counted); execution's include decode.
- Outputs equal the cached script tracks up to track-id renaming on 165 of 183 points.  Of the 18 with a
  mismatching video, 14 prune (mostly s=1, the largest ILPs: the 0.1 s time limit, as with CalDoT2 te32) and 4
  do not (consistent with canvas-order detector noise, as with caldot1-y05 te06); not examined one by one.
- Four pruning configurations first failed on Gurobi's 2-session web license ("Too many sessions") and passed on
  the driver's retry pass.
