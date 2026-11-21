# execution2: pipeline-parallel test-pass engine

Replaces `execution/`. Decisions below were agreed in review on 2026-11-20 (issue numbers in brackets).

## Interface

```
python execution2/main.py --config <yaml> --videos <dir> --out <dir>
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
  - Decode creates it.
  - Classify reads it and copies frames to its GPU.
  - Prune never sees it; the handle just passes through.
  - Pack reads it and frees it.
- Canvas block, one per video: pack creates it, detect reads it and frees it.
- Detect gets CPU numpy BGR from the block directly, so there is no GPU-to-CPU canvas copy.
- Blocks are keyed by the video's full path.
- Startup check [15A]:
  - The worst-case number of videos in flight is the queue sizes plus the process counts from decode to detect.
  - Worst-case bytes are that count × the largest video's frame-block plus canvas-block size, using OpenCV metadata.
  - `main` refuses to start if this exceeds free `/dev/shm`.
  - Container `shm_size` is to be raised in `docker-compose.yml`. Restarting the container is the author's call.

## Decode [4A, 18A]

- OpenCV for now, wrapped in one function so swapping the decoder is local.
  Decided 2026-11-20 after the step-1 smoke test:
  - On one caldot1 video, decord matched OpenCV exactly: 900/900 frames identical, and random access returned the same frames.
  - Uncontended, 1 thread each, decord was faster: 1.6 s against 3.1 s for all frames.
  - The author chose to keep OpenCV unless decoding turns out to be the bottleneck. `execution2/tools/compare_decoders.py` (modes `pixels` and `timings`) re-checks this before any switch.
- Frames needed, matching p020 ([p020:277-281](../scripts/p020_exec_classify.py)): sampled `idx % s == 0`, plus the last frame when it isn't already sampled, plus for each sampled frame `idx-1`, or `idx+1` for frame 0 (diff channel).
- p020 asserts that frames read == `CAP_PROP_FRAME_COUNT`; execution2 keeps that assert.
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

Move these pure functions from the scripts into `polyis/` with no logic change. Both the scripts and `execution2` import them.
- p020: `load_model`, `classify_batch`
- p022: the ILP core of `process_video`, split from its file I/O
- p030: CPU canvas drawing helpers (so canvases match p030 by construction)
- p050: `unpack_detections`
- p060: the per-video tracking loop

## Build order

1. ~~Decoder comparison [14A, 18A]~~: done as a smoke test; OpenCV kept (see Decode).
2. Before/after tests [11A], then the move to `polyis/` [5A].
3. ~~`config.py`, `frames.py`, `pipeline.py` with fake-stage tests [12A]~~: done (`tests/execution2/`).
   - numpy views over shared memory hold no buffer export, so `SharedMemory.close()` would unmap memory under a surviving view. `frames.py` detects escaped views by the mmap's reference count and refuses to close.
   - The video edge cases below (empty dir, non-video file, unreadable or 1-frame video, zero collages, duplicate stems) belong to steps 4–5, where main and the stages exist.

   Originally planned:
   - process counts 1×1, 1×N, N×1, N×M
   - an exception and a SIGKILL in each stage, with no leftover `/dev/shm` blocks
   - backpressure
   - empty dir, non-video file, unreadable video, 1-frame video, zero collages, duplicate stems
4. ~~Stages: decode, classify, prune, pack, detect, track~~: done (`execution2/stages.py`, `tests/execution2/test_stages.py`).
   - Decisions: `params.canvas_format` chooses the detector's input: `jpeg` passes each canvas through `cv2.imencode('.jpg')`/`cv2.imdecode`, exactly the pixels p040 reads back from p030's files; `raw` passes the drawn canvas (first chosen as [20B], made a mode on 2026-11-20); [21A] `polyis.models.detector.build_detector` is shared with `get_detector`; p020's model loading moved to `polyis.classify.load_classifier`.
   - Checked against the scripts themselves (loaders redirected to temp files): prune equals p022's `process_video` exactly; pack's index maps and offset lookups equal p030's `compress` exactly, and every drawn tile equals its source tile; track equals p060's `track` file for file; classify equals p020's batching with a fake model.
   - With `canvas_format: jpeg`, pack's canvases equal p030's decoded JPEGs pixel for pixel.
   - Known differences from the scripts: with `canvas_format: raw`, canvases are not JPEG-compressed; within a frame, boxes are merged in collage order (p050 follows p040's `os.listdir` order); classify picks its model optimization from the first video it receives.
   - A padded polyomino larger than the canvas (e.g. `canvas_scale` < 1) aborts inside the C `pack()`, as it would in p030; the engine reports it as a worker death.
5. ~~`main.py`: dummy-video timing and writing outputs~~: done (`python -m execution2.main`, `tests/execution2/test_main.py`).
   - Startup checks before any worker starts: video files only (other files ignored), no empty dir, no two videos with the same stem, every video openable with >= 2 frames, worst-case `/dev/shm` use within free space.
   - Results are kept in memory and written after the last arrival, so writing cannot stall the timed run. Prints `{"videos", "runtime_s", "frames"}` as one JSON line.
6. ~~Stage-by-stage diff against p020→p060 on 2–3 test videos of one dataset [13A]~~: done (`execution2/tools/compare_scripts.py`, reports in `measurements-results/execution2-compare/`).
   - caldot1-y05 te00–te02, real classifier and detector, two configs: s=4 without pruning, and s=8 with mistrack tolerance 0.8 (`execution2/configs/`), both `canvas_format: jpeg`, compared with the cached April 2026 script outputs (read only).
   - Each stage fed the script's previous output: classify, prune, and pack (index maps, offset lookups, JPEG canvases) are identical to p020, p022, and p030; detect has the same box count in every frame, with coordinates within 1.2e-4 px of p050 (batching order); track fed p050's detections is identical to p060.
   - Bit-identity (2026-11-20): our detector on our JPEG canvases, batched exactly as p040 batched its files and merged in p050's order (both read from p040's `detections.jsonl`), reproduces p040's raw detections, p050's per-frame detections, and p060's `tracking.jsonl` byte for byte on all six cases. The only differences in normal runs come from that order, which is p040's `os.listdir` order of files execution2 never writes.
   - End to end through `main` (multi-process): every output equals p060's tracks up to a consistent renaming of track ids, with every box matched within 9.2e-5 px. The renaming comes from box order within a frame (collage order here, p040's `os.listdir` order there), which sets SORT's id numbering.
   - The first `/dev/shm` estimate refused `queue_maxsize: 2` on ace's then 8 GiB (it charged every in-flight video for both blocks and assumed one canvas per stored frame).
     It now counts the two block lifetimes separately (frames: decode through pack; canvases: pack through detect), with exact block sizes and one canvas per sampled frame: 5.0 GiB instead of 10.5 GiB for the s=4 config over all 60 caldot1-y05 test videos.
   - ace's `polyis` container was recreated with `shm_size: 64gb` on 2026-11-20, from `/work/cwkt/projects/tetris` (the old compose directory `/work/cwkt/projects/polyis` is empty); `duckdb` was pip-installed again because the image predates it.

## Benchmark (2026-11-20)

`execution2/tools/benchmark.py`, results in `measurements-results/execution2-benchmark/`. All 60 caldot1-y05 test videos (54,000 frames) on ace GPUs 2-3, in throwaway containers whose `/polyis-cache` is a scratch directory and whose real cache is mounted read-only, so no old result can be overwritten. Videos were read once beforehand. ace's load average was 12-17 (other users) during the runs.

| config | p020-p060 (sum of stage wall clocks) | execution2, 2 GPUs: process / first-to-last result | execution2, 1 GPU: process / first-to-last result |
|---|---|---|---|
| s=4, no pruning | 547.7 s | 96.7 s / 66.2 s | 93.3 s / 59.8 s |
| s=8, tolerance 0.8 | 430.5 s | 74.1 s / 35.9 s | 72.3 s / 30.9 s |

- The scripts' stage times include each stage's process start and model load; execution2's "process" includes them too (about 30-40 s), and its first-to-last time excludes them (dummy-video timing).
- p030 alone took 274 s and 168 s: drawing and writing JPEGs serially, video by video.
- One GPU was slightly faster than two here, so this workload is not limited by the classifier and detector sharing a GPU.
- Outputs vs the scripts' fresh outputs from the same run: 119 of 120 equal up to track-id renaming. The exception (te06, s=4) is the canvas-order effect: its detector boxes differ by up to 0.057 px in collage order, and in p040's order te06 is bit-identical (raw detections, per-frame detections, tracks).
- Single runs; repeatability and the best process counts per stage are not measured yet.

