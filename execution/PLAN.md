# Pipeline-Parallel Execution Engine

## Overview

This directory replaces **Phase 4** (test pass) of `scripts/run.sh` with a
pipeline-parallel implementation.  Instead of running each stage
(p020 -> p022 -> p030 -> p040 -> p050 -> p060) sequentially and writing
intermediate results to disk, all stages run as concurrent processes
connected by `torch.multiprocessing` queues.  Video frames stay on GPU
from decode through detection, eliminating intermediate disk I/O.

## Pipeline Topology

```
                  Shared GPU Frame Buffer (per video)
                  Filled by Decoder, read by Classify + Compress
                  Freed after Compress finishes each video

[Decoder]        CPU decode + GPU transfer, frame-level streaming
    | queue: VideoStart, FrameBatch, VideoEnd
    v
[Classify]       Single thread, batch_size=16, GPU inference (ShuffleNet05 fp16)
    | queue: VideoClassifications (all frames for one video)
    v
[Prune]          CONDITIONAL: only when tracking_accuracy_threshold != null
    |            CPU ILP solver (Gurobi), whole-video batch
    | queue: VideoClassifications (pruned)
    v
[Compress]       Cython group_tiles + pack on CPU, GPU tensor tile rendering
    | queue: CollageReady (one per collage, canvas stays on GPU)
    v
[Detect+Unpack]  1 detector model, detection + coordinate remapping (p050 merged)
    | queue: VideoDetections (accumulated per-frame detections)
    v
[Track]          Sequential SORT/ByteTrack/OC-SORT tracking
    | queue: TrackingResult (in-memory)
    v
[Save to disk]   After timer stops
```

## Design Decisions

### Parallelism granularity
- **Frame-level streaming**: Decoder -> Classify (frames stream in batches of 16)
- **Video-level pipeline**: Classify -> Prune -> Compress -> Detect -> Track
  (different videos overlap across stages)
- **Why not frame-level end-to-end**: p022 (ILP solver), p030 (bin packing), and
  p060 (tracking) all require the complete video before they can proceed.  These
  are inherent batch barriers.

### GPU memory (24 GB target)
- Classifier model: ~5 MB (ShuffleNet05 fp16)
- Detector model: 2-10 GB (Ultralytics / RetinaNet depending on dataset)
- Frame buffer per video: 1-3 GB (depending on resolution and frame count)
- CUDA contexts: ~200 MB x 5 processes = ~1 GB
- At most 2 videos in flight simultaneously (one classifying, one compressing)

### Frames on GPU
- Video frames are decoded on CPU (OpenCV), immediately transferred to a GPU
  tensor buffer, and never return to CPU until the detector stage needs
  numpy BGR input (one `.cpu()` call per collage).
- p030 (Compress) tile rendering is rewritten in PyTorch: source tiles are
  copied from the GPU frame buffer to a GPU canvas tensor using torch slicing.
- The detector still expects numpy BGR arrays (all active datasets use
  Ultralytics or RetinaNet, both PyTorch-based but with numpy input APIs).

### p022 bypass
- When `tracking_accuracy_threshold` is `None`, the Prune process is not
  spawned.  The Classify -> Compress queue is wired directly.

### p050 merged into p040
- Coordinate uncompression is pure CPU arithmetic (<1 ms per collage).
  Running it in the same process as detection avoids an extra queue hop.

### No intermediate disk I/O
- Classification grids, canvases, detections, and tracking results all flow
  through in-memory queues.  Only the final tracking results are saved to
  disk, and only after the throughput timer stops.

### Preload mode
- `--preload` decodes ALL videos into GPU memory before the pipeline starts.
  The timer begins when the first frame enters the Classify stage, so decode
  time is excluded from throughput measurement.
- Warning: with many videos or high-resolution data, GPU memory may be
  insufficient.  Falls back gracefully if tensors cannot be allocated.

## Timing

- **Timer start**: First `FrameBatch` enters the Classify process
  (after model loading, warmup, and optional preload).
- **Timer stop**: Last `TrackingResult` is collected by the pipeline
  orchestrator.
- **Excluded**: Model loading, warmup (16 classifier batches + 2 detector
  batches), and disk save of final results.

## File Layout

| File                          | Responsibility |
|-------------------------------|----------------|
| `__init__.py`                 | Package marker |
| `pipeline.py`                 | Message types (7 NamedTuples), `PipelineConfig`, `run_pipeline()`, result saving |
| `main.py`                     | CLI (`--test`/`--valid`/`--preload`), config iteration, Pareto filtering, ProgressBar |
| `stage_decode.py`             | CPU decode, BGR->RGB, GPU transfer, sample_rate filtering, preload mode |
| `stage_classify.py`           | GPU classify_batch (reads from shared buffer), model loading + warmup |
| `stage_prune.py`              | ILP pruning (group_tiles_all + solve_ilp), pass-through when threshold=null |
| `stage_compress.py`           | Cython group_tiles + pack, GPU tensor tile rendering |
| `stage_detect_uncompress.py`  | Detector inference + p050 unpack_detections merged |
| `stage_track.py`              | Sequential tracker, in-memory result collection |

## Queue Message Protocol

```
video_queue      (main -> Decoder):     str (video filename) | None
decode_queue     (Decoder -> Classify):  VideoStart | FrameBatch | VideoEnd | None
classify_queue   (Classify -> Prune/Compress): VideoClassifications | None
prune_queue      (Prune -> Compress):    VideoClassifications | None
compress_queue   (Compress -> Detect):   CollageReady | None
detect_queue     (Detect -> Track):      VideoDetections | None
result_queue     (Track -> main):        TrackingResult | None
```

`None` is the shutdown sentinel.  It propagates through the pipeline:
when a stage receives `None`, it sends `None` to its output queue and exits.

## Shared GPU Frame Buffer

Each video gets its own GPU tensor `[num_needed_frames, H, W, 3]` (uint8).
The Decoder allocates it, fills it with decoded frames, and passes the
reference through `VideoStart`.  The Classify and Compress stages read from
it by buffer position index.  After Compress finishes all collages for a
video, it drops the reference and GPU memory is reclaimed.

The buffer contains both sampled frames and their previous frames (needed
for the diff channel in classification).  A mapping from absolute frame
index to buffer position is maintained per video.

## Outer Orchestration

- One pipeline per GPU, round-robin across available GPUs (via `ProgressBar`).
- Each pipeline processes all videos for one (dataset, parameter_combo) pair.
- Parameter combos are iterated from `configs/global.yaml` with Pareto
  filtering applied when running `--test`.
- Matches the same parameter iteration and filtering logic as the original
  scripts.

## Code Reuse from Original Scripts

| Import | Source |
|--------|--------|
| `load_model()` | `scripts/p020_exec_classify.py` |
| `select_model_optimization()` | `polyis/train/select_model_optimization.py` |
| `group_tiles()` | `polyis/pack/group_tiles.pyx` (Cython) |
| `pack()` | `polyis/pack/pack.pyx` (Cython) |
| `group_tiles_all()` | `polyis/pack/adapters.pyx` (Cython) |
| `solve_ilp()` | `polyis/sample/ilp/c/gurobi.py` |
| `get_detector()`, `detect_batch()` | `polyis/models/detector.py` |
| `unpack_detections()` | `scripts/p050_exec_uncompress.py` |
| `create_tracker()`, `register_tracked_detections()` | `polyis/utilities.py` |
| `build_pareto_combo_filter()` | `polyis/pareto.py` |
| `ProgressBar` | `polyis/utilities.py` |

The only substantial rewrite is p030 rendering: tile-copy loops are
reimplemented in PyTorch tensor operations instead of numpy.

## Known Risks and Mitigations

1. **CUDA context overhead**: ~200 MB per spawned process (5 processes = ~1 GB).
   Mitigation: acceptable on 24 GB GPU.  CPU-only stages (Prune, Track)
   could be threads instead, but processes provide cleaner isolation.

2. **Cython raw pointers**: `group_tiles()` returns a C pointer as uint64.
   Must not cross process boundaries.  Mitigation: both `group_tiles()` and
   `pack()` run in the same Compress process.

3. **Detector numpy input**: All detector backends expect `list[NPImage]` in
   BGR.  Canvas must be `.cpu().numpy()[:,:,::-1]`.  Small overhead relative
   to inference time.

4. **Queue deadlocks**: Unbounded queues (default `maxsize=0`) prevent
   producer/consumer deadlocks.  Memory is bounded by frame buffer sizing,
   not queue depth.

5. **Edge-repeat in GPU rendering**: The tile-copy edge-repeat logic
   (for boundary tiles where source and destination sizes differ) must match
   the CPU implementation exactly.  Needs pixel-exact testing.

## Usage

```bash
# Inside the Docker container:

# Run on test videoset with Pareto filtering (replaces Phase 4 of run.sh)
python execution/main.py --test

# Run with preloaded video frames (throughput benchmarking)
python execution/main.py --test --preload

# Run on valid videoset (full parameter grid, no Pareto filtering)
python execution/main.py --valid
```
