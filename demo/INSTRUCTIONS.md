# Instructions: Generate Data and View the Animation

This document walks through every step from a fresh clone to a working visualization in the browser.

---

## 0. Prerequisites

- **Local machine**: macOS or Linux with `ssh`, `rsync` (GNU rsync recommended), Python 3, and a modern browser (Chrome/Safari/Firefox).
- **Remote machine**: SSH alias `ace` configured in `~/.ssh/config`, with the `polyis` Docker container already running (`docker ps` should show it). The project is mounted at `/polyis` inside the container.
- **Required existing artifacts on the remote** (for the default dataset):
  - Videos at `/polyis-data/datasets/caldot1-y05/test/*.mp4`.
  - Groundtruth tracking at `/polyis-cache/caldot1-y05/execution/<video>/002_naive/tracking.jsonl` (produced by `p002_preprocess_groundtruth_tracking.py`).
  - Either `/polyis-cache/caldot1-y05/indexing/track_rate/bytetrackcython_60/accuracy.npy` already exists (fast path — instant) **or** `/polyis-cache/caldot1-y05/indexing/det/*.detections.jsonl` exists (slow path — script will run p016 aggregation across the training set, takes minutes to hours).

> Check whether the fast path is available:
> ```bash
> ssh ace 'docker exec polyis bash -c "ls /polyis-cache/caldot1-y05/indexing/track_rate/bytetrackcython_60/accuracy.npy"'
> ```
> If it prints the path, you're set. If not, the script will build it on first run.

---

## 1. Push code to the remote

From your local repo root:

```bash
./sync
```

This rsyncs everything under `viz/` (configured in `configs/sync.yaml`) and all the usual source directories to `ace:/work/cwkt/projects/polyis/`.

---

## 2. Generate the data on the remote (inside Docker)

The data-generation script is **non-destructive** — it refuses to overwrite an
existing `data/` directory unless you pass `--overwrite`, and it never touches
the standard cache paths under `/polyis-cache/.../indexing/`.

### 2a. Recommended demo (busy intersection)

The bundled demo uses `jnc0/te01.mp4` frames 704–767 — a 4-way intersection with
~6 vehicles per frame, picked by `scan_activity.py` (see §2c) as the highest-scoring
64-frame window across all configured datasets. It produces a compelling pruning
demonstration: 0 → 292 polyominoes discarded as M goes from 0 to 1, and the
canvas count drops from 16 → 6.

```bash
ssh ace 'docker exec polyis bash -c "cd /polyis && python demo/generate_data.py --dataset jnc0 --video te01.mp4 --frame-start 704 --overwrite"'
```

Expected output (key lines):

```
[setup] Dataset=jnc0 videoset=test video=te01.mp4
[frames] Reading frame indices 704..767
[frames] src=(720,1080) target=(720,1080) grid=(12,18)
[polyominoes] total=404 across 64 frames
[max_rate_table] shape=(12, 18, 11)
[M=0.00] discarded=0 / 404   canvases=16
[M=0.10] discarded=150 / 404 canvases=11
[M=0.50] discarded=237 / 404 canvases=8
[M=1.00] discarded=292 / 404 canvases=6
[done] Wrote data to /polyis/demo/data
```

If you re-run, pass `--overwrite` (otherwise the script refuses to clobber an
existing `data/` dir).

#### Other good demos

The scan (§2c) ranks these as the busiest spread-out 64-frame windows. Pick one
that matches the visual density you want:

| Score | Dataset | Video | Frames | Polys | Per frame | Spread |
|---|---|---|---|---|---|---|
| 290.0 | jnc0 | te01.mp4 | 704–767 | 399 | 6.23 | 45.4% (12×18) |
| 282.1 | jnc0 | te04.mp4 | 224–287 | 382 | 5.97 | 47.7% (12×18) |
| 283.3 | jnc6 | te12.mp4 | 608–671 | 391 | 6.11 | 44.9% (12×18) |
| 269.4 | jnc2 | te01.mp4 | 448–511 | 400 | 6.25 | 34.7% (12×18) |
| 108.1 | caldot2-y05 | te16.mp4 | 352–415 | 182 | 2.84 | 18.8% (8×12)  |
| 73.0  | caldot1-y05 | te33.mp4 | 224–287 | 123 | 1.92 | 18.8% (8×12)  |

`jnc*` videos are intersections (busiest); `caldot*` are highways (sparser);
`ams-y05` is mid-density. Pass `--dataset`/`--video`/`--frame-start` to switch.

### 2b. Custom run — pick a different video, range, or M values

All defaults are overridable. Full CLI:

| Flag | Default | Notes |
|---|---|---|
| `--dataset` | `caldot1-y05` | Any dataset listed in `configs/global.yaml` |
| `--video` | first `.mp4` in `--videoset` | e.g. `te01.mp4` |
| `--videoset` | `test` | `train` / `valid` / `test` |
| `--frame-start` | `0` | Starting frame index |
| `--num-frames` | `64` | Should be 64 for the 8x8 layout |
| `--sample-rate` | `1` | Stride between sampled frames |
| `--tile-size` | `60` | Tile size in pixels |
| `--tracker` | `bytetrackcython` | Must have indexing artifacts |
| `--tilepadding` | `none` | One of `none/plus/tr/bl/square` |
| `--canvas-scale` | `1.0` | Canvas grid scale vs. source grid |
| `--relevance-threshold` | `0.5` | Recorded in meta.json (unused for groundtruth) |
| `--packing-mode` | `first_fit` | `first_fit/best_fit/easiest_fit` |
| `--time-limit` | `10.0` | ILP seconds per M value |
| `--m-values` | `0.0,0.1,...,1.0` | Comma-separated, in `[0,1]` |
| `--output-dir` | `demo/data` | Where to write outputs |
| `--overwrite` | off | Pass to clobber an existing data dir |
| `--image-scale` | `0.25` | Frame/polyomino PNGs are saved at this fraction of original pixel size (~16x smaller files at 0.25). JSON coords stay in original space; the browser stretches the smaller PNGs back up. Set to `1.0` for full-resolution. |

Example — try `te01.mp4` (lots of activity from frame 0):

```bash
ssh ace 'docker exec polyis bash -c "cd /polyis && python demo/generate_data.py --video te01.mp4 --frame-start 0 --overwrite"'
```

Example — finer M slider (21 values, slower):

```bash
ssh ace 'docker exec polyis bash -c "cd /polyis && python demo/generate_data.py --frame-start 192 --m-values 0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0 --overwrite"'
```

### 2c. Picking a frame range with enough activity — `scan_activity.py`

To search across all configured datasets and surface the best windows, run:

```bash
ssh ace 'docker exec polyis bash -c "cd /polyis && python demo/scan_activity.py"'
```

This slides a 64-frame window in steps of 32 over every test video, builds the
polyomino set per window, and ranks by a composite score (`total_polys × (0.5 +
0.5 × spatial_spread_fraction)`) so it favors **busy AND well-distributed**
windows over single-road-segment scenes.

Useful flags:

```
--datasets jnc0 caldot1-y05   # restrict to specific datasets
--num-frames 64               # window size (match the viz)
--stride 16                   # finer scan (slower)
--min-polyominoes 30          # exclude near-empty windows
--max-polyominoes 400         # exclude pathologically crowded windows
--top-k 8                     # how many top windows per dataset to print
```

Output is a ranked table per dataset plus a global top-N. Pick a row and feed
its `dataset`, `video`, and `start` to `generate_data.py`.

---

## 3. Pull the generated data back to local

```bash
rsync -av --delete ace:/work/cwkt/projects/polyis/demo/data/ demo/data/
```

Expected size: ~30–80 MB depending on video resolution and polyomino count.
Contents:

```
data/
├── frames/{frame_idx}.png          # 64 source-frame PNGs (tile-aligned, scaled per --image-scale)
├── polyominoes/{frame_idx}_{i}.png # RGBA cutouts with alpha mask
├── polyominoes.json                # offsets, sizes, outlines, image paths
├── pruning.json                    # M -> list of [f, i] discarded
├── packing.json                    # M -> list of {canvas_idx, polyominoes: [...]}
├── detections.json                 # naive detections + per-M canvas bbox positions (null when polyomino was pruned)
├── max_rate_table_viz.npy          # derived; not used by the viz directly
├── meta.json                       # all parameters and dimensions
└── raw_indexing/                   # only present if accuracy.npy was missing
```

---

## 4. View the visualization

From the project root:

```bash
cd demo
python3 -m http.server 8765
```

Then open <http://localhost:8765/index.html> in your browser.

**Controls**:

- **Previous / Next** buttons or **←/→ arrow keys**: step through the 4 stages.
- **M slider**: enabled only in stages 3 and 4. Snaps in 0.1 increments (or whatever step you generated with). Drag to see how the discarded set and canvas count change.

**Stage progression**:

1. **Initial frames** — 64 frames laid out in an 8×8 grid.
2. **Relevance classification** — frames dim; relevant polyominoes pop with green outlines.
3. **Polyomino pruning** — polyominoes the ILP would discard at the current M get a red border (and the image dims into the background).
4. **Polyomino packing** — frames + discarded polyominoes vanish; surviving polyominoes translate into a horizontal row of canvases. The viewBox zooms in so canvases fill the display.
5. **Detect (on canvases)** — blue detection bounding boxes appear on the packed canvases, showing what the detector would see.
6. **Unpack (detections on source)** — canvases fade out; source frames return at full opacity; detection bboxes translate back to their original frame positions, demonstrating the round-trip of the pipeline.

When you're done:

```bash
lsof -ti:8765 | xargs kill
```

---

## 5. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `Videoset directory ... does not exist` | The remote doesn't have that dataset under `/polyis-data/datasets/`. Check `configs/global.yaml` for valid names. |
| `Detection directory ... does not exist` (during accuracy build) | `p011_tune_detect.py` hasn't been run on that dataset. Run it first, or pick a dataset where `accuracy.npy` already exists. |
| `Tracking results not found: ... 002_naive/tracking.jsonl` | Run `p002_preprocess_groundtruth_tracking.py` for that dataset first. |
| `Naive detection file not found: ... 002_naive/detection.jsonl` | The naive-detection preprocessing stage hasn't been run for this video. Run it before generating the viz. |
| Animation looks empty (no polyominoes) | The 64-frame window has too few detections. See §2c to pick a busier range. |
| Animation looks cluttered | Window is too busy. Try a shorter `--frame-start` offset or a different video. |
| ILP times out before finding optimum | Bump `--time-limit`. The default 10 s is enough for typical 64-frame windows; very dense scenes may need 30–60 s. |
| Slider jumps too coarsely | Re-run with a finer `--m-values` list (e.g. 21 values at step 0.05). |
| Browser shows a blank page | Check the JS console for fetch errors. If `data/meta.json` returns 404, the data step didn't complete or wasn't pulled back. |
| `data/` exists, script refuses to run | Pass `--overwrite`, or `rm -rf demo/data` first. |

---

## 6. Quick one-liner (after setup)

For everyday regeneration once you've already configured everything:

```bash
./sync && \
  ssh ace 'docker exec polyis bash -c "cd /polyis && python demo/generate_data.py --dataset jnc0 --video te01.mp4 --frame-start 704 --overwrite"' && \
  rsync -av --delete ace:/work/cwkt/projects/polyis/demo/data/ demo/data/ && \
  (cd demo && python3 -m http.server 8765)
```

Then open <http://localhost:8765/index.html>.
