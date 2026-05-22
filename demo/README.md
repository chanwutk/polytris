# PolyIS Execution-Engine Operator Animation

Self-contained D3 visualization of the 3 execution-engine operators
(relevance classification → pruning → packing) from the PolyIS paper.

## Regenerate the data (on the remote)

```bash
./sync
ssh ace 'cd <repo_path> && docker exec polyis ./run viz/demo/generate_data.py'
rsync -av --delete ace:<repo_path>/viz/demo/data/ viz/demo/data/
```

CLI overrides for `generate_data.py`:

| Flag | Default | Notes |
|---|---|---|
| `--dataset` | `caldot1-y05` | Any dataset listed in `configs/global.yaml` |
| `--video` | first .mp4 in test set | Specific video file name |
| `--frame-start` | `0` | Starting frame index |
| `--num-frames` | `64` | Must be a square number for 8x8 layout |
| `--sample-rate` | `1` | Stride between sampled frames |
| `--tile-size` | `60` | Tile size in pixels |
| `--tracker` | `bytetrackcython` | Must have indexing artifacts (`accuracy.npy` or detections) |
| `--m-values` | `0.0,0.1,...,1.0` | Comma-separated, 11 values by default |
| `--time-limit` | `10.0` | ILP solver seconds per M |
| `--overwrite` | off | Pass to clobber an existing data dir |

### Demo data

The bundled demo data was generated with:

```bash
./run viz/demo/generate_data.py --dataset jnc0 --video te01.mp4 --frame-start 704 --overwrite
```

`jnc0/te01.mp4` frames 704–767 is a busy 4-way intersection with ~6 vehicles
per frame (404 polyominoes total). Pruning produces a clean monotonic
progression: 0 → 292 discarded across M=0…1, with the canvas count dropping
from 16 → 6 — a much more compelling demonstration than a quiet highway scene.

To find other good windows, run `scan_activity.py` (see INSTRUCTIONS.md §2c).
For sparser scenes, pass `--dataset caldot1-y05` or `caldot2-y05` instead.

The script refuses to overwrite the standard `max_rate_table.npy`. If the dataset's
`accuracy.npy` does not exist on the remote, the script aggregates per-tile
tracking accuracy from the training set into `data/raw_indexing/` instead of the
shared cache path, then writes a viz-only `data/max_rate_table_viz.npy`.

## View the animation

Serve the `viz/demo/` directory and open `index.html`:

```bash
cd viz/demo && python -m http.server 8000
# then visit http://localhost:8000/index.html
```

Controls:
- **Previous / Next** (or ←/→): step through the 4 stages.
- **M slider**: enabled in stages 3 and 4 only. Adjusts the mistrack rate tolerance.

## Data layout

```
data/
├── frames/{frame_idx}.png          # tile-aligned source frames
├── polyominoes/{frame_idx}_{i}.png # RGBA polyomino cutouts with alpha mask
├── polyominoes.json                # { polyominoes: [{f, i, y, x, height, width, outline, image}, ...] }
├── pruning.json                    # { "0.00": [[f, i], ...], "0.10": [...], ... }
├── packing.json                    # { "0.00": [{canvas_idx, polyominoes: [{f, i, y, x}, ...]}, ...], ... }
├── max_rate_table_viz.npy          # derived; not used by the viz directly
├── meta.json                       # all parameters + frame/canvas dimensions
└── raw_indexing/                   # (only present when accuracy.npy was missing)
```

`data/` is gitignored — regenerate it via the script above.
