# Demo Visualization — Design Notes

This document describes the architecture and design decisions behind the
`./demo` visualization. The visualization is an interactive D3.js animation
that walks through the PolyIS execution engine's pipeline on real data
(64 consecutive frames from a single video).

> Companion docs:
> - `README.md` — quick "how do I run this" reference.
> - `INSTRUCTIONS.md` — full step-by-step pipeline (data generation + viewing).

---

## 1. Purpose

The PolyIS execution engine has three core operators — **Relevance
Classification**, **Polyomino Pruning**, and **Polyomino Packing**. After
the detector runs on the packed canvases, an **Unpack** step maps detection
bounding boxes back to the original source frames.

The visualization animates this entire round-trip on real data so that a
viewer can:

- See exactly what the system does to a video segment.
- Adjust the mistrack-rate tolerance `M` interactively and watch pruning,
  canvas count, and surviving detections change in real time.
- Step through each stage of the pipeline with prev/next controls (or
  arrow keys).

The page is fully static after data generation — no server-side rendering,
no per-request computation. All per-`M` outcomes are precomputed and shipped
in JSON files. The browser only does layout, opacity, and translate tweens.

---

## 2. The six stages

| # | Name | What you see |
|---|---|---|
| 1 | Initial frames | All 64 source frames laid out in a 5-column grid at full opacity. |
| 2 | Relevance classification | Frames dim to 32% opacity; per-frame *polyominoes* (connected sets of relevant tiles) pop with green outlines. |
| 3 | Polyomino pruning | Polyominoes the ILP would discard at the current `M` get a red outline and their image content fades to match the dimmed background. The M slider becomes active. |
| 4 | Polyomino packing | Frames disappear. Surviving polyominoes glide from their frame positions into a 5-column grid of canvases; outlines turn white to read clearly against the canvas background. |
| 5 | Detect (on canvases) | Detection bounding boxes (deep pink) appear on the packed canvases at the positions the detector would output. |
| 6 | Unpack (detections on source) | The canvas grid melts back into the 5-column frame grid; surviving polyominoes return to their source positions; detection bboxes glide back to their original locations overlaid on the now-fully-opaque source frames. |

Stages 1–3, 6 share the **frame-grid viewBox**; stages 4–5 share the
**canvas-grid viewBox** (per-`M`, sized to fit the canvases for the current
`M` exactly). The viewBox transitions smoothly along with the elements.

---

## 3. Pipeline (`generate_data.py`)

Everything the browser shows comes from a single Python script that runs on
the remote inside the polyis Docker container. The script is deliberately
non-destructive — it refuses to clobber an existing `data/` directory unless
`--overwrite` is passed, and it **never** writes back to the standard
`/polyis-cache/.../indexing/` paths.

Inputs (read-only):
- The chosen video file (`store.dataset(dataset, videoset, video)`).
- Groundtruth tracks (`load_tracking_results`) — used for the relevance bitmap.
- Naive detector outputs (`cache.exec(dataset, 'naive', video, 'detection.jsonl')`).
- The dataset's `accuracy.npy` (`cache.index(dataset, 'track_rates', ...)`).

Pipeline steps:

1. **Extract 64 frames** from the chosen video. Resize to tile-aligned
   dimensions if the source isn't already tile-aligned. Save downscaled PNGs
   (default `--image-scale 1/6`) to `data/frames/{f}.png`.
2. **Compute relevance bitmap** per frame from groundtruth tracks via
   `mark_detections` (same logic as `scripts/p021_exec_classify_correct.py`).
3. **Group tiles into polyominoes** via `group_tiles_all` from
   `polyis.pack.adapters`. Yields `tile_to_polyomino_id[F,H,W]` plus per-frame
   polyomino lengths.
4. **Cut polyomino PNGs** with an alpha mask — each polyomino's bounding box
   is cropped from the source frame, tiles outside the polyomino get alpha=0.
   A tile-aligned outline polyline is computed for the green/white border.
   Saved as `data/polyominoes/{f}_{i}.png`.
5. **Compute the per-M max_rate_table** from `accuracy.npy`. The shipped
   `accuracy.npy` covers the canonical thresholds the pipeline normally uses
   (`{0.30..1.00}`); for the demo we derive a custom table indexed by our 11
   `M` values (mistrack tolerance = `1 - accuracy_threshold`). The derived
   table is saved to `data/max_rate_table_viz.npy` only (never overwrites
   the standard cache path).
6. **For each M ∈ {0.0, 0.1, …, 1.0}**: run the Gurobi ILP solver
   (`solve_ilp` from `polyis.sample.ilp.c.gurobi`) with a 10-second wall-clock
   budget. Record per-frame selected polyomino IDs → `discarded[M]`. Then
   build per-frame `PolyominoArray`s of only the selected polyominoes and
   pass them to `pack()` from `polyis.pack.pack`. Record canvas layouts.
7. **Pair detections with polyominoes**: for each naive detection bbox, find
   its containing polyomino via `tile_to_polyomino_id` lookups on the bbox's
   tile range. Discard detections that land on background tiles (the
   pipeline would never see them).
8. **For each M, compute per-detection canvas bbox**: if the detection's
   polyomino is in some canvas at this `M`, shift the bbox by
   `(canvas_origin − source_origin)` of that polyomino. Otherwise record
   `null`.
9. **Emit JSON files** + `meta.json` recording every parameter.

### Output files

```
data/
├── frames/{f}.png                # source PNGs at --image-scale
├── polyominoes/{f}_{i}.png       # RGBA cutouts with alpha-masked tiles
├── polyominoes.json              # [{f, i, y, x, height, width, outline, image}, ...]
├── pruning.json                  # M -> [[f, i], ...]  (discarded)
├── packing.json                  # M -> [{canvas_idx, polyominoes: [{f, i, y, x}, ...]}]
├── detections.json               # {detections: [...], canvas_bboxes: {M -> [{canvas_idx, bbox} | null]}}
├── max_rate_table_viz.npy        # derived per-M table (not consumed by the viz)
├── meta.json                     # all parameters + dimensions + counts
└── raw_indexing/                 # only present if accuracy.npy was missing on remote
```

The `image-scale` default of **1/6** trades ~36× file size for a barely
visible quality hit. JSON coordinates remain in original (tile-aligned)
pixel space; the browser stretches the smaller PNGs back up at render time,
which means `image-scale` is purely a download/perf knob and never affects
positioning math.

### Picking a frame range with enough activity (`scan_activity.py`)

The first 64 frames of most videos are nearly empty (the chosen video,
`jnc0/te01.mp4`, happens to be a busy intersection where frames 704–767
have ~6 vehicles per frame). `scan_activity.py` slides a 64-frame window
in steps of 32 across every test video in every configured dataset, builds
the polyomino set per window, and ranks by a composite "busy AND spread out"
score:

```
score = total_polyominoes × (0.5 + 0.5 × spatial_spread_fraction)
```

This avoids picking windows where everything overlaps a single road segment.

---

## 4. Front-end architecture

Three files at the root of `demo/`:

- **`index.html`** — DOM shell. Header, stage breadcrumb (`<ol id="stages-nav">`),
  controls bar, legend, and a single `<svg id="stage">` that contains the
  entire scene. CSS/JS are linked with `?v=NN` cache-buster query strings.
- **`styles.css`** — layout + per-stage color theme via CSS custom properties.
- **`animation.js`** — bootstraps state, loads JSON, renders SVG, runs
  the animation. No build step; loaded directly by the browser.

D3 v7 is loaded from CDN; no other runtime dependencies.

### State

A single mutable `state` object holds:

- The four loaded JSON payloads (`polyominoes`, `pruning`, `packing`, `detections`).
- Precomputed lookups built once on load:
  - `frameOrder: Map<frameIdx, arrayIdx>`
  - `framePos: Map<arrayIdx, {x, y}>`
  - `canvasPositions: Map<mKey, {canvases, positions, viewBox}>`
  - `discardedSets: Map<mKey, Set<"f_i">>`
  - `detectionCanvasPositions: Map<mKey, Map<"f_id", {canvas_idx, x, y, width, height}>>`
- `stage` (1–6) and `mValue`.

### Render layers (back to front)

```
<svg>
  <g class="frames-layer">      …  <g class="frame">  per source frame
  <g class="canvases-layer">    …  <g class="canvas"> per canvas (stages 4–5 only)
  <g class="polyominoes-layer"> …  <g class="polyomino"> per polyomino
  <g class="detections-layer">  …  <rect class="detection"> per detection
</svg>
```

Detections are drawn last so they always sit on top of everything else.
Each polyomino's `<g>` contains an `<image>` (the RGBA cutout) and a nested
`<g class="polyomino-edges">` with one `<line class="polyomino-edge">`
per outline segment.

### Identity-keyed binding

Every `.data()` call uses an explicit key function so D3 maintains element
identity across re-renders:

| Element | Key |
|---|---|
| frame | `d.frameIdx` |
| canvas | `d.canvas_idx` |
| polyomino | `${d.f}_${d.i}` |
| detection | `${d.f}_${d.id}` |
| outline edge (per polyomino) | line position |

Per-`M` lookups (canvas positions, detection canvas bboxes, discarded sets)
are all `Map`s keyed by identity strings — never by DOM-iteration index.
This prevents "identity swap" artifacts where two elements appear to trade
places during rapid M-slider scrubbing.

### `transitionElement` — the universal transition helper

Almost every visible attribute change goes through one helper function:

```js
transitionElement(sel, t, opacity, applyPosition)
```

Where `applyPosition(s)` is a callback that sets the positional attrs on
either a selection or a transition (e.g. `s => s.attr('x', x).attr('y', y)`
for rects or `s => s.attr('transform', '…')` for groups).

The helper enforces four cases, all sharing the same `d3.transition()`
instance `t` (so every animated element finishes on the same frame):

| Target opacity | Previous state | Behavior |
|---|---|---|
| `0` | already `is-hidden` (mid-fade) | **early return** — let the in-flight fade and its `on('end')` snap complete naturally. Avoids visible position jumps. |
| `0` | rendered opacity is 0, no class | mark `is-hidden`, snap position, set opacity 0 (no tween). |
| `0` | currently visible | mark `is-hidden`, fade opacity to 0, then snap position via the transition's `.on('end', …)`. |
| `>0` | invisible (class or rendered opacity 0) | clear `is-hidden`, `sel.interrupt()` to cancel any leftover scheduled work, snap to new position, fade opacity to target. |
| `>0` | currently visible | tween position + opacity together. |

#### Why the `is-hidden` class

Reading the live opacity attribute is unreliable during rapid M scrubbing
because a fade-out can be mid-flight (opacity = 0.5) when the next slider
event arrives. The class is set **synchronously** at the start of every
exit transition, so the next call always knows the *logical* state of the
element regardless of where its rendered opacity is.

#### Why `on('end')` instead of a chained `transition().duration(0)`

The chained-transition approach was originally used for the
"fade-then-snap" step. The problem was that a chained transition's start
time is fixed at the parent's *natural* end time. If the user interrupts
the parent partway through, the snap can still fire later and overwrite a
fresh position set by a subsequent entry. The `.on('end', fn)` callback
only fires on natural completion, so any new transition (which calls
`sel.interrupt()`) cleanly cancels it.

### Stage transitions in detail

```
Stage 1 (frames full, polyominoes hidden, no canvases, no detections)
   │
   ├─ Next ─▶ Stage 2: frames fade to 0.32; polyominoes snap to frame positions, fade in.
   │                  Slider remains disabled — pruning is M-independent here.
   │
   ├─ Next ─▶ Stage 3: polyomino borders may turn red and their images dim per current M.
   │                  Slider becomes active. No movement.
   │
   ├─ Next ─▶ Stage 4: frame opacity goes to 0; surviving polyominoes glide to canvas
   │                  positions; canvas backgrounds enter; viewBox zooms to canvas grid.
   │                  Polyomino borders turn white.
   │
   ├─ Next ─▶ Stage 5: detections snap to canvas positions (with opacity 0) then fade in.
   │                  Everything else holds steady.
   │
   └─ Next ─▶ Stage 6: viewBox returns to frame grid; frames fade up to full opacity;
                      surviving polyominoes glide back to source frame positions;
                      detections glide back to their source-frame positions (both stay
                      at opacity 1 across this transition). Discarded polyominoes
                      stay invisible.
```

Prev transitions are exact inverses.

### M slider behavior

- Active in stages 3–6 (any post-classification stage).
- Step size is derived from the JSON's `m_values` array (matching the
  Python writer's `f"{m:.2f}"` key format).
- On `input` events (live drag), `applyStage(state.stage, true)` is called
  with the new `M`. The shared `t` transition smoothly retargets every
  element. Mid-flight transitions are interrupted cleanly via the
  `is-hidden` class + `sel.interrupt()` pattern described above.

---

## 5. Visual style

### Colors (CSS custom properties)

| Variable | Hex | Used for |
|---|---|---|
| `--accent` | `#2c7a3e` | polyomino outlines in stages 2–3; legend swatch in those stages |
| `--discarded` | `#c4302b` | discarded polyomino outlines in stage 3 |
| `--packed` | `#ffffff` | polyomino outlines in stages 4–6 |
| `--detection` | `#ff1493` | detection bbox outlines |
| `--frame-border` | `#cdd5dd` | placeholder rect behind each frame image |
| `--canvas-bg` | `#e8eef3` | canvas background fill |
| `--canvas-border` | `#b6c3cd` | canvas background stroke |
| `--muted` | `#5b6770` | controls labels, meta footer |

The "Relevant polyomino" legend swatch tracks the active stage: gray-ish
green by default, white once stage ≥ 4 (toggled via a `stage-packed` class
on `<body>` from `updateControls`).

### Layout constants

| Constant | Value | Notes |
|---|---|---|
| `COLUMNS` | `5` | Used for both the 8x8→5×13 frame grid and the canvas grid. |
| `FRAME_GAP` | `12` px | Gap between adjacent frame cells. |
| `CANVAS_GAP` | `24` px | Gap between adjacent canvas cells. |
| `TRANSITION_MS` | `600` | Shared duration for the `t` transition (ease: `cubicInOut`). |
| `STAGE_DIMMED_OPACITY` | `0.32` | Dimmed-frame opacity in stages 2–3 and dimmed-image opacity for discarded polyominoes in stage 3. |

### Stage breadcrumb

The header shows all 6 stage names horizontally separated by `→`. The
current stage is plain black (`var(--fg)`); the others are
`#b6bdc4` (gray). No bolding. Rendered from the `STAGES` array via
`buildStagesNav` so adding/removing stages requires editing exactly one
array.

---

## 6. Deployment

A GitHub Action (`.github/workflows/deploy-demo.yml`) publishes `./demo`
to the `gh-pages` branch on every push to `main` that touches `demo/**`
(or the workflow file). It uses `peaceiris/actions-gh-pages@v4` with
`force_orphan: true` so the deploy branch stays a single commit. Manual
runs are also available via the Actions tab.

The `demo/data/` directory is **tracked in git** (not ignored), so the
PNGs and JSON ride along on every deploy without any extra build step.

---

## 7. Extending the visualization

### Adding a new stage

1. Add an entry to `STAGES` in `animation.js`.
2. Add a branch (or branches) to `applyStage` for the new stage's
   opacity / position rules for each layer.
3. If the new stage shows a layer that didn't previously exist, add a new
   `g.something-layer` in `renderInitial` and an enter/each block.

### Adding a new data dimension

The Python pipeline already maintains JSON files keyed by the M value.
To add another parameter (e.g. tile-size sweep): extend the JSON schema
so each entry is keyed by a tuple, and add a parallel slider/dropdown in
the controls. `transitionElement`'s logic is independent of which
parameter changed.

### Changing the visualization target dataset/video

Re-run `generate_data.py` with the appropriate flags:

```bash
./run demo/generate_data.py --dataset <name> --video <file.mp4> \
  --frame-start <n> --overwrite
```

Or run `scan_activity.py` first to find the busiest window:

```bash
./run demo/scan_activity.py
```

Pick a row from its ranked output, then plug the values into
`generate_data.py`.

---

## 8. Known limitations & quirks

- **Detector mismatch with relevance**: relevance bitmaps come from
  groundtruth tracks, while detection bboxes come from the naive detector.
  Detections landing on background tiles (no covering polyomino) are
  filtered out at data-generation time, so the viz always shows a
  self-consistent subset.
- **Discrete M values**: the slider snaps to the M values present in the
  generated data (default 11 evenly spaced values 0.0–1.0). Generating
  finer steps is a script re-run away (`--m-values 0.0,0.05,...,1.0`),
  but bigger JSON files.
- **ILP time budget**: each M pass has a 10-second Gurobi wall-clock
  budget. For very dense windows the solver may return an early
  feasible solution rather than the optimum. Adjustable via
  `--time-limit`.
- **Browser cache**: `index.html` references `styles.css?v=NN` and
  `animation.js?v=NN`; bump the integer after any frontend edit to force
  a refetch.
- **No mobile support**: the canvas-grid layout assumes a desktop-class
  viewport. On narrow screens the SVG will horizontally scroll within
  the `<main>` container.
