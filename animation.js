// D3 animation of the PolyIS execution engine's 3 operators.
//
// Stages:
//   1. Initial 8x8 layout of frame thumbnails.
//   2. Relevance: frames dimmed; polyominoes drawn with green outlines.
//   3. Pruning: polyominoes that the ILP would discard at the current M turn red.
//   4. Packing: frames + discarded polyominoes hidden; surviving polyominoes
//      translate from their frame positions into a horizontal row of canvases.
//
// Opacity fades use FADE_TRANSITION_MS; position/viewBox moves use MOVE_TRANSITION_MS.
// The M slider is disabled outside stages 3-4 and snaps in 0.1 increments.

const STAGES = [
  { id: 1, name: 'Initial frames' },
  { id: 2, name: 'Relevance classification' },
  { id: 3, name: 'Polyomino pruning' },
  { id: 4, name: 'Polyomino packing' },
  { id: 5, name: 'Detect (on canvases)' },
  { id: 6, name: 'Unpack (detections on source)' },
];

const FADE_TRANSITION_MS = 400;
const MOVE_TRANSITION_MS = 900;
// Number of columns for both the frame grid (stages 1-3, 6) and the canvas grid (stages 4-5).
const COLUMNS = 5;
const FRAME_GAP = 12;
const CANVAS_GAP = 24;
const STAGE_DIMMED_OPACITY = 0.32;
const DATA_DIR = 'data';

const state = {
  stage: 1,
  mValue: 0.5,
  meta: null,
  polyominoes: [],
  pruning: {},       // M-key -> array of [f, i]
  packing: {},       // M-key -> array of {canvas_idx, polyominoes: [...]}
  detections: [],    // array of {f, id, bbox, polyomino, score?}
  detectionCanvasBboxes: {},   // M-key -> array (parallel to detections) of {canvas_idx, bbox} or null
  // Lookups built from the data on load
  polyominoByKey: new Map(),          // "f_i" -> polyomino record
  frameOrder: new Map(),              // frame_idx -> array_idx (position in 8x8 grid)
  framePos: new Map(),                // array_idx -> {x, y} top-left in SVG units
  canvasPositions: new Map(),         // M-key -> { canvases: [{x, y, w, h}], positions: Map("f_i" -> {x, y}) }
  discardedSets: new Map(),           // M-key -> Set of "f_i"
  detectionCanvasPositions: new Map(),  // M-key -> array of {canvas_idx, x1, y1, x2, y2} (absolute SVG coords) or null
};

const svg = d3.select('#stage');
const stagesNavEl = document.getElementById('stages-nav');
const mSliderEl = document.getElementById('m-slider');
const mReadoutEl = document.getElementById('m-readout');
const canvasCountEl = document.getElementById('canvas-count');
const metaLineEl = document.getElementById('meta-line');
const prevBtn = document.getElementById('prev-btn');
const nextBtn = document.getElementById('next-btn');

init();

async function init() {
  const [meta, polyominoes, pruning, packing, detections] = await Promise.all([
    d3.json(`${DATA_DIR}/meta.json`),
    d3.json(`${DATA_DIR}/polyominoes.json`),
    d3.json(`${DATA_DIR}/pruning.json`),
    d3.json(`${DATA_DIR}/packing.json`),
    d3.json(`${DATA_DIR}/detections.json`).catch(() => ({ detections: [], canvas_bboxes: {} })),
  ]);

  state.meta = meta;
  state.polyominoes = polyominoes.polyominoes || [];
  state.pruning = pruning;
  state.packing = packing;
  state.detections = detections.detections || [];
  state.detectionCanvasBboxes = detections.canvas_bboxes || {};

  meta.frame_indices.forEach((frameIdx, arrayIdx) => {
    state.frameOrder.set(frameIdx, arrayIdx);
  });

  computeFrameLayout();
  precomputePackingPositions();
  precomputeDiscardedSets();
  precomputeDetectionCanvasPositions();

  // Index polyominoes for fast lookup.
  for (const p of state.polyominoes) {
    state.polyominoByKey.set(`${p.f}_${p.i}`, p);
  }

  // Snap M slider to the available M values from the data.
  const mValues = (meta.m_values || []).map(v => Number(v));
  if (mValues.length > 0) {
    mSliderEl.min = String(Math.min(...mValues));
    mSliderEl.max = String(Math.max(...mValues));
    // Heuristic: assume an evenly-spaced grid; pick step from the first gap.
    const sorted = [...mValues].sort((a, b) => a - b);
    if (sorted.length >= 2) mSliderEl.step = String(+(sorted[1] - sorted[0]).toFixed(4));
    // Default to the available M value closest to 0.5.
    const target = 0.5;
    state.mValue = sorted.reduce((best, v) =>
      Math.abs(v - target) < Math.abs(best - target) ? v : best, sorted[0]);
    mSliderEl.value = String(state.mValue);
  }

  setupSvg();
  renderInitial();
  buildStagesNav();
  updateControls();
  attachHandlers();
  updateMetaLine();
}

function computeFrameLayout() {
  // Lay frames out in COLUMNS-wide grid, sized in source-pixel units. Add a small gap.
  const { width: fw, height: fh } = state.meta.frame_dims;
  const rows = Math.ceil(state.meta.num_frames / COLUMNS);
  const cols = Math.min(COLUMNS, state.meta.num_frames);
  for (let arrayIdx = 0; arrayIdx < state.meta.num_frames; arrayIdx++) {
    const col = arrayIdx % COLUMNS;
    const row = Math.floor(arrayIdx / COLUMNS);
    state.framePos.set(arrayIdx, {
      x: col * (fw + FRAME_GAP),
      y: row * (fh + FRAME_GAP),
    });
  }
  state.frameGridWidth = cols * fw + (cols - 1) * FRAME_GAP;
  state.frameGridHeight = rows * fh + (rows - 1) * FRAME_GAP;
}

function precomputePackingPositions() {
  // Lay canvases in a COLUMNS-wide grid. Each M has its own per-M viewBox sized to
  // exactly fit that M's canvas count, so the SVG zooms appropriately as M changes.
  const { width: cw, height: ch } = state.meta.canvas_dims;
  for (const [mKey, canvases] of Object.entries(state.packing)) {
    const canvasMeta = [];
    const polyPos = new Map();
    canvases.forEach((canvas, idx) => {
      const col = idx % COLUMNS;
      const row = Math.floor(idx / COLUMNS);
      const cx = col * (cw + CANVAS_GAP);
      const cy = row * (ch + CANVAS_GAP);
      canvasMeta.push({ canvas_idx: canvas.canvas_idx, x: cx, y: cy, w: cw, h: ch });
      for (const p of canvas.polyominoes) {
        polyPos.set(`${p.f}_${p.i}`, { x: cx + p.x, y: cy + p.y });
      }
    });
    const cols = Math.min(COLUMNS, canvases.length || 1);
    const rows = Math.max(1, Math.ceil(canvases.length / COLUMNS));
    const gridW = cols * cw + (cols - 1) * CANVAS_GAP;
    const gridH = rows * ch + (rows - 1) * CANVAS_GAP;
    state.canvasPositions.set(mKey, {
      canvases: canvasMeta,
      positions: polyPos,
      viewBox: `0 0 ${gridW} ${gridH}`,
    });
  }
}

function precomputeDiscardedSets() {
  for (const [mKey, entries] of Object.entries(state.pruning)) {
    const set = new Set();
    for (const [f, i] of entries) set.add(`${f}_${i}`);
    state.discardedSets.set(mKey, set);
  }
}

function precomputeDetectionCanvasPositions() {
  // Translate canvas-local detection bboxes into SVG-space coordinates, keyed by
  // detection identity (`${f}_${id}`) so applyStage looks them up by identity
  // rather than by DOM index — eliminates any risk of identity swapping when D3
  // iterates the SVG selection in a different order than `state.detections`.
  for (const [mKey, bboxList] of Object.entries(state.detectionCanvasBboxes)) {
    const canvasInfo = state.canvasPositions.get(mKey);
    const outMap = new Map();
    bboxList.forEach((entry, idx) => {
      if (!entry || !canvasInfo) return;
      const canvas = canvasInfo.canvases[entry.canvas_idx];
      if (!canvas) return;
      const det = state.detections[idx];
      if (!det) return;
      const [x1, y1, x2, y2] = entry.bbox;
      outMap.set(`${det.f}_${det.id}`, {
        canvas_idx: entry.canvas_idx,
        x: canvas.x + x1,
        y: canvas.y + y1,
        width: x2 - x1,
        height: y2 - y1,
      });
    });
    state.detectionCanvasPositions.set(mKey, outMap);
  }
}

function setupSvg() {
  // Per-stage viewBox: stages 1-3, 6 frame the COLUMNS-wide frame grid; stages 4-5 use
  // the canvas grid (per-M to fit each M's canvas count tightly).
  state.viewBoxFrames = `0 0 ${state.frameGridWidth} ${state.frameGridHeight}`;
  // Fallback viewBox when no canvas data is available (shouldn't happen with valid input).
  state.viewBoxCanvases = state.viewBoxFrames;
  svg.attr('viewBox', state.viewBoxFrames)
     .attr('preserveAspectRatio', 'xMidYMid meet');
}

function renderInitial() {
  // Frames layer (one image per array_idx, identified by frame_idx).
  const framesLayer = svg.append('g').attr('class', 'frames-layer');
  const frameData = state.meta.frame_indices.map((frameIdx, arrayIdx) => ({
    frameIdx, arrayIdx, ...state.framePos.get(arrayIdx),
  }));
  framesLayer.selectAll('g.frame')
    .data(frameData, d => d.frameIdx)
    .enter()
    .append('g')
    .attr('class', 'frame')
    .attr('transform', d => `translate(${d.x}, ${d.y})`)
    // Explicit initial opacity so D3's first opacity tween reads `1` (not null,
    // which would interpolate from 0 and flash the frames to near-invisible).
    .attr('opacity', 1)
    .each(function (d) {
      const g = d3.select(this);
      g.append('rect')
        .attr('class', 'frame-bg')
        .attr('width', state.meta.frame_dims.width)
        .attr('height', state.meta.frame_dims.height);
      g.append('image')
        .attr('class', 'frame-image')
        .attr('href', `${DATA_DIR}/frames/${d.frameIdx}.png`)
        .attr('width', state.meta.frame_dims.width)
        .attr('height', state.meta.frame_dims.height)
        .attr('opacity', 1);
    });

  // Canvas-row layer (background rects for stage 4). Hidden in other stages.
  svg.append('g').attr('class', 'canvases-layer');

  // Polyominoes layer drawn on top so outlines/overlays sit above the frames.
  const polyLayer = svg.append('g').attr('class', 'polyominoes-layer');

  polyLayer.selectAll('g.polyomino')
    .data(state.polyominoes, d => `${d.f}_${d.i}`)
    .enter()
    .append('g')
    .attr('class', 'polyomino')
    .attr('transform', d => {
      const arrayIdx = state.frameOrder.get(d.f);
      const pos = state.framePos.get(arrayIdx);
      return `translate(${pos.x + d.x}, ${pos.y + d.y})`;
    })
    .attr('opacity', 0)
    .each(function (d) {
      const g = d3.select(this);
      g.append('image')
        .attr('href', `${DATA_DIR}/${d.image}`)
        .attr('width', d.width)
        .attr('height', d.height);
      // Edge outlines. Each edge is a separate <line> so we can stroke disjoint polyominoes.
      const edgeSel = g.append('g').attr('class', 'polyomino-edges');
      edgeSel.selectAll('line.polyomino-edge')
        .data(d.outline)
        .enter()
        .append('line')
        .attr('class', 'polyomino-edge')
        .attr('x1', e => e[0])
        .attr('y1', e => e[1])
        .attr('x2', e => e[2])
        .attr('y2', e => e[3]);
    });

  // Detection bboxes layer (drawn last → on top). One <rect> per detection; we mutate its
  // (x, y) + opacity in applyStage to keep transitions cheap.
  const detLayer = svg.append('g').attr('class', 'detections-layer');
  detLayer.selectAll('rect.detection')
    .data(state.detections, d => `${d.f}_${d.id}`)
    .enter()
    .append('rect')
    .attr('class', 'detection')
    .attr('width', d => d.bbox[2] - d.bbox[0])
    .attr('height', d => d.bbox[3] - d.bbox[1])
    .attr('opacity', 0);
}

function attachHandlers() {
  prevBtn.addEventListener('click', () => goToStage(state.stage - 1));
  nextBtn.addEventListener('click', () => goToStage(state.stage + 1));
  mSliderEl.addEventListener('input', () => {
    state.mValue = Number(mSliderEl.value);
    mReadoutEl.textContent = state.mValue.toFixed(2);
    updateControls();
    if (state.stage >= 3) applyStage(state.stage, /*transition*/ true);
  });
  window.addEventListener('keydown', e => {
    if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
    if (e.key === 'ArrowRight') goToStage(state.stage + 1);
    else if (e.key === 'ArrowLeft') goToStage(state.stage - 1);
  });
}

function goToStage(stage) {
  if (stage < 1 || stage > STAGES.length) return;
  state.stage = stage;
  updateControls();
  applyStage(stage, true);
}

function buildStagesNav() {
  // One <li> per stage; updateControls toggles `.current` on the active one.
  stagesNavEl.replaceChildren();
  for (const s of STAGES) {
    const li = document.createElement('li');
    li.dataset.stage = String(s.id);
    li.textContent = s.name;
    stagesNavEl.appendChild(li);
  }
}

function updateControls() {
  // Highlight the active stage in the breadcrumb; others stay gray.
  const items = stagesNavEl.querySelectorAll('li');
  items.forEach(li => {
    li.classList.toggle('current', Number(li.dataset.stage) === state.stage);
  });
  prevBtn.disabled = state.stage === 1;
  nextBtn.disabled = state.stage === STAGES.length;

  // Stage 4 onward swaps the polyomino outline color (green → white). Reflect
  // that in the legend swatch so the guide always matches the live styling.
  document.body.classList.toggle('stage-packed', state.stage >= 4);

  const sliderActive = state.stage >= 3;
  mSliderEl.disabled = !sliderActive;
  mReadoutEl.textContent = state.mValue.toFixed(2);

  const mKey = formatMKey(state.mValue);
  const canvasCount = state.canvasPositions.get(mKey)?.canvases.length;
  if (state.stage === 4 && canvasCount != null) {
    canvasCountEl.textContent = `${canvasCount} canvases at M=${state.mValue.toFixed(2)}`;
  } else if (state.stage === 3) {
    const discarded = state.discardedSets.get(mKey)?.size ?? 0;
    canvasCountEl.textContent = `${discarded} polyominoes discarded at M=${state.mValue.toFixed(2)}`;
  } else if (state.stage === 5 || state.stage === 6) {
    const kept = (state.detectionCanvasPositions.get(mKey) || new Map()).size;
    const total = state.detections.length;
    canvasCountEl.textContent = `${kept} / ${total} detections kept at M=${state.mValue.toFixed(2)}`;
  } else {
    canvasCountEl.textContent = '';
  }
}

function applyStage(stage, withTransition) {
  const tFade = withTransition
    ? d3.transition().duration(FADE_TRANSITION_MS).ease(d3.easeCubicInOut)
    : null;
  const tMove = withTransition
    ? d3.transition().duration(MOVE_TRANSITION_MS).ease(d3.easeCubicInOut)
    : null;
  const mKey = formatMKey(state.mValue);
  const discarded = state.discardedSets.get(mKey) || new Set();
  const canvasInfo = state.canvasPositions.get(mKey) || { canvases: [], positions: new Map() };

  // Transition the viewBox: stages 4-5 zoom into the per-M canvas grid; stages 1-3 and 6
  // use the COLUMNS-wide frame grid. Falls back to the frame viewBox if the per-M packing
  // data is somehow missing.
  const useCanvasViewBox = stage === 4 || stage === 5;
  const targetViewBox = useCanvasViewBox
    ? (canvasInfo.viewBox || state.viewBoxCanvases)
    : state.viewBoxFrames;
  if (tMove) {
    svg.transition(tMove).attr('viewBox', targetViewBox);
  } else {
    svg.attr('viewBox', targetViewBox);
  }

  // Frames opacity per stage.
  // Stage 6 (unpack) brings frames back at full opacity — the final "result" view.
  const frameOpacity = stage === 1 ? 1
                     : stage === 2 ? STAGE_DIMMED_OPACITY
                     : stage === 3 ? STAGE_DIMMED_OPACITY
                     : stage === 6 ? 1
                     : 0;
  const framesSel = svg.selectAll('g.frame');
  (tFade ? framesSel.transition(tFade) : framesSel)
    .attr('opacity', frameOpacity);

  // Canvas backgrounds: visible in stages 4 and 5 (packing & detect).
  const showCanvases = stage === 4 || stage === 5;
  const canvasesLayer = svg.select('g.canvases-layer');
  const canvasBgs = canvasesLayer.selectAll('g.canvas')
    .data(showCanvases ? canvasInfo.canvases : [], d => d.canvas_idx);

  if (tFade) canvasBgs.exit().transition(tFade).attr('opacity', 0).remove();
  else canvasBgs.exit().attr('opacity', 0).remove();

  const canvasEnter = canvasBgs.enter()
    .append('g')
    .attr('class', 'canvas')
    .attr('transform', d => `translate(${d.x}, ${d.y})`)
    .attr('opacity', 0);
  canvasEnter.append('rect')
    .attr('class', 'canvas-bg')
    .attr('width', d => d.w)
    .attr('height', d => d.h);
  canvasEnter.append('text')
    .attr('class', 'canvas-label')
    .attr('x', 6)
    .attr('y', 14)
    .text(d => `Canvas ${d.canvas_idx}`);

  canvasEnter.merge(canvasBgs).each(function (d) {
    const g = d3.select(this);
    const targetOpacity = showCanvases ? 1 : 0;
    transitionElement(g, tFade, tMove, targetOpacity, s => s.attr('transform', `translate(${d.x}, ${d.y})`));
  });

  // Polyominoes: position, opacity, color depending on stage + M.
  const polySel = svg.selectAll('g.polyomino');
  polySel.each(function (d) {
    const key = `${d.f}_${d.i}`;
    const isDiscarded = discarded.has(key);
    const arrayIdx = state.frameOrder.get(d.f);
    const framePos = state.framePos.get(arrayIdx);
    const canvasPos = canvasInfo.positions.get(key);

    let opacity = 0;
    // imageOpacity dims only the cutout PNG (keeping the red border crisp on discarded ones).
    let imageOpacity = 1;
    let translateX = framePos.x + d.x;
    let translateY = framePos.y + d.y;

    if (stage === 1) {
      opacity = 0;
    } else if (stage === 2) {
      opacity = 1;
    } else if (stage === 3) {
      opacity = 1;
      if (isDiscarded) imageOpacity = STAGE_DIMMED_OPACITY;
    } else if (stage === 4 || stage === 5 || stage === 6) {
      // Discarded polyominoes never reach the canvas in the pipeline → stay invisible
      // for all post-pruning stages.
      if (isDiscarded || !canvasPos) {
        opacity = 0;
      } else {
        opacity = 1;
        if (stage === 4 || stage === 5) {
          // Packed view: polyominoes sit at their canvas positions.
          translateX = canvasPos.x;
          translateY = canvasPos.y;
        }
        // Stage 6 (unpack): polyominoes return to their source frame slots — the default
        // translateX/Y = framePos + d.x/y already encodes that.
      }
    }

    const g = d3.select(this);
    const targetTransform = `translate(${translateX}, ${translateY})`;
    transitionElement(g, tFade, tMove, opacity, s => s.attr('transform', targetTransform));

    // Image opacity tweens smoothly regardless of group state (it only matters when
    // the group is visible, which the user can see).
    const imgSel = g.select('image');
    const imgTarget = tFade ? imgSel.transition(tFade) : imgSel;
    imgTarget.attr('opacity', imageOpacity);

    // Outline color based on discarded state — no fill overlay (just the red border).
    const showDiscardedStyle = (stage === 3 || stage === 4 || stage === 5) && isDiscarded;
    // Surviving polyominoes in the post-pack stages (4 packing, 5 detect, 6 unpack) get a
    // white border so the pink detection bboxes drawn over them read clearly. Stage 6
    // shows the unpacked detections on top of the source frames — keeping the polyomino
    // border white visually links the bboxes back to their packing context.
    const showPackedStyle = (stage === 4 || stage === 5 || stage === 6) && !isDiscarded;
    g.selectAll('line.polyomino-edge')
      .classed('discarded', showDiscardedStyle)
      .classed('packed', showPackedStyle);
  });

  // Detection bboxes: visible in stages 5 (on canvases) and 6 (back on source frames).
  // Look up each detection's canvas position by identity (Map keyed by `${f}_${id}`),
  // not by DOM index — that way the bound datum `d` is always paired with the right
  // canvas entry even if D3's selection order ever drifts from `state.detections`.
  const detCanvasMap = state.detectionCanvasPositions.get(mKey) || new Map();
  const detSel = svg.selectAll('rect.detection');
  detSel.each(function (d) {
    const detKey = `${d.f}_${d.id}`;
    const canvasEntry = detCanvasMap.get(detKey);
    const isKept = canvasEntry != null;

    let opacity = 0;
    let x = d.bbox[0];
    let y = d.bbox[1];

    if (stage === 5 && isKept) {
      // Position on the packed canvas.
      x = canvasEntry.x;
      y = canvasEntry.y;
      opacity = 1;
    } else if (stage === 6 && isKept) {
      // Back to source frame position (offset by the frame's slot in the grid).
      const arrayIdx = state.frameOrder.get(d.f);
      const framePos = state.framePos.get(arrayIdx);
      x = framePos.x + d.bbox[0];
      y = framePos.y + d.bbox[1];
      opacity = 1;
    }

    const sel = d3.select(this);
    transitionElement(sel, tFade, tMove, opacity, s => s.attr('x', x).attr('y', y));
  });
}

function formatMKey(m) {
  // Match the key format the Python writer uses: 2 decimal places, e.g. "0.10".
  return m.toFixed(2);
}

// Universal transition rule for any element that has an `opacity` plus some
// positional attribute(s):
//
//   - appearing  (invisible → visible): snap position synchronously with opacity
//       still 0, then fade opacity to its new value
//   - disappearing (visible → invisible): fade opacity to 0 first via the shared
//       transition; only after it's invisible do we snap the position
//   - staying visible (visible → visible): tween position + opacity together
//   - staying invisible: snap position; opacity tween (if any) is a no-op
//
// We track *logical* visibility via the `is-hidden` class rather than reading
// the rendered opacity, because a mid-flight fade-out could otherwise look like
// "visible-→-visible" (opacity 0.5 → 1) and cause the element to move while
// re-entering. The class is set/cleared at the start of every transition so any
// rapid follow-up call sees the correct logical state.
function transitionElement(sel, tFade, tMove, opacity, applyPosition) {
  const renderedOpacity = +sel.attr('opacity') || 0;
  const logicallyHidden = sel.classed('is-hidden');

  if (opacity === 0) {
    // TARGET: invisible.
    if (logicallyHidden) {
      // Already in (or finished) a fade-out. Leave it alone — snapping the
      // position now while opacity is still mid-fade would make the element
      // visibly jump to its default location. The original exit's on('end')
      // callback will snap the position once the fade-out completes.
      return;
    }
    sel.classed('is-hidden', true);
    if (renderedOpacity === 0) {
      // Was already fully invisible (just hadn't been marked logically hidden
      // yet — e.g. very first applyStage call). Snap position immediately; no
      // visible change since opacity stays 0.
      sel.interrupt();
      applyPosition(sel);
      sel.attr('opacity', 0);
    } else {
      // Was visible; fade out, then snap position when fully transparent.
      if (tFade) {
        sel.transition(tFade)
           .attr('opacity', 0)
           .on('end', function () { applyPosition(d3.select(this)); });
      } else {
        sel.attr('opacity', 0);
        applyPosition(sel);
      }
    }
  } else {
    // TARGET: visible.
    if (logicallyHidden || renderedOpacity === 0) {
      // Was invisible (either logically or rendered). Cancel any in-flight
      // transition first so a leftover on('end') snap can't override us, then
      // snap to the new position and fade opacity in.
      sel.classed('is-hidden', false);
      sel.interrupt();
      applyPosition(sel);
      const target = tFade ? sel.transition(tFade) : sel;
      target.attr('opacity', opacity);
    } else {
      // Visible → visible: position uses the slower move transition. When both
      // position and opacity change, share that transition so parallel tweens
      // do not interrupt each other.
      const target = tMove ? sel.transition(tMove) : (tFade ? sel.transition(tFade) : sel);
      applyPosition(target);
      target.attr('opacity', opacity);
    }
  }
}

function updateMetaLine() {
  if (!state.meta) return;
  const m = state.meta;
  metaLineEl.textContent = `${m.dataset} · ${m.video} · frames ${m.frame_indices[0]}-${m.frame_indices[m.frame_indices.length - 1]} (sr=${m.sample_rate}) · tile=${m.tile_size} · tracker=${m.tracker} · packing=${m.packing_mode}`;
}
