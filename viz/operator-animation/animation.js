// D3 animation of the PolyIS execution engine's 3 operators.
//
// Stages:
//   1. Initial 8x8 layout of frame thumbnails.
//   2. Relevance: frames dimmed; polyominoes drawn with green outlines.
//   3. Pruning: polyominoes that the ILP would discard at the current M turn red.
//   4. Packing: frames + discarded polyominoes hidden; surviving polyominoes
//      translate from their frame positions into a horizontal row of canvases.
//
// All stage transitions are D3 transitions with ~600 ms ease-in-out.
// The M slider is disabled outside stages 3-4 and snaps in 0.1 increments.

const STAGES = [
  { id: 1, name: 'Initial frames' },
  { id: 2, name: 'Relevance classification' },
  { id: 3, name: 'Polyomino pruning' },
  { id: 4, name: 'Polyomino packing' },
];

const TRANSITION_MS = 600;
const FRAMES_PER_ROW = 8;
const FRAME_GAP = 12;
const CANVAS_GAP = 24;
const STAGE_DIMMED_OPACITY = 0.32;
const DATA_DIR = 'data';

const state = {
  stage: 1,
  mValue: 0.0,
  meta: null,
  polyominoes: [],
  pruning: {},       // M-key -> array of [f, i]
  packing: {},       // M-key -> array of {canvas_idx, polyominoes: [...]}
  // Lookups built from the data on load
  polyominoByKey: new Map(),          // "f_i" -> polyomino record
  frameOrder: new Map(),              // frame_idx -> array_idx (position in 8x8 grid)
  framePos: new Map(),                // array_idx -> {x, y} top-left in SVG units
  canvasPositions: new Map(),         // M-key -> { canvases: [{x, y, w, h}], positions: Map("f_i" -> {x, y}) }
  discardedSets: new Map(),           // M-key -> Set of "f_i"
};

const svg = d3.select('#stage');
const stageNumEl = document.getElementById('stage-number');
const stageNameEl = document.getElementById('stage-name');
const mSliderEl = document.getElementById('m-slider');
const mReadoutEl = document.getElementById('m-readout');
const canvasCountEl = document.getElementById('canvas-count');
const metaLineEl = document.getElementById('meta-line');
const prevBtn = document.getElementById('prev-btn');
const nextBtn = document.getElementById('next-btn');

init();

async function init() {
  const [meta, polyominoes, pruning, packing] = await Promise.all([
    d3.json(`${DATA_DIR}/meta.json`),
    d3.json(`${DATA_DIR}/polyominoes.json`),
    d3.json(`${DATA_DIR}/pruning.json`),
    d3.json(`${DATA_DIR}/packing.json`),
  ]);

  state.meta = meta;
  state.polyominoes = polyominoes.polyominoes || [];
  state.pruning = pruning;
  state.packing = packing;

  meta.frame_indices.forEach((frameIdx, arrayIdx) => {
    state.frameOrder.set(frameIdx, arrayIdx);
  });

  computeFrameLayout();
  precomputePackingPositions();
  precomputeDiscardedSets();

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
    state.mValue = sorted[0];
    mSliderEl.value = String(state.mValue);
  }

  setupSvg();
  renderInitial();
  updateControls();
  attachHandlers();
  updateMetaLine();
}

function computeFrameLayout() {
  // Lay frames out in an 8-column grid, sized in source-pixel units. Add a small gap.
  const { width: fw, height: fh } = state.meta.frame_dims;
  const rows = Math.ceil(state.meta.num_frames / FRAMES_PER_ROW);
  for (let arrayIdx = 0; arrayIdx < state.meta.num_frames; arrayIdx++) {
    const col = arrayIdx % FRAMES_PER_ROW;
    const row = Math.floor(arrayIdx / FRAMES_PER_ROW);
    state.framePos.set(arrayIdx, {
      x: col * (fw + FRAME_GAP),
      y: row * (fh + FRAME_GAP),
    });
  }
  state.frameGridWidth = FRAMES_PER_ROW * fw + (FRAMES_PER_ROW - 1) * FRAME_GAP;
  state.frameGridHeight = rows * fh + (rows - 1) * FRAME_GAP;
}

function precomputePackingPositions() {
  const { width: cw, height: ch } = state.meta.canvas_dims;
  for (const [mKey, canvases] of Object.entries(state.packing)) {
    const canvasMeta = [];
    const polyPos = new Map();
    canvases.forEach((canvas, idx) => {
      const cx = idx * (cw + CANVAS_GAP);
      const cy = 0;
      canvasMeta.push({ canvas_idx: canvas.canvas_idx, x: cx, y: cy, w: cw, h: ch });
      for (const p of canvas.polyominoes) {
        polyPos.set(`${p.f}_${p.i}`, { x: cx + p.x, y: cy + p.y });
      }
    });
    state.canvasPositions.set(mKey, { canvases: canvasMeta, positions: polyPos });
  }
}

function precomputeDiscardedSets() {
  for (const [mKey, entries] of Object.entries(state.pruning)) {
    const set = new Set();
    for (const [f, i] of entries) set.add(`${f}_${i}`);
    state.discardedSets.set(mKey, set);
  }
}

function setupSvg() {
  // Per-stage viewBox: stages 1-3 frame the 8x8 grid; stage 4 zooms into the canvas row
  // so canvases fill the available width rather than huddling in the corner of a huge SVG.
  const maxCanvases = Math.max(
    1,
    ...Object.values(state.packing).map(canvases => canvases.length),
  );
  const { width: cw, height: ch } = state.meta.canvas_dims;
  const canvasRowWidth = maxCanvases * cw + (maxCanvases - 1) * CANVAS_GAP;
  state.viewBoxFrames = `0 0 ${state.frameGridWidth} ${state.frameGridHeight}`;
  state.viewBoxCanvases = `0 0 ${canvasRowWidth} ${ch}`;
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
}

function attachHandlers() {
  prevBtn.addEventListener('click', () => goToStage(state.stage - 1));
  nextBtn.addEventListener('click', () => goToStage(state.stage + 1));
  mSliderEl.addEventListener('input', () => {
    state.mValue = Number(mSliderEl.value);
    mReadoutEl.textContent = state.mValue.toFixed(2);
    if (state.stage === 3 || state.stage === 4) applyStage(state.stage, /*transition*/ true);
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

function updateControls() {
  const info = STAGES[state.stage - 1];
  stageNumEl.textContent = String(info.id);
  stageNameEl.textContent = info.name;
  prevBtn.disabled = state.stage === 1;
  nextBtn.disabled = state.stage === STAGES.length;

  const sliderActive = state.stage === 3 || state.stage === 4;
  mSliderEl.disabled = !sliderActive;
  mReadoutEl.textContent = state.mValue.toFixed(2);

  const mKey = formatMKey(state.mValue);
  const canvasCount = state.canvasPositions.get(mKey)?.canvases.length;
  if (state.stage === 4 && canvasCount != null) {
    canvasCountEl.textContent = `${canvasCount} canvases at M=${state.mValue.toFixed(2)}`;
  } else if (state.stage === 3) {
    const discarded = state.discardedSets.get(mKey)?.size ?? 0;
    canvasCountEl.textContent = `${discarded} polyominoes discarded at M=${state.mValue.toFixed(2)}`;
  } else {
    canvasCountEl.textContent = '';
  }
}

function applyStage(stage, withTransition) {
  const t = withTransition
    ? d3.transition().duration(TRANSITION_MS).ease(d3.easeCubicInOut)
    : null;
  const mKey = formatMKey(state.mValue);
  const discarded = state.discardedSets.get(mKey) || new Set();
  const canvasInfo = state.canvasPositions.get(mKey) || { canvases: [], positions: new Map() };

  // Transition the viewBox so the canvas row fills the display in stage 4.
  const targetViewBox = stage === 4 ? state.viewBoxCanvases : state.viewBoxFrames;
  if (t) {
    svg.transition(t).attr('viewBox', targetViewBox);
  } else {
    svg.attr('viewBox', targetViewBox);
  }

  // Frames opacity per stage.
  const frameOpacity = stage === 1 ? 1
                     : stage === 2 ? STAGE_DIMMED_OPACITY
                     : stage === 3 ? STAGE_DIMMED_OPACITY
                     : 0;
  const framesSel = svg.selectAll('g.frame');
  (t ? framesSel.transition(t) : framesSel)
    .attr('opacity', frameOpacity);

  // Canvas backgrounds: only visible in stage 4.
  const canvasesLayer = svg.select('g.canvases-layer');
  const canvasBgs = canvasesLayer.selectAll('g.canvas')
    .data(stage === 4 ? canvasInfo.canvases : [], d => d.canvas_idx);

  canvasBgs.exit().transition().duration(TRANSITION_MS).attr('opacity', 0).remove();

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

  canvasEnter.merge(canvasBgs)
    .transition().duration(TRANSITION_MS)
    .attr('opacity', stage === 4 ? 1 : 0)
    .attr('transform', d => `translate(${d.x}, ${d.y})`);

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
    } else if (stage === 4) {
      // In stage 4, surviving polyominoes move to canvas positions; discarded disappear.
      if (isDiscarded || !canvasPos) {
        opacity = 0;
      } else {
        opacity = 1;
        translateX = canvasPos.x;
        translateY = canvasPos.y;
      }
    }

    const g = d3.select(this);
    const groupTarget = t ? g.transition(t) : g;
    groupTarget.attr('opacity', opacity)
               .attr('transform', `translate(${translateX}, ${translateY})`);

    const imgSel = g.select('image');
    const imgTarget = t ? imgSel.transition(t) : imgSel;
    imgTarget.attr('opacity', imageOpacity);

    // Outline color based on discarded state — no fill overlay (just the red border).
    const showDiscardedStyle = (stage === 3 || stage === 4) && isDiscarded;
    g.selectAll('line.polyomino-edge').classed('discarded', showDiscardedStyle);
  });
}

function formatMKey(m) {
  // Match the key format the Python writer uses: 2 decimal places, e.g. "0.10".
  return m.toFixed(2);
}

function updateMetaLine() {
  if (!state.meta) return;
  const m = state.meta;
  metaLineEl.textContent = `${m.dataset} · ${m.video} · frames ${m.frame_indices[0]}-${m.frame_indices[m.frame_indices.length - 1]} (sr=${m.sample_rate}) · tile=${m.tile_size} · tracker=${m.tracker} · packing=${m.packing_mode}`;
}
