"""Object and polyomino statistics behind the ground-truth-tile packing experiment (r1o4_oracle_packing).

Run in a container whose /polyis-cache is one dataset's oracle-packing scratch cache (bombe), after
measurements-gpu/r1o4_oracle_packing.py finished for that dataset and stride. For each padding, it reads the
canvases p030 wrote (033_compressed_frames index maps and offset lookups) and the ground-truth boxes of the
sampled frames, and writes measurements-results/r1o4_oracle_packing/sr<stride>/stats/<dataset>.json with:
  - objects: boxes per sampled frame, box size relative to the tile, how many tiles a box overlaps,
    and the context ratio (area of the tiles a box overlaps / box area) that unpadded tiles leave around it;
  - per padding: canvases, canvas tiles, occupied tiles, packing efficacy (occupied / canvas tiles),
    polyominoes, tiles per polyomino, and objects per polyomino.
"""
import argparse
import json
import math
from pathlib import Path

import duckdb
import numpy as np

from evaluation.p110_accuracy_compute import resolve_groundtruth_tracking_dataset
from measurements.common import DATASET_LABELS
from polyis.io import cache
from polyis.utilities import build_param_str

TILE_SIZE = 60
PADDINGS = ('none', 'square')
# Corner paddings some deployed operating points use; included when their results exist.
EXTRA_PADDINGS = ('tr', 'bl')
RESULTS = Path('/polyis/measurements-results/r1o4_oracle_packing')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=list(DATASET_LABELS), required=True)
    parser.add_argument('--stride', type=int, default=16)
    return parser.parse_args()


def config_name(stride: int, padding: str) -> str:
    # The configuration name r1o4_oracle_packing.py gave p030/p040/p050.
    return build_param_str(classifier='OracleGT', tilesize=TILE_SIZE, sample_rate=stride, tilepadding=padding,
                           canvas_scale=1.0, tracker=None, tracking_accuracy_threshold=None, relevance_threshold=0.5)


def box_rows(path: Path, stride: int, width: int, height: int) -> np.ndarray:
    """Rows (frame, x1, y1, x2, y2) of the ground-truth boxes on sampled frames, clipped to the tiled grid."""
    rows = []
    # One JSON line per frame (file I/O needs the loop).
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['frame_idx'] % stride == 0:
                rows.extend([row['frame_idx'], *t[1:5]] for t in row['tracks'])
    boxes = np.array(rows, dtype=np.float64).reshape(-1, 5)
    # p030 tiles only whole 60-pixel cells, as oracle_bitmap does through mark_detections.
    grid_w, grid_h = width // TILE_SIZE * TILE_SIZE, height // TILE_SIZE * TILE_SIZE
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, grid_w)
    boxes[:, [2, 4]] = boxes[:, [2, 4]].clip(0, grid_h)
    return boxes


def object_stats(boxes: np.ndarray) -> dict:
    """Box sizes in tile units, tiles overlapped per box, and the unpadded context ratio."""
    w, h = boxes[:, 3] - boxes[:, 1], boxes[:, 4] - boxes[:, 2]
    keep = (w > 0) & (h > 0)
    x1, y1, x2, y2, w, h = boxes[keep, 1], boxes[keep, 2], boxes[keep, 3], boxes[keep, 4], w[keep], h[keep]
    # Tiles a box overlaps: from the tile holding its first pixel to the tile holding its last.
    nx = np.floor((x2 - 1e-9) / TILE_SIZE) - np.floor(x1 / TILE_SIZE) + 1
    ny = np.floor((y2 - 1e-9) / TILE_SIZE) - np.floor(y1 / TILE_SIZE) + 1
    tiles = nx * ny
    # Area of those tiles over the box area: how much surrounding image an isolated box keeps without padding.
    context = tiles * TILE_SIZE ** 2 / (w * h)
    longest = np.maximum(w, h) / TILE_SIZE
    return dict(boxes=int(keep.sum()),
                median_width_tiles=float(np.median(w) / TILE_SIZE),
                median_height_tiles=float(np.median(h) / TILE_SIZE),
                median_longest_side_tiles=float(np.median(longest)),
                fraction_within_one_tile=float(np.mean(tiles == 1)),
                mean_tiles_overlapped=float(tiles.mean()),
                median_tiles_overlapped=float(np.median(tiles)),
                median_context_ratio=float(np.median(context)))


def canvas_stats(directory: Path) -> dict:
    """Canvases, canvas tiles, occupied tiles and polyominoes of one 033_compressed_frames configuration."""
    canvases = canvas_tiles = occupied = polyominoes = 0
    # One index map and one offset lookup per canvas (file I/O needs the loop).
    for path in sorted((directory / 'index_maps').glob('*.npy')):
        index_map = np.load(path)
        canvases += 1
        canvas_tiles += index_map.size
        occupied += int(np.count_nonzero(index_map))
        # One offset-lookup line per polyomino placed on this canvas.
        with open(directory / 'offset_lookups' / f'{path.stem}.jsonl') as stream:
            polyominoes += sum(1 for _ in stream)
    return dict(canvases=canvases, canvas_tiles=canvas_tiles, occupied_tiles=occupied, polyominoes=polyominoes)


def main():
    args = parse_args()
    gt_dataset = resolve_groundtruth_tracking_dataset(args.dataset)
    out = RESULTS / f'sr{args.stride}'
    # Video sizes and sampled-frame counts come from the experiment's own records, identical across paddings.
    paddings = list(PADDINGS) + [p for p in EXTRA_PADDINGS if (out / p / f'{args.dataset}.json').exists()]
    results = {p: json.loads((out / p / f'{args.dataset}.json').read_text()) for p in paddings}
    videos = results['none']['videos']
    assert all(results[p]['videos'].keys() == videos.keys() for p in paddings)
    frames = sum(v['sampled_frames'] for v in videos.values())
    grid_tiles = sum(v['sampled_frames'] * (v['grid_width'] // TILE_SIZE) * (v['grid_height'] // TILE_SIZE)
                     for v in videos.values())

    # Ground-truth boxes of every sampled frame, clipped to each video's tiled grid.
    boxes = np.concatenate([box_rows(cache.exec(gt_dataset, 'groundtruth', video, 'tracking.jsonl'),
                                     args.stride, v['width'], v['height']) for video, v in videos.items()])
    objects = object_stats(boxes)
    objects['boxes_per_frame'] = objects['boxes'] / frames

    # Canvas counts of each padding, summed over videos with DuckDB.
    con = duckdb.connect()
    con.execute('CREATE TABLE canvas(padding VARCHAR, video VARCHAR, canvases BIGINT, canvas_tiles BIGINT, '
                'occupied_tiles BIGINT, polyominoes BIGINT)')
    for padding in paddings:
        for video in videos:
            s = canvas_stats(Path(cache.exec(args.dataset, 'comp-frames', video, config_name(args.stride, padding))))
            con.execute('INSERT INTO canvas VALUES (?, ?, ?, ?, ?, ?)',
                        [padding, video, s['canvases'], s['canvas_tiles'], s['occupied_tiles'], s['polyominoes']])
    per_padding = {}
    for row in con.execute("""SELECT padding, sum(canvases), sum(canvas_tiles), sum(occupied_tiles), sum(polyominoes)
                              FROM canvas GROUP BY padding""").fetchall():
        padding, canvases, canvas_tiles, occupied, polyominoes = row
        # The canvas counts must agree with the experiment's own records.
        assert canvases == sum(v['canvases'] for v in results[padding]['videos'].values()), padding
        per_padding[padding] = dict(
            canvases=int(canvases), canvases_per_frame=canvases / frames,
            canvas_tiles=int(canvas_tiles), occupied_tiles=int(occupied),
            packing_efficacy=occupied / canvas_tiles,
            occupied_tiles_per_frame=occupied / frames, occupied_fraction_of_grid=occupied / grid_tiles,
            polyominoes=int(polyominoes), polyominoes_per_frame=polyominoes / frames,
            tiles_per_polyomino=occupied / polyominoes, objects_per_polyomino=objects['boxes'] / polyominoes)
    # Unpadded occupied tiles are exactly the relevant tiles; padding adds the rest.
    per_padding['square']['padding_tiles'] = per_padding['square']['occupied_tiles'] - per_padding['none']['occupied_tiles']
    per_padding['square']['polyomino_merge_factor'] = per_padding['none']['polyominoes'] / per_padding['square']['polyominoes']
    assert not math.isnan(per_padding['square']['polyomino_merge_factor'])

    result = dict(dataset=args.dataset, label=DATASET_LABELS[args.dataset], stride=args.stride, tile_size=TILE_SIZE,
                  sampled_frames=frames, grid_tiles=grid_tiles, objects=objects, padding=per_padding)
    path = out / 'stats' / f'{args.dataset}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
