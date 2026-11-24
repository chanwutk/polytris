"""R1.O2: rebuild LEAP's per-clip track exports from a parsed_results directory with the exporter the scored runs used.

The scored exports in /sota-results/<dataset>/leap/<clip>.json were written by reconstruct_tracks_json in
export-leap-outputs.py (ace:/work/alenachao/tunnel/work/scripts/). That script runs its own export loop at import
time over hard-coded paths, so this script parses its source and executes only its import lines and the
reconstruct_tracks_json definition, then calls that function on every <clip>_results.json and
<clip>_tracks.json of --parsed-dir. The default preserves the original function; --clip-negative-starts skips
pre-video frame writes without changing pattern interpolation or LEAP's predicted intervals. --resample-patterns
replaces the original function with resample_tracks_json, which linearly resamples every assigned pattern to its
interval, where the original truncates patterns longer than the interval. The frame lists are written as one gzipped JSON line per clip
({"video_name": ..., "frames": [...]}), so Tetris's measurements/r1o2_leap_otif_rerun.py can compare them with the scored
exports and score them. Runs in the LEAP image (numpy 1.24.3, as the scored container had).
"""
import argparse
import ast
import gzip
import hashlib
import json
import os
import tempfile

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="R1.O2: rebuild LEAP's track exports with export-leap-outputs.py's function")
    parser.add_argument('--exporter', required=True)
    parser.add_argument('--parsed-dir', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--clip-negative-starts', action='store_true')
    parser.add_argument('--resample-patterns', action='store_true')
    return parser.parse_args()


def load_exporter(path: str, clip_negative_starts: bool = False):
    """reconstruct_tracks_json from the exporter's source, without running the script's module-level loop."""
    # The script's syntax tree.
    with open(path) as f:
        tree = ast.parse(f.read(), filename=path)
    # Its imports and the one function; everything else (the hard-coded export loop) is left out.
    keep = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
            or (isinstance(node, ast.FunctionDef) and node.name == 'reconstruct_tracks_json')]
    assert sum(isinstance(n, ast.FunctionDef) for n in keep) == 1, f'{path}: reconstruct_tracks_json not found once'
    if clip_negative_starts:
        # LEAP can predict starts before zero, including indices smaller than the export's negative-index range.
        inserted = 0
        function = next(node for node in keep if isinstance(node, ast.FunctionDef))
        for loop in (node for node in ast.walk(function) if isinstance(node, ast.For)):
            for index, statement in enumerate(loop.body):
                if isinstance(statement, ast.Assign) and any(
                        isinstance(target, ast.Name) and target.id == 'frame_idx' for target in statement.targets):
                    loop.body.insert(index + 1, ast.parse('if frame_idx < 0: continue').body[0])
                    inserted += 1
                    break
        assert inserted == 1, f'{path}: expected exactly one frame assignment, found {inserted}'
    # Execute them in a fresh namespace.
    namespace: dict = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=keep, type_ignores=[])), path, 'exec'), namespace)
    return namespace['reconstruct_tracks_json']


def resample(pattern: list, length: int) -> np.ndarray:
    """`pattern`'s boxes resampled to `length` boxes, its first and last boxes on the first and last frames."""
    # The pattern as a (boxes, 4) array; a pattern without boxes has nothing to place.
    boxes = np.asarray(pattern, dtype=float)
    if boxes.ndim != 2 or boxes.shape[0] == 0 or boxes.shape[1] < 4:
        raise ValueError(f'a motion pattern needs at least one [x1, y1, x2, y2] box, got shape {boxes.shape}')
    # A one-box pattern has no motion to interpolate, so the box stays in place over the whole interval.
    if boxes.shape[0] == 1:
        return np.repeat(boxes[:, :4], length, axis=0)
    # Each frame's position along the pattern, endpoints aligned, as the original's shorter-pattern branch places them.
    positions = np.linspace(0, boxes.shape[0] - 1, length)
    # Interpolate each coordinate between the neighbouring pattern boxes.
    return np.stack([np.interp(positions, np.arange(boxes.shape[0]), boxes[:, c]) for c in range(4)], axis=1)


def resample_tracks_json(results_file: str, tracks_file: str, output_dir: str, video_name: str,
                         clip_negative_starts: bool = False):
    """The original reconstruct_tracks_json, except that every pattern is resampled to its interval.

    The original resamples a pattern shorter than its interval and truncates a longer one; this resamples both,
    so the reconstructed box always runs from the pattern's first box to its last. Everything else is kept: the
    frame list ends at the latest interval end, LEAP's object IDs are kept, a repeated ID on a frame keeps its first
    box, starts before zero index from the end of the list unless clip_negative_starts drops them, and the output
    is <output_dir>/<video_name>.json.
    """
    # LEAP's resolved objects ({id: [[start, end], pattern index, ...]}) and the clip's motion patterns.
    with open(results_file) as f:
        resolved_tuple = json.load(f)
    with open(tracks_file) as f:
        tracks = json.load(f)
    # One detection list per frame, up to the latest interval end.
    max_frame = max((value[0][1] for value in resolved_tuple.values()), default=0)
    frame_detections = [[] for _ in range(max_frame)]
    for obj_id, value in resolved_tuple.items():
        # The object's predicted interval [start, end) and its assigned pattern.
        start_frame, end_frame = value[0]
        traj_idx = value[1]
        if not 0 <= traj_idx < len(tracks):
            raise ValueError(f'{results_file}: object {obj_id} has pattern index {traj_idx}, '
                             f'but {tracks_file} has {len(tracks)} patterns')
        # An empty or inverted interval places no boxes, as in the original.
        required_length = end_frame - start_frame
        if required_length <= 0:
            continue
        # One box per frame of the interval.
        for idx, bbox in enumerate(resample(tracks[traj_idx], required_length)):
            frame_idx = start_frame + idx
            # Frames before the video are dropped only when asked; otherwise they index from the end, as originally.
            if clip_negative_starts and frame_idx < 0:
                continue
            # A repeated object ID on a frame keeps its first box.
            if any(d['track_id'] == int(obj_id) for d in frame_detections[frame_idx]):
                continue
            frame_detections[frame_idx].append({
                'left': float(bbox[0]), 'top': float(bbox[1]), 'right': float(bbox[2]), 'bottom': float(bbox[3]),
                'class': 'car', 'score': 1.0, 'track_id': int(obj_id),
            })
    # One JSON file per clip, as the original writes it.
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f'{video_name}.json'), 'w') as f:
        json.dump(frame_detections, f, indent=2)


def main():
    # Command-line options.
    args = parse_args()
    # The scored runs' exporter function, or the resampling one, and the digest of the exporter source, for the record.
    if args.resample_patterns:
        def reconstruct(results_file, tracks_file, output_dir, video_name):
            resample_tracks_json(results_file, tracks_file, output_dir, video_name, args.clip_negative_starts)
    else:
        reconstruct = load_exporter(args.exporter, clip_negative_starts=args.clip_negative_starts)
    with open(args.exporter, 'rb') as f:
        exporter_sha256 = hashlib.sha256(f.read()).hexdigest()
    # Clips of the parsed results directory, by their results file.
    names = sorted(f[:-len('_results.json')] for f in os.listdir(args.parsed_dir) if f.endswith('_results.json'))
    assert names, f'no <clip>_results.json in {args.parsed_dir}'
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for name in names:
            # The exporter writes <tmp>/<clip>.json and prints a line; read the file back.
            reconstruct(os.path.join(args.parsed_dir, f'{name}_results.json'),
                        os.path.join(args.parsed_dir, f'{name}_tracks.json'), tmp, name)
            with open(os.path.join(tmp, f'{name}.json')) as f:
                row = dict(video_name=name, exporter_sha256=exporter_sha256, frames=json.load(f))
                if args.clip_negative_starts:
                    row['negative_frame_policy'] = 'drop_before_zero'
                if args.resample_patterns:
                    row['pattern_policy'] = 'resample_to_interval'
                rows.append(row)
    # One gzipped line per clip, written to a temporary name first.
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with gzip.open(args.out + '.tmp', 'wt') as f:
        f.writelines(json.dumps(r) + '\n' for r in rows)
    os.replace(args.out + '.tmp', args.out)
    print(f'wrote {args.out}: {len(rows)} clips')


if __name__ == '__main__':
    main()
