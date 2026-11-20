#!/usr/local/bin/python
"""R7.O3: prepare Seiden's inputs from Tetris's saved reference-pipeline outputs.

Seiden (Bang et al., PVLDB 16) runs its target detector offline: its index reads every frame's detections from
`<video_data>/<name>/tasti_labels.csv` and looks up only the frames it samples, and its I-frame anchors come from
`<video_data>/<name>/video.mp4`. Its notebooks charge the detector as (frames sampled) x (measured per-frame detector
time). Each test clip is a separate video, which Tetris, OTIF, and LEAP also process independently, so this script
writes one Seiden input per clip, without changing Seiden's code:

  <clip>/video.mp4          the test clip itself (copied unchanged, so it keeps its own I-frames).
  <clip>/tasti_labels.csv   the reference detector's boxes for that clip (002_naive/detection.jsonl), one row per box;
                            columns frame, object_name (car), confidence, xmin, ymin, xmax, ymax.
  manifest.json             clip order, frame counts, and offsets in the concatenated test split (the order the scorer
                            stitches answers in), and the detector time per frame charged for each frame Seiden labels.

The charge is the reference detector's per-image time when it runs batched, as in Tetris's and OTIF's pipelines (batch
4): the mean 'detect' op of Tetris's stage-040 detection over every canvas of the split's videos. Canvases have the
frame's resolution, so this is the batched cost of one frame on the same detector and GPU. The unbatched per-frame time
of the reference pipeline (002_naive/detection_runtime.jsonl) is recorded alongside for reference.

Run inside ace's polyis container: python measurements/r7o3_seiden_prepare.py --datasets caldot1-y05
"""
import argparse
import csv
import glob
import json
import shutil
import subprocess
from pathlib import Path

from evaluation.manifests import list_split_videos
from polyis.io import cache, store


# Where Seiden's mounted video_data directory lives inside the polyis container.
OUT_ROOT = Path('/polyis-cache/seiden/video_data')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', required=True)
    parser.add_argument('--out', type=Path, default=OUT_ROOT)
    return parser.parse_args()


def read_jsonl(path) -> list:
    """Every line of a JSONL file, parsed."""
    with open(path) as f:
        return [json.loads(line) for line in f]


def op_times(records, op: str) -> list:
    """The times of one op from runtime records stored as lists of {op, time}."""
    return [next(o['time'] for o in ops if o['op'] == op) for ops in records]


def prepare(dataset: str, out_root: Path) -> dict:
    """Write each test clip's video.mp4 and tasti_labels.csv, and the dataset's manifest.json; returns the manifest."""
    out = out_root / dataset
    # Start from an empty directory so no input from an earlier layout remains.
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    clips, unbatched_ms, batched_ms, offset = [], [], [], 0
    for video in list_split_videos(dataset, 'test'):
        # The reference detector's per-frame boxes and timings for this clip.
        frames = read_jsonl(cache.exec(dataset, 'naive', video, 'detection.jsonl'))
        assert [r['frame_idx'] for r in frames] == list(range(len(frames))), f'{dataset} {video}: frames not dense'
        unbatched_ms += op_times(read_jsonl(cache.exec(dataset, 'naive', video, 'detection_runtime.jsonl')), 'detect')
        # Tetris's batched detection of this clip's canvases, over every configuration.
        for path in glob.glob(str(cache.execution(dataset, video, '040_compressed_detections', '*', 'runtimes.jsonl'))):
            batched_ms += op_times(read_jsonl(path), 'detect')
        clip = out / video.split('.')[0]
        clip.mkdir()
        with open(clip / 'tasti_labels.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame', 'object_name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax'])
            for r in frames:
                for x1, y1, x2, y2, score in r['detections']:
                    writer.writerow([r['frame_idx'], 'car', score, x1, y1, x2, y2])
        shutil.copyfile(store.dataset(dataset, 'test', video), clip / 'video.mp4')
        # The clip's video must hold exactly the labeled frames.
        probed = subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0', '-show_entries',
                                 'stream=nb_read_frames', '-of', 'csv=p=0', str(clip / 'video.mp4')],
                                capture_output=True, text=True, check=True).stdout.strip()
        assert int(probed) == len(frames), f'{dataset} {video}: video has {probed} frames, labels cover {len(frames)}'
        clips.append(dict(video=video, dir=clip.name, offset=offset, frames=len(frames)))
        offset += len(frames)
    assert batched_ms, f'{dataset}: no stage-040 detection runtimes'
    manifest = dict(dataset=dataset, split='test', frames=offset, clips=clips,
                    detector_ms_per_frame=sum(batched_ms) / len(batched_ms),
                    detector_timing_source='040_compressed_detections/*/runtimes.jsonl, mean batched detect op per canvas',
                    detector_ms_per_frame_unbatched=sum(unbatched_ms) / len(unbatched_ms))
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f"{dataset}: {offset} frames in {len(clips)} clips, detector {manifest['detector_ms_per_frame']:.2f} ms/frame "
          f"batched ({manifest['detector_ms_per_frame_unbatched']:.2f} unbatched)", flush=True)
    return manifest


def main():
    args = parse_args()
    for dataset in args.datasets:
        prepare(dataset, args.out)


if __name__ == '__main__':
    main()
