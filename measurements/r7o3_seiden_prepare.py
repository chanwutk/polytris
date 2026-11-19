#!/usr/local/bin/python
"""R7.O3: prepare Seiden's inputs from Tetris's saved reference-pipeline outputs.

Seiden (Bang et al., PVLDB 16) runs its target detector offline: its index reads every frame's detections from
`<video_data>/<name>/tasti_labels.csv` and looks up only the frames it samples, and its I-frame anchors come from
`<video_data>/<name>/video.mp4`. Its notebooks charge the detector as (frames sampled) x (measured per-frame detector
time). This script builds those inputs for one dataset's test split, without changing Seiden's code:

  video.mp4         the test clips concatenated in manifest order with stream copy (keeps every clip's I-frames).
  tasti_labels.csv  the reference detector's boxes (002_naive/detection.jsonl), one row per box, frames renumbered
                    globally; columns frame, object_name (car), confidence, xmin, ymin, xmax, ymax.
  manifest.json     clip order and frame offsets, total frames, and the reference detector's measured time per frame
                    (mean of the 'detect' op over every test frame, 002_naive/detection_runtime.jsonl) that the
                    Seiden run charges per sampled frame.

Run inside ace's polyis container: python measurements/r7o3_seiden_prepare.py --datasets caldot1-y05
"""
import argparse
import csv
import json
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


def read_jsonl(path: Path) -> list:
    """Every line of a JSONL file, parsed."""
    with open(path) as f:
        return [json.loads(line) for line in f]


def prepare(dataset: str, out_root: Path) -> dict:
    """Write video.mp4, tasti_labels.csv, and manifest.json for one dataset's test split; returns the manifest."""
    out = out_root / dataset
    out.mkdir(parents=True, exist_ok=True)
    videos = list_split_videos(dataset, 'test')
    clips, detect_ms, offset = [], [], 0
    with open(out / 'tasti_labels.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'object_name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax'])
        for video in videos:
            # The reference detector's per-frame boxes and timings for this clip.
            frames = read_jsonl(cache.exec(dataset, 'naive', video, 'detection.jsonl'))
            runtime = read_jsonl(cache.exec(dataset, 'naive', video, 'detection_runtime.jsonl'))
            assert len(frames) == len(runtime), f'{dataset} {video}: {len(frames)} frames, {len(runtime)} timings'
            # Frames must be dense and in order so global numbering is offset + frame_idx.
            assert [r['frame_idx'] for r in frames] == list(range(len(frames))), f'{dataset} {video}: frames not dense'
            for r in frames:
                for x1, y1, x2, y2, score in r['detections']:
                    writer.writerow([offset + r['frame_idx'], 'car', score, x1, y1, x2, y2])
            # Detector time per frame (ms), the 'detect' op of each frame's timing record.
            detect_ms += [next(op['time'] for op in ops if op['op'] == 'detect') for ops in runtime]
            clips.append(dict(video=video, offset=offset, frames=len(frames)))
            offset += len(frames)
    # Concatenate the clips in the same order with stream copy, so each clip keeps its own I-frames.
    listing = out / 'concat.txt'
    listing.write_text(''.join(f"file '{store.dataset(dataset, 'test', c['video'])}'\n" for c in clips))
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', str(listing),
                    '-c', 'copy', str(out / 'video.mp4')], check=True)
    # The concatenated video must hold exactly the labeled frames.
    probed = subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0', '-show_entries',
                             'stream=nb_read_frames', '-of', 'csv=p=0', str(out / 'video.mp4')],
                            capture_output=True, text=True, check=True).stdout.strip()
    assert int(probed) == offset, f'{dataset}: video.mp4 has {probed} frames, labels cover {offset}'
    manifest = dict(dataset=dataset, split='test', frames=offset, clips=clips,
                    detector_ms_per_frame=sum(detect_ms) / len(detect_ms),
                    detector_timing_source='002_naive/detection_runtime.jsonl, mean detect op over all test frames')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f"{dataset}: {offset} frames in {len(clips)} clips, detector {manifest['detector_ms_per_frame']:.2f} ms/frame",
          flush=True)
    return manifest


def main():
    args = parse_args()
    for dataset in args.datasets:
        prepare(dataset, args.out)


if __name__ == '__main__':
    main()
