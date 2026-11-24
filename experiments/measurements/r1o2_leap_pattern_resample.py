"""R1.O2: LEAP's assigned-pattern HOTA with every pattern resampled to its interval instead of truncated.

LEAP's received exporter (export-leap-outputs.py) linearly resamples a motion pattern shorter than the object's
predicted interval but truncates a longer one, so the reconstructed box stops partway along the pattern. This script
re-exports the reported LEAP runs of all seven datasets from their saved per-clip outputs (parsed_results, copied
from bombe to RESULTS/inputs), once with the received exporter and once with export_tracks.py --resample-patterns,
and scores the pattern reconstruction of each with the canonical (padded) TrackEval window.

Checks before any new number is used:
  - the received re-export equals the saved export of the reported run, clip by clip;
  - its pattern HOTA reproduces the recorded pattern HOTA of the reported run.
The raw and linear reconstructions do not use patterns, so their recorded HOTA is reused.

Run in ace's polyis container from /polyis:  python experiments/measurements/r1o2_leap_pattern_resample.py
Output: RESULTS/results.json (per dataset: interval/pattern counts, recorded, received, and resampled HOTA, and the
best reconstruction).
"""
import argparse
import gzip
import json
import multiprocessing as mp
import subprocess
import sys
from pathlib import Path

import duckdb

from experiments.evaluation.manifests import list_split_videos
from experiments.evaluation.p140_otif_transform import extract_video_id, transform_tracking_json
from experiments.measurements.common import output_path, write_json
from experiments.measurements.r1o2_leap_otif_rerun import run_trackeval

ROOT = Path(__file__).resolve().parents[1]
MEASURED = ROOT / 'measurements-results'
RESULTS = MEASURED / 'r1o2_leap_pattern_resample'
EXPORT_TOOL = ROOT / 'leap/eval/export_tracks.py'
EXPORTER = RESULTS / 'inputs' / 'export-leap-outputs.py'
# Per dataset: label, the reported run's saved export, whether that export dropped pre-video frames, and where the
# reported run's padded HOTA per reconstruction is recorded (file, key path to the {reading: {HOTA, ...}} section).
DATASETS = {
    'caldot1-y05': dict(label='CalDoT1', export='r1o2_leap_unmasked/raw/caldot1-y05/leap_rerun_export.jsonl.gz',
                        clip_negative_starts=True, recorded=('r1o2_leap_unmasked/comparison/results.json',
                                                             ('datasets', 'CalDoT1', 'versions', 'unmasked', 'hota'))),
    'caldot2-y05': dict(label='CalDoT2', export='r1o2_leap_unmasked/raw/caldot2-y05/leap_rerun_export.jsonl.gz',
                        clip_negative_starts=True, recorded=('r1o2_leap_unmasked/comparison/results.json',
                                                             ('datasets', 'CalDoT2', 'versions', 'unmasked', 'hota'))),
    'ams-y05': dict(label='Amsterdam', export='r1o2_leap_regions_v2/raw/ams-y05/leap_rerun_export.jsonl.gz',
                    clip_negative_starts=True, recorded=('r1o2_leap_regions_v2/comparison/results.json',
                                                         ('datasets', 'Amsterdam', 'versions', 'regions_v2', 'hota'))),
    **{f'jnc{c}': dict(label=f'B3D{i}', export=f'r1o2_leap_otif_rerun/raw/jnc{c}/leap_rerun_export.jsonl.gz',
                       clip_negative_starts=False,
                       recorded=('b3d_human_gt/leap_rerun/results-jnc0-jnc2-jnc6-jnc7.json',
                                 ('datasets', f'B3D{i}', 'hota')))
       for i, c in enumerate('0267', 1)},
}
# The B3D record keeps four decimals; the others keep full precision.
TOLERANCE = {'B3D': 5e-5, 'other': 1e-9}
EXPORT_COLUMNS = "{'video_name': 'VARCHAR', 'frames': 'JSON'}"


def parse_args():
    parser = argparse.ArgumentParser(description="R1.O2: LEAP pattern HOTA with patterns resampled to their intervals")
    parser.add_argument('--datasets', nargs='+', default=list(DATASETS))
    parser.add_argument('--workers', type=int, default=7)
    return parser.parse_args()


def recorded_hota(dataset: str) -> dict:
    """The reported run's recorded HOTA section ({reading: {'HOTA': ...}}) for one dataset."""
    # Walk the key path inside the recording file.
    path, keys = DATASETS[dataset]['recorded']
    section = json.loads((MEASURED / path).read_text())
    for key in keys:
        section = section[key]
    return section


def export(dataset: str, variant: str) -> Path:
    """Export the dataset's saved parsed results with the received exporter or with resampling; return the file."""
    # Output next to the other results, one gzipped JSON line per clip.
    out = RESULTS / 'exports' / dataset / f'{variant}.jsonl.gz'
    command = [sys.executable, str(EXPORT_TOOL), '--exporter', str(EXPORTER),
               '--parsed-dir', str(RESULTS / 'inputs' / dataset / 'parsed_results'), '--out', str(out)]
    # The reported run's pre-video frame policy, and the resampling option for the new variant.
    if DATASETS[dataset]['clip_negative_starts']:
        command.append('--clip-negative-starts')
    if variant == 'resampled':
        command.append('--resample-patterns')
    subprocess.run(command, check=True)
    return out


def read_export(path: Path) -> dict[str, list]:
    """{LEAP clip name: frame list} of an export file."""
    # Read every clip line; a clip name may appear only once.
    with gzip.open(path, 'rt') as f:
        rows = [json.loads(line) for line in f]
    names = [row['video_name'] for row in rows]
    assert len(names) == len(set(names)), f'{path}: repeated clips'
    return {row['video_name']: row['frames'] for row in rows}


def interval_counts(dataset: str) -> dict:
    """How many objects' patterns are shorter than, as long as, or longer than their intervals, and how many intervals
    are empty or inverted (the counts the resampling change acts on)."""
    # Every clip's resolved objects ({id: [[start, end], pattern index, ...]}) and pattern lengths, joined in DuckDB.
    parsed = RESULTS / 'inputs' / dataset / 'parsed_results'
    with duckdb.connect() as con:
        objects, patterns = [], []
        for results in sorted(parsed.glob('*_results.json')):
            clip = results.name[:-len('_results.json')]
            objects += [(clip, obj, v[0][0], v[0][1], v[1]) for obj, v in json.loads(results.read_text()).items()]
            tracks = json.loads((parsed / f'{clip}_tracks.json').read_text())
            patterns += [(clip, index, len(pattern)) for index, pattern in enumerate(tracks)]
        con.execute('CREATE TABLE objects (clip VARCHAR, id VARCHAR, s BIGINT, e BIGINT, pattern BIGINT)')
        con.executemany('INSERT INTO objects VALUES (?, ?, ?, ?, ?)', objects)
        con.execute('CREATE TABLE patterns (clip VARCHAR, pattern BIGINT, boxes BIGINT)')
        con.executemany('INSERT INTO patterns VALUES (?, ?, ?)', patterns)
        row = con.execute("""
            SELECT count(*) AS objects,
                   count(*) FILTER (o.e <= o.s) AS empty_or_inverted,
                   count(*) FILTER (o.e > o.s AND p.boxes < o.e - o.s) AS shorter,
                   count(*) FILTER (o.e > o.s AND p.boxes = o.e - o.s) AS equal,
                   count(*) FILTER (o.e > o.s AND p.boxes > o.e - o.s) AS longer,
                   count(*) FILTER (p.boxes IS NULL) AS missing_pattern
            FROM objects o LEFT JOIN patterns p ON p.clip = o.clip AND p.pattern = o.pattern
        """).fetchone()
        names = ('objects', 'empty_or_inverted', 'shorter', 'equal', 'longer', 'missing_pattern')
    return dict(zip(names, row))


def write_tracks(dataset: str, variant: str, frames: dict[str, list]) -> dict:
    """Write one variant's pattern tracks per test clip through p140's LEAP transform; return its TrackEval task."""
    # Our test clips and LEAP's names for them (te07.mp4 is LEAP's 7).
    videos = sorted(list_split_videos(dataset, 'test'))
    names = {str(extract_video_id(video)): video for video in videos}
    assert set(frames) == set(names), f'{dataset}: export clips {sorted(frames)} differ from test clips {sorted(names)}'
    base = RESULTS / 'tracks' / dataset
    for leap_name, video in names.items():
        # The clip's frame list as a JSON file, then the transform that unwraps starts and clips ends to the video.
        source = base / 'exports' / variant / f'{leap_name}.json'
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(json.dumps(frames[leap_name]))
        transform_tracking_json(str(source), str(base / variant / video / 'tracking.jsonl'), dataset, video,
                                unwrap_leap=True)
    # One TrackEval task against the root dataset's ground truth, as the reported runs were scored.
    gt_dir = str(Path(output_path(dataset, 'ground truth', '', videos[0])).parents[2])
    return dict(dataset=dataset, variant=variant, videos=videos, gt_dir=gt_dir,
                gt_file=str(Path('003_groundtruth', 'tracking.jsonl')), track_dir=str(base / variant),
                track_file='tracking.jsonl', output_dir=str(base / 'trackeval' / variant))


def main():
    args = parse_args()
    tasks, sections = [], {}
    for dataset in args.datasets:
        config = DATASETS[dataset]
        # Re-export with both exporters.
        received = read_export(export(dataset, 'received'))
        resampled = read_export(export(dataset, 'resampled'))
        # The received re-export must be the reported run's saved export, clip by clip.
        saved = read_export(MEASURED / config['export'])
        assert received.keys() == saved.keys(), f'{dataset}: re-exported clips differ from the saved export'
        changed = sorted(name for name in saved if received[name] != saved[name])
        assert not changed, f'{dataset}: re-export differs from the saved export on clips {changed}'
        # Pattern tracks and TrackEval tasks for both variants.
        tasks += [write_tracks(dataset, 'received', received), write_tracks(dataset, 'resampled', resampled)]
        sections[dataset] = dict(label=config['label'], counts=interval_counts(dataset),
                                 clips_changed_by_resampling=sum(received[n] != resampled[n] for n in received))
        print(f"{config['label']}: exports ready; {sections[dataset]['counts']}", flush=True)
    # Score every variant in parallel.
    with mp.Pool(max(1, min(args.workers, len(tasks)))) as pool:
        measured = pool.map(run_trackeval, tasks)
    for dataset, section in sections.items():
        # The recorded HOTA of the reported run, and the two pattern readings.
        recorded = recorded_hota(dataset)
        scores = {row['variant']: row['metrics']['HOTA'] for row in measured if row['dataset'] == dataset}
        tolerance = TOLERANCE['B3D' if section['label'].startswith('B3D') else 'other']
        difference = scores['received']['HOTA'] - recorded['pattern']['HOTA']
        assert abs(difference) <= tolerance, \
            f"{section['label']}: received pattern HOTA {scores['received']['HOTA']} does not reproduce " \
            f"the recorded {recorded['pattern']['HOTA']}"
        # The best reconstruction with resampled patterns (raw never wins; linear is unchanged).
        best = max(('linear', 'pattern'),
                   key=lambda r: recorded['linear']['HOTA'] if r == 'linear' else scores['resampled']['HOTA'])
        section.update(recorded={r: recorded[r]['HOTA'] for r in ('raw', 'linear', 'pattern')},
                       received_pattern=scores['received'], resampled_pattern=scores['resampled'],
                       received_minus_recorded=difference, best_reconstruction=best,
                       best_hota=recorded['linear']['HOTA'] if best == 'linear' else scores['resampled']['HOTA'],
                       recorded_best_reconstruction=max(('linear', 'pattern'), key=lambda r: recorded[r]['HOTA']))
        print(f"{section['label']}: pattern HOTA recorded {recorded['pattern']['HOTA']:.6f}, "
              f"received {scores['received']['HOTA']:.6f}, resampled {scores['resampled']['HOTA']:.6f}; "
              f"linear {recorded['linear']['HOTA']:.6f}; best {best}", flush=True)
    # Results with the inputs that produced them.
    write_json(RESULTS, 'results.json', dict(
        method=dict(exporter=str(EXPORTER.relative_to(ROOT)), export_tool=str(EXPORT_TOOL.relative_to(ROOT.parent)),
                    scoring='TrackEval HOTA, canonical padded window, root ground truth (B3D: human-verified)',
                    reused='raw and linear HOTA from the recorded reported runs (they do not use patterns)'),
        datasets={s['label']: s for s in sections.values()}))


if __name__ == '__main__':
    main()
