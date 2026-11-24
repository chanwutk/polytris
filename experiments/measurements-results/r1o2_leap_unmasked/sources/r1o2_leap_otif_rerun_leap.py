"""R1.O2: LEAP's inference on one dataset's test clips, with an export of the state LEAP keeps per clip.

Runs inside the LEAP container (python 3.10, the scored container's conda env) with the working directory set to a
copy of LEAP's checkout, because LEAP reads masks, re-identification configs, and pretrained weights relative to it
and writes visualizations there. measurements-gpu/r1o2_leap_otif_rerun.py builds the container and calls this.

LEAP's own code is not changed. This script repeats the multi-video block of LEAP's main.py (its `__main__`: seed,
clear_dir, then run_session on every .mp4 of --video_dir in sorted file-name order, in one process, so the state
LEAP's global video_details carries from one clip to the next is the same as in the scored runs) and, after each
run_session returns, reads video_details and writes:

  objects       per predicted object: LEAP's id, the parsed interval [start, end), motion-pattern index, and class
                id (parse_intervals' result, as LEAP saves in <clip>_results.json), the raw interval list and the
                matched-pattern dict of resolved_tuple, and the sparse observations resolved_tuple holds, one
                [x1, y1, x2, y2, class_id, frame] per sampled frame on which the object was detected
  frames_sampled  the frames LEAP sampled in this clip (video_details.frame_sampled grows across clips; the slice
                this clip appended)
  timings       LEAP's own timers for the clip (as in its speed.tsv) and this script's wall clock of run_session
  patterns      the motion patterns LEAP loaded (<clip>_tracks.json), stored once per dataset with a hash per clip

The export is two gzipped JSON-lines files: <prefix>.jsonl.gz with one line per clip, and <prefix>_run.jsonl.gz with
one line holding LEAP's arguments, the loaded config, the clip order, the total wall clock, and the motion patterns.
Object crops (resolved_tuple's third field) are not exported.
"""
import argparse
import datetime
import gzip
import hashlib
import json
import os
import sys
import time


# LEAP's arguments shared by every dataset in the scored runs (from the run logs: --active_log prints 'Sampled
# Frame', the test split, and one visible GPU, which is cuda:0 inside the container).
COMMON_LEAP_ARGS = ['--type', 'test', '--active_log', '--device', '0']


def parse_args():
    parser = argparse.ArgumentParser(description="R1.O2: rerun LEAP's inference on one dataset and export its state")
    parser.add_argument('--config', required=True)
    parser.add_argument('--weight', required=True)
    parser.add_argument('--reid-weight', required=True)
    parser.add_argument('--motion-dir', default=None)
    parser.add_argument('--retinanet', action='store_true')
    parser.add_argument('--no-mask', action='store_true')
    parser.add_argument('--video-dir', required=True)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--export-prefix', required=True)
    return parser.parse_args()


def leap_argv(args) -> list[str]:
    """The argument list of LEAP's main.py for this run, as the scored runs passed it."""
    # Paths of the run's inputs and of LEAP's own outputs (logs and parsed results go to the work directory).
    argv = ['--config', args.config, '--weight', args.weight, '--reid_weight', args.reid_weight,
            '--video_dir', args.video_dir, '--log_dir', os.path.join(args.work_dir, 'logs'),
            '--output_dir', os.path.join(args.work_dir, 'outputs'), *COMMON_LEAP_ARGS]
    # CalDoT used LEAP's default motion directory (fixed_files/preprocessed); the others a separate one.
    if args.motion_dir:
        argv += ['--motion_dir', args.motion_dir]
    # B3D ran RetinaNet instead of YOLOv5x.
    if args.retinanet:
        argv += ['--use_retinanet']
    # LEAP's store_false option disables its otherwise-enabled image mask.
    if getattr(args, 'no_mask', False):
        argv += ['--use_mask']
    return argv


def plain(value):
    """JSON fallback for NumPy scalars and arrays in LEAP's state."""
    # NumPy arrays become lists, NumPy scalars Python numbers.
    if hasattr(value, 'tolist'):
        return value.tolist()
    # Anything else is a bug in the export, not something to stringify silently.
    raise TypeError(f'cannot export {type(value)}')


def digest(value) -> str:
    """SHA-256 of a JSON-serializable value in canonical form."""
    # Canonical JSON (sorted keys, no spaces) so equal values hash equally.
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), default=plain).encode()).hexdigest()


def export_clip(video_details, video_name: str, video_file: str, sampled_before: int, wall_s: float,
                tracks_path: str) -> tuple[dict, list]:
    """One clip's export from LEAP's global state right after run_session returned, and its motion patterns."""
    # run_session's parse_intervals result: {id: [[start, end], pattern index, class id]}.
    parsed = video_details.return_tuple
    # resolved_tuple: {id: [observations, matched patterns, crops, intervals]}; crops are not exported.
    resolved = video_details.resolved_tuple
    # Both describe the same objects.
    assert set(parsed) == set(resolved), f'{video_name}: parsed and resolved objects differ'
    # One record per object, in id order.
    objects = [dict(id=int(i), interval=parsed[i][0], pattern=parsed[i][1], class_id=parsed[i][2],
                    observations=resolved[i][0], matched=resolved[i][1], intervals_raw=resolved[i][3])
               for i in sorted(resolved)]
    # The motion patterns LEAP wrote for this clip (the exporter reads them by pattern index).
    with open(tracks_path) as f:
        patterns = json.load(f)
    # LEAP's per-clip timers (seconds), as it writes them to speed.tsv, and this script's wall clock.
    timings = dict(run_session_wall_s=wall_s, detector_s=video_details.detector_time,
                   differencer_s=video_details.frame_differencer_time, reid_s=video_details.reid_time,
                   match_s=video_details.match_time, decode_s=video_details.decode_time,
                   frames_filtered=video_details.differencor)
    # The slice of frame_sampled this clip appended.
    clip = dict(video_name=video_name, video_file=video_file, frames_sampled=video_details.frame_sampled[sampled_before:],
                timings=timings, patterns_sha256=digest(patterns), objects=objects)
    return clip, patterns


def main():
    # Command-line options of this wrapper.
    args = parse_args()
    # LEAP's modules import from the working directory, which must be the LEAP checkout copy.
    sys.path.insert(0, os.getcwd())
    # LEAP's entry module and its global state (main.py imports video_details from settings.settings).
    import main as leap_main
    from loguru import logger
    from settings.settings import video_details
    # LEAP's parser and config loader, exactly as its __main__ uses them.
    argv = leap_argv(args)
    leap_args = leap_main.make_parser().parse_args(argv)
    assert not getattr(args, 'no_mask', False) or not leap_args.use_mask
    cfgs = leap_main.getYaml(leap_args.config)
    # The config as loaded, before the loop overwrites video_name and video_path.
    config = dict(cfgs)
    # LEAP's log file, named by hour as its __main__ names it.
    os.makedirs(leap_args.log_dir, exist_ok=True)
    logger.add(os.path.join(leap_args.log_dir, '%s.log' % (datetime.datetime.now().strftime('%Y-%m-%d_%H'))))
    # Seed and output folders, as in LEAP's __main__.
    leap_main.set_seed(cfgs['seed'])
    leap_main.clear_dir(cfgs, leap_args)
    logger.info('Clean Folders')
    # The clips in LEAP's order (sorted file names, so 0, 1, 10, 11, ..., 2, ...).
    video_files = sorted([f for f in os.listdir(leap_args.video_dir) if f.endswith('.mp4')])
    logger.info(f'Found {len(video_files)} videos in {leap_args.video_dir}')
    # Per-clip exports and the motion patterns, which must be the same for every clip.
    clips, patterns = [], None
    start = time.time()
    for f in video_files:
        # The clip's name and path, set as LEAP's __main__ sets them.
        video_name = os.path.splitext(f)[0]
        logger.info(f'Processing video: {video_name}')
        cfgs['video_name'] = video_name
        cfgs['video_path'] = os.path.join(leap_args.video_dir, f)
        # frame_sampled is never reset by LEAP; remember where this clip's frames begin.
        sampled_before = len(video_details.frame_sampled)
        # LEAP's own per-clip run, timed by wall clock.
        clip_start = time.time()
        leap_main.run_session(cfgs, leap_args, logger)
        wall_s = time.time() - clip_start
        # Read the state run_session left behind; the motion patterns are the file LEAP just wrote.
        tracks_path = os.path.join(leap_args.output_dir, 'parsed_results', cfgs['full_name'], f'{video_name}_tracks.json')
        clip, clip_patterns = export_clip(video_details, video_name, f, sampled_before, wall_s, tracks_path)
        clips.append(clip)
        # Every clip loads the same pattern file, so keep it once and check the rest against it.
        patterns = clip_patterns if patterns is None else patterns
        assert digest(clip_patterns) == clip['patterns_sha256'] == digest(patterns), f'{video_name}: patterns changed'
    # The run record: what LEAP was given, in which order it saw the clips, and the patterns every clip used.
    run = dict(leap_argv=argv, mask_enabled=bool(leap_args.use_mask), config=config, video_files=video_files, total_wall_s=time.time() - start,
               patterns_sha256=digest(patterns), patterns=patterns)
    # One line per clip, and one line for the run.
    os.makedirs(os.path.dirname(os.path.abspath(args.export_prefix)), exist_ok=True)
    with gzip.open(f'{args.export_prefix}.jsonl.gz', 'wt') as f:
        f.writelines(json.dumps(clip, default=plain) + '\n' for clip in clips)
    with gzip.open(f'{args.export_prefix}_run.jsonl.gz', 'wt') as f:
        f.write(json.dumps(run, default=plain) + '\n')
    print(f'wrote {args.export_prefix}*.jsonl.gz: {len(clips)} clips, {sum(len(c["objects"]) for c in clips)} objects')


if __name__ == '__main__':
    main()
