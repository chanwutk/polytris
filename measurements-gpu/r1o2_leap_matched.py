"""Rebuild LEAP preparation and extraction with each dataset's Tetris detector.

Host-side, on bombe. One invocation per dataset/GPU, in a detached tmux session.
Reuses the recorded environment, launcher, and stage timing harness; all new
artifacts and results have a separate root. Successful stages resume on restart.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace


# The existing host-side harness supplies the launcher, stage timing, and file formats.
SPEC = importlib.util.spec_from_file_location('leap_rerun', Path(__file__).with_name('r1o2_leap_otif_rerun.py'))
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)
ORIGINAL_SCRATCH, ORIGINAL_RAW = BASE.SCRATCH, BASE.RAW
SCRATCH = BASE.ROOT / 'scratch' / 'r1o2_leap_matched'
RAW = BASE.TETRIS / 'measurements-results' / 'r1o2_leap_matched' / 'raw'
ADAPTER = f'{BASE.LEAP_TOOLS}/run/matched_detector.py'
MOTION_RUNNER = f'{BASE.LEAP_TOOLS}/prep/motion_patterns.py'
# The adapter as the host sees it, for its hash in the run manifest.
ADAPTER_HOST = BASE.TETRIS / ADAPTER.removeprefix('/polyis/')
DATASETS = {'caldot1-y05': 'caldot1', 'caldot2-y05': 'caldot2', 'ams-y05': 'amsterdam'}


def parse_args():
    parser = argparse.ArgumentParser(description='Full LEAP rerun with Tetris detectors (dry run unless --run)')
    parser.add_argument('--dataset', required=True, choices=list(DATASETS))
    parser.add_argument('--gpu', required=True, type=int, choices=list(BASE.GPU_CPUS))
    parser.add_argument('--run', action='store_true')
    return parser.parse_args()


def configure(dataset):
    # Keep all new writes separate while reusing the original, read-only code and video inputs.
    BASE.SCRATCH, BASE.RAW, BASE.RUNNER = SCRATCH, RAW, ADAPTER
    c = dict(BASE.LEAP_DATASETS[dataset])
    c.update(labels=c['full_name'], workspace_motion=True)
    BASE.LEAP_DATASETS = {dataset: c}
    weight = f'/polyis-data/training/ultralytics/{DATASETS[dataset]}/weights/yolov5x6/weights/best.pt'
    BASE.DETECTOR_WEIGHTS = {'yolov5': f'models/yolov5/{dataset}.pt'}
    # Reuse stage command generation, replacing only the detector boundary and the new model directory name.
    original_command = BASE.leap_prep_command

    def command(ds, step):
        text, output = original_command(ds, step)
        if step.startswith('labels_'):
            text = text.replace('python video_labeling/video_tracker.py', f'python {ADAPTER} label')
            text = text.replace('--yolo_weight weights/yolov5x.pt', f'--yolo_weight {weight}')
        elif step == 'reid_train':
            new_output = output.rsplit('/', 1)[0] + '/' + c['full_name']
            text, output = text.replace(output, new_output), new_output
        elif step == 'motion_patterns':
            root = f'/scratch/prep/leap/{ds}'
            text = (f'python {MOTION_RUNNER} --video-name {c["full_name"]} '
                    f'--label-dir {root}/labels --normalized-label-dir {root}/motion-labels '
                    f'--output-dir {root}/motion_patterns')
        return text, output

    BASE.leap_prep_command = command
    BASE.common_mounts = lambda: [
        '-v', f'{SCRATCH}/inputs/leap-dataset:{BASE.LEAP_DATASET_MOUNT}:ro',
        '-v', f'{ORIGINAL_SCRATCH}/inputs/LEAP:/leap:ro',
        '-v', f'{ORIGINAL_SCRATCH}/videos:/scratch/videos:ro',
        '-v', f'/usr/bin/time:{BASE.GNU_TIME}:ro',
        '-e', 'PYTHONPATH=/leap', '-e', 'OMP_NUM_THREADS=10', '-e', 'MKL_NUM_THREADS=10']
    # Comparisons with the old checkpoint's preparation are not an expected equality test.
    BASE.compare_prep = lambda *args: None
    return c, weight


def setup(dataset, c, weight):
    # Resolve and hash the exact mirrored checkpoint, rather than relying on a filename.
    host_weight = BASE.ROOT / 'mirror' / weight.removeprefix('/')
    checkpoint_sha = hashlib.sha256(host_weight.read_bytes()).hexdigest()
    workspace = SCRATCH / 'inputs' / 'leap-dataset'
    for folder in ('configs', 'models/yolov5', 'labels', 'logs', 'motion_patterns'):
        (workspace / folder).mkdir(parents=True, exist_ok=True)
    # Copy only configuration, and link learned inputs to this rerun's fresh preparation outputs.
    config = ORIGINAL_SCRATCH / 'inputs' / 'leap-dataset' / 'configs' / c['config']
    shutil.copyfile(config, workspace / 'configs' / c['config'])
    links = {
        workspace / 'models/yolov5' / f'{dataset}.pt': weight,
        workspace / 'labels' / c['full_name']: f"/scratch/prep/leap/{dataset}/labels/{c['full_name']}",
        workspace / 'logs' / c['full_name']: f"/scratch/prep/leap/{dataset}/reid/{c['full_name']}",
        workspace / 'motion_patterns' / c['full_name']: f"/scratch/prep/leap/{dataset}/motion_patterns/{c['full_name']}",
    }
    for path, target in links.items():
        if path.is_symlink():
            assert os.readlink(path) == target, f'Unexpected input link: {path}'
        else:
            path.symlink_to(target)
    # Keep the original scored exports available for accuracy and timing comparisons.
    (RAW / dataset).mkdir(parents=True, exist_ok=True)
    for name in ('leap_scored.jsonl.gz', 'leap_scored_run.jsonl.gz', 'leap_scored_export.jsonl.gz'):
        shutil.copyfile(ORIGINAL_RAW / dataset / name, RAW / dataset / name)
    manifest = json.loads((ORIGINAL_RAW / 'manifest.json').read_text())
    manifest['matched_detector'] = dict(
        checkpoint=weight, sha256=checkpoint_sha, dataset=dataset,
        config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        adapter_sha256=hashlib.sha256(ADAPTER_HOST.read_bytes()).hexdigest(),
        inference='polyis/models/ultralytics.py defaults; FP32; class 0 -> LEAP class 2',
        preparation='fresh labels, crops, re-identification training, and motion patterns; no distillation')
    manifest['motion_pattern_policy'] = 'clip-local training track IDs are namespaced before original clustering'
    manifest['export_policy'] = 'drop negative frame writes before original pattern export; clip video ends in p140'
    # Dataset manifests avoid races between the independent queues; the common manifest is environment provenance.
    existing = RAW / dataset / 'manifest.json'
    if existing.exists():
        assert json.loads(existing.read_text())['matched_detector'] == manifest['matched_detector'], 'Run inputs changed'
    BASE.write_json(existing, manifest)
    BASE.write_json(RAW / 'manifest.json', {k: v for k, v in manifest.items() if k != 'matched_detector'})
    print(json.dumps(manifest['matched_detector'], indent=2), flush=True)


def check_stage(dataset, step, output):
    # Fail the queue when a stage silently exits without its required artifacts.
    if step.startswith('labels_'):
        split = step.removeprefix('labels_')
        expected = {p.stem for p in (ORIGINAL_SCRATCH / 'videos' / dataset / split).glob('*.mp4')}
        files = list(output.glob(f'*/{split}/label_*.txt'))
        assert {p.stem.removeprefix('label_') for p in files} == expected, f'{step}: missing clip labels'
        assert any(p.stat().st_size for p in files), f'{step}: all labels empty'
    elif step == 'reid_data':
        for folder in ('image_train', 'image_query', 'image_test'):
            assert any((output / folder).glob('*.jpg')), f'Missing re-identification images: {folder}'
    elif step == 'reid_train':
        assert (output / 'model_final.pth').exists(), 'Missing trained re-identification model'
    else:
        patterns = list(output.glob('*_tracks_clustered.npy'))
        assert patterns, 'Missing motion patterns'
        # Inspect the NumPy header without needing NumPy in the host-side launcher or unpickling objects.
        for path in patterns:
            with path.open('rb') as f:
                assert f.read(6) == b'\x93NUMPY', f'Invalid motion-pattern file: {path}'
                major, _ = f.read(2)
                assert major in (1, 2, 3), f'Unsupported NumPy format: {major}'
                size = int.from_bytes(f.read(2 if major == 1 else 4), 'little')
                header = ast.literal_eval(f.read(size).decode('utf-8' if major == 3 else 'latin1'))
            assert header['shape'] and header['shape'][0] > 0, f'Empty motion-pattern library: {path}'


def needs_extraction(launch_path, motion_changed):
    # A failed extraction or changed library must not be hidden by an existing launch record.
    if not launch_path.exists():
        return True
    launch = json.loads(launch_path.read_text())
    return motion_changed or launch['exit'] != 0 or launch['contaminated']


def run(args):
    c, weight = configure(args.dataset)
    if not args.run:
        print(f'Isolated outputs: {RAW}; scratch: {SCRATCH}; checkpoint: {weight}')
        for step in BASE.LEAP_PREP_STEPS:
            print(step, BASE.leap_prep_command(args.dataset, step))
        return
    setup(args.dataset, c, weight)
    stage_args = SimpleNamespace(datasets=[args.dataset], gpu=args.gpu, run=True,
                                 force=False, system='leap', steps=None)
    motion_changed = False
    for step in BASE.LEAP_PREP_STEPS:
        stage_args.steps = [step]
        record_path = RAW / 'prep/leap' / args.dataset / f'{step}.json'
        previous = json.loads(record_path.read_text()) if record_path.exists() else None
        stage_args.force = bool(previous and previous.get('contaminated'))
        if step == 'motion_patterns':
            command, _ = BASE.leap_prep_command(args.dataset, step)
            stage_args.force |= bool(previous and previous['command'] != command)
            motion_changed = previous is None or previous['exit'] != 0 or stage_args.force
        BASE.prep(stage_args)
        record = json.loads(record_path.read_text())
        assert record['exit'] == 0 and not record['contaminated'], f'Failed or contaminated stage: {step}'
        check_stage(args.dataset, step, BASE.host_path(record['output'], None))
    # Extraction always reads this run's freshly trained re-identification model and patterns through the links above.
    launch_path = RAW / args.dataset / 'leap_rerun_launch.json'
    if needs_extraction(launch_path, motion_changed):
        BASE.leap(stage_args)
    launch = json.loads(launch_path.read_text())
    assert launch['exit'] == 0 and not launch['contaminated'], 'Failed or contaminated extraction'
    # Reuse the original exporter without touching any of the scored files.
    parsed = Path(launch['work_dir']) / 'outputs/parsed_results' / c['full_name']
    command = ['docker', 'run', '--rm', '--cpus', '2', '-v', f'{SCRATCH}:/scratch',
               '-v', f'{ORIGINAL_SCRATCH}/inputs/scripts:/exporter:ro', '-v', f'{BASE.TETRIS}:/polyis',
               BASE.LEAP_IMAGE, 'python', f'{BASE.LEAP_TOOLS}/eval/export_tracks.py',
               '--exporter', '/exporter/export-leap-outputs.py',
               '--parsed-dir', '/scratch/' + str(parsed.relative_to(SCRATCH)),
               '--out', '/polyis/' + str((RAW / args.dataset / 'leap_rerun_export.jsonl.gz').relative_to(BASE.TETRIS)),
               '--clip-negative-starts']
    subprocess.run(command, check=True)
    BASE.gpus(stage_args)
    # Accuracy is CPU-only and runs on ace, where the original ground truth and TrackEval cache live.
    score = ['docker', 'exec', '-w', '/polyis', 'polyis', 'python', 'measurements/r1o2_leap_otif_rerun.py',
             '--datasets', args.dataset, '--raw-dir', '/polyis/' + str(RAW.relative_to(BASE.TETRIS)),
             '--output-dir', '/polyis/measurements-results/r1o2_leap_matched/summary', '--workers', '3']
    subprocess.run(['ssh', 'ace', shlex.join(score)], check=True)
    BASE.write_json(RAW / args.dataset / 'complete.json', dict(dataset=args.dataset, gpu=args.gpu, scored=True))


if __name__ == '__main__':
    run(parse_args())
