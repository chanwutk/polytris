#!/usr/local/bin/python
"""R9.O2: re-timing plan for the preparation work that has no timing records. DRY RUN unless --run.

Prints, per dataset, the commands that would time the following, CPU-only steps first:

  p110           scoring of every validation configuration (CPU; the measurement only estimates it
                 from the test split's scoring)
  p012, p013     training-data creation and pruning (GPU; p012's seek and decode are untimed and
                 p013 writes no timings)
  detector_load  detector construction as p040 does it once per configuration (GPU; not recorded)
  p011           reference detection on the training split (GPU; optional, since per-frame times
                 exist and only detector loading is missing)
  p016           mistrack profiling (CPU, 40 workers; optional, since it takes seconds)
  otif, leap     OTIF's and LEAP's preparation (printed only)

Every polyis command runs with CONFIG pointing at a copy of configs/global.yaml whose CACHE_DIR is
<scratch>/cache and whose DATASETS holds only the dataset being timed, so it writes only under the
scratch directory. Inputs that a step needs from earlier stages (p011's detections, the validation
tracking outputs) are copied from the real cache first; videos and detector weights are read in
place. Every executable step lists the paths it writes, and --run refuses a step that would write
outside the scratch directory; nothing under /polyis-cache, /polyis-data, /otif-dataset, or
/sota-results is written.

OTIF and LEAP steps are printed, never executed: they run in those projects' own environments
outside this container, their arguments come from the projects' READMEs, and each is marked with
what must be checked against the reported runs first. Their outputs are directed to
--host-scratch, a host path, after copying the inputs there.

With --run, the polyis steps execute in order; each step's output goes to <scratch>/logs/, and its
wall clock, exit code, and a GPU snapshot before and after are appended to <scratch>/timings.jsonl
(detector_load also appends its per-load times to <scratch>/detector_load.jsonl). A GPU step
refuses to start while the GPU is in use (cached timings were taken on idle GPUs) unless
--allow-busy is given. measurements/r9o2_preparation_cost.py does not read these files yet.
"""
import argparse
import json
import os
import shlex
import socket
import subprocess
import time
from pathlib import Path

import yaml

from evaluation.manifests import list_split_videos
from evaluation.p110_accuracy_compute import resolve_groundtruth_tracking_dataset
from measurements.common import DATASET_LABELS
from polyis.io import cache
from polyis.utilities import get_config


# Roots of real data: read by the plan, never written.
PROTECTED_ROOTS = ('/polyis-cache', '/polyis-data', '/otif-dataset', '/sota-results')
# OTIF's dataset names for the datasets that have an OTIF-format copy under /otif-dataset (B3D has none).
OTIF_NAMES = {'caldot1-y05': 'caldot1', 'caldot2-y05': 'caldot2', 'ams-y05': 'amsterdam'}
# OTIF's per-dataset tracker thresholds from its README's tuning commands.
OTIF_THRESHOLDS = {'caldot1': '{"Thresholds":[0,1,1,1,1,1,1]}', 'caldot2': '{"Thresholds":[0,1,1,1,1,1,1]}',
                   'amsterdam': '{"Thresholds":[0,0,1,1,1,1,1]}'}
# Host paths on ace of the baselines' code and OTIF's dataset (outside this container).
OTIF_HOST_ROOT = '/work/cwkt/projects/_archive/otif'
OTIF_HOST_DATASET = '/work/cwkt/data/otif-dataset'
LEAP_HOST_ROOT = '/work/alenachao/tunnel/work/LEAP'
# Steps this plan knows, in execution order (CPU-only steps first), and the ones run by default.
STEPS = ('p110', 'p012', 'p013', 'detector_load', 'p011', 'p016', 'otif', 'leap')
DEFAULT_STEPS = ('p110', 'p012', 'p013', 'detector_load', 'otif', 'leap')
# Steps that read p011's detections from the cache.
NEEDS_DETECTIONS = ('p012', 'p016')
# A GPU counts as busy above either level (nvidia-smi units: percent, MiB).
BUSY_UTILIZATION_PERCENT = 5
BUSY_MEMORY_MIB = 500
# Detector constructions per detector_load run: the first includes CUDA start-up, the rest are warm, as in
# p040's long-lived workers that construct one detector per configuration.
DETECTOR_LOADS = 3
# The detector_load program: construct the dataset's detector as p040 does and append the times to a JSONL file.
DETECTOR_LOAD_PROGRAM = '''
import json, sys, time
import torch
import polyis.models.detector
dataset, out, loads = sys.argv[1], sys.argv[2], int(sys.argv[3])
start = time.time()
torch.zeros(1, device='cuda')
torch.cuda.synchronize()
cuda_init_s = time.time() - start
times = []
for _ in range(loads):
    start = time.time()
    detector = polyis.models.detector.get_detector(dataset, 0, batch_size=16, num_images=0)
    torch.cuda.synchronize()
    times.append(time.time() - start)
    del detector
    torch.cuda.empty_cache()
with open(out, 'a') as f:
    f.write(json.dumps(dict(dataset=dataset, cuda_init_s=cuda_init_s, load_s=times)) + '\\n')
'''


def parse_args():
    parser = argparse.ArgumentParser(description='R9.O2: re-time preparation stages into a scratch cache (dry run unless --run)')
    parser.add_argument('--datasets', nargs='+', default=list(DATASET_LABELS), choices=list(DATASET_LABELS))
    parser.add_argument('--steps', nargs='+', default=list(DEFAULT_STEPS), choices=STEPS)
    parser.add_argument('--gpu', type=int, default=4)
    parser.add_argument('--scratch', default='/tmp/r9o2-retime')
    parser.add_argument('--host-scratch', default='/work/cwkt/scratch/r9o2-retime')
    parser.add_argument('--allow-busy', action='store_true')
    parser.add_argument('--run', action='store_true')
    return parser.parse_args()


def check_scratch(scratch: str, cache_dir: str) -> Path:
    """The scratch directory as a Path, after refusing any location inside real data or the real cache."""
    # Keep the path as given; checks use its resolved form.
    path = Path(scratch)
    # A relative path would depend on the working directory.
    assert path.is_absolute(), f'scratch {scratch} must be an absolute path'
    # Resolve symbolic links and '..' before comparing.
    resolved = path.resolve()
    # Refuse the real roots themselves and anything below them.
    for root in (*PROTECTED_ROOTS, cache_dir):
        protected = Path(root).resolve()
        assert resolved != protected and protected not in resolved.parents, f'scratch {scratch} is inside {root}'
    return path


def inside(path: str | Path, scratch: Path) -> bool:
    """Whether `path` is the scratch directory or below it, after resolving symbolic links and '..'."""
    # Compare resolved paths so links and '..' cannot escape.
    resolved, root = Path(path).resolve(), scratch.resolve()
    return resolved == root or root in resolved.parents


def scratch_config(dataset: str, scratch: Path) -> dict:
    """configs/global.yaml with the cache moved into the scratch directory and only `dataset` configured."""
    # The checked-in configuration, as the pipeline reads it.
    config = get_config()
    # Every cache path in polyis derives from DATA.CACHE_DIR.
    config['DATA']['CACHE_DIR'] = str(scratch / 'cache') + '/'
    # The tuning and scoring scripts loop over EXEC.DATASETS; one dataset per run keeps timings per dataset.
    config['EXEC']['DATASETS'] = [dataset]
    return config


def executable(step: str, dataset: str, argv: list[str], writes: list[Path], gpu: bool, cwd: str = '/polyis',
               env: dict | None = None, mkdir: list[Path] | None = None) -> dict:
    """One step that --run executes in this container."""
    # Paths are kept as strings so the plan prints and serializes as is.
    return dict(step=step, dataset=dataset, cwd=cwd, env=env or {}, argv=argv, writes=[str(p) for p in writes],
                mkdir=[str(p) for p in mkdir or []], gpu=gpu, manual=False)


def polyis_steps(dataset: str, steps: list[str], scratch: Path, gpu: int) -> list[dict]:
    """The executable polyis steps for one dataset, in STEPS order, with the copies their inputs need."""
    # p013 prunes what p012 creates; the cached training data is not copied (gigabytes per dataset).
    assert 'p013' not in steps or 'p012' in steps, 'p013 needs p012 in the same run'
    # The scratch cache every polyis step writes into.
    scratch_cache = scratch / 'cache'
    # CONFIG is resolved relative to configs/, and an absolute path overrides that directory. GPU steps see only the
    # chosen GPU and CPU steps none, so a CPU step cannot take a GPU by accident.
    config = str(scratch / 'configs' / f'{dataset}.yaml')
    gpu_env, cpu_env = dict(CONFIG=config, CUDA_VISIBLE_DEVICES=str(gpu)), dict(CONFIG=config, CUDA_VISIBLE_DEVICES='')
    # Steps in execution order.
    plan = []
    # p110 scores the validation tracking outputs against the reference's, both copied from the cache.
    if 'p110' in steps:
        # Where p110 reads the tracking outputs in the scratch cache.
        target = scratch_cache / dataset / 'execution'
        # p110 checks that the ground-truth dataset's execution directory exists, although validation does not read it.
        groundtruth = scratch_cache / resolve_groundtruth_tracking_dataset(dataset) / 'execution'
        # Copy every validation video's tracking.jsonl (the reference's and each configuration's), keeping relative paths.
        plan.append(executable('copy validation tracking outputs', dataset, cwd=str(cache.execution(dataset)),
                               argv=['find', *list_split_videos(dataset, 'valid'), '-name', 'tracking.jsonl', '-exec',
                                     'cp', '--parents', '-t', str(target), '{}', '+'],
                               writes=[target], mkdir=[target, groundtruth], gpu=False))
        # Score every validation configuration, as p110 --valid does in the pipeline.
        plan.append(executable('p110', dataset, ['python', 'evaluation/p110_accuracy_compute.py', '--valid'],
                               writes=[scratch_cache], env=cpu_env, gpu=False))
    # p012 and p016 read p011's detections; without a fresh p011 run, copy the cached ones.
    if any(s in steps for s in NEEDS_DETECTIONS) and 'p011' not in steps:
        # p011's output directory in the real cache, and its parent in the scratch cache.
        source = Path(cache.index(dataset, 'det'))
        target = scratch_cache / source.relative_to(Path(cache.root(dataset)).parent).parent
        plan.append(executable('copy cached p011 detections', dataset, ['cp', '-r', str(source), str(target)],
                               writes=[target], mkdir=[target], gpu=False))
    # The remaining polyis steps, each a whole-process wall clock: (argv, uses a GPU, paths written).
    load_times = scratch / 'detector_load.jsonl'
    commands = dict(
        p011=(['python', 'scripts/p011_tune_detect.py'], True, [scratch_cache]),
        p012=(['python', 'scripts/p012_tune_create_training_data.py'], True, [scratch_cache]),
        p013=(['python', 'scripts/p013_tune_optimize_training_data.py'], True, [scratch_cache]),
        detector_load=(['python', '-c', DETECTOR_LOAD_PROGRAM, dataset, str(load_times), str(DETECTOR_LOADS)], True,
                       [load_times]),
        p016=(['python', 'scripts/p016_tune_track_rate.py', '--num_workers', '40'], False, [scratch_cache]))
    # p011 runs before the stages that read its detections; the rest keep STEPS order.
    for step in sorted((s for s in steps if s in commands), key=lambda s: (s != 'p011', STEPS.index(s))):
        # Each command with its environment.
        argv, uses_gpu, writes = commands[step]
        plan.append(executable(step, dataset, argv, writes=writes, gpu=uses_gpu, env=gpu_env if uses_gpu else cpu_env))
    return plan


def manual(step: str, dataset: str, cwd: str, command: str, check: str) -> dict:
    """A printed-only step that runs outside this container."""
    # No argv: never executed by run().
    return dict(step=step, dataset=dataset, cwd=cwd, env={}, argv=None, command=command, check=check, writes=[],
                mkdir=[], gpu=False, manual=True)


def otif_steps(dataset: str, host_scratch: str) -> list[dict]:
    """OTIF's preparation for one dataset, following its README, with outputs under the host scratch copy."""
    # OTIF's own name for the dataset, if it has an OTIF-format copy.
    name = OTIF_NAMES.get(dataset)
    # B3D has no OTIF-format dataset under /otif-dataset.
    if name is None:
        return [manual('otif', dataset, '-', '-', f'no OTIF-format copy of {dataset} under {OTIF_HOST_DATASET}; '
                                                  'locate the converted data used for the reported OTIF runs')]
    # The scratch copy of OTIF's dataset and the dataset's tracker thresholds.
    root = f'{host_scratch}/otif-dataset'
    data = f'{root}/dataset/{name}'
    thresholds = shlex.quote(OTIF_THRESHOLDS[name])
    check = ('UNVERIFIED: take <detector> and <WxH> from the detector_cfg column of '
             f'/sota-results/<name>/otif_<name>.csv and confirm the step list with the reported runs')
    # Copy, then each preparation step of OTIF's README, timed by /usr/bin/time.
    return [
        manual('otif copy', dataset, '-', f'mkdir -p {root}/dataset && rsync -a {OTIF_HOST_DATASET}/dataset/{name}/ {data}/ '
                                          f'&& rsync -a {OTIF_HOST_DATASET}/yolov3/ {root}/yolov3/',
               'copies the inputs so every OTIF output lands in the scratch copy'),
        manual('otif detect train', dataset, f'{OTIF_HOST_ROOT}/pipeline2',
               f'/usr/bin/time -v go run 10_detect.go {root}/ {name} {data}/train/video/ {data}/train/<detector>/ <WxH>', check),
        manual('otif tracks train', dataset, f'{OTIF_HOST_ROOT}/pipeline2',
               f"/usr/bin/time -v go run iou-tracker2.go {root}/ {name} {data}/train/video/ {data}/train/<detector>/ "
               f"{data}/train/tracks/ iou '' {thresholds}",
               'section 7 trains the recurrent tracker on the training split; rnn/train.py reads the tracker split, '
               'so point it at train'),
        manual('otif extract frames', dataset, f'{OTIF_HOST_ROOT}/pipeline',
               f'/usr/bin/time -v python 16_extract_frames.py {root}/ {name} <detector>', check),
        manual('otif train proxy', dataset, f'{OTIF_HOST_ROOT}/pipeline',
               f'/usr/bin/time -v python 20_train_segmentation.py {root}/ {name} car', check),
        manual('otif train tracker', dataset, f'{OTIF_HOST_ROOT}/rnn', f'/usr/bin/time -v python train.py {root}/ {name}', check),
        manual('otif tune valid', dataset, f'{OTIF_HOST_ROOT}/pipeline2',
               f'/usr/bin/time -v go run 30_speed_accuracy.go {root}/ {name} ours-simple <detector> 416_256_-1 {thresholds} 10 '
               f'{host_scratch}/otif/{name}-valid-ours-simple.txt', check),
    ]


def leap_steps(dataset: str, host_scratch: str) -> list[dict]:
    """LEAP's preparation for one dataset, following its README, run in a scratch copy of the LEAP checkout."""
    copy = f'{host_scratch}/leap/{dataset}/LEAP'
    check = ('UNVERIFIED: confirm with the LEAP operator which steps, arguments, and splits the reported runs used; '
             'these scripts write inside the checkout, hence the copy')
    # Copy the checkout, then each preparation step of LEAP's README, timed by /usr/bin/time.
    return [
        manual('leap copy', dataset, '-', f'mkdir -p {copy} && rsync -a --exclude outputs --exclude logs {LEAP_HOST_ROOT}/ {copy}/',
               'LEAP writes labels, re-identification data, and models inside its checkout'),
        manual('leap pseudo labels', dataset, copy,
               f'/usr/bin/time -v python video_tracker.py --video_path <train videos> --video_name {dataset} '
               f'--mask_path masks/<mask> --save_path {copy}/labels --fps 15', check),
        manual('leap reid data', dataset, copy, '/usr/bin/time -v python ./vehicle_reid/generate_reid.py', check),
        manual('leap reid train', dataset, copy, '/usr/bin/time -v bash ./vehicle_reid/run.sh', check),
        manual('leap motion patterns', dataset, copy, '/usr/bin/time -v python track_cluster.py', check),
    ]


def plan(args, scratch: Path) -> list[dict]:
    """Every step for the requested datasets and steps, in execution order."""
    # Steps of every dataset, in order.
    steps = []
    # Per dataset: the polyis steps, then the printed baseline steps.
    for dataset in args.datasets:
        steps += polyis_steps(dataset, args.steps, scratch, args.gpu)
        # OTIF's steps, if requested.
        if 'otif' in args.steps:
            steps += otif_steps(dataset, args.host_scratch)
        # LEAP's steps, if requested.
        if 'leap' in args.steps:
            steps += leap_steps(dataset, args.host_scratch)
    return steps


def describe(step: dict) -> str:
    """One printable line per step: where it runs, its environment, and its command."""
    # The command as a shell line, its environment prefix, and where it runs.
    command = step['command'] if step['manual'] else shlex.join(step['argv'])
    env = ' '.join(f'{k}={shlex.quote(v)}' for k, v in step['env'].items())
    where = 'MANUAL (outside the polyis container)' if step['manual'] else f"polyis container, {'GPU' if step['gpu'] else 'CPU'}"
    line = f"[{step['dataset']}] {step['step']}: {where}, cd {step['cwd']} && {env + ' ' if env else ''}{command}"
    # Printed-only steps carry what to verify first.
    return line + (f"\n    check: {step['check']}" if step.get('check') else '')


def gpu_snapshot(gpu: int) -> dict:
    """Utilization and memory of one GPU from nvidia-smi (a driver query; nothing runs on the GPU)."""
    # Query the driver only.
    out = subprocess.run(['nvidia-smi', f'--id={gpu}', '--query-gpu=utilization.gpu,memory.used',
                          '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True).stdout
    # One line: "<utilization>, <memory>".
    utilization, memory = (int(v) for v in out.strip().split(','))
    return dict(utilization_percent=utilization, memory_used_mib=memory)


def busy(snapshot: dict) -> bool:
    """Whether another job is using the GPU, which would distort the timing."""
    # Either level means another job holds the GPU.
    return snapshot['utilization_percent'] > BUSY_UTILIZATION_PERCENT or snapshot['memory_used_mib'] > BUSY_MEMORY_MIB


def run(steps: list[dict], scratch: Path, gpu: int, allow_busy: bool = False):
    """Execute the polyis steps in order, appending one timing record per step to <scratch>/timings.jsonl."""
    # Check every step's writes before running any, so a bad plan writes nothing.
    outside = [(s['step'], p) for s in steps if not s['manual'] for p in s['writes'] + s['mkdir'] if not inside(p, scratch)]
    assert not outside, f'steps would write outside {scratch}: {outside}'
    # Every step's output goes to its own log.
    (scratch / 'logs').mkdir(parents=True, exist_ok=True)
    for step in steps:
        # Printed-only steps run elsewhere.
        if step['manual']:
            continue
        # Write this dataset's scratch config before its first step.
        config_path = scratch / 'configs' / f"{step['dataset']}.yaml"
        # Configs live next to the scratch cache.
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.safe_dump(scratch_config(step['dataset'], scratch), f)
        # Directories the step writes into must exist first.
        for directory in step['mkdir']:
            os.makedirs(directory, exist_ok=True)
        # GPU state before the step, kept with its timing.
        before = gpu_snapshot(gpu)
        # A shared GPU would make the timing incomparable with the cached ones.
        assert not (step['gpu'] and busy(before)) or allow_busy, \
            f"GPU {gpu} is in use ({before}); rerun when it is idle, or pass --allow-busy to time it anyway"
        # This step's log file.
        log = scratch / 'logs' / f"{step['dataset']}_{step['step'].replace(' ', '_')}.log"
        # Wall clock of the whole step.
        start = time.time()
        # Run the step with its environment on top of ours, its output to its log.
        with open(log, 'w') as out:
            result = subprocess.run(step['argv'], cwd=step['cwd'], env={**os.environ, **step['env']}, stdout=out,
                                    stderr=subprocess.STDOUT)
        end = time.time()
        # One timing record per step, written before checking the exit code so failures are kept too.
        record = dict(step=step['step'], dataset=step['dataset'], argv=step['argv'], env=step['env'], start=start, end=end,
                      seconds=end - start, returncode=result.returncode, host=socket.gethostname(), gpu=gpu,
                      uses_gpu=step['gpu'], gpu_before=before, gpu_after=gpu_snapshot(gpu), log=str(log))
        # Append the record to the scratch timings file.
        with open(scratch / 'timings.jsonl', 'a') as f:
            f.write(json.dumps(record) + '\n')
        print(f"{step['dataset']} {step['step']}: {record['seconds']:.1f} s, exit {result.returncode}")
        # Later steps depend on earlier ones.
        assert result.returncode == 0, f"{step['step']} failed for {step['dataset']}; see {log}"


def main():
    # Command-line options.
    args = parse_args()
    # The real cache is protected even when it is not one of the usual roots.
    scratch = check_scratch(args.scratch, get_config()['DATA']['CACHE_DIR'])
    # Every step for the requested datasets.
    steps = plan(args, scratch)
    # Always show the plan; only --run executes the polyis steps.
    print(f"{'RUN' if args.run else 'DRY RUN'}: {len(steps)} steps, scratch {scratch}, GPU {args.gpu}")
    # Print each step, executable or not.
    for step in steps:
        print(describe(step))
    # Execute only on request.
    if args.run:
        run(steps, scratch, args.gpu, args.allow_busy)


if __name__ == '__main__':
    main()
