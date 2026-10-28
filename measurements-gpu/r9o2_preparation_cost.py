#!/usr/local/bin/python
"""R9.O2: re-time the preparation work that has no timing records. DRY RUN unless --run.

Prints, per dataset, the commands that would time the following, in this order (p110 first, then p011 when
requested, then the steps that read its detections):

  p110           scoring of every validation configuration (CPU; the measurement only estimates it from the test
                 split's scoring) with P110_CPU_COUNT CPUs, so 16 TrackEval processes. Only validation videos whose
                 every tracking output (the reference's and each configuration's) exists in the source cache are
                 scored; on CalDoT2 a 2026-05-29 rerun deleted the sweep outputs of va00-va09 (REVISION.md
                 finding 5), so p110 there scores 50 of 60 videos.
  p110_density   the same scoring at the deployed density, PYTHON_CPU_COUNT = the CPUs this process may use, so 0.8
                 TrackEval processes per CPU as on ace (64 on 80); a cross-check, not run by default
  p012, p013     training-data creation and pruning (GPU; p012's seek and decode are untimed and p013 writes no
                 timings)
  detector_load  detector construction with get_detector as p040 calls it once per configuration, without p040's
                 warm-up (GPU; not recorded). Its batch_size and num_images differ from p040's, but only polyis's
                 yolov3 detector reads them, and these datasets use ultralytics and RetinaNet.
  p011           reference detection on the training split (GPU; per-frame times exist, this adds each video's
                 whole-process span)
  p016           mistrack profiling (CPU) with P016_RETIMING_WORKERS worker, so its wall clock is its process
                 seconds (the measurement's estimate is an upper bound from p016's 40-worker elapsed time)
  otif, leap     OTIF's and LEAP's preparation (printed only)

Every polyis command runs with CONFIG pointing at a copy of configs/global.yaml whose CACHE_DIR is
<scratch>/cache and whose DATASETS holds only the dataset being timed, so it writes only under the
scratch directory. Inputs that a step needs from earlier stages (p011's detections, the validation
tracking outputs) are copied from the source cache (the configured CACHE_DIR) first; videos and
detector weights are read in place. Every executable step lists the paths it writes, and --run
refuses a step that would write outside the scratch directory; nothing under /polyis-cache,
/polyis-data, /otif-dataset, or /sota-results is written.

On bombe the plan runs inside measurements-gpu/bombe_container.sh, driven by
measurements-gpu/r9o2_preparation_retiming.sh: /polyis-cache is the launcher's staging cache (the
inputs pushed from ace's cache; the source, never written), /polyis-data is the read-only mirror,
the only GPU is cuda:0 (--gpu 0), and --scratch is /scratch/retime on the host's scratch directory.

OTIF and LEAP steps are printed, never executed: they run in those projects' own environments
outside this container, their arguments come from the projects' READMEs, and each is marked with
what must be checked against the reported runs first. Their outputs are directed to
--host-scratch, a host path, after copying the inputs there.

With --run, the polyis steps execute in order; each step's output goes to <scratch>/logs/, and one
record per step is appended to <scratch>/timings.jsonl: its wall clock and exit code, the CPU
seconds of its processes (getrusage of the reaped children), GPU snapshots before and after,
`interference` (the busy CPUs of this process's CPU set and of the host from /proc/stat over the
step, minus the step's own CPU use, and the launcher's samples, about every 2 s, of foreign
processes on the GPU during the step), and `placement` (CPU set, physical GPU, and launcher run). The creation and
modification times of the step's outputs (STEP_OUTPUTS) go to <scratch>/files.jsonl under the
record's id; measurements/r9o2_preparation_cost.py derives each unit's process time from them.
detector_load also appends its per-load times to <scratch>/detector_load.jsonl. A GPU step waits
up to IDLE_WAIT_S for its GPU to show 0% and under 100 MiB, and refuses to start otherwise
(cached timings were taken on idle GPUs) unless --allow-busy is given.
"""
import argparse
import glob
import json
import os
import resource
import shlex
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path

import yaml

from evaluation.manifests import build_query_task_manifest, list_split_videos
from evaluation.p110_accuracy_compute import resolve_groundtruth_tracking_dataset, resolve_prediction_relative_path
# file_times: GNU stat with creation times, the one implementation the plan and the R9.O2 measurement share.
from measurements.common import DATASET_LABELS, file_times
from polyis.io import cache, store
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
# Steps this plan knows, in execution order (p011 is moved first among the stages that follow p110, since p012 and
# p016 read its detections), and the ones run by default.
STEPS = ('p110', 'p110_density', 'p012', 'p013', 'detector_load', 'p011', 'p016', 'otif', 'leap')
DEFAULT_STEPS = ('p110', 'p012', 'p013', 'detector_load', 'otif', 'leap')
# Steps that read p011's detections from the cache.
NEEDS_DETECTIONS = ('p012', 'p016')
# A GPU is idle only at 0% utilization and under 100 MiB used (nvidia-smi units: percent, MiB).
IDLE_UTILIZATION_PERCENT = 0
IDLE_MEMORY_BELOW_MIB = 100
# How long a GPU step waits for its GPU to become idle (utilization is a trailing average, so it lingers for a moment
# after the previous step exits), and how often it looks.
IDLE_WAIT_S = 30.0
IDLE_POLL_S = 2.0
# Detector constructions per detector_load run: the first includes CUDA start-up, the rest are warm, as in
# p040's long-lived workers that construct one detector per configuration.
DETECTOR_LOADS = 5
# p016's workers in the re-timing: one, so the step's wall clock is the process seconds of its profiles (the
# measurement checks the record against its own P016_RETIMING_WORKERS; a test keeps the two equal).
P016_RETIMING_WORKERS = 1
# CPU count p110 sees (PYTHON_CPU_COUNT, Python 3.13): it runs int(0.8 x count) TrackEval processes at once, 16 here,
# under the 40 CPUs of a bombe NUMA node, so a task's span is not stretched by waiting for a CPU. p110_density sets it
# to the CPUs this process may use instead.
P110_CPU_COUNT = 20
# Where the launcher (measurements-gpu/bombe_container.sh) writes its host-side samples, as mounted in its container.
LAUNCHER_SAMPLES = Path('/scratch/samples')
# Outputs whose creation and modification times each step records, as globs under the scratch cache's dataset root.
STEP_OUTPUTS = {
    # One directory per scored configuration, created when its task starts, and its per-video and combined results.
    'p110': [f"evaluation/{cache.EVAL_STAGE_MAP['acc']}/raw/valid/*",
             f"evaluation/{cache.EVAL_STAGE_MAP['acc']}/raw/valid/*/*.json"],
    'p110_density': [f"evaluation/{cache.EVAL_STAGE_MAP['acc']}/raw/valid/*",
                     f"evaluation/{cache.EVAL_STAGE_MAP['acc']}/raw/valid/*/*.json"],
    # One detections file per training video, opened after the detector is loaded.
    'p011': [f"indexing/{cache.INDEX_STAGE_MAP['det']}/*.detections.jsonl"],
    # One runtime file per training video and tile size, opened when the video's task starts.
    'p012': [f"indexing/{cache.INDEX_STAGE_MAP['training']}/runtime/tilesize_*/*_creating_training_data.jsonl"],
    # The per-video relevance bitmaps p013 combines and its statistics.
    'p013': [f"indexing/{cache.INDEX_STAGE_MAP['never-relevant']}/*"],
    # Each profile's per-video partial counts and its aggregate outputs.
    'p016': [f"indexing/{cache.INDEX_STAGE_MAP['track_rates']}/*/*",
             f"indexing/{cache.INDEX_STAGE_MAP['track_rates']}/*/partial/*.npy"],
}
# Small JSON outputs whose content is kept in the step's record (p013's files-removed statistics).
SUMMARY_SUFFIXES = {'p013': '_optimization_stats.json'}
# The detector_load program: construct the dataset's detector with get_detector, as p040 does without its warm-up,
# and append the times to a JSONL file.
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
    begin = time.time()
    detector = polyis.models.detector.get_detector(dataset, 0, batch_size=16, num_images=0)
    torch.cuda.synchronize()
    times.append(time.time() - begin)
    del detector
    torch.cuda.empty_cache()
with open(out, 'a') as f:
    f.write(json.dumps(dict(dataset=dataset, start=start, end=time.time(), device=torch.cuda.get_device_name(0),
                            cuda_init_s=cuda_init_s, load_s=times)) + '\\n')
'''


def parse_args():
    parser = argparse.ArgumentParser(description='R9.O2: re-time preparation stages into a scratch cache (dry run unless --run)')
    parser.add_argument('--datasets', nargs='+', default=list(DATASET_LABELS), choices=list(DATASET_LABELS))
    parser.add_argument('--steps', nargs='+', default=list(DEFAULT_STEPS), choices=STEPS)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--scratch', default='/scratch/retime')
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


def scratch_config(dataset: str, scratch: Path, data: dict | None = None) -> dict:
    """configs/global.yaml with the cache moved into the scratch directory, only `dataset` configured, and `data`'s
    DATA entries overridden."""
    # The checked-in configuration, as the pipeline reads it.
    config = get_config()
    # Every cache path in polyis derives from DATA.CACHE_DIR.
    config['DATA']['CACHE_DIR'] = str(scratch / 'cache') + '/'
    # Step-specific data locations (p110's directory of covered validation videos).
    config['DATA'].update(data or {})
    # The tuning and scoring scripts loop over EXEC.DATASETS; one dataset per run keeps timings per dataset.
    config['EXEC']['DATASETS'] = [dataset]
    return config


def executable(step: str, dataset: str, argv: list[str], writes: list[Path], gpu: bool, cwd: str = '/polyis',
               env: dict | None = None, mkdir: list[Path] | None = None, config: dict | None = None,
               meta: dict | None = None) -> dict:
    """One step that --run executes in this container; `config` overrides DATA entries of its CONFIG file and `meta`
    is kept in its timing record."""
    # Paths are kept as strings so the plan prints and serializes as is.
    return dict(step=step, dataset=dataset, cwd=cwd, env=env or {}, argv=argv, writes=[str(p) for p in writes],
                mkdir=[str(p) for p in mkdir or []], gpu=gpu, manual=False, config=config or {}, meta=meta or {})


def covered_validation_videos(dataset: str) -> tuple[list[str], list[str]]:
    """Validation videos whose every tracking output p110 reads exists in the source cache, and the other videos."""
    # p110's own task manifest: every configured variant and the reference on every validation video.
    manifest = build_query_task_manifest(datasets=[dataset], include_naive=True, videosets=['valid'])
    # Each row's tracking file, by p110's own path rule (per-row path building, not data analysis).
    paths = manifest.apply(lambda row: str(cache.execution(dataset, row['video'],
                                                           resolve_prediction_relative_path(row.to_dict()))), axis=1)
    # A video is covered when all of its rows' files exist.
    complete = paths.map(os.path.exists).groupby(manifest['video']).all()
    return sorted(complete.index[complete]), sorted(complete.index[~complete])


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
    # The p110 runs requested and the CPU count each sees: the re-timing's, and the deployed density's (every CPU this
    # process may use, as p110 sees all of ace's CPUs).
    scorings = {s: c for s, c in (('p110', P110_CPU_COUNT), ('p110_density', len(os.sched_getaffinity(0))))
                if s in steps}
    # p110 scores the validation tracking outputs against the reference's, both copied from the cache.
    if scorings:
        # Only videos with every tracking output can be scored (p110 asserts that each one exists).
        covered, excluded = covered_validation_videos(dataset)
        assert covered, f'{dataset}: no validation video has every tracking output under {cache.execution(dataset)}'
        # Where p110 reads the tracking outputs in the scratch cache.
        target = scratch_cache / dataset / 'execution'
        # p110 checks that the ground-truth dataset's execution directory exists, although validation does not read it.
        groundtruth = scratch_cache / resolve_groundtruth_tracking_dataset(dataset) / 'execution'
        # Copy every covered video's tracking.jsonl (the reference's and each configuration's), keeping relative paths;
        # copy-on-write clones where the file system supports them.
        plan.append(executable('copy validation tracking outputs', dataset, cwd=str(cache.execution(dataset)),
                               argv=['find', *covered, '-name', 'tracking.jsonl', '-exec', 'cp', '--parents',
                                     '--reflink=auto', '-t', str(target), '{}', '+'],
                               writes=[target], mkdir=[target, groundtruth], gpu=False))
        # p110 scores every video of the split's directory; with some excluded, a directory of links to the covered
        # videos stands in for it, through p110's own config.
        p110_config, data = config, {}
        if excluded:
            videos_dir = scratch / 'datasets' / dataset / 'valid'
            plan.append(executable('link covered validation videos', dataset,
                                   ['ln', '-sf', '-t', str(videos_dir), *(str(store.dataset(dataset, 'valid', v))
                                                                          for v in covered)],
                                   writes=[videos_dir], mkdir=[videos_dir], gpu=False))
            p110_config, data = str(scratch / 'configs' / f'{dataset}.p110.yaml'), dict(DATASETS_DIR=str(scratch / 'datasets'))
        # Score every validation configuration, as p110 --valid does in the pipeline, once per requested CPU count.
        for step, cpu_count in scorings.items():
            plan.append(executable(step, dataset, ['python', 'evaluation/p110_accuracy_compute.py', '--valid'],
                                   writes=[scratch_cache], env=dict(cpu_env, CONFIG=p110_config,
                                                                    PYTHON_CPU_COUNT=str(cpu_count)),
                                   gpu=False, config=data, meta=dict(videos=covered, excluded_videos=excluded)))
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
        p016=(['python', 'scripts/p016_tune_track_rate.py', '--num_workers', str(P016_RETIMING_WORKERS)], False,
              [scratch_cache]))
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
    # Excluded validation videos are named, so the plan shows what p110 leaves out.
    if step.get('meta', {}).get('excluded_videos'):
        line += f"\n    scores {len(step['meta']['videos'])} videos; excluded (incomplete outputs): " \
                f"{' '.join(step['meta']['excluded_videos'])}"
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
    return snapshot['utilization_percent'] > IDLE_UTILIZATION_PERCENT or snapshot['memory_used_mib'] >= IDLE_MEMORY_BELOW_MIB


def wait_idle(gpu: int) -> dict:
    """The first idle snapshot of the GPU within IDLE_WAIT_S, else the last busy one."""
    # Look until the GPU is idle or the wait is over.
    deadline = time.time() + IDLE_WAIT_S
    snapshot = gpu_snapshot(gpu)
    while busy(snapshot) and time.time() < deadline:
        time.sleep(IDLE_POLL_S)
        snapshot = gpu_snapshot(gpu)
    return snapshot


def children_cpu_s() -> float:
    """User and system CPU seconds of every reaped child process of this one (and their reaped children)."""
    # getrusage counts a child once it has exited and been waited for, as subprocess.run does.
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def cpu_jiffies() -> dict[int, tuple[int, int]]:
    """{cpu: (busy, total)} clock ticks of every CPU since boot, from /proc/stat (a container sees the host's)."""
    # Ticks per CPU.
    jiffies = {}
    with open('/proc/stat') as f:
        for line in f:
            name, *values = line.split()
            # Per-CPU lines only ('cpu' alone is the sum over CPUs).
            if not (name.startswith('cpu') and name[3:].isdigit()):
                continue
            # The first eight fields; guest time is already inside user time.
            user, nice, system, idle, iowait, irq, softirq, steal = (int(v) for v in values[:8])
            busy_ticks = user + nice + system + irq + softirq + steal
            jiffies[int(name[3:])] = (busy_ticks, busy_ticks + idle + iowait)
    return jiffies


def launcher_samples(start: float, end: float) -> dict:
    """The launcher's host-side samples (about every 2 s) during [start, end]: how many, how many saw a foreign process
    on this GPU, and the largest GPU utilization and load average among them (None outside the launcher)."""
    # The launcher names its samples after its run id; without it (not in the launcher) nothing was sampled.
    run_id = os.environ.get('BOMBE_RUN')
    path = LAUNCHER_SAMPLES / f'{run_id}.jsonl'
    if not run_id or not path.exists():
        return dict(gpu_samples=None, gpu_foreign_samples=None, gpu_util_max=None, load_1m_max=None)
    # Whole lines only: the sampler may be writing the last one.
    text = path.read_text()
    lines = text.splitlines()[:None if text.endswith('\n') else -1]
    # The samples inside the step's window (the launcher stamps them to the second).
    samples = [s for s in map(json.loads, filter(None, lines))
               if start - 1 <= datetime.fromisoformat(s['t']).timestamp() <= end + 1]
    return dict(gpu_samples=len(samples), gpu_foreign_samples=sum(s['foreign'] > 0 for s in samples),
                gpu_util_max=max((s['util'] for s in samples), default=None),
                load_1m_max=max((s['load_1m'] for s in samples), default=None))


def interference(before: dict, after: dict, seconds: float, own_cpu_s: float, start: float, end: float) -> dict:
    """CPU and GPU use during a step besides the step's own: the busy CPUs of this process's CPU set (the GPU's NUMA
    node in the launcher) and of the host, the step's own CPUs, and the launcher's GPU samples."""
    # Clock ticks per second of /proc/stat, and the CPUs this process may run on.
    ticks, mine = os.sysconf('SC_CLK_TCK'), os.sched_getaffinity(0)
    # Average busy CPUs of a set over the step.
    duration = max(seconds, 1e-9)
    busy_cpus = lambda cpus: sum(after[c][0] - before[c][0] for c in cpus if c in before) / ticks / duration
    node, own = busy_cpus(mine & set(after)), own_cpu_s / duration
    # Others' use of this CPU set is what is busy beyond the step's own processes.
    return dict(cpu_set_size=len(mine), cpu_set_busy_cpus=round(node, 2), own_cpus=round(own, 2),
                other_busy_cpus=round(node - own, 2), host_busy_cpus=round(busy_cpus(after.keys()), 2),
                **launcher_samples(start, end))


def placement() -> dict:
    """Where this process runs: its CPU set, and the launcher's physical GPU, CPU list, and run id (None elsewhere)."""
    # The CPU set as a sorted list keeps the record self-contained even outside the launcher.
    return dict(cpus=sorted(os.sched_getaffinity(0)), host=socket.gethostname(), physical_gpu=os.environ.get('BOMBE_GPU'),
                launcher_cpus=os.environ.get('BOMBE_CPUS'), launcher_run=os.environ.get('BOMBE_RUN'))


def output_times(step: dict, scratch: Path) -> list[dict]:
    """Creation and modification times of the step's outputs (STEP_OUTPUTS), paths relative to the scratch cache."""
    # The step's output globs under its dataset's scratch cache root.
    root = scratch / 'cache' / step['dataset']
    paths = sorted({p for pattern in STEP_OUTPUTS.get(step['step'], []) for p in glob.glob(str(root / pattern))})
    # Timestamps of every output at once.
    times = file_times(paths)
    return [dict(path=os.path.relpath(p, scratch / 'cache'), birth_s=b, mtime_s=m) for p, (b, m) in sorted(times.items())]


def run(steps: list[dict], scratch: Path, gpu: int, allow_busy: bool = False):
    """Execute the polyis steps in order, appending one timing record per step to <scratch>/timings.jsonl and its
    outputs' timestamps to <scratch>/files.jsonl."""
    # Check every step's writes before running any, so a bad plan writes nothing.
    outside = [(s['step'], p) for s in steps if not s['manual'] for p in s['writes'] + s['mkdir'] if not inside(p, scratch)]
    assert not outside, f'steps would write outside {scratch}: {outside}'
    # Every step's output goes to its own log.
    (scratch / 'logs').mkdir(parents=True, exist_ok=True)
    for step in steps:
        # Printed-only steps run elsewhere.
        if step['manual']:
            continue
        # A step that reads a scratch config gets it written first (its dataset, its DATA overrides).
        if 'CONFIG' in step['env']:
            config_path = Path(step['env']['CONFIG'])
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                yaml.safe_dump(scratch_config(step['dataset'], scratch, step['config']), f)
        # Directories the step writes into must exist first.
        for directory in step['mkdir']:
            os.makedirs(directory, exist_ok=True)
        # GPU state before the step, kept with its timing; a GPU step waits for its GPU to be idle first.
        before = wait_idle(gpu) if step['gpu'] else gpu_snapshot(gpu)
        # A shared GPU would make the timing incomparable with the cached ones.
        assert not (step['gpu'] and busy(before)) or allow_busy, \
            f"GPU {gpu} is in use ({before}); rerun when it is idle, or pass --allow-busy to time it anyway"
        # This step's log file.
        log = scratch / 'logs' / f"{step['dataset']}_{step['step'].replace(' ', '_')}.log"
        # CPU seconds of reaped children and every CPU's busy ticks before the step.
        cpu_before, jiffies_before = children_cpu_s(), cpu_jiffies()
        # Wall clock of the whole step.
        start = time.time()
        # Run the step with its environment on top of ours, its output to its log.
        with open(log, 'w') as out:
            result = subprocess.run(step['argv'], cwd=step['cwd'], env={**os.environ, **step['env']}, stdout=out,
                                    stderr=subprocess.STDOUT)
        end = time.time()
        # The step's own CPU seconds and everything else that ran on the CPUs meanwhile.
        cpu_s = children_cpu_s() - cpu_before
        load = interference(jiffies_before, cpu_jiffies(), end - start, cpu_s, start, end)
        # One id ties the record to its outputs' timestamps; the launcher's run id keeps ids of containers apart, whose
        # host name and pid (1) are the same.
        record_id = f"{socket.gethostname()}-{os.environ.get('BOMBE_RUN', 'local')}-{os.getpid()}-{int(start * 1e6)}"
        outputs = output_times(step, scratch)
        # Small JSON summaries the step wrote (p013's statistics), kept in the record.
        suffix = SUMMARY_SUFFIXES.get(step['step'])
        summaries = {o['path']: json.loads((scratch / 'cache' / o['path']).read_text())
                     for o in outputs if suffix and o['path'].endswith(suffix)}
        # One timing record per step, written before checking the exit code so failures are kept too.
        record = dict(record_id=record_id, step=step['step'], dataset=step['dataset'], argv=step['argv'], env=step['env'],
                      start=start, end=end, seconds=end - start, returncode=result.returncode, host=socket.gethostname(),
                      gpu=gpu, uses_gpu=step['gpu'], gpu_before=before, gpu_after=gpu_snapshot(gpu), log=str(log),
                      cpu_s=cpu_s, interference=load, placement=placement(), meta=step['meta'], outputs=summaries)
        # The outputs' timestamps under the record's id.
        with open(scratch / 'files.jsonl', 'a') as f:
            f.writelines(json.dumps(dict(record_id=record_id, dataset=step['dataset'], step=step['step'], **o)) + '\n'
                         for o in outputs)
        # Append the record to the scratch timings file.
        with open(scratch / 'timings.jsonl', 'a') as f:
            f.write(json.dumps(record) + '\n')
        print(f"{step['dataset']} {step['step']}: {record['seconds']:.1f} s, exit {result.returncode}, "
              f"{len(outputs)} outputs, other busy CPUs {load['other_busy_cpus']}, "
              f"foreign GPU samples {load['gpu_foreign_samples']}", flush=True)
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
    print(f"{'RUN' if args.run else 'DRY RUN'}: {len(steps)} steps, scratch {scratch}, GPU {args.gpu}", flush=True)
    # Print each step, executable or not.
    for step in steps:
        print(describe(step))
    # Execute only on request.
    if args.run:
        run(steps, scratch, args.gpu, args.allow_busy)


if __name__ == '__main__':
    main()
