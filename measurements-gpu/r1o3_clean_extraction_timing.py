#!/usr/local/bin/python
"""R1.O3 clean extraction timing (GPU): the reference pipeline and Tetris's operating points rerun on idle GPUs.

REVISION.md finding 3: p121's throughput boundary is asymmetric (it counts the reference's per-frame decode but not
Tetris's decodes, and counts Tetris's canvas reads), and the cached timing logs were taken on ace's GPUs under other
users' load. This job reruns, on idle bombe GPUs pinned to their NUMA node, the pipeline's own scripts unchanged, so
measurements/r1o3_clean_extraction_timing.py can report throughput at several boundaries from clean logs.

What runs, per dataset (a "unit" is one launcher container, measurements-gpu/bombe_container.sh):
  reference  preprocess/p002_preprocess_naive_detection.py --test, then preprocess/p002_preprocess_naive_tracking.py
             --test: the two scripts that write <video>/002_naive/{detection,tracking}_runtime.jsonl, the files
             evaluation/p120_throughput_gather.py lists as stages 001 and 002 of the reference ('naive'). Before them,
             the unit decodes every video with cv2 (the default VideoCapture p002 opens, cap.read() per frame, the call
             p002 times as 'read'), timed on its own, DECODE_PASSES times in alternating video order (one pass in
             attempts made before the serial phase): the decode cost both sides are charged at boundary (3).
  tetris     scripts/p020_exec_classify.py, p022_exec_prune_polyominoes.py (pruning configurations only),
             p030_exec_compress.py, p040_exec_detect.py, p050_exec_uncompress.py, p060_exec_track.py, each --test: the
             scripts that write the 020-060 runtime files p120 lists for one Tetris configuration. Each stage selects
             its work through the test pass's Pareto filter (polyis.pareto.build_pareto_combo_filter), so the unit's
             cache holds a pareto_params.csv with only the unit's configuration (its row copied verbatim from ace's).
  workers    polyis.utilities.ProgressBar starts one worker process per task and runs at most num_workers at once:
             the number of visible GPUs for p002 (both), p020, p040, and p060 (1 here, 8 in the paper's runs on ace),
             cpu_count // 4 = 20 for p022, 20 for p030, and cpu_count // 2 = 40 for p050 (cpu_count is 80 on both
             hosts). A Tetris stage has one task per configuration, which processes the videos one after another in
             sorted order, so a unit's Tetris stage runs one worker process; p002 has one task per video, and with one
             visible GPU those run one at a time. Nothing inside a unit runs concurrently here, while the paper's runs
             ran up to 8-40 tasks of a stage at once (every Pareto configuration).
  p022       Gurobi's license has expired (BOMBE.md), so p022 cannot solve. When gurobipy cannot create a model in
             the unit's container, the unit copies ace's cached p022 outputs (score.jsonl and runtime.jsonl, staged
             by `stage`) into its cache instead of running p022, and its manifest records p022_source 'cache'; with
             a working license p022 runs and p022_source is 'rerun' (its output may then differ from the April cache,
             REVISION.md finding 6).
  operating  per dataset, the fastest Tetris test configuration within 5% and within 10% HOTA loss of the reference
  points     (measurements.common.fastest_tetris on the cached tradeoff tables, as r7o1_hota_matched_speedup does).

Phases:
  parallel   every dataset's reference unit and one unit per distinct operating point, on all test videos. Datasets
             are assigned to --gpus by longest-processing-time first (estimated from the cached throughput); each GPU
             runs its datasets' units one after another, so the GPUs run datasets in parallel. GPUs 0-2 share NUMA
             node 0, so these units compete for its CPUs; this phase measures that interference.
  serial     the parallel units' work again (all test videos, names serial-*), one unit at a time on its dataset's
             GPU, each started only on an idle node: the headline numbers.
  solo       per dataset, the reference unit and the 5% operating point's unit on CALIBRATION_VIDEOS evenly spaced test
             videos (all of them for B3D), one unit at a time on the dataset's GPU, each on an idle node: a second
             clean run of those videos, against which the serial phase's repeatability is measured.
  smoke      the solo units cut to one video, to check the setup; never aggregated.

Isolation and records, per unit:
  - The launcher refuses a GPU that is not at 0% and under 100 MiB with no compute process (exit 3; `run` waits
    BUSY_WAIT_S and retries, up to BUSY_RETRIES times), and `run` refuses to start unless every --gpus GPU is idle.
  - The launcher pins the container to the GPU's NUMA node's CPUs (taskset) and records the CPU set, GPU samples,
    the load average, and `contaminated` (another process used the GPU).
  - Serial, solo, and smoke units start only when the node's CPUs were at most NODE_IDLE_MAX busy over NODE_CHECK_S
    (`run` waits BUSY_WAIT_S between checks, up to BUSY_RETRIES times; unit.json records the check), and their attempt
    is clean only when other processes used at most FOREIGN_CPU_MAX of the node, on average over the timed steps
    (from the job's CPU samples, since no other unit of this job runs then).
  - An attempt that is contaminated or, outside the parallel phase, not clean on the CPU side is retried; a unit gets
    at most MAX_ATTEMPTS attempts per invocation of `run --redo`, and MAX_ATTEMPTS in all without --redo. A failed
    job is not retried. Every attempt is kept; measurements/r1o3_clean_extraction_timing.py chooses among them.
  - Each unit has its own cache (DATA.CACHE_DIR), config (CONFIG), and videos directory (DATA.DATASETS_DIR, symlinks
    to the read-only mirror), so stages that clear their output directories never touch another unit's outputs.
    ProgressBar writes its worker logs to the hard-coded /polyis-cache/LOGS/<script> and clears that directory at
    every stage start; `launch` mounts a per-unit directory there (WORKER_LOGS). Attempts made before that change
    shared the track's scratch polyis-cache/LOGS and kept no worker logs.
  - Videos are read once before timing (page cache warm, timed and recorded), so decoding is measured from memory.
  - Each stage process's wall clock (perf_counter around the subprocess), the per-op runtime logs as the stage wrote
    them (gzip-compressed, at their cache-relative paths, with each file's last-write time), digests of each stage's
    outputs, and /proc/stat samples of the unit's NUMA node and the other node once per second.

Subcommands:
  plan   (ace, polyis container) choose the operating points and write raw/plan.json: units, videos and frame counts,
         configurations with their Pareto rows, and the inputs with ace's SHA-256.
  stage  (ace host) push the cached p022 outputs the plan lists to bombe's scratch inputs (ace's key works on bombe).
  run    (bombe host, in tmux) copy the plan's indexing inputs from bombe's mirror (checking SHA-256), then run the
         phases through the launcher.
  job    (inside a launcher container; started by run) run one unit.

Writes measurements-results/r1o3_clean_extraction_timing/raw/ (never ace's or the mirror's cache):
  plan.json
  units/<unit>/attempt-<n>/manifest.json     the unit, its environment, stage records, decode, outputs, status
  units/<unit>/attempt-<n>/launcher.json     the launcher's run summary (GPU, NUMA node, CPUs, contaminated, ...)
  units/<unit>/attempt-<n>/samples.jsonl.gz  the launcher's 1 s samples of the unit's GPU and the load average
  units/<unit>/attempt-<n>/gpus.csv.gz       the launcher's 1 s nvidia-smi samples of every GPU
  units/<unit>/attempt-<n>/cpu.jsonl.gz      per second: /proc/stat busy fractions of the unit's CPUs and the other
                                             CPUs, and the container's own share of the unit's CPUs (cgroup)
  units/<unit>/attempt-<n>/decode.jsonl.gz   reference units: per decode pass and video, frames, seconds, threads
  units/<unit>/attempt-<n>/logs/<cache-relative runtime file>.gz
  units/<unit>/attempt-<n>/worker-logs/      ProgressBar's worker logs (*.log, gitignored)
  units/<unit>/attempt-<n>/*.log             launcher and stage output (gitignored)
measurements/r1o3_clean_extraction_timing.py aggregates them.

Usage:
  ace:   docker exec polyis bash -c 'cd /polyis && python measurements-gpu/r1o3_clean_extraction_timing.py plan'
  ace:   python3 measurements-gpu/r1o3_clean_extraction_timing.py stage
  bombe: tmux new-session -d -s r1o3 'cd /work/cwkt/projects/tetris &&
           python3 measurements-gpu/r1o3_clean_extraction_timing.py run --phases parallel serial solo >> <log> 2>&1'
"""
import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


# This measurement's name, shared with measurements/r1o3_clean_extraction_timing.py.
NAME = 'r1o3_clean_extraction_timing'
# The checkout (/work/cwkt/projects/tetris on ace and bombe, /polyis inside a container).
CHECKOUT = Path(__file__).resolve().parent.parent
# This script, relative to the checkout, as the launcher runs it.
SCRIPT = f'measurements-gpu/{NAME}.py'
# Raw outputs inside the measurement's result directory.
RAW = CHECKOUT / 'measurements-results' / NAME / 'raw'
# The plan written by `plan` and read by every other subcommand and by the aggregator.
PLAN_FILE = 'plan.json'
# Unit attempts: raw/units/<unit>/attempt-<n>/, each with a manifest written last by the job.
UNITS_DIR = 'units'
ATTEMPT_PREFIX = 'attempt-'
MANIFEST = 'manifest.json'
# Files the launcher's samples are copied to, inside an attempt directory.
LAUNCHER_SUMMARY = 'launcher.json'
LAUNCHER_SAMPLES = 'samples.jsonl.gz'
LAUNCHER_GPUS = 'gpus.csv.gz'
LAUNCHER_LOG = 'launcher.log'
# Files the job writes inside an attempt directory.
CPU_SAMPLES = 'cpu.jsonl.gz'
DECODE_FILE = 'decode.jsonl.gz'
LOGS_DIR = 'logs'
# bombe: the revision root, this track's scratch directory (mounted at /scratch by the launcher), and the mirror.
BOMBE_ROOT = Path('/data/cwkt/data/bombe-revision')
SCRATCH_HOST = BOMBE_ROOT / 'scratch' / NAME
MIRROR_CACHE = BOMBE_ROOT / 'mirror' / 'polyis-cache'
SCRATCH_CONTAINER = Path('/scratch')
# The launcher, and the mirror of ace's polyis-data inside its containers.
LAUNCHER = CHECKOUT / 'measurements-gpu' / 'bombe_container.sh'
CONTAINER_DATASETS = '/polyis-data/datasets'
# ace's cache on the ace host (mounted at /polyis-cache in ace's polyis container), and bombe's name seen from ace.
ACE_CACHE = Path('/data/cwkt/data/polyis-cache')
BOMBE_HOST = 'bombe.millennium.berkeley.edu'
# HOTA-loss bounds of the operating points (names as in r7o1_hota_matched_speedup), and the one calibrated solo.
BOUNDS = ('0.05', '0.10')
CALIBRATION_BOUND = '0.05'
# Test videos per solo calibration unit (a dataset with fewer runs all of them).
CALIBRATION_VIDEOS = 6
# Phases `run` can execute, in order; every phase but parallel runs one unit at a time on an idle node.
PHASES = ('smoke', 'parallel', 'serial', 'solo')
# Phases whose units are copies of another phase's units (smoke: the solo units cut to one video; serial: the
# parallel units).
DERIVED_PHASES = {'smoke': 'solo', 'serial': 'parallel'}
# The bombe GPUs this track may use.
GPUS = (0, 1, 2)
# Launcher exit codes: refused because the GPU was busy, and refused because of a bad argument.
LAUNCHER_BUSY = 3
LAUNCHER_BAD_ARGS = 2
# Waiting for a busy GPU or node: seconds between checks and how many retries before the unit fails.
BUSY_WAIT_S = 60
BUSY_RETRIES = 60
# Attempts per unit: an unclean attempt is retried until this many exist (per invocation with --redo).
MAX_ATTEMPTS = 2
# Idle node: at most this busy share of the GPU's NUMA node over NODE_CHECK_S seconds before a one-at-a-time unit
# starts, and at most FOREIGN_CPU_MAX used by other processes on average over its timed steps for the attempt to count
# as clean (an idle bombe node shows about 0.01; the launcher's host-side samplers are part of it).
NODE_IDLE_MAX = 0.05
NODE_CHECK_S = 10.0
FOREIGN_CPU_MAX = 0.05
# Steps of a unit that are not timed work (CpuSampler labels): container setup, page-cache warm-up, and harvesting.
UNTIMED_STEPS = ('setup', 'preread', 'harvest')
# Decode passes per reference unit, in alternating video order (the median pass is the decode cost).
DECODE_PASSES = 5
# Per-unit directory (in the unit's scratch directory) mounted at the container's /polyis-cache/LOGS, where
# ProgressBar writes its worker logs, and the attempt subdirectory they are copied to.
WORKER_LOGS = 'worker-logs'
# Rough per-video and per-stage process costs, used only to balance datasets across GPUs.
EST_VIDEO_OVERHEAD_S = 10.0
EST_STAGE_OVERHEAD_S = 30.0
# Reference stages: p120's stage name, the script, the stage directory, the runtime file, and the output files whose
# digests are recorded (all in the runtime file's directory).
REFERENCE_STAGES = (
    ('001_preprocess_groundtruth_detection', 'preprocess/p002_preprocess_naive_detection.py', '002_naive',
     'detection_runtime.jsonl', ('detection.jsonl',)),
    ('002_preprocess_groundtruth_tracking', 'preprocess/p002_preprocess_naive_tracking.py', '002_naive',
     'tracking_runtime.jsonl', ('tracking.jsonl',)),
)
# Tetris stages, in pipeline order, with the same fields.
TETRIS_STAGES = (
    ('020_exec_classify', 'scripts/p020_exec_classify.py', '020_relevancy', 'runtime.jsonl', ('score.jsonl',)),
    ('022_exec_prune_polyominoes', 'scripts/p022_exec_prune_polyominoes.py', '022_pruned_polyominoes', 'runtime.jsonl',
     ('score.jsonl',)),
    ('030_exec_compress', 'scripts/p030_exec_compress.py', '033_compressed_frames', 'runtime.jsonl', ()),
    ('040_exec_detect', 'scripts/p040_exec_detect.py', '040_compressed_detections', 'runtimes.jsonl', ('detections.jsonl',)),
    ('050_exec_uncompress', 'scripts/p050_exec_uncompress.py', '050_uncompressed_detections', 'runtime.jsonl',
     ('detections.jsonl',)),
    ('060_exec_track', 'scripts/p060_exec_track.py', '060_uncompressed_tracks', 'runtimes.jsonl', ('tracking.jsonl',)),
)
# Every stage by name: (script, stage directory, runtime file, output files).
STAGES = {s[0]: s[1:] for s in (*REFERENCE_STAGES, *TETRIS_STAGES)}
# The pruning stage, which runs only for configurations with a tracking-accuracy threshold.
PRUNE_STAGE = '022_exec_prune_polyominoes'
# Read size when warming the page cache.
PREREAD_CHUNK = 16 << 20
# PCI devices in sysfs, where a GPU's NUMA node CPUs are listed.
SYSFS_PCI = Path('/sys/bus/pci/devices')
# Seconds between /proc/stat samples, and the container's cgroup CPU accounting (cgroup v2).
CPU_SAMPLE_S = 1.0
CGROUP_CPU_STAT = Path('/sys/fs/cgroup/cpu.stat')


def parse_args():
    parser = argparse.ArgumentParser(description='R1.O3: reference and Tetris operating points rerun on idle bombe GPUs')
    sub = parser.add_subparsers(dest='command', required=True)
    plan = sub.add_parser('plan')
    plan.add_argument('--datasets', nargs='+', default=None)
    sub.add_parser('stage')
    run = sub.add_parser('run')
    run.add_argument('--gpus', nargs='+', type=int, default=list(GPUS))
    run.add_argument('--phases', nargs='+', choices=PHASES, default=['parallel', 'solo'])
    run.add_argument('--datasets', nargs='+', default=None)
    run.add_argument('--redo', action='store_true')
    job = sub.add_parser('job')
    job.add_argument('--unit', required=True)
    return parser.parse_args()


def now_iso() -> str:
    """The current UTC time to the millisecond."""
    # One clock format across every record of this job.
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


def sha256(path: Path) -> str:
    """SHA-256 of a file, read in chunks."""
    # Hash in chunks so large outputs do not load into memory at once.
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(PREREAD_CHUNK), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data):
    """Write one JSON file with stable key order, via a temporary file so readers never see a partial one."""
    # Write next to the target, then rename atomically.
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


def write_jsonl_gz(path: Path, records: list[dict]):
    """One JSON line per record, gzip-compressed."""
    # Text mode so json.dumps' strings are written directly.
    with gzip.open(path, 'wt') as f:
        f.writelines(json.dumps(r) + '\n' for r in records)


def gzip_copy(src: Path, dst: Path):
    """Copy a file into a gzip-compressed one, creating the directory."""
    # The destination's directory may not exist yet.
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Stream the bytes through gzip.
    with open(src, 'rb') as fin, gzip.open(dst, 'wb') as fout:
        shutil.copyfileobj(fin, fout)


def parse_cpu_list(text: str) -> list[int]:
    """CPUs of a list such as '0-19,40-59' (sysfs local_cpulist, taskset)."""
    cpus: list[int] = []
    # Each comma-separated part is a CPU or an inclusive range.
    for part in text.strip().split(','):
        if not part:
            continue
        lo, _, hi = part.partition('-')
        cpus.extend(range(int(lo), int(hi or lo) + 1))
    return cpus


def evenly_spaced(items: list[str], k: int) -> list[str]:
    """k items of a sorted list, evenly spaced from its first to its last (all of them when there are at most k)."""
    # Short lists are taken whole.
    if len(items) <= k:
        return list(items)
    # One item: the first.
    if k == 1:
        return [items[0]]
    # Indices round(i * (n - 1) / (k - 1)) are distinct for k <= n and include both ends.
    return [items[round(i * (len(items) - 1) / (k - 1))] for i in range(k)]


def assign_gpus(costs: dict[str, float], gpus: list[int]) -> dict[str, int]:
    """Datasets to GPUs, longest estimated cost first onto the least-loaded GPU (ties: the earlier GPU, then the name)."""
    # Datasets in decreasing cost, ties by name, so the assignment is deterministic.
    order = sorted(costs, key=lambda d: (-costs[d], d))
    load = {g: 0.0 for g in gpus}
    assignment = {}
    # Greedy longest-processing-time scheduling (a handful of datasets).
    for d in order:
        g = min(gpus, key=lambda x: (load[x], gpus.index(x)))
        assignment[d] = g
        load[g] += costs[d]
    return assignment


def unit_name(phase: str, dataset: str, system: str, variant_id: str | None) -> str:
    """A unit's name: phase, dataset, and 'reference' or the Tetris configuration."""
    # The configuration names a Tetris unit; the reference has none.
    return f'{phase}-{dataset}-{variant_id if system == "tetris" else "reference"}'


def plan_units(datasets: dict[str, dict]) -> list[dict]:
    """The parallel and solo units of every planned dataset (see the module docstring)."""
    units = []
    # A handful of units per dataset, in execution order on its GPU.
    for d, info in datasets.items():
        # Distinct operating points in bound order (one configuration can be the point of both bounds).
        variants = list(dict.fromkeys(info['bounds'][b]['variant_id'] for b in BOUNDS))
        for phase, videos, systems in (
                ('parallel', list(info['videos']), [('reference', None)] + [('tetris', v) for v in variants]),
                ('solo', info['calibration_videos'],
                 [('reference', None), ('tetris', info['bounds'][CALIBRATION_BOUND]['variant_id'])])):
            units += [dict(unit=unit_name(phase, d, s, v), phase=phase, dataset=d, system=s, variant_id=v, videos=videos)
                      for s, v in systems]
    return units


def phase_units(plan: dict, phase: str, datasets: list[str]) -> list[dict]:
    """The plan's units of one phase for `datasets`. Derived phases copy another phase's units under their own names:
    smoke units are the solo units cut to their first video, serial units the parallel units."""
    source = DERIVED_PHASES.get(phase, phase)
    units = [u for u in plan['units'] if u['phase'] == source and u['dataset'] in datasets]
    if phase in DERIVED_PHASES:
        units = [dict(u, unit=unit_name(phase, u['dataset'], u['system'], u['variant_id']), phase=phase,
                      videos=u['videos'][:1] if phase == 'smoke' else u['videos']) for u in units]
    return units


def unit_cost(plan: dict, unit: dict) -> float:
    """A unit's estimated seconds from the cached throughput, for balancing only."""
    info = plan['datasets'][unit['dataset']]
    # The unit's frames at the cached throughput of its system.
    frames = sum(info['videos'][v] for v in unit['videos'])
    if unit['system'] == 'reference':
        return frames / info['reference']['fps'] + EST_VIDEO_OVERHEAD_S * len(unit['videos'])
    fps = info['configs'][unit['variant_id']]['fps']
    return frames / fps + EST_STAGE_OVERHEAD_S * len(TETRIS_STAGES)


def attempts(unit: str, raw: Path = RAW) -> list[Path]:
    """A unit's attempt directories under `raw`, in attempt order."""
    # Attempts are numbered from 1; sort numerically.
    root = raw / UNITS_DIR / unit
    found = [p for p in root.glob(f'{ATTEMPT_PREFIX}*') if p.name[len(ATTEMPT_PREFIX):].isdigit()]
    return sorted(found, key=lambda p: int(p.name[len(ATTEMPT_PREFIX):]))


def others_on_node(path: Path) -> float | None:
    """Mean share of the unit's CPUs that processes outside its container used over its timed steps, from an
    attempt's CPU samples (None without usable samples). Outside the parallel phase no other unit of this job runs,
    so this is other tracks' and users' use."""
    if not path.exists():
        return None
    # One sample per second; setup, warm-up, and harvesting are not timed work.
    with gzip.open(path, 'rt') as f:
        shares = [r['unit_cpus_busy'] - r['own_busy'] for r in map(json.loads, f)
                  if r['step'] not in UNTIMED_STEPS and None not in (r['unit_cpus_busy'], r['own_busy'])]
    return sum(shares) / len(shares) if shares else None


def attempt_outcome(path: Path) -> dict:
    """An attempt's job status, the launcher's contaminated flag (None when either record is missing), and other
    processes' share of its node over its timed steps."""
    # The job's manifest is written last, the launcher summary after the container exits.
    manifest = path / MANIFEST
    summary = path / LAUNCHER_SUMMARY
    status = json.loads(manifest.read_text())['status'] if manifest.exists() else None
    contaminated = json.loads(summary.read_text())['contaminated'] if summary.exists() else None
    return dict(status=status, contaminated=contaminated, others_on_node=others_on_node(path / CPU_SAMPLES))


def usable(outcome: dict, phase: str) -> bool:
    """Whether an attempt needs no retry: it finished, no other process used its GPU, and, outside the parallel phase
    (where this job's other units share the node by design), other processes left its node idle enough."""
    # A missing CPU record cannot show an idle node.
    cpu_clean = phase == 'parallel' or (outcome['others_on_node'] is not None
                                        and outcome['others_on_node'] <= FOREIGN_CPU_MAX)
    return outcome['status'] == 'ok' and outcome['contaminated'] is False and cpu_clean


# ---------------------------------------------------------------------------------------------------- plan (ace)

def pareto_row(path: Path, variant_id: str) -> tuple[str, str, dict]:
    """The header line, the line, and the parsed row of `variant_id` in a pareto_params.csv; exactly one must match."""
    # Keep the lines verbatim, so the unit's filter file parses exactly as ace's.
    lines = path.read_text().splitlines()
    header = next(csv.reader([lines[0]]))
    rows = [(line, dict(zip(header, next(csv.reader([line]))))) for line in lines[1:] if line.strip()]
    matches = [(line, row) for line, row in rows if row['variant_id'] == variant_id]
    assert len(matches) == 1, f'{path}: {len(matches)} rows for {variant_id}'
    return lines[0], matches[0][0], matches[0][1]


def config_params(row: dict) -> dict:
    """A Pareto row's configuration as typed values (the CSV writes integers as floats and no threshold as empty)."""
    # Integers come back from the CSV as '60.0'; a missing threshold is the empty string.
    threshold = row['tracking_accuracy_threshold']
    return dict(classifier=row['classifier'], tilesize=int(float(row['tilesize'])),
                sample_rate=int(float(row['sample_rate'])),
                tracking_accuracy_threshold=float(threshold) if threshold != '' else None,
                relevance_threshold=float(row['relevance_threshold']), tilepadding=row['tilepadding'],
                canvas_scale=float(row['canvas_scale']), tracker=row['tracker'])


def make_plan(args):
    """Choose the operating points and write raw/plan.json (run inside ace's polyis container)."""
    # Pipeline and measurement code is imported here, since `run` executes this file on a host without it.
    import duckdb
    from measurements.common import (DATASET_LABELS, fastest_tetris, file_record, host_info, load_test_tradeoff,
                                     load_tradeoff_points)
    from polyis.io import cache, store
    from polyis.utilities import build_param_str, get_video_frame_count

    datasets = args.datasets or list(DATASET_LABELS)
    # Cached tradeoff tables, and the fastest configuration within each bound (the rule r7o1 and r9o1 use).
    tradeoffs = {d: load_test_tradeoff(d) for d in datasets}
    con = duckdb.connect()
    load_tradeoff_points(con, 'points', tradeoffs)
    chosen = {b: fastest_tetris(con, 'points', float(b)) for b in BOUNDS}
    cache_root = Path(cache.CACHE_DIR)
    out: dict[str, dict] = {}
    # A handful of datasets and configurations; their videos are listed, not analysed.
    for d in datasets:
        _, naive = tradeoffs[d]
        # Test videos and their frame counts as p130 counts them.
        videos = sorted(v for v in os.listdir(store.dataset(d, 'test')) if v.endswith('.mp4'))
        frames = {v: get_video_frame_count(d, v) for v in videos}
        assert sum(frames.values()) == int(naive['frame_count']), f'{d}: frame counts differ from the tradeoff table'
        pareto_path = cache.eval(d, 'pareto-params') / 'pareto_params.csv'
        configs, inputs = {}, {}
        # One configuration per distinct operating point.
        for b in BOUNDS:
            variant_id = chosen[b][d]['config']
            if variant_id in configs:
                continue
            header, line, row = pareto_row(pareto_path, variant_id)
            params = config_params(row)
            # The configuration's directory name must be its variant id, as p060 and p120 name it.
            assert build_param_str(**params) == variant_id, f'{d}: {variant_id} does not encode its Pareto row'
            prunes = params['tracking_accuracy_threshold'] is not None
            # p022's directory name drops tile padding and canvas scale (as p022 builds it).
            prune_param = build_param_str(**{k: params[k] for k in ('classifier', 'tilesize', 'sample_rate', 'tracker',
                                                                     'tracking_accuracy_threshold', 'relevance_threshold')}) if prunes else None
            configs[variant_id] = dict(params=params, pareto_header=header, pareto_line=line, prunes=prunes,
                                       prune_param=prune_param, fps=chosen[b][d]['fps'], hota=chosen[b][d]['accuracy'])
            # Indexing inputs of the stages: classifier, its compilation benchmark, the never-relevant mask, and the
            # mistrack table p022 reads.
            ts, cls = params['tilesize'], params['classifier']
            paths = [cache.index(d, 'training', 'results', f'{cls}_{ts}', 'model_best.pth'),
                     cache.index(d, 'training', 'results', f'{cls}_{ts}', 'model_compilation.jsonl'),
                     cache.index(d, 'never-relevant', f'{ts}_all.npy')]
            if prunes:
                paths.append(cache.index(d, 'track_rates', f'{params["tracker"]}_{ts}', 'max_rate_table.npy'))
                # ace's p022 outputs, copied into a unit when p022 cannot run.
                paths += [cache.exec(d, 'pruned-polyominoes', v, prune_param, 'score', name)
                          for v in videos for name in ('score.jsonl', 'runtime.jsonl')]
            for p in paths:
                record = file_record(p)
                assert record['exists'], f'{d}: input {p} missing'
                inputs[str(Path(p).relative_to(cache_root))] = dict(sha256=record['sha256'], bytes=record['bytes'],
                                                                     staged_from='ace' if '/execution/' in str(p) else 'mirror')
        out[d] = dict(label=DATASET_LABELS[d], videos=frames, frame_count=sum(frames.values()),
                      calibration_videos=evenly_spaced(videos, CALIBRATION_VIDEOS),
                      reference=dict(fps=float(naive['throughput_fps']), hota=float(naive['HOTA_HOTA'])),
                      bounds={b: dict(variant_id=chosen[b][d]['config'], fps=chosen[b][d]['fps'],
                                      hota=chosen[b][d]['accuracy'], bound=chosen[b][d]['bound']) for b in BOUNDS},
                      configs=configs, inputs=inputs,
                      tradeoff=file_record(cache.eval(d, 'tradeoff', 'tradeoff.csv')), pareto=file_record(pareto_path))
    plan = dict(created=now_iso(), host=host_info(), bounds=list(BOUNDS), calibration_bound=CALIBRATION_BOUND,
                calibration_videos=CALIBRATION_VIDEOS, datasets=out, units=plan_units(out))
    RAW.mkdir(parents=True, exist_ok=True)
    write_json(RAW / PLAN_FILE, plan)
    print(f'wrote {RAW / PLAN_FILE}: {len(plan["units"])} units')


# ---------------------------------------------------------------------------------------------------- stage (ace host)

def stage_inputs(args):
    """Push the plan's ace-only inputs (cached p022 outputs) to bombe's scratch inputs, from the ace host."""
    plan = json.loads((RAW / PLAN_FILE).read_text())
    # Cache-relative paths of the inputs that only ace has.
    rels = sorted(r for info in plan['datasets'].values() for r, rec in info['inputs'].items() if rec['staged_from'] == 'ace')
    if not rels:
        print('nothing to stage')
        return
    # rsync reads the list from a file; it lives next to the plan and is gitignored as a .log file.
    listing = RAW / 'stage-files.log'
    listing.write_text('\n'.join(rels) + '\n')
    # The destination must exist before rsync recreates the relative paths under it.
    subprocess.run(['ssh', BOMBE_HOST, 'mkdir', '-p', str(SCRATCH_HOST / 'inputs')], check=True)
    # One rsync of the listed files, read-only on ace's side.
    subprocess.run(['rsync', '-a', f'--files-from={listing}', f'{ACE_CACHE}/', f'{BOMBE_HOST}:{SCRATCH_HOST}/inputs/'],
                   check=True)
    print(f'staged {len(rels)} files to {BOMBE_HOST}:{SCRATCH_HOST}/inputs')


# ---------------------------------------------------------------------------------------------------- run (bombe host)

class Log:
    """Timestamped lines to stdout, safe across the per-GPU threads."""

    def __init__(self):
        # One lock for all threads.
        self.lock = threading.Lock()

    def __call__(self, text: str):
        # Print whole lines under the lock and flush, since stdout goes to a file in tmux.
        with self.lock:
            print(f'{now_iso()} {text}', flush=True)


def gpu_state() -> dict[int, dict]:
    """Every GPU's utilization, memory, UUID, and whether a compute process is on it (nvidia-smi on the host)."""
    # One query for the GPUs and one for compute processes.
    rows = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                          capture_output=True, text=True, check=True).stdout.strip().splitlines()
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'],
                          capture_output=True, text=True, check=True).stdout.split()
    state = {}
    for row in rows:
        index, uuid, util, mem = [x.strip() for x in row.split(',')]
        state[int(index)] = dict(uuid=uuid, util=float(util), mem=float(mem), apps=apps.count(uuid))
    return state


def prepare_inputs(plan: dict, datasets: list[str], log: Log):
    """Copy the plan's mirror inputs into scratch inputs/ and check every input's SHA-256 against ace's."""
    inputs = SCRATCH_HOST / 'inputs'
    # A few files per dataset, plus the staged p022 outputs.
    for d in datasets:
        for rel, rec in plan['datasets'][d]['inputs'].items():
            dst = inputs / rel
            # Mirror inputs are copied (a copy, never a hard link: the mirror stays untouched).
            if rec['staged_from'] == 'mirror' and not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(MIRROR_CACHE / rel, dst)
            # ace-only inputs must have been pushed by `stage`.
            assert dst.exists(), f'{dst} missing: run `{SCRIPT} stage` on ace'
            assert sha256(dst) == rec['sha256'], f'{dst} differs from ace (SHA-256)'
        log(f'{d}: {len(plan["datasets"][d]["inputs"])} inputs match ace')


def launcher_run_id(log_path: Path) -> str | None:
    """The run id the launcher printed ('run <id> <ISO time>: GPU ...'), or None when it refused to start."""
    # The launcher prints its start line once, before the command's output.
    for line in log_path.read_text(errors='replace').splitlines():
        match = re.match(r'^run (g\d+-\d{8}T\d{6}) \d{4}-', line)
        if match:
            return match.group(1)
    return None


def node_cpus(gpu: int) -> list[int]:
    """The CPUs of a GPU's NUMA node, from sysfs as the launcher finds them."""
    # nvidia-smi writes the bus id as 00000000:B2:00.0, sysfs names the device 0000:b2:00.0.
    bus = subprocess.run(['nvidia-smi', '--query-gpu=pci.bus_id', '--format=csv,noheader', '-i', str(gpu)],
                         capture_output=True, text=True, check=True).stdout.strip()
    domain, rest = bus.split(':', 1)
    return parse_cpu_list((SYSFS_PCI / f'{int(domain, 16):04x}:{rest.lower()}' / 'local_cpulist').read_text())


def node_busy(cpus: list[int], seconds: float = NODE_CHECK_S) -> float:
    """Busy share of `cpus` over the next `seconds`, with CpuSampler's /proc/stat accounting."""
    # Two readings `seconds` apart; their differences, summed over the CPUs.
    before = CpuSampler.read()
    time.sleep(seconds)
    after = CpuSampler.read()
    busy = sum(after[c][0] - before[c][0] for c in cpus)
    total = sum(after[c][1] - before[c][1] for c in cpus)
    # No elapsed jiffies cannot show an idle node.
    return busy / total if total else 1.0


def wait_for_idle_node(gpu: int, log: Log) -> dict | None:
    """Wait until the GPU's NUMA node is at most NODE_IDLE_MAX busy over NODE_CHECK_S, checking every BUSY_WAIT_S;
    returns the passing check, or None when the node stayed busy for BUSY_RETRIES checks."""
    cpus = node_cpus(gpu)
    for retry in range(BUSY_RETRIES + 1):
        busy = node_busy(cpus)
        if busy <= NODE_IDLE_MAX:
            return dict(node_busy=busy, node_cpus=cpus, checked=now_iso(), retries=retry)
        log(f'GPU {gpu}\'s node {busy:.3f} busy (retry {retry + 1}/{BUSY_RETRIES}); waiting {BUSY_WAIT_S} s')
        time.sleep(BUSY_WAIT_S)
    return None


def launch(unit: dict, gpu: int, attempt: int, log: Log, gate: dict | None = None) -> dict:
    """One attempt of a unit through the launcher; waits for a busy GPU. `gate` is the idle-node check that preceded
    it (recorded in unit.json, hence in the manifest). Returns the attempt's outcome."""
    out = RAW / UNITS_DIR / unit['unit'] / f'{ATTEMPT_PREFIX}{attempt}'
    out.mkdir(parents=True, exist_ok=True)
    # The unit's scratch directory starts empty (stages delete their own outputs, so nothing may survive).
    unit_dir = SCRATCH_HOST / 'units' / unit['unit']
    shutil.rmtree(unit_dir, ignore_errors=True)
    unit_dir.mkdir(parents=True)
    # ProgressBar's worker logs go to this unit's own directory, mounted where ProgressBar writes them.
    (unit_dir / WORKER_LOGS).mkdir()
    env = dict(os.environ, BOMBE_DOCKER_ARGS=f'-v {unit_dir / WORKER_LOGS}:/polyis-cache/LOGS')
    write_json(unit_dir / 'unit.json', dict(unit, attempt=attempt, gpu=gpu, raw=str(out.relative_to(RAW)), gate=gate))
    command = f'python {SCRIPT} job --unit {unit["unit"]}'
    # Retry while the launcher refuses a busy GPU (run through bash, so it needs no exec permission).
    for retry in range(BUSY_RETRIES + 1):
        with open(out / LAUNCHER_LOG, 'w') as f:
            status = subprocess.run(['bash', str(LAUNCHER), str(gpu), str(SCRATCH_HOST), command], stdout=f,
                                    stderr=subprocess.STDOUT, cwd=CHECKOUT, env=env).returncode
        if status != LAUNCHER_BUSY:
            break
        log(f'GPU {gpu} busy before {unit["unit"]} (retry {retry + 1}/{BUSY_RETRIES}); waiting {BUSY_WAIT_S} s')
        time.sleep(BUSY_WAIT_S)
    assert status != LAUNCHER_BAD_ARGS, f'launcher rejected its arguments for {unit["unit"]}: see {out / LAUNCHER_LOG}'
    run_id = launcher_run_id(out / LAUNCHER_LOG)
    # Copy the launcher's samples of this run next to the job's records.
    if run_id is not None:
        samples = SCRATCH_HOST / 'samples'
        if (samples / f'{run_id}.json').exists():
            shutil.copy2(samples / f'{run_id}.json', out / LAUNCHER_SUMMARY)
        for src, dst in ((f'{run_id}.jsonl', LAUNCHER_SAMPLES), (f'{run_id}.gpus.csv', LAUNCHER_GPUS)):
            if (samples / src).exists():
                gzip_copy(samples / src, out / dst)
    outcome = dict(attempt_outcome(out), launcher_exit=status, run=run_id)
    log(f'{unit["unit"]} attempt {attempt} on GPU {gpu}: {outcome}')
    return outcome


def run_unit(unit: dict, gpu: int, redo: bool, log: Log) -> str:
    """Run a unit until it has a usable attempt (see `usable`). Without --redo, a unit with a usable attempt is done,
    and a unit gets at most MAX_ATTEMPTS attempts in all; with --redo, up to MAX_ATTEMPTS new ones. Returns 'ok', or
    why it stopped: 'failed' (the job failed; not retried, since that failure is in the pipeline, not the GPU),
    'unclean' (every new attempt was contaminated or ran on a busy node), 'exhausted' (no usable attempt and no
    attempts left without --redo), or 'node-busy' (the node never became idle)."""
    phase = unit['phase']
    previous = attempts(unit['unit'], RAW)
    if not redo:
        outcomes = [attempt_outcome(p) for p in previous]
        if any(usable(o, phase) for o in outcomes):
            log(f'{unit["unit"]}: done ({len(previous)} attempts)')
            return 'ok'
        if len(previous) >= MAX_ATTEMPTS:
            log(f'{unit["unit"]}: no usable attempt in {len(previous)}, none left without --redo: {outcomes[-1]}')
            return 'exhausted'
    first = len(previous) + 1
    count = MAX_ATTEMPTS if redo else MAX_ATTEMPTS - len(previous)
    for attempt in range(first, first + count):
        # Every phase but parallel starts only on an idle node.
        gate = None
        if phase != 'parallel':
            gate = wait_for_idle_node(gpu, log)
            if gate is None:
                return 'node-busy'
        outcome = launch(unit, gpu, attempt, log, gate)
        if outcome['status'] != 'ok':
            return 'failed'
        if usable(outcome, phase):
            return 'ok'
    return 'unclean'


def run(args):
    """Copy inputs, check the GPUs, and run the requested phases through the launcher (on bombe)."""
    log = Log()
    plan = json.loads((RAW / PLAN_FILE).read_text())
    datasets = args.datasets or list(plan['datasets'])
    unknown = sorted(set(datasets) - set(plan['datasets']))
    assert not unknown, f'datasets not in the plan: {unknown}'
    bad = sorted(set(args.gpus) - set(GPUS))
    assert not bad, f'this track may use GPUs {GPUS} only, not {bad}'
    # Refuse to start unless every GPU of this run is idle.
    state = gpu_state()
    busy = {g: state[g] for g in args.gpus if state[g]['util'] > 0 or state[g]['mem'] >= 100 or state[g]['apps'] > 0}
    assert not busy, f'GPUs busy, not starting: {busy}'
    prepare_inputs(plan, datasets, log)
    # Every planned dataset to a GPU by estimated parallel-phase cost, so a dataset keeps its GPU in every phase and
    # whichever --datasets are run; the other phases reuse each dataset's GPU.
    costs = {d: sum(unit_cost(plan, u) for u in phase_units(plan, 'parallel', [d])) for d in plan['datasets']}
    assignment = assign_gpus(costs, args.gpus)
    log(f'assignment {assignment} (estimated seconds {dict((d, round(c)) for d, c in costs.items())})')
    results: dict[str, str] = {}
    for phase in [p for p in PHASES if p in args.phases]:
        units = phase_units(plan, phase, datasets)
        if phase == 'parallel':
            # One thread per GPU runs its datasets' units in plan order; the GPUs run in parallel.
            def worker(gpu):
                for u in units:
                    if assignment[u['dataset']] == gpu:
                        results[u['unit']] = run_unit(u, gpu, args.redo, log)
            threads = [threading.Thread(target=worker, args=(g,)) for g in args.gpus]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        else:
            # Other phases run one unit at a time on their dataset's GPU, so no other unit of this job is running.
            for u in sorted(units, key=lambda u: (args.gpus.index(assignment[u['dataset']]), u['dataset'])):
                results[u['unit']] = run_unit(u, assignment[u['dataset']], args.redo, log)
        log(f'phase {phase} done: {sum(v == "ok" for v in results.values())} of {len(results)} units ok so far')
    log(f'results {json.dumps(results, sort_keys=True)}')


# ---------------------------------------------------------------------------------------------------- job (container)

class CpuSampler:
    """Busy fraction of `cpus` and of every other CPU from /proc/stat, and the share of `cpus` this container used
    (its cgroup's usage_usec), once per CPU_SAMPLE_S, labeled with the step. The container runs only on `cpus`, so
    unit_cpus_busy - own_busy is roughly what other processes used of the unit's NUMA node."""

    def __init__(self, cpus: list[int]):
        # The unit's CPUs, the current step label, and the samples so far.
        self.cpus = set(cpus)
        self.step = 'setup'
        self.samples: list[dict] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    @staticmethod
    def cgroup_usage_us() -> int | None:
        """This container's CPU time in microseconds (cgroup v2 cpu.stat), or None when unavailable."""
        # The container's own cgroup is mounted at /sys/fs/cgroup (private cgroup namespace).
        try:
            with open(CGROUP_CPU_STAT) as f:
                for line in f:
                    if line.startswith('usage_usec '):
                        return int(line.split()[1])
        except OSError:
            return None
        return None

    @staticmethod
    def read() -> dict[int, tuple[int, int]]:
        """{cpu: (busy jiffies, total jiffies)} from /proc/stat (busy = total - idle - iowait)."""
        stats = {}
        # Lines 'cpuN user nice system idle iowait irq softirq steal ...'; the aggregate 'cpu' line is skipped.
        with open('/proc/stat') as f:
            for line in f:
                name, *values = line.split()
                if name.startswith('cpu') and name != 'cpu':
                    v = [int(x) for x in values[:8]]
                    stats[int(name[3:])] = (sum(v) - v[3] - v[4], sum(v))
        return stats

    def loop(self):
        # Differences of consecutive readings, summed over the unit's CPUs and over the rest.
        before, usage_before, clock_before = self.read(), self.cgroup_usage_us(), time.monotonic_ns()
        while not self.stop_event.wait(CPU_SAMPLE_S):
            after, usage_after, clock_after = self.read(), self.cgroup_usage_us(), time.monotonic_ns()
            sums = {True: [0, 0], False: [0, 0]}
            for cpu, (busy, total) in after.items():
                b0, t0 = before.get(cpu, (busy, total))
                sums[cpu in self.cpus][0] += busy - b0
                sums[cpu in self.cpus][1] += total - t0
            # The container's CPU time over the interval, as a fraction of the unit CPUs' capacity.
            capacity_us = (clock_after - clock_before) / 1000 * len(self.cpus)
            own = (usage_after - usage_before) / capacity_us if None not in (usage_after, usage_before) else None
            self.samples.append(dict(t_ns=time.time_ns(), step=self.step,
                                     unit_cpus_busy=sums[True][0] / sums[True][1] if sums[True][1] else None,
                                     other_cpus_busy=sums[False][0] / sums[False][1] if sums[False][1] else None,
                                     own_busy=own))
            before, usage_before, clock_before = after, usage_after, clock_after

    def stop(self) -> list[dict]:
        # End the loop and return every sample.
        self.stop_event.set()
        self.thread.join()
        return self.samples


def gurobi_available() -> dict:
    """Whether gurobipy can create a model here (a license is needed for p022's instance sizes), with its error."""
    # A separate process, so a license failure cannot leave state in this one.
    result = subprocess.run([sys.executable, '-c', 'import gurobipy; gurobipy.Model()'], capture_output=True, text=True)
    return dict(available=result.returncode == 0, error=result.stderr.strip().splitlines()[-1] if result.returncode else None)


def build_config(unit_dir: Path, dataset: str) -> Path:
    """The unit's copy of configs/global.yaml: this dataset only, the unit's cache, and the unit's videos directory."""
    # PyYAML is in the polyis image.
    import yaml
    config = yaml.safe_load((CHECKOUT / 'configs' / 'global.yaml').read_text())
    config['EXEC']['DATASETS'] = [dataset]
    config['DATA']['CACHE_DIR'] = f'{unit_dir / "cache"}/'
    config['DATA']['DATASETS_DIR'] = str(unit_dir / 'datasets')
    path = unit_dir / 'global.yaml'
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def prepare_unit(unit: dict, info: dict, unit_dir: Path, copy_p022: bool) -> list[str]:
    """The unit's videos directory (symlinks to the mirror) and cache inputs; returns the copied inputs."""
    d = unit['dataset']
    # Videos: a symlink per unit video, so every stage lists exactly the unit's videos.
    videos_dir = unit_dir / 'datasets' / d / 'test'
    videos_dir.mkdir(parents=True)
    for v in unit['videos']:
        target = Path(CONTAINER_DATASETS) / d / 'test' / v
        assert target.exists(), f'{target} missing from the mirror'
        (videos_dir / v).symlink_to(target)
    cache_root = unit_dir / 'cache'
    copied = []
    # The p022 parameter directory of the unit's configuration (None for the reference and unpruned configurations).
    prune_param = info['configs'][unit['variant_id']]['prune_param'] if unit['system'] == 'tetris' else None
    # Tetris units copy the indexing inputs, and the cached p022 outputs of their own configuration and videos when
    # p022 does not run; reference units need none. p022 inputs are <dataset>/execution/<video>/<022 dir>/<param>/...
    for rel, rec in info['inputs'].items():
        parts = rel.split('/')
        if rec['staged_from'] == 'ace':
            wanted = copy_p022 and parts[2] in unit['videos'] and parts[4] == prune_param
        else:
            wanted = unit['system'] == 'tetris'
        if not wanted:
            continue
        dst = cache_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SCRATCH_CONTAINER / 'inputs' / rel, dst)
        copied.append(rel)
    # The test pass's Pareto filter: the unit's configuration only, its row verbatim from ace's file.
    if unit['system'] == 'tetris':
        config = info['configs'][unit['variant_id']]
        pareto = cache_root / d / 'evaluation' / '095_pareto_params' / 'pareto_params.csv'
        pareto.parent.mkdir(parents=True, exist_ok=True)
        pareto.write_text(config['pareto_header'] + '\n' + config['pareto_line'] + '\n')
    return copied


def preread(unit: dict, unit_dir: Path) -> list[dict]:
    """Read every unit video once (page cache warm), timed; returns per-video bytes and seconds."""
    records = []
    # Whole files in large chunks; the bytes are discarded.
    for v in unit['videos']:
        path = unit_dir / 'datasets' / unit['dataset'] / 'test' / v
        start, size = time.perf_counter_ns(), 0
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(PREREAD_CHUNK), b''):
                size += len(chunk)
        records.append(dict(video=v, bytes=size, seconds=(time.perf_counter_ns() - start) / 1e9))
    return records


def decode_videos(unit: dict, unit_dir: Path, passes: int = DECODE_PASSES) -> list[dict]:
    """Decode every unit video `passes` times with cv2 as p002 reads it (the default VideoCapture, whose FFmpeg
    thread count is recorded, and cap.read() per frame), timed per call, in sorted video order on even passes and
    reversed on odd ones; one record per pass and video."""
    import cv2
    records = []
    for decode_pass in range(passes):
        # Alternating the order keeps one video from always following the warm-up or another video.
        for v in unit['videos'] if decode_pass % 2 == 0 else unit['videos'][::-1]:
            path = unit_dir / 'datasets' / unit['dataset'] / 'test' / v
            # Opening the container is timed apart from the per-frame reads.
            start = time.perf_counter_ns()
            cap = cv2.VideoCapture(str(path))
            assert cap.isOpened(), f'cannot open {path}'
            opened = time.perf_counter_ns()
            frames, read_ns = 0, 0
            # Every frame; the final failed read is not a frame and, as in p002, not counted.
            while True:
                t0 = time.perf_counter_ns()
                ok, _ = cap.read()
                t1 = time.perf_counter_ns()
                if not ok:
                    break
                frames += 1
                read_ns += t1 - t0
            # The decoder's threads (p002 opens the same default capture in the same container).
            threads = cap.get(cv2.CAP_PROP_N_THREADS)
            cap.release()
            records.append(dict(video=v, decode_pass=decode_pass, frames=frames, read_s=read_ns / 1e9,
                                open_s=(opened - start) / 1e9, total_s=(time.perf_counter_ns() - start) / 1e9,
                                threads=threads))
    return records


def stage_files(cache_root: Path, dataset: str, video: str, stage: str) -> list[Path]:
    """The runtime files a stage wrote for one video in a unit's cache (one per configuration)."""
    _, stage_dir, runtime, _ = STAGES[stage]
    # Runtime files sit at or below the stage directory.
    return sorted((cache_root / dataset / 'execution' / video / stage_dir).rglob(runtime))


def run_stage(stage: str, config_path: Path, log_path: Path) -> dict:
    """One stage script as a subprocess with the unit's config; its wall clock and exit code."""
    script = STAGES[stage][0]
    env = dict(os.environ, CONFIG=str(config_path))
    start_ns = time.time_ns()
    t0 = time.perf_counter_ns()
    # The script exactly as the pipeline runs it, on the test split.
    with open(log_path, 'w') as f:
        status = subprocess.run([sys.executable, script, '--test'], cwd=CHECKOUT, env=env, stdout=f,
                                stderr=subprocess.STDOUT).returncode
    t1 = time.perf_counter_ns()
    return dict(stage=stage, script=script, args=['--test'], start_ns=start_ns, end_ns=time.time_ns(),
                wall_s=(t1 - t0) / 1e9, exit=status, ran=True)


def harvest(unit: dict, cache_root: Path, stages: list[str], out: Path) -> dict:
    """Copy every runtime file into out/logs/ (gzip, cache-relative path) and digest each stage's outputs."""
    outputs = {}
    for stage in stages:
        outputs[stage] = {}
        for v in unit['videos']:
            files = stage_files(cache_root, unit['dataset'], v, stage)
            assert len(files) == 1, f'{stage} {v}: {len(files)} runtime files'
            runtime = files[0]
            rel = runtime.relative_to(cache_root)
            gzip_copy(runtime, out / LOGS_DIR / f'{rel}.gz')
            # Output digests in the runtime file's directory, the runtime file's last write (when the stage finished
            # this video), and the canvas count for packing.
            record = dict(runtime=str(rel), files={}, mtime_ns=runtime.stat().st_mtime_ns)
            for name in STAGES[stage][3]:
                path = runtime.parent / name
                record['files'][name] = dict(sha256=sha256(path), bytes=path.stat().st_size) if path.exists() else None
            if stage == '030_exec_compress':
                record['canvases'] = len(list((runtime.parent / 'images').glob('*.jpg')))
            outputs[stage][v] = record
    return outputs


def job(args):
    """Run one unit inside a launcher container (see the module docstring)."""
    unit_dir = SCRATCH_CONTAINER / 'units' / args.unit
    unit = json.loads((unit_dir / 'unit.json').read_text())
    plan = json.loads((RAW / PLAN_FILE).read_text())
    info = plan['datasets'][unit['dataset']]
    out = RAW / unit['raw']
    out.mkdir(parents=True, exist_ok=True)
    cpus = sorted(os.sched_getaffinity(0))
    sampler = CpuSampler(cpus)
    manifest = dict(unit=unit, started=now_iso(), started_ns=time.time_ns(), status='failed', error=None, stages=[],
                    environment=dict(hostname=socket.gethostname(), python=sys.version.split()[0], cpus=cpus,
                                     bombe_gpu=os.environ.get('BOMBE_GPU'), bombe_cpus=os.environ.get('BOMBE_CPUS'),
                                     bombe_run=os.environ.get('BOMBE_RUN'),
                                     nvidia_smi=subprocess.run(['nvidia-smi', '-L'], capture_output=True,
                                                               text=True).stdout.strip()),
                    files=dict(cpu=CPU_SAMPLES, logs=LOGS_DIR))
    try:
        # Which stages run: the reference's two, or Tetris's with p022 only for a pruning configuration.
        if unit['system'] == 'reference':
            stages = [s[0] for s in REFERENCE_STAGES]
            p022_source = None
        else:
            prunes = info['configs'][unit['variant_id']]['prunes']
            stages = [s[0] for s in TETRIS_STAGES if prunes or s[0] != PRUNE_STAGE]
            gurobi = gurobi_available() if prunes else None
            manifest['gurobi'] = gurobi
            p022_source = None if not prunes else ('rerun' if gurobi['available'] else 'cache')
        manifest['p022_source'] = p022_source
        config_path = build_config(unit_dir, unit['dataset'])
        manifest['inputs_copied'] = prepare_unit(unit, info, unit_dir, copy_p022=p022_source == 'cache')
        cache_root = unit_dir / 'cache'
        # Page cache warm-up, then (reference units) the separately timed decode.
        sampler.step = 'preread'
        manifest['preread'] = preread(unit, unit_dir)
        if unit['system'] == 'reference':
            sampler.step = 'decode'
            decode = decode_videos(unit, unit_dir, DECODE_PASSES)
            write_jsonl_gz(out / DECODE_FILE, decode)
            manifest['files']['decode'] = DECODE_FILE
            # Every pass must see the frames the plan counts, or it measured something else.
            wrong = {(r['video'], r['decode_pass']): r['frames'] for r in decode if r['frames'] != info['videos'][r['video']]}
            assert not wrong, f'decoded frame counts differ from the plan: {wrong}'
        # Each stage in order; p022 from the cache is recorded as not run.
        for stage in stages:
            if stage == PRUNE_STAGE and p022_source == 'cache':
                manifest['stages'].append(dict(stage=stage, ran=False, source='cache'))
                continue
            sampler.step = stage
            record = run_stage(stage, config_path, out / f'{stage}.log')
            manifest['stages'].append(record)
            assert record['exit'] == 0, f'{stage} exited {record["exit"]} (see {out / f"{stage}.log"})'
            # A stage whose worker failed still exits 0, so its outputs are checked: one runtime file per video.
            counts = {v: len(stage_files(cache_root, unit['dataset'], v, stage)) for v in unit['videos']}
            wrong = {v: n for v, n in counts.items() if n != 1}
            assert not wrong, f'{stage}: runtime files per video {wrong} (see {out / f"{stage}.log"})'
        sampler.step = 'harvest'
        manifest['outputs'] = harvest(unit, cache_root, stages, out)
        manifest['status'] = 'ok'
    except Exception as e:
        # The failure is recorded; the orchestrator does not retry a failed job.
        manifest['error'] = f'{type(e).__name__}: {e}'
        raise
    finally:
        # Worker logs of the pipeline's ProgressBar (the per-unit directory `launch` mounts at /polyis-cache/LOGS),
        # kept as .log files (gitignored) for diagnosis.
        logs = unit_dir / WORKER_LOGS
        if logs.exists():
            for f in logs.rglob('*'):
                if f.is_file():
                    dst = out / WORKER_LOGS / f'{f.relative_to(logs)}.log'
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dst)
        write_jsonl_gz(out / CPU_SAMPLES, sampler.stop())
        manifest['finished'] = now_iso()
        manifest['finished_ns'] = time.time_ns()
        # The manifest is written last: its presence marks a finished attempt.
        write_json(out / MANIFEST, manifest)


def main(args):
    # Dispatch to the subcommand.
    {'plan': make_plan, 'stage': stage_inputs, 'run': run, 'job': job}[args.command](args)


if __name__ == '__main__':
    main(parse_args())
