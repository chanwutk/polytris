import csv
import gzip
import hashlib
import json
import os
import shutil
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from evaluation.p121_throughput_compute import (QUERY_DATA_ACCESSORS, aggregate_per_op, filter_excluded_runtime_ops,
                                                parse_runtime_file)
from measurements import r1o3_clean_extraction_timing as ct
from measurements.common import OP_LIST_PATHS


# The GPU job, as the aggregator loads it.
gpu = ct.gpu

# One dataset with two test videos (frames) and one calibration video; two Tetris configurations: A without pruning
# (the 5% point) and B with pruning (the 10% point). Both read the same p020 output, as configurations sharing a
# classifier, tile size, and sampling rate do.
DATASET = 'jnc0'
VIDEOS = {'te01.mp4': 3, 'te04.mp4': 2}
VIDEO_INDEX = {v: i for i, v in enumerate(VIDEOS)}
CALIBRATION = ['te01.mp4']
A = 'Net_60_1_r025_none_s100_sort'
B = 'Net_60_1_040_r050_none_s100_sort'
PRUNE_PARAM_B = 'Net_60_1_040_r050_sort'
BOUNDS = {'0.05': A, '0.10': B}
# Canvases per video.
CANVASES = 1
# Every op's time: its base ms x the run's scale x (1 + op jitter x op index) x (1 + video jitter x video index), so
# every op and video has its own ratio between two runs.
SCALES = {('clean', 'serial'): 1.0, ('paper', 'serial'): 1.5, ('clean', 'parallel'): 1.25, ('clean', 'solo'): 0.8}
OP_JITTER = {('clean', 'serial'): 0.0, ('paper', 'serial'): 0.1, ('clean', 'parallel'): 0.05, ('clean', 'solo'): 0.02}
VIDEO_JITTER = {('clean', 'serial'): 0.0, ('paper', 'serial'): 0.2, ('clean', 'parallel'): 0.1, ('clean', 'solo'): 0.3}
# The cached p022 outputs carry the paper's times in every unit.
PAPER = ('paper', 'serial')
# Measured decode, ms per frame of each pass, per phase, scaled per video by DECODE_VIDEO_JITTER: serial decodes three
# times; the earlier phases decoded once, in the old record format without pass or thread fields.
DECODE_MS = {'serial': [1.0, 1.4, 0.8], 'parallel': [1.2], 'solo': [0.9]}
DECODE_VIDEO_JITTER = 0.1
# Stage process wall clocks (s) in the manifests.
WALLS = {'001_preprocess_groundtruth_detection': 10.0, '002_preprocess_groundtruth_tracking': 2.0, '020_exec_classify': 3.0,
         '030_exec_compress': 1.0, '040_exec_detect': 2.0, '050_exec_uncompress': 0.5, '060_exec_track': 0.5}
# Ops p121 must drop, written down independently of the aggregator: everywhere, and per stage.
P121_IGNORED = {'total_frame_time', 'read_frame', 'save_canvas', 'save_mapping_files'}
P121_EXCLUDED = {'020_exec_classify': {'read', 'retrieve'}, '030_exec_compress': {'save_collage', 'pack_all_total'}}
# Counted ops that decode video and that read canvases.
DECODE = {('001_preprocess_groundtruth_detection', 'read')}
CANVAS_READ = {('040_exec_detect', 'read')}
# CPU samples: the unit's own share of its node, and the node's busy share per phase (serial units ran on an idle
# node, foreign 0.02; the others did not, 0.3).
OWN_BUSY = 0.2
NODE_BUSY = {'serial': 0.22, 'parallel': 0.5, 'solo': 0.5}


def records(stage: str, frames: int) -> list[list[tuple[str, object]]]:
    """A stage's runtime records for one video, each an op list of (op, base ms) in the stage's own shape."""
    per_frame = {'001_preprocess_groundtruth_detection': [('read', 2.0), ('detect', 40.0)],
                 '002_preprocess_groundtruth_tracking': [('convert_detections', 0.1), ('tracker_update', 0.5),
                                                         ('interpolate_trajectory', 0.4)],
                 '060_exec_track': [('convert_detections', 0.1), ('tracker_update', 0.2), ('interpolate_trajectory', 0.2)]}
    if stage in per_frame:
        return [per_frame[stage]] * frames
    if stage == '020_exec_classify':
        return [[('inference', 4.0), ('send', 1.0)], [('read', 50.0), ('retrieve', 5.0)]]
    if stage == '022_exec_prune_polyominoes':
        return [[('prepare_bitmaps', 1.0), ('group_tiles', 2.0), ('build_ilp', 3.0), ('solve_ilp', 4.0), ('extract_grids', 1.0)]]
    if stage == '030_exec_compress':
        return ([[('get_classifications', 0.1), ('create_bitmap', 0.2), ('group_tiles', 0.2)]] * frames
                + [[('pack_batch', 0.5)]]
                + [[('initialize_canvas', 1.0), ('render_copy', 2.0), ('save_canvas', 7.0), ('save_mapping_files', 3.0),
                    ('save_collage', 11.0)]] * CANVASES)
    if stage == '040_exec_detect':
        return [[('image_file', 'c.jpg'), ('read', 1.0), ('detect', 10.0)]] * CANVASES
    assert stage == '050_exec_uncompress'
    return [[('unpack_detections', 0.3)] * CANVASES + [('sort_frames', 0.2)]]


# Every op name, indexed for the per-op jitter.
ALL_STAGES = [s[0] for s in (*gpu.REFERENCE_STAGES, *gpu.TETRIS_STAGES)]
OP_INDEX = {op: i for i, op in enumerate(sorted({op for s in ALL_STAGES for ops in records(s, 1) for op, _ in ops}))}


def factor(key: tuple[str, str], op: str, video: str) -> float:
    """The time factor of an op of a video in run `key` (side, phase)."""
    return SCALES[key] * (1 + OP_JITTER[key] * OP_INDEX[op]) * (1 + VIDEO_JITTER[key] * VIDEO_INDEX[video])


def line(stage: str, ops: list[tuple[str, object]], key: tuple[str, str], video: str) -> str:
    """One runtime-log line: the op list (times scaled for run `key`) wrapped as the stage writes it."""
    items = [{'op': op, 'time': t * factor(key, op, video) if isinstance(t, float) else t} for op, t in ops]
    if stage in ('002_preprocess_groundtruth_tracking', '060_exec_track'):
        return json.dumps({'frame_idx': 0, 'runtime': items, 'num_detections': 1})
    if stage == '022_exec_prune_polyominoes':
        return json.dumps({'runtime': items, 'time_limit': None})
    if stage == '030_exec_compress':
        return json.dumps({'step': 'group_tiles', 'runtime': items})
    return json.dumps(items)


def op_seconds(stage: str, video: str, key: tuple[str, str]) -> dict[str, dict]:
    """Oracle: per numeric op of a stage's video in run `key`, its seconds, items, and whether p121 counts it."""
    out: dict[str, dict] = {}
    for ops in records(stage, VIDEOS[video]):
        for op, t in ops:
            if not isinstance(t, float):
                continue
            entry = out.setdefault(op, dict(seconds=0.0, items=0,
                                            counted=op not in P121_IGNORED and op not in P121_EXCLUDED.get(stage, ())))
            entry['seconds'] += t * factor(key, op, video) / 1000
            entry['items'] += 1
    return out


def rel_path(stage: str, video: str, variant: str | None) -> str:
    """The cache-relative runtime path of a stage, as the pipeline names it."""
    base = f'{DATASET}/execution/{video}'
    param = (variant or '').replace('_sort', '')
    return {'001_preprocess_groundtruth_detection': f'{base}/002_naive/detection_runtime.jsonl',
            '002_preprocess_groundtruth_tracking': f'{base}/002_naive/tracking_runtime.jsonl',
            '020_exec_classify': f'{base}/020_relevancy/Net_60_1/score/runtime.jsonl',
            '022_exec_prune_polyominoes': f'{base}/022_pruned_polyominoes/{PRUNE_PARAM_B}/score/runtime.jsonl',
            '030_exec_compress': f'{base}/033_compressed_frames/{param}/runtime.jsonl',
            '040_exec_detect': f'{base}/040_compressed_detections/{param}/runtimes.jsonl',
            '050_exec_uncompress': f'{base}/050_uncompressed_detections/{param}/runtime.jsonl',
            '060_exec_track': f'{base}/060_uncompressed_tracks/{variant}/runtimes.jsonl'}[stage]


def unit_stages(system: str, variant: str | None) -> list[str]:
    """The stages a unit runs."""
    if system == 'reference':
        return [s[0] for s in gpu.REFERENCE_STAGES]
    return [s[0] for s in gpu.TETRIS_STAGES if variant == B or s[0] != gpu.PRUNE_STAGE]


def write_log(path: Path, stage: str, video: str, key: tuple[str, str], compress: bool, frames: int | None = None):
    """A stage's runtime log for one video in run `key`, gzip-compressed (harvested) or plain (ace's cache)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = ''.join(line(stage, ops, key, video) + '\n' for ops in records(stage, frames or VIDEOS[video]))
    if compress:
        with gzip.open(path, 'wt') as f:
            f.write(text)
    else:
        path.write_text(text)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def iso(t: float) -> str:
    """An epoch time as the launcher's date -Is writes it (with a UTC offset)."""
    return datetime.fromtimestamp(t, timezone.utc).astimezone().isoformat(timespec='seconds')


def decode_total(phase: str, video: str, decode_pass: int) -> float:
    """Oracle: the measured decode seconds of a video in one pass."""
    return VIDEOS[video] * DECODE_MS[phase][decode_pass] * (1 + DECODE_VIDEO_JITTER * VIDEO_INDEX[video]) / 1000


def plan() -> dict:
    """The GPU job's plan for the synthetic dataset."""
    configs = {A: dict(prunes=False, prune_param=None, fps=0.0, hota=0.8), B: dict(prunes=True, prune_param=PRUNE_PARAM_B, fps=0.0, hota=0.76)}
    info = dict(label='B3D1', videos=VIDEOS, frame_count=sum(VIDEOS.values()), calibration_videos=CALIBRATION,
                reference=dict(fps=0.0, hota=0.8), configs=configs, inputs={},
                bounds={b: dict(variant_id=v, fps=0.0, hota=0.8, bound=0.7) for b, v in BOUNDS.items()})
    # Throughput as the tradeoff table would hold it: frames over the paper's p121 seconds.
    frames = sum(VIDEOS.values())
    info['reference']['fps'] = frames / oracle('serial', 'paper', 'reference', None, list(VIDEOS))['p121']
    for b, v in BOUNDS.items():
        info['bounds'][b]['fps'] = configs[v]['fps'] = frames / oracle('serial', 'paper', 'tetris', v, list(VIDEOS))['p121']
    datasets = {DATASET: info}
    return dict(created='2026-10-27T00:00:00+00:00', datasets=datasets, units=gpu.plan_units(datasets))


def write_attempt(raw: Path, cache_root: Path, unit: dict, attempt: int, contaminated: bool = False,
                  base_t: float = 1.8e9, node_busy: float | None = None) -> Path:
    """One finished attempt of a unit: manifest, harvested logs, decode, CPU and launcher samples. Serial units (the
    headline) also write ace's cached logs and outputs (te04's tracking differs) and, as the job does from the serial
    phase on, each runtime file's last write and the idle-node check."""
    out = raw / gpu.UNITS_DIR / unit['unit'] / f'{gpu.ATTEMPT_PREFIX}{attempt}'
    out.mkdir(parents=True)
    phase, variant = unit['phase'], unit['variant_id']
    key = ('clean', phase)
    stages = unit_stages(unit['system'], variant)
    outputs, manifest_stages = {}, []
    for i, stage in enumerate(stages):
        outputs[stage] = {}
        cached_022 = stage == gpu.PRUNE_STAGE
        start = base_t + 10 * i
        for j, video in enumerate(unit['videos']):
            rel = rel_path(stage, video, variant)
            # p022 from the cache logs the paper's times, in both the harvested and the cached copy.
            write_log(out / gpu.LOGS_DIR / f'{rel}.gz', stage, video, PAPER if cached_022 else key, True)
            files = {}
            for name in gpu.STAGES[stage][3]:
                text = f'{stage} {video} output'
                files[name] = dict(sha256=digest(text), bytes=len(text))
                if phase == 'serial':
                    cached = cache_root / Path(rel).parent / name
                    cached.parent.mkdir(parents=True, exist_ok=True)
                    cached.write_text(text if not (video == 'te04.mp4' and name == 'tracking.jsonl') else 'other')
            outputs[stage][video] = dict(runtime=rel, files=files)
            if phase == 'serial':
                # The j-th video's runtime file was last written 1 + j s after its stage started.
                outputs[stage][video]['mtime_ns'] = int((start + 1 + j) * 1e9)
                write_log(cache_root / rel, stage, video, PAPER, False)
        manifest_stages.append(dict(stage=stage, ran=False, source='cache') if cached_022 else
                               dict(stage=stage, ran=True, start_ns=int(start * 1e9), end_ns=int((start + 5) * 1e9),
                                    wall_s=WALLS[stage], exit=0))
    files = dict(cpu=gpu.CPU_SAMPLES, logs=gpu.LOGS_DIR)
    if unit['system'] == 'reference':
        files['decode'] = gpu.DECODE_FILE
        rows = []
        for p in range(len(DECODE_MS[phase])):
            for v in unit['videos']:
                total = decode_total(phase, v, p)
                row = dict(video=v, frames=VIDEOS[v], read_s=total - 0.0001, open_s=0.0001, total_s=total)
                rows.append(dict(row, decode_pass=p, threads=16.0) if phase == 'serial' else row)
        gpu.write_jsonl_gz(out / gpu.DECODE_FILE, rows)
    gate = dict(node_busy=0.01) if phase == 'serial' else None
    manifest = dict(unit=dict(unit, attempt=attempt, gate=gate), status='ok', stages=manifest_stages, outputs=outputs,
                    files=files, p022_source='cache' if variant == B else None)
    (out / gpu.MANIFEST).write_text(json.dumps(manifest))
    (out / gpu.LAUNCHER_SUMMARY).write_text(json.dumps(dict(
        gpu=1, numa_node=0, cpus='0-19,40-59', contaminated=contaminated, foreign_samples=3 if contaminated else 0,
        start='s', end='e')))
    # GPU samples: one inside every stage window at 50% (60% in the first stage), and one outside all windows at 99%.
    samples = [dict(t=iso(base_t + 10 * i + 1), util=60.0 if i == 0 else 50.0, mem_mib=1000.0, pids=[], foreign=0, load_1m=4.0)
               for i in range(len(stages))] + [dict(t=iso(base_t - 100), util=99.0, mem_mib=0.0, pids=[], foreign=0, load_1m=8.0)]
    gpu.write_jsonl_gz(out / gpu.LAUNCHER_SAMPLES, samples)
    # nvidia-smi's padded CSV of every GPU, its last line cut by the sampler's exit.
    with gzip.open(out / gpu.LAUNCHER_GPUS, 'wt') as f:
        f.write('2026/10/27 06:00:00.000, 1, 50, 1000\n2026/10/27 06:00:00.000, 3, 100, 10785\n'
                '2026/10/27 06:00:01.000, 1, 70, 1200\n2026/10/27 06:00:01.000, 3\n')
    # CPU samples: one per step, 10 s apart from base_t; this unit uses OWN_BUSY of its node.
    busy = NODE_BUSY[phase] if node_busy is None else node_busy
    gpu.write_jsonl_gz(out / gpu.CPU_SAMPLES, [
        dict(t_ns=int((base_t + 10 * j + 1) * 1e9), step=s, unit_cpus_busy=busy, other_cpus_busy=0.1, own_busy=OWN_BUSY)
        for j, s in enumerate(['preread', *stages])])
    return out


def write_p121_table(cache_root: Path):
    """p121's per-op table for the paper's cached logs, made by p121's own parser."""
    path = cache_root / DATASET / 'evaluation' / '080_throughput' / 'measurements' / 'query_execution_per_op.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ['op', 'time', 'count', 'stage', 'dataset', 'videoset', 'video', 'variant', 'variant_id', 'classifier',
               'tilesize', 'tilepadding', 'sample_rate', 'tracking_accuracy_threshold', 'canvas_scale', 'tracker']
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for system, variant in [('reference', None), ('tetris', A), ('tetris', B)]:
            for stage in unit_stages(system, variant):
                for video in VIDEOS:
                    per_op = aggregate_per_op(filter_excluded_runtime_ops(parse_runtime_file(
                        str(cache_root / rel_path(stage, video, variant)), stage, QUERY_DATA_ACCESSORS[stage]), stage))
                    for r in per_op.itertuples():
                        writer.writerow([r.op, r.time, r.count, stage, DATASET, 'test', video,
                                         'naive' if variant is None else 'polytris', variant or 'naive'] + [''] * 7)
    return path


def all_units(p: dict) -> list[dict]:
    """The plan's units of every aggregated phase."""
    return [u for phase in ct.PHASES for u in gpu.phase_units(p, phase, [DATASET])]


def unit_named(p: dict, name: str) -> dict:
    return next(u for u in all_units(p) if u['unit'] == name)


@pytest.fixture
def world(tmp_path):
    """The raw directory (plan and attempts) and ace's cache (logs, outputs, p121 table)."""
    raw, cache_root = tmp_path / 'raw', tmp_path / 'cache'
    raw.mkdir()
    p = plan()
    (raw / gpu.PLAN_FILE).write_text(json.dumps(p))
    for k, unit in enumerate(all_units(p)):
        # Each unit runs in its own time window, except A's parallel unit, which runs alongside the parallel reference
        # (k = 0), as units on two GPUs of the parallel phase do.
        if unit['unit'] == f'parallel-{DATASET}-{A}':
            # A contaminated first attempt in its own window, then the clean one beside the parallel reference.
            write_attempt(raw, cache_root, unit, 1, contaminated=True, base_t=1.8e9 + 50000)
            write_attempt(raw, cache_root, unit, 2, base_t=1.8e9)
        else:
            write_attempt(raw, cache_root, unit, 1, base_t=1.8e9 + 1000 * k)
    write_p121_table(cache_root)
    return raw, cache_root, p


def oracle(phase: str, side: str, system: str, variant: str | None, videos: list[str]) -> dict[str, float]:
    """Expected boundary seconds of a system over `videos`, with the decode boundaries' fastest and slowest pass."""
    key = (side, phase)
    totals = dict(p121=0.0, decode_ops=0.0, canvas_read=0.0)
    for stage in unit_stages(system, variant):
        for v in videos:
            # p022 from the cache always carries the paper's times.
            for op, e in op_seconds(stage, v, PAPER if stage == gpu.PRUNE_STAGE else key).items():
                if e['counted']:
                    totals['p121'] += e['seconds']
                    totals['decode_ops'] += e['seconds'] if (stage, op) in DECODE else 0.0
                    totals['canvas_read'] += e['seconds'] if (stage, op) in CANVAS_READ else 0.0
    # The reference's logged read in this side's run; the measured decode's passes (the paper's one decode is its
    # logged read).
    logged = sum(op_seconds('001_preprocess_groundtruth_detection', v, key)['read']['seconds'] for v in videos)
    passes = ([sum(decode_total(phase, v, p) for v in videos) for p in range(len(DECODE_MS[phase]))]
              if side == 'clean' else [logged])
    excluded = totals['p121'] - totals['decode_ops']
    out = dict(p121=totals['p121'], decode_excluded=excluded, one_decode_logged=excluded + logged)
    for name, d in (('', statistics.median(passes)), ('_min', min(passes)), ('_max', max(passes))):
        out[f'one_decode{name}'] = excluded + d
        out[f'one_decode_no_canvas_read{name}'] = excluded + d - totals['canvas_read']
    return out


def op_ratio(a: tuple[str, str], b: tuple[str, str], stage: str, op: str, videos: list[str]) -> float:
    """Oracle: an op's seconds in run a over run b on `videos` (p022 from the cache is the paper's in both)."""
    ka, kb = (PAPER, PAPER) if stage == gpu.PRUNE_STAGE else (a, b)
    return (sum(op_seconds(stage, v, ka)[op]['seconds'] for v in videos)
            / sum(op_seconds(stage, v, kb)[op]['seconds'] for v in videos))


# ------------------------------------------------------------------------------------------------ the GPU job's helpers


def test_stage_tables_match_p120():
    # Every stage the GPU job runs has p121's op-list path, and the stage names are p120's.
    from evaluation.p121_throughput_compute import STAGE_TO_SCRIPT_PATH
    assert set(gpu.STAGES) <= set(OP_LIST_PATHS) and set(gpu.STAGES) <= set(STAGE_TO_SCRIPT_PATH)
    assert [s[0] for s in gpu.TETRIS_STAGES] == ['020_exec_classify', '022_exec_prune_polyominoes', '030_exec_compress',
                                                 '040_exec_detect', '050_exec_uncompress', '060_exec_track']
    # The scripts exist in this checkout.
    assert all((gpu.CHECKOUT / s[0]).exists() for s in gpu.STAGES.values())


def test_parse_cpu_list():
    assert gpu.parse_cpu_list('0-2,40-41') == [0, 1, 2, 40, 41]
    assert gpu.parse_cpu_list('7\n') == [7]
    assert gpu.parse_cpu_list('') == []


@pytest.mark.parametrize('n, k', [(60, 6), (5, 6), (6, 6), (7, 6), (10, 1), (2, 2), (61, 6)])
def test_evenly_spaced(n, k):
    items = [f'te{i:02d}' for i in range(n)]
    got = gpu.evenly_spaced(items, k)
    # min(n, k) distinct items in order, with both ends when k > 1.
    assert len(got) == min(n, k) == len(set(got)) and got == sorted(got)
    assert got[0] == items[0] and (k == 1 or got[-1] == items[-1])


def test_evenly_spaced_sixty():
    assert gpu.evenly_spaced([f'te{i:02d}' for i in range(60)], 6) == ['te00', 'te12', 'te24', 'te35', 'te47', 'te59']


def test_assign_gpus_lpt_and_ties():
    # Longest first onto the least loaded GPU; equal loads go to the earlier GPU, equal costs by name.
    costs = {'ams': 45, 'cal1': 28, 'cal2': 32, 'b1': 8, 'b2': 8, 'b3': 8, 'b4': 8}
    got = gpu.assign_gpus(costs, [0, 1, 2])
    assert got == {'ams': 0, 'cal2': 1, 'cal1': 2, 'b1': 2, 'b2': 1, 'b3': 2, 'b4': 1}
    assert gpu.assign_gpus({'x': 1, 'y': 1}, [2, 0]) == {'x': 2, 'y': 0}
    assert gpu.assign_gpus({}, [0]) == {}


def test_plan_units_dedupes_and_calibrates():
    info = dict(videos={'te00.mp4': 1, 'te01.mp4': 1}, calibration_videos=['te00.mp4'],
                bounds={'0.05': dict(variant_id='X'), '0.10': dict(variant_id='X')})
    units = gpu.plan_units({'d': info})
    # One Tetris unit when both bounds pick the same configuration; solo runs the 5% point on the calibration videos.
    assert [u['unit'] for u in units] == ['parallel-d-reference', 'parallel-d-X', 'solo-d-reference', 'solo-d-X']
    assert units[0]['videos'] == ['te00.mp4', 'te01.mp4'] and units[3]['videos'] == ['te00.mp4']
    assert units[0]['variant_id'] is None and units[1]['system'] == 'tetris'


def test_phase_units_derived_phases():
    p = plan()
    smoke = gpu.phase_units(p, 'smoke', [DATASET])
    assert [u['unit'] for u in smoke] == [f'smoke-{DATASET}-reference', f'smoke-{DATASET}-{A}']
    assert all(u['videos'] == CALIBRATION[:1] and u['phase'] == 'smoke' for u in smoke)
    # Serial units are the parallel units, renamed, on all of their videos.
    serial = gpu.phase_units(p, 'serial', [DATASET])
    parallel = gpu.phase_units(p, 'parallel', [DATASET])
    assert [u['unit'] for u in serial] == [f'serial-{DATASET}-reference', f'serial-{DATASET}-{A}', f'serial-{DATASET}-{B}']
    assert [dict(u, unit=None, phase=None) for u in serial] == [dict(u, unit=None, phase=None) for u in parallel]
    assert all(u['phase'] == 'serial' and u['videos'] == list(VIDEOS) for u in serial)
    assert gpu.phase_units(p, 'parallel', ['other']) == [] and gpu.phase_units(p, 'serial', ['other']) == []


def test_unit_cost_uses_cached_throughput():
    p = plan()
    ref = gpu.unit_cost(p, p['units'][0])
    assert ref == pytest.approx(5 / p['datasets'][DATASET]['reference']['fps'] + 2 * gpu.EST_VIDEO_OVERHEAD_S)


def test_launcher_run_id(tmp_path):
    log = tmp_path / 'launcher.log'
    log.write_text('run g1-20261027T061018 2026-10-27T06:10:18-07:00: GPU 1 ...\ncommand: x\nrun g1-20261027T061018 exit 0\n')
    assert gpu.launcher_run_id(log) == 'g1-20261027T061018'
    log.write_text('GPU 1 busy (utilization 5%, 0 MiB, or a compute process); not starting\n')
    assert gpu.launcher_run_id(log) is None


def write_cpu(path: Path, rows: list[tuple[str, float | None, float | None]]):
    """An attempt's CPU samples: (step, node busy share, own share) per second."""
    gpu.write_jsonl_gz(path, [dict(t_ns=i * 10**9, step=s, unit_cpus_busy=b, other_cpus_busy=0.0, own_busy=o)
                              for i, (s, b, o) in enumerate(rows)])


def test_attempts_numeric_order_and_outcome(tmp_path):
    for n in (10, 2, 1):
        (tmp_path / gpu.UNITS_DIR / 'u' / f'{gpu.ATTEMPT_PREFIX}{n}').mkdir(parents=True)
    (tmp_path / gpu.UNITS_DIR / 'u' / 'attempt-x').mkdir()
    got = gpu.attempts('u', tmp_path)
    assert [p.name for p in got] == ['attempt-1', 'attempt-2', 'attempt-10']
    # Missing records read as unknown; present ones as written.
    assert gpu.attempt_outcome(got[0]) == dict(status=None, contaminated=None, others_on_node=None)
    (got[0] / gpu.MANIFEST).write_text(json.dumps(dict(status='ok')))
    (got[0] / gpu.LAUNCHER_SUMMARY).write_text(json.dumps(dict(contaminated=True)))
    write_cpu(got[0] / gpu.CPU_SAMPLES, [('020_exec_classify', 0.5, 0.2)])
    assert gpu.attempt_outcome(got[0]) == dict(status='ok', contaminated=True, others_on_node=pytest.approx(0.3))


def test_others_on_node_skips_untimed_steps_and_gaps(tmp_path):
    path = tmp_path / gpu.CPU_SAMPLES
    # Setup, warm-up, and harvesting are not timed; a sample without the cgroup share is unusable.
    write_cpu(path, [('setup', 0.9, 0.0), ('preread', 0.9, 0.0), ('decode', 0.3, 0.2), ('001_x', 0.5, 0.2),
                     ('001_x', 0.5, None), ('harvest', 0.9, 0.0)])
    assert gpu.others_on_node(path) == pytest.approx((0.1 + 0.3) / 2)
    write_cpu(path, [('setup', 0.9, 0.0)])
    assert gpu.others_on_node(path) is None
    assert gpu.others_on_node(tmp_path / 'missing.jsonl.gz') is None


@pytest.mark.parametrize('outcome, phase, expected', [
    (dict(status='ok', contaminated=False, others_on_node=0.01), 'serial', True),
    (dict(status='ok', contaminated=False, others_on_node=gpu.FOREIGN_CPU_MAX), 'solo', True),
    (dict(status='ok', contaminated=False, others_on_node=0.06), 'serial', False),
    (dict(status='ok', contaminated=False, others_on_node=None), 'serial', False),
    (dict(status='ok', contaminated=False, others_on_node=0.4), 'parallel', True),
    (dict(status='ok', contaminated=True, others_on_node=0.0), 'parallel', False),
    (dict(status='ok', contaminated=None, others_on_node=0.0), 'serial', False),
    (dict(status='failed', contaminated=False, others_on_node=0.0), 'serial', False),
])
def test_usable(outcome, phase, expected):
    assert gpu.usable(outcome, phase) is expected


def test_pareto_row_and_params(tmp_path):
    path = tmp_path / 'pareto_params.csv'
    header = 'dataset,variant_id,classifier,tilesize,sample_rate,tracking_accuracy_threshold,relevance_threshold,tilepadding,canvas_scale,tracker'
    rows = ['d,X,Net,60.0,4.0,,0.25,bl,1.0,sort', 'd,Y,Net,60.0,8.0,0.8,0.5,tr,1.0,sort', 'd,Y,Net,60.0,8.0,0.8,0.5,tr,1.0,sort']
    path.write_text('\n'.join([header, *rows]) + '\n')
    h, line_, row = gpu.pareto_row(path, 'X')
    # The lines come back verbatim; the parameters typed (no threshold is None).
    assert (h, line_) == (header, rows[0])
    assert gpu.config_params(row) == dict(classifier='Net', tilesize=60, sample_rate=4, tracking_accuracy_threshold=None,
                                          relevance_threshold=0.25, tilepadding='bl', canvas_scale=1.0, tracker='sort')
    assert gpu.config_params(gpu.pareto_row(path, 'X')[2])['tilesize'] == 60
    with pytest.raises(AssertionError, match='2 rows'):
        gpu.pareto_row(path, 'Y')
    with pytest.raises(AssertionError, match='0 rows'):
        gpu.pareto_row(path, 'Z')


def test_stage_files_and_harvest(tmp_path):
    cache_root = tmp_path / 'cache'
    unit = dict(dataset=DATASET, videos=['te01.mp4'])
    d = cache_root / DATASET / 'execution' / 'te01.mp4' / '033_compressed_frames' / 'P'
    (d / 'images').mkdir(parents=True)
    (d / 'runtime.jsonl').write_text('{"runtime": []}\n')
    os.utime(d / 'runtime.jsonl', ns=(1, 1_793_000_000_123_456_789))
    for i in range(3):
        (d / 'images' / f'{i}.jpg').write_bytes(b'x')
    t = cache_root / DATASET / 'execution' / 'te01.mp4' / '060_uncompressed_tracks' / 'V'
    t.mkdir(parents=True)
    (t / 'runtimes.jsonl').write_text('{"frame_idx": 0, "runtime": []}\n')
    (t / 'tracking.jsonl').write_text('tracks\n')
    assert gpu.stage_files(cache_root, DATASET, 'te01.mp4', '030_exec_compress') == [d / 'runtime.jsonl']
    assert gpu.stage_files(cache_root, DATASET, 'te01.mp4', '040_exec_detect') == []
    out = tmp_path / 'out'
    got = gpu.harvest(unit, cache_root, ['030_exec_compress', '060_exec_track'], out)
    # The logs keep their cache-relative paths, gzip-compressed; the canvases are counted, outputs digested, and the
    # runtime file's last write recorded.
    rel = f'{DATASET}/execution/te01.mp4/033_compressed_frames/P/runtime.jsonl'
    assert gzip.open(out / gpu.LOGS_DIR / f'{rel}.gz', 'rt').read() == '{"runtime": []}\n'
    assert got['030_exec_compress']['te01.mp4'] == dict(runtime=rel, files={}, canvases=3, mtime_ns=1_793_000_000_123_456_789)
    assert got['060_exec_track']['te01.mp4']['files']['tracking.jsonl'] == dict(sha256=digest('tracks\n'), bytes=7)
    # A second configuration's runtime file makes the stage ambiguous.
    (cache_root / DATASET / 'execution' / 'te01.mp4' / '033_compressed_frames' / 'Q').mkdir()
    (cache_root / DATASET / 'execution' / 'te01.mp4' / '033_compressed_frames' / 'Q' / 'runtime.jsonl').write_text('')
    with pytest.raises(AssertionError, match='2 runtime files'):
        gpu.harvest(unit, cache_root, ['030_exec_compress'], tmp_path / 'out2')


def test_prepare_unit(tmp_path, monkeypatch):
    # Container paths point into the test directory.
    monkeypatch.setattr(gpu, 'SCRATCH_CONTAINER', tmp_path / 'scratch')
    monkeypatch.setattr(gpu, 'CONTAINER_DATASETS', str(tmp_path / 'mirror'))
    for v in VIDEOS:
        (tmp_path / 'mirror' / DATASET / 'test').mkdir(parents=True, exist_ok=True)
        (tmp_path / 'mirror' / DATASET / 'test' / v).write_bytes(b'video')
    index = f'{DATASET}/indexing/training/results/Net_60/model_best.pth'
    p022 = {v: f'{DATASET}/execution/{v}/022_pruned_polyominoes/{PRUNE_PARAM_B}/score/score.jsonl' for v in VIDEOS}
    other = f'{DATASET}/execution/te01.mp4/022_pruned_polyominoes/{PRUNE_PARAM_B}x/score/score.jsonl'
    inputs = {index: dict(staged_from='mirror'), other: dict(staged_from='ace'), **{r: dict(staged_from='ace') for r in p022.values()}}
    for rel in inputs:
        (tmp_path / 'scratch' / 'inputs' / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / 'scratch' / 'inputs' / rel).write_text(rel)
    info = dict(inputs=inputs, configs={B: dict(prune_param=PRUNE_PARAM_B, pareto_header='h,variant_id', pareto_line=f'x,{B}')})
    unit = dict(dataset=DATASET, system='tetris', variant_id=B, videos=['te01.mp4'])
    copied = gpu.prepare_unit(unit, info, tmp_path / 'u1', copy_p022=True)
    # Only the unit's video, its configuration's p022 outputs, and the indexing input; the filter file verbatim.
    assert sorted(copied) == sorted([index, p022['te01.mp4']])
    assert os.readlink(tmp_path / 'u1' / 'datasets' / DATASET / 'test' / 'te01.mp4') == str(tmp_path / 'mirror' / DATASET / 'test' / 'te01.mp4')
    assert sorted(os.listdir(tmp_path / 'u1' / 'datasets' / DATASET / 'test')) == ['te01.mp4']
    assert (tmp_path / 'u1' / 'cache' / DATASET / 'evaluation' / '095_pareto_params' / 'pareto_params.csv').read_text() == f'h,variant_id\nx,{B}\n'
    # Without copying p022 only the indexing input; reference units copy nothing and write no filter.
    assert gpu.prepare_unit(unit, info, tmp_path / 'u2', copy_p022=False) == [index]
    ref = dict(dataset=DATASET, system='reference', variant_id=None, videos=list(VIDEOS))
    assert gpu.prepare_unit(ref, info, tmp_path / 'u3', copy_p022=False) == []
    assert not (tmp_path / 'u3' / 'cache' / DATASET / 'evaluation').exists()
    # A video missing from the mirror fails.
    with pytest.raises(AssertionError, match='missing from the mirror'):
        gpu.prepare_unit(dict(ref, videos=['te99.mp4']), info, tmp_path / 'u4', copy_p022=False)


def test_prepare_inputs_checks_every_input(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu, 'SCRATCH_HOST', tmp_path / 'scratch')
    monkeypatch.setattr(gpu, 'MIRROR_CACHE', tmp_path / 'mirror')
    mirror_rel, ace_rel = f'{DATASET}/indexing/m.npy', f'{DATASET}/execution/te01.mp4/022/x/score/score.jsonl'
    (tmp_path / 'mirror' / mirror_rel).parent.mkdir(parents=True)
    (tmp_path / 'mirror' / mirror_rel).write_text('model')
    p = dict(datasets={DATASET: dict(inputs={mirror_rel: dict(sha256=digest('model'), staged_from='mirror'),
                                             ace_rel: dict(sha256=digest('scores'), staged_from='ace')})})
    # An ace-only input `stage` did not push fails.
    with pytest.raises(AssertionError, match='stage'):
        gpu.prepare_inputs(p, [DATASET], gpu.Log())
    (tmp_path / 'scratch' / 'inputs' / ace_rel).parent.mkdir(parents=True)
    (tmp_path / 'scratch' / 'inputs' / ace_rel).write_text('scores')
    gpu.prepare_inputs(p, [DATASET], gpu.Log())
    # The mirror input was copied, not linked.
    copy = tmp_path / 'scratch' / 'inputs' / mirror_rel
    assert copy.read_text() == 'model' and not copy.is_symlink() and copy.stat().st_ino != (tmp_path / 'mirror' / mirror_rel).stat().st_ino
    # A copy that differs from ace's fails.
    (tmp_path / 'scratch' / 'inputs' / ace_rel).write_text('changed')
    with pytest.raises(AssertionError, match='differs from ace'):
        gpu.prepare_inputs(p, [DATASET], gpu.Log())


def fake_run(outputs: dict[tuple, str]):
    """A stand-in for subprocess.run that answers commands by their argument tuples."""
    def run(cmd, **kwargs):
        return SimpleNamespace(stdout=outputs[tuple(cmd)], returncode=0)
    return run


def test_gpu_state_parsing(monkeypatch):
    monkeypatch.setattr(gpu.subprocess, 'run', fake_run({
        ('nvidia-smi', '--query-gpu=index,uuid,utilization.gpu,memory.used', '--format=csv,noheader,nounits'):
            '0, GPU-a, 0, 0\n1, GPU-b, 37, 2001\n',
        ('nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'): 'GPU-b\nGPU-b\n'}))
    assert gpu.gpu_state() == {0: dict(uuid='GPU-a', util=0.0, mem=0.0, apps=0), 1: dict(uuid='GPU-b', util=37.0, mem=2001.0, apps=2)}


def test_node_cpus(tmp_path, monkeypatch):
    # nvidia-smi's bus id 00000000:B2:00.0 is sysfs's 0000:b2:00.0.
    monkeypatch.setattr(gpu, 'SYSFS_PCI', tmp_path)
    (tmp_path / '0000:b2:00.0').mkdir()
    (tmp_path / '0000:b2:00.0' / 'local_cpulist').write_text('20-21,60\n')
    monkeypatch.setattr(gpu.subprocess, 'run', fake_run({
        ('nvidia-smi', '--query-gpu=pci.bus_id', '--format=csv,noheader', '-i', '5'): '00000000:B2:00.0\n'}))
    assert gpu.node_cpus(5) == [20, 21, 60]


def test_node_busy(monkeypatch):
    # Busy and total jiffies per CPU, before and after; CPU 9 is not the node's.
    readings = iter([{0: (10, 100), 1: (0, 100), 9: (0, 100)}, {0: (40, 200), 1: (10, 200), 9: (100, 200)}])
    monkeypatch.setattr(gpu.CpuSampler, 'read', staticmethod(lambda: next(readings)))
    monkeypatch.setattr(gpu.time, 'sleep', lambda s: None)
    assert gpu.node_busy([0, 1], 1.0) == pytest.approx((30 + 10) / 200)
    # No elapsed jiffies cannot show an idle node.
    readings = iter([{0: (0, 0)}, {0: (0, 0)}])
    assert gpu.node_busy([0], 1.0) == 1.0


def test_wait_for_idle_node(monkeypatch):
    monkeypatch.setattr(gpu, 'node_cpus', lambda g: [0, 1])
    monkeypatch.setattr(gpu, 'BUSY_WAIT_S', 0)
    monkeypatch.setattr(gpu, 'BUSY_RETRIES', 2)
    shares = iter([0.5, gpu.NODE_IDLE_MAX])
    monkeypatch.setattr(gpu, 'node_busy', lambda cpus: next(shares))
    gate = gpu.wait_for_idle_node(1, gpu.Log())
    assert gate['node_busy'] == gpu.NODE_IDLE_MAX and gate['retries'] == 1 and gate['node_cpus'] == [0, 1]
    # A node busy for every check gives up after BUSY_RETRIES retries.
    calls = []
    monkeypatch.setattr(gpu, 'node_busy', lambda cpus: calls.append(1) or 0.9)
    assert gpu.wait_for_idle_node(1, gpu.Log()) is None and len(calls) == 3


def test_build_config(tmp_path):
    import yaml
    path = gpu.build_config(tmp_path, DATASET)
    config = yaml.safe_load(path.read_text())
    assert config['EXEC']['DATASETS'] == [DATASET]
    assert config['DATA']['CACHE_DIR'] == f'{tmp_path / "cache"}/' and config['DATA']['DATASETS_DIR'] == str(tmp_path / 'datasets')
    # Everything else is the checkout's configuration.
    base = yaml.safe_load((gpu.CHECKOUT / 'configs' / 'global.yaml').read_text())
    assert config['EXEC']['SAMPLE_RATES'] == base['EXEC']['SAMPLE_RATES'] and config['OPS'] == base['OPS']


def write_video(path: Path, frames: int):
    """A small mp4 with `frames` frames."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 15, (64, 48))
    for i in range(frames):
        writer.write(np.full((48, 64, 3), i * 40, dtype=np.uint8))
    writer.release()


def test_decode_videos(tmp_path):
    # Two videos of five and three frames, decoded in three passes, the middle one in reverse order.
    for v, n in (('te01.mp4', 5), ('te04.mp4', 3)):
        write_video(tmp_path / 'datasets' / DATASET / 'test' / v, n)
    unit = dict(dataset=DATASET, videos=['te01.mp4', 'te04.mp4'])
    got = gpu.decode_videos(unit, tmp_path, 3)
    assert [(r['decode_pass'], r['video']) for r in got] == [(0, 'te01.mp4'), (0, 'te04.mp4'), (1, 'te04.mp4'),
                                                            (1, 'te01.mp4'), (2, 'te01.mp4'), (2, 'te04.mp4')]
    assert all(r['frames'] == {'te01.mp4': 5, 'te04.mp4': 3}[r['video']] for r in got)
    assert all(0 < r['read_s'] <= r['total_s'] and r['open_s'] > 0 and r['threads'] >= 0 for r in got)
    # The page-cache warm-up reads the whole file.
    path = tmp_path / 'datasets' / DATASET / 'test' / 'te01.mp4'
    assert gpu.preread(dict(dataset=DATASET, videos=['te01.mp4']), tmp_path)[0]['bytes'] == path.stat().st_size


def test_run_stage(tmp_path, monkeypatch):
    # A stand-in stage script records the CONFIG it saw and exits with a code.
    script = tmp_path / 'stage.py'
    script.write_text('import os, sys\nopen(sys.argv[0] + ".config", "w").write(os.environ["CONFIG"] + " " + sys.argv[1])\n'
                      'sys.exit(int(os.environ.get("EXIT", "0")))\n')
    monkeypatch.setattr(gpu, 'STAGES', {'s': (str(script), 'd', 'r', ())})
    got = gpu.run_stage('s', tmp_path / 'global.yaml', tmp_path / 's.log')
    assert got['exit'] == 0 and got['ran'] and got['wall_s'] > 0 and got['end_ns'] >= got['start_ns']
    assert (tmp_path / 'stage.py.config').read_text() == f'{tmp_path / "global.yaml"} --test'
    monkeypatch.setenv('EXIT', '3')
    assert gpu.run_stage('s', tmp_path / 'global.yaml', tmp_path / 's.log')['exit'] == 3


def test_cpu_sampler():
    # One sample after a second, with fractions in [0, 1], labeled with the current step.
    sampler = gpu.CpuSampler(sorted(os.sched_getaffinity(0)))
    sampler.step = 'x'
    import time
    time.sleep(1.3)
    samples = sampler.stop()
    assert samples and samples[0]['step'] == 'x'
    assert 0 <= samples[0]['unit_cpus_busy'] <= 1
    assert samples[0]['own_busy'] is None or samples[0]['own_busy'] >= 0


# ------------------------------------------------------------------------------------------------ orchestration


FAKE_LAUNCHER = '''import json, os, pathlib, sys
gpu, scratch, command = sys.argv[1:4]
# Count the calls; the first one finds the GPU busy.
calls = pathlib.Path({calls!r})
n = int(calls.read_text()) if calls.exists() else 0
calls.write_text(str(n + 1))
if n == 0:
    print('GPU busy; not starting')
    sys.exit(3)
if os.environ.get('FAKE_BAD_ARGS'):
    sys.exit(2)
# The job writes its manifest into the attempt directory unit.json names.
unit = command.split('--unit ')[1]
info = json.loads((pathlib.Path(scratch) / 'units' / unit / 'unit.json').read_text())
(pathlib.Path({raw!r}) / info['raw'] / 'manifest.json').write_text(json.dumps(dict(status='ok')))
run = f'g{{gpu}}-20261027T061018'
print(f'run {{run}} 2026-10-27T06:10:18-07:00: GPU {{gpu}}')
samples = pathlib.Path(scratch) / 'samples'
samples.mkdir(exist_ok=True)
(samples / f'{{run}}.json').write_text(json.dumps(dict(contaminated=False, docker_args=os.environ['BOMBE_DOCKER_ARGS'])))
(samples / f'{{run}}.jsonl').write_text('{{}}\\n')
(samples / f'{{run}}.gpus.csv').write_text('x\\n')
'''


def test_launch_retries_a_busy_gpu_and_copies_records(tmp_path, monkeypatch):
    raw, scratch = tmp_path / 'raw', tmp_path / 'scratch'
    # A bash launcher, as the real one, handing its arguments to a Python stand-in.
    (tmp_path / 'launcher.py').write_text(FAKE_LAUNCHER.format(calls=str(tmp_path / 'calls'), raw=str(raw)))
    (tmp_path / 'launcher.sh').write_text(f'exec {sys.executable} {tmp_path / "launcher.py"} "$@"\n')
    monkeypatch.setattr(gpu, 'LAUNCHER', tmp_path / 'launcher.sh')
    monkeypatch.setattr(gpu, 'RAW', raw)
    monkeypatch.setattr(gpu, 'SCRATCH_HOST', scratch)
    monkeypatch.setattr(gpu, 'BUSY_WAIT_S', 0)
    unit = dict(unit='serial-d-reference', phase='serial', dataset='d', system='reference', variant_id=None, videos=['v'])
    # A stale scratch directory of the unit is cleared first.
    (scratch / 'units' / unit['unit'] / 'cache').mkdir(parents=True)
    outcome = gpu.launch(unit, 1, 3, gpu.Log(), dict(node_busy=0.01))
    assert (tmp_path / 'calls').read_text() == '2'
    assert outcome == dict(status='ok', contaminated=False, others_on_node=None, launcher_exit=0, run='g1-20261027T061018')
    out = raw / gpu.UNITS_DIR / unit['unit'] / 'attempt-3'
    # The launcher's records are copied; the worker logs mount names the unit's own directory.
    summary = json.loads((out / gpu.LAUNCHER_SUMMARY).read_text())
    assert summary['docker_args'] == f'-v {scratch / "units" / unit["unit"] / gpu.WORKER_LOGS}:/polyis-cache/LOGS'
    assert gzip.open(out / gpu.LAUNCHER_SAMPLES, 'rt').read() == '{}\n' and (out / gpu.LAUNCHER_GPUS).exists()
    # unit.json carries the attempt, GPU, raw directory, and the idle-node check.
    written = json.loads((scratch / 'units' / unit['unit'] / 'unit.json').read_text())
    assert written == dict(unit, attempt=3, gpu=1, raw=f'{gpu.UNITS_DIR}/{unit["unit"]}/attempt-3', gate=dict(node_busy=0.01))
    assert not (scratch / 'units' / unit['unit'] / 'cache').exists() and (scratch / 'units' / unit['unit'] / gpu.WORKER_LOGS).is_dir()
    # A launcher that rejects its arguments fails the run.
    monkeypatch.setenv('FAKE_BAD_ARGS', '1')
    with pytest.raises(AssertionError, match='rejected its arguments'):
        gpu.launch(unit, 1, 4, gpu.Log())


def write_outcome(path: Path, status: str, contaminated: bool, others: float):
    """An attempt's records as run_unit reads them: manifest, launcher summary, and one timed CPU sample."""
    path.mkdir(parents=True)
    (path / gpu.MANIFEST).write_text(json.dumps(dict(status=status)))
    (path / gpu.LAUNCHER_SUMMARY).write_text(json.dumps(dict(contaminated=contaminated)))
    write_cpu(path / gpu.CPU_SAMPLES, [('020_exec_classify', OWN_BUSY + others, OWN_BUSY)])


CLEAN, BUSY_NODE, DIRTY_GPU, FAILED = ('ok', False, 0.01), ('ok', False, 0.3), ('ok', True, 0.01), ('failed', False, 0.01)


@pytest.mark.parametrize('phase, redo, before, script, gate, result, launched', [
    # A contaminated attempt is retried; a busy node does not matter in the parallel phase.
    ('parallel', False, [], [DIRTY_GPU, BUSY_NODE], True, 'ok', [1, 2]),
    # Elsewhere a busy node is retried, and every attempt waits for an idle node first.
    ('serial', False, [], [BUSY_NODE, CLEAN], True, 'ok', [1, 2]),
    ('solo', False, [], [BUSY_NODE, DIRTY_GPU], True, 'unclean', [1, 2]),
    # A failed job is not retried.
    ('serial', False, [], [FAILED], True, 'failed', [1]),
    # Without --redo: a usable attempt is enough, MAX_ATTEMPTS attempts are all, and fewer leave the rest.
    ('serial', False, [BUSY_NODE, CLEAN], [], True, 'ok', []),
    ('serial', False, [BUSY_NODE, DIRTY_GPU], [], True, 'exhausted', []),
    ('serial', False, [FAILED], [CLEAN], True, 'ok', [2]),
    # With --redo: up to MAX_ATTEMPTS new attempts, after any number.
    ('serial', True, [CLEAN], [CLEAN], True, 'ok', [2]),
    ('serial', True, [BUSY_NODE, BUSY_NODE], [BUSY_NODE, BUSY_NODE], True, 'unclean', [3, 4]),
    # A node that never becomes idle launches nothing.
    ('serial', False, [], [], False, 'node-busy', []),
])
def test_run_unit(tmp_path, monkeypatch, phase, redo, before, script, gate, result, launched):
    raw = tmp_path / 'raw'
    monkeypatch.setattr(gpu, 'RAW', raw)
    unit = dict(unit=f'{phase}-d-reference', phase=phase)
    for n, outcome in enumerate(before, 1):
        write_outcome(raw / gpu.UNITS_DIR / unit['unit'] / f'attempt-{n}', *outcome)
    calls, gates = [], []

    def launch(u, g, attempt, log, gate_record=None):
        # Each call writes the next scripted attempt and reads it back as launch does.
        path = raw / gpu.UNITS_DIR / u['unit'] / f'attempt-{attempt}'
        write_outcome(path, *script[len(calls)])
        calls.append(attempt)
        gates.append(gate_record)
        return gpu.attempt_outcome(path)

    monkeypatch.setattr(gpu, 'launch', launch)
    monkeypatch.setattr(gpu, 'wait_for_idle_node', lambda g, log: dict(node_busy=0.0) if gate else None)
    assert gpu.run_unit(unit, 1, redo, gpu.Log()) == result
    assert calls == launched
    # The parallel phase never waits for an idle node; the others pass the check to every attempt.
    assert gates == [None if phase == 'parallel' else dict(node_busy=0.0)] * len(launched)


def test_run_keeps_each_dataset_on_its_gpu(monkeypatch, tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    # Two datasets: the synthetic one and a copy with a hundred times its frames.
    p = plan()
    big = dict(p['datasets'][DATASET], videos={v: n * 100 for v, n in VIDEOS.items()})
    p['datasets']['big'] = big
    p['units'] = gpu.plan_units(p['datasets'])
    (raw / gpu.PLAN_FILE).write_text(json.dumps(p))
    monkeypatch.setattr(gpu, 'RAW', raw)
    monkeypatch.setattr(gpu, 'gpu_state', lambda: {g: dict(util=0.0, mem=0.0, apps=0) for g in range(7)})
    monkeypatch.setattr(gpu, 'prepare_inputs', lambda plan_, datasets, log: None)
    ran = []
    monkeypatch.setattr(gpu, 'run_unit', lambda u, g, redo, log: ran.append((u['unit'], g, redo)) or 'ok')
    gpu.run(SimpleNamespace(gpus=[0, 1], phases=['serial', 'solo'], datasets=None, redo=False))
    # The larger dataset goes to GPU 0; one-at-a-time phases run in GPU order, serial before solo.
    assert [(u, g) for u, g, _ in ran] == [(f'serial-big-{x}', 0) for x in ('reference', A, B)] + \
        [(f'serial-{DATASET}-{x}', 1) for x in ('reference', A, B)] + \
        [(f'solo-big-{x}', 0) for x in ('reference', A)] + [(f'solo-{DATASET}-{x}', 1) for x in ('reference', A)]
    # A run of one dataset keeps its GPU, and --redo is passed on.
    ran.clear()
    gpu.run(SimpleNamespace(gpus=[0, 1], phases=['parallel'], datasets=[DATASET], redo=True))
    assert sorted(ran) == sorted((f'parallel-{DATASET}-{x}', 1, True) for x in ('reference', A, B))
    # Busy GPUs, GPUs of other tracks, and unknown datasets are refused.
    with pytest.raises(AssertionError, match='may use GPUs'):
        gpu.run(SimpleNamespace(gpus=[3], phases=['serial'], datasets=None, redo=False))
    with pytest.raises(AssertionError, match='not in the plan'):
        gpu.run(SimpleNamespace(gpus=[0], phases=['serial'], datasets=['nope'], redo=False))
    monkeypatch.setattr(gpu, 'gpu_state', lambda: {g: dict(util=0.0, mem=0.0, apps=int(g == 1)) for g in range(7)})
    with pytest.raises(AssertionError, match='GPUs busy'):
        gpu.run(SimpleNamespace(gpus=[0, 1], phases=['serial'], datasets=None, redo=False))


STUB_STAGE = '''import json, os, pathlib, sys, yaml
# A stand-in stage: one runtime file and the listed outputs per video of the unit's config, unless told to fail.
config = yaml.safe_load(open(os.environ['CONFIG']))
cache = pathlib.Path(config['DATA']['CACHE_DIR'])
for dataset in config['EXEC']['DATASETS']:
    for video in sorted(os.listdir(pathlib.Path(config['DATA']['DATASETS_DIR']) / dataset / 'test')):
        d = cache / dataset / 'execution' / video / {stage_dir!r} / {param!r}
        d.mkdir(parents=True, exist_ok=True)
        if not {fail!r}:
            (d / {runtime!r}).write_text(json.dumps([dict(op='x', time=1.0)]) + chr(10))
        for name in {outputs!r}:
            (d / name).write_text(video)
'''


def job_world(tmp_path, monkeypatch, system: str, failing: str | None = None):
    """A unit ready for gpu.job: plan, unit.json, mirror videos, staged p022 inputs, and stub stage scripts."""
    raw, scratch, mirror = tmp_path / 'raw', tmp_path / 'scratch', tmp_path / 'mirror'
    for name, value in (('RAW', raw), ('SCRATCH_CONTAINER', scratch), ('CONTAINER_DATASETS', str(mirror)), ('DECODE_PASSES', 2)):
        monkeypatch.setattr(gpu, name, value)
    monkeypatch.setattr(gpu, 'gurobi_available', lambda: dict(available=False, error='License expired'))
    for v, n in VIDEOS.items():
        write_video(mirror / DATASET / 'test' / v, n)
    # The cached p022 outputs of B, staged from ace.
    inputs = {}
    for v in VIDEOS:
        for name in ('score.jsonl', 'runtime.jsonl'):
            rel = f'{DATASET}/execution/{v}/022_pruned_polyominoes/{PRUNE_PARAM_B}/score/{name}'
            (scratch / 'inputs' / rel).parent.mkdir(parents=True, exist_ok=True)
            (scratch / 'inputs' / rel).write_text(json.dumps(dict(runtime=[dict(op='solve_ilp', time=1.0)])) + '\n')
            inputs[rel] = dict(staged_from='ace')
    configs = {B: dict(prunes=True, prune_param=PRUNE_PARAM_B, pareto_header='h,variant_id', pareto_line=f'x,{B}')}
    raw.mkdir()
    (raw / gpu.PLAN_FILE).write_text(json.dumps(dict(datasets={DATASET: dict(videos=VIDEOS, configs=configs, inputs=inputs)})))
    # Stub stage scripts in place of the pipeline's, writing where stage_files looks.
    stages = {}
    for name, (script, stage_dir, runtime, outputs) in gpu.STAGES.items():
        stub = tmp_path / 'stubs' / f'{name}.py'
        stub.parent.mkdir(exist_ok=True)
        stub.write_text(STUB_STAGE.format(stage_dir=stage_dir, param='' if stage_dir == '002_naive' else 'P',
                                          runtime=runtime, outputs=outputs, fail=name == failing))
        stages[name] = (str(stub), stage_dir, runtime, outputs)
    monkeypatch.setattr(gpu, 'STAGES', stages)
    unit = dict(unit=f'serial-{DATASET}-{system}', phase='serial', dataset=DATASET, system=system,
                variant_id=B if system == 'tetris' else None, videos=list(VIDEOS))
    unit_dir = scratch / 'units' / unit['unit']
    unit_dir.mkdir(parents=True)
    (unit_dir / 'unit.json').write_text(json.dumps(dict(unit, attempt=1, raw=f'units/{unit["unit"]}/attempt-1')))
    # A worker log as ProgressBar leaves it in the mounted directory.
    (unit_dir / gpu.WORKER_LOGS / 'p020_exec_classify').mkdir(parents=True)
    (unit_dir / gpu.WORKER_LOGS / 'p020_exec_classify' / 'worker_0.log').write_text('worker')
    return raw / 'units' / unit['unit'] / 'attempt-1', unit


def test_job_reference_unit(tmp_path, monkeypatch):
    out, unit = job_world(tmp_path, monkeypatch, 'reference')
    gpu.job(SimpleNamespace(unit=unit['unit']))
    manifest = json.loads((out / gpu.MANIFEST).read_text())
    assert manifest['status'] == 'ok' and manifest['error'] is None and manifest['p022_source'] is None
    assert [s['stage'] for s in manifest['stages']] == [s[0] for s in gpu.REFERENCE_STAGES]
    assert all(s['ran'] and s['exit'] == 0 for s in manifest['stages'])
    # Two decode passes of both videos, every frame.
    decode = [json.loads(x) for x in gzip.open(out / gpu.DECODE_FILE, 'rt')]
    assert sorted((r['decode_pass'], r['video'], r['frames']) for r in decode) == \
        sorted((p, v, n) for p in range(2) for v, n in VIDEOS.items())
    # Harvested logs, outputs, worker logs, and CPU samples.
    rel = manifest['outputs']['001_preprocess_groundtruth_detection']['te01.mp4']['runtime']
    assert rel == f'{DATASET}/execution/te01.mp4/002_naive/detection_runtime.jsonl' and (out / gpu.LOGS_DIR / f'{rel}.gz').exists()
    assert (out / gpu.WORKER_LOGS / 'p020_exec_classify' / 'worker_0.log.log').read_text() == 'worker'
    assert (out / gpu.CPU_SAMPLES).exists() and manifest['files']['decode'] == gpu.DECODE_FILE


def test_job_tetris_unit_with_cached_p022(tmp_path, monkeypatch):
    out, unit = job_world(tmp_path, monkeypatch, 'tetris')
    gpu.job(SimpleNamespace(unit=unit['unit']))
    manifest = json.loads((out / gpu.MANIFEST).read_text())
    assert manifest['status'] == 'ok' and manifest['p022_source'] == 'cache' and not manifest['gurobi']['available']
    # p022 is recorded as not run, its cached outputs copied and harvested; no decode for Tetris.
    stages = {s['stage']: s for s in manifest['stages']}
    assert list(stages) == [s[0] for s in gpu.TETRIS_STAGES]
    assert stages[gpu.PRUNE_STAGE] == dict(stage=gpu.PRUNE_STAGE, ran=False, source='cache')
    assert len(manifest['inputs_copied']) == 2 * len(VIDEOS) and 'decode' not in manifest['files']
    assert set(manifest['outputs'][gpu.PRUNE_STAGE]) == set(VIDEOS)
    # The unit's Pareto filter holds its configuration only.
    pareto = tmp_path / 'scratch' / 'units' / unit['unit'] / 'cache' / DATASET / 'evaluation' / '095_pareto_params' / 'pareto_params.csv'
    assert pareto.read_text() == f'h,variant_id\nx,{B}\n'


def test_job_records_a_failed_stage(tmp_path, monkeypatch):
    # A stage that exits 0 without writing its runtime files fails the job, which still writes its manifest.
    out, unit = job_world(tmp_path, monkeypatch, 'tetris', failing='030_exec_compress')
    with pytest.raises(AssertionError, match='runtime files per video'):
        gpu.job(SimpleNamespace(unit=unit['unit']))
    manifest = json.loads((out / gpu.MANIFEST).read_text())
    assert manifest['status'] == 'failed' and 'runtime files per video' in manifest['error']
    assert [s['stage'] for s in manifest['stages']][-1] == '030_exec_compress'
    assert (out / gpu.WORKER_LOGS / 'p020_exec_classify' / 'worker_0.log.log').exists()


# ------------------------------------------------------------------------------------------------ the aggregation


def test_compute_boundaries_against_oracle(world):
    raw, cache_root, p = world
    reports = ct.compute(raw, cache_root, None)
    throughput = reports['throughput.json']['datasets']['B3D1']
    for bound, variant in BOUNDS.items():
        entry = throughput[bound]
        assert entry['config'] == variant and entry['frames'] == 5 and entry['videos'] == 2 and entry['decode_passes'] == 3
        for side in ('clean', 'paper'):
            ref = oracle('serial', side, 'reference', None, list(VIDEOS))
            tet = oracle('serial', side, 'tetris', variant, list(VIDEOS))
            for b in ('p121', 'decode_excluded', 'one_decode', 'one_decode_no_canvas_read', 'one_decode_logged'):
                got = entry[side][b]
                assert got['reference_s'] == pytest.approx(ref[b]) and got['tetris_s'] == pytest.approx(tet[b])
                assert got['speedup'] == pytest.approx(ref[b] / tet[b])
                assert got['tetris_fps'] == pytest.approx(5 / tet[b]) and got['reference_fps'] == pytest.approx(5 / ref[b])
            # The decode boundaries' range: the speedup at the fastest and the slowest pass.
            for b in ct.DECODE_BAND_BOUNDARIES:
                ends = sorted([ref[f'{b}_min'] / tet[f'{b}_min'], ref[f'{b}_max'] / tet[f'{b}_max']])
                assert entry[side][b]['speedup_range'] == pytest.approx(ends)
                assert (ends[0] < ends[1]) == (side == 'clean')
        # Wall clock: the stage processes; B's p022 from the cache adds its logged seconds and is flagged.
        ref_wall = WALLS['001_preprocess_groundtruth_detection'] + WALLS['002_preprocess_groundtruth_tracking']
        tet_wall = sum(WALLS[s] for s in unit_stages('tetris', A))
        if variant == B:
            tet_wall += sum(e['seconds'] for v in VIDEOS for e in op_seconds(gpu.PRUNE_STAGE, v, PAPER).values())
        assert entry['clean']['wall_clock']['reference_s'] == pytest.approx(ref_wall)
        assert entry['clean']['wall_clock']['tetris_s'] == pytest.approx(tet_wall)
        assert entry['paper']['wall_clock'] is None
        # Without startup: each stage's two videos were written 1 s apart, so 2 s per stage; p022 as above.
        steady = entry['clean']['wall_clock_steady']
        assert steady['reference_s'] == pytest.approx(2 * 2.0)
        assert steady['tetris_s'] == pytest.approx(5 * 2.0 + tet_wall - sum(WALLS[s] for s in unit_stages('tetris', A)))
        assert entry['paper']['wall_clock_steady'] is None
        assert entry['p022_wall_from_cache'] == (variant == B) and entry['p022_source'] == ('cache' if variant == B else None)
        # Serial units ran on an idle node and were not contaminated.
        assert entry['contaminated'] is False and entry['cpu_clean'] is True
    table = reports['throughput.json']['speedup_table']
    assert set(table) == set(BOUNDS) and set(table['0.05']) == set(ct.BOUNDARIES)
    assert table['0.05']['wall_clock']['B3D1'].keys() == {'clean'} == table['0.05']['wall_clock_steady']['B3D1'].keys()
    assert table['0.10']['p121']['B3D1']['paper'] == pytest.approx(
        oracle('serial', 'paper', 'reference', None, list(VIDEOS))['p121'] / oracle('serial', 'paper', 'tetris', B, list(VIDEOS))['p121'])


def test_compute_checks_and_per_op(world):
    raw, cache_root, _ = world
    reports = ct.compute(raw, cache_root, None)
    checks = reports['conditions.json']['checks']
    assert checks['port_check']['mismatches'] == 0 and checks['port_check']['rows_compared'] > 0
    assert checks['paper_throughput_vs_tradeoff']['max_relative_difference'] < 1e-9
    # Clean reference units: the parallel and serial units' two videos each and the solo unit's one.
    assert checks['frames'] == dict(videos=5, all_frames=True)
    # Digests compare the headline units only; te04's tracking differs from the cache.
    digests = {(r['unit'], r['stage'], r['file']): (r['identical'], r['videos']) for r in checks['outputs_identical_to_cache']}
    assert {u for u, _, _ in digests} == {f'serial-{DATASET}-{x}' for x in ('reference', A, B)}
    assert digests[(f'serial-{DATASET}-{A}', '060_exec_track', 'tracking.jsonl')] == (1, 2)
    assert digests[(f'serial-{DATASET}-{A}', '020_exec_classify', 'score.jsonl')] == (2, 2)
    assert digests[(f'serial-{DATASET}-reference', '002_preprocess_groundtruth_tracking', 'tracking.jsonl')] == (1, 2)
    ops = {(r['variant_id'], r['stage'], r['op']): r for r in reports['per_op.json']['ops']}
    # Every op's clean/paper ratio is its own (per-op and per-video jitter), as the oracle has it.
    for (variant, stage, op), r in ops.items():
        assert r['clean_over_paper'] == pytest.approx(op_ratio(('clean', 'serial'), PAPER, stage, op, list(VIDEOS)))
        assert r['items_match'] and r['clean_items'] == sum(op_seconds(stage, v, PAPER)[op]['items'] for v in VIDEOS)
    # p020's read is reported but not counted; p022 from the cache equals the paper.
    assert ops[(A, '020_exec_classify', 'read')]['counted'] is False
    assert ops[(B, gpu.PRUNE_STAGE, 'solve_ilp')]['clean_over_paper'] == pytest.approx(1.0)
    assert ops[('naive', '001_preprocess_groundtruth_detection', 'detect')]['clean_items'] == 5
    assert len({round(r['clean_over_paper'], 9) for r in ops.values()}) > 10
    # Ops without numeric times (p040's image_file) are not ops.
    assert (A, '040_exec_detect', 'image_file') not in ops


def comparison(reports: dict, a: str, b: str) -> dict:
    return next(c for c in reports['interference.json']['comparisons'] if (c['a'], c['b']) == (a, b))


@pytest.mark.parametrize('a, scope, videos, variants', [
    ('parallel', 'unit', list(VIDEOS), {'0.05': A, '0.10': B}),
    ('solo', 'calibration', CALIBRATION, {'0.05': A}),
])
def test_compute_comparisons(world, a, scope, videos, variants):
    raw, cache_root, _ = world
    c = comparison(ct.compute(raw, cache_root, None), a, 'serial')
    assert c['scope'] == scope
    assert {(r['dataset'], r['bound']) for r in c['boundaries']} == {(DATASET, bound) for bound in variants}
    for r in c['boundaries']:
        ref_a, ref_b = (oracle(ph, 'clean', 'reference', None, videos) for ph in (a, 'serial'))
        tet_a, tet_b = (oracle(ph, 'clean', 'tetris', variants[r['bound']], videos) for ph in (a, 'serial'))
        for k in ('p121', 'decode_excluded', 'one_decode', 'one_decode_no_canvas_read', 'one_decode_logged'):
            assert r[k]['reference_ratio'] == pytest.approx(ref_a[k] / ref_b[k])
            assert r[k]['tetris_ratio'] == pytest.approx(tet_a[k] / tet_b[k])
            assert r[k]['speedup_a'] == pytest.approx(ref_a[k] / tet_a[k]) and r[k]['speedup_b'] == pytest.approx(ref_b[k] / tet_b[k])
        # Wall clock covers whole units only; without startup, only units with write times (serial).
        assert (r['wall_clock'] is not None) == (scope == 'unit') and r['wall_clock_steady'] is None
    # Every op's own ratio on the scope's videos, for the configurations both phases ran.
    assert {r['variant_id'] for r in c['ops']} == {'naive', *variants.values()}
    for r in c['ops']:
        assert r['a_over_b'] == pytest.approx(op_ratio(('clean', a), ('clean', 'serial'), r['stage'], r['op'], videos))
        assert r['items_match']
    # The measured decode (median pass) and p002's logged read.
    d = c['decode'][0]
    assert d['measured_a_over_b'] == pytest.approx(
        statistics.median(sum(decode_total(a, v, p) for v in videos) for p in range(len(DECODE_MS[a])))
        / statistics.median(sum(decode_total('serial', v, p) for v in videos) for p in range(3)))
    assert d['logged_a_over_b'] == pytest.approx(op_ratio(('clean', a), ('clean', 'serial'), '001_preprocess_groundtruth_detection', 'read', videos))
    # Foreign CPU of both phases' units: serial on an idle node; parallel A beside the parallel reference.
    foreign = {(r['phase'], r['variant_id']): r for r in c['foreign_cpu']}
    assert set(foreign) == {(ph, v) for ph in (a, 'serial') for v in ('naive', *variants.values())}
    assert all(foreign[('serial', v)]['foreign_on_node_mean'] == pytest.approx(0.02) for v in ('naive', *variants.values()))
    if a == 'parallel':
        assert foreign[('parallel', 'naive')]['foreign_on_node_mean'] == pytest.approx(0.1)
        assert foreign[('parallel', A)]['foreign_on_node_mean'] == pytest.approx((2 * 0.1 + 3 * 0.3) / 5)
        assert foreign[('parallel', B)]['foreign_on_node_mean'] == pytest.approx(0.3)
    else:
        assert foreign[('solo', A)]['foreign_on_node_mean'] == pytest.approx(0.3) and not foreign[('solo', A)]['cpu_clean']


def test_decode_spread(world):
    raw, cache_root, _ = world
    rows = {r['phase']: r for r in ct.compute(raw, cache_root, None)['interference.json']['decode_passes']}
    assert set(rows) == {'parallel', 'serial', 'solo'}
    s = rows['serial']
    totals = sorted(sum(decode_total('serial', v, p) for v in VIDEOS) for p in range(3))
    assert s['passes'] == 3 and s['threads'] == [16.0]
    assert s['median_ms_per_frame'] == pytest.approx(1000 * totals[1] / 5)
    assert (s['fastest_ms_per_frame'], s['slowest_ms_per_frame']) == pytest.approx((1000 * totals[0] / 5, 1000 * totals[2] / 5))
    assert s['spread'] == pytest.approx((totals[2] - totals[0]) / totals[1])
    assert s['logged_read_ms_per_frame'] == pytest.approx(
        1000 * sum(op_seconds('001_preprocess_groundtruth_detection', v, ('clean', 'serial'))['read']['seconds'] for v in VIDEOS) / 5)
    # The old single-pass records have no thread count.
    assert rows['parallel']['passes'] == 1 and rows['parallel']['spread'] == 0 and rows['parallel']['threads'] == [None]


def test_compute_conditions(world):
    raw, cache_root, _ = world
    units = {u['unit']: u for u in ct.compute(raw, cache_root, None)['conditions.json']['units']}
    assert len(units) == 8
    a = units[f'parallel-{DATASET}-{A}']
    assert a['attempt'] == 'attempt-2' and a['contaminated'] is False and a['gpu'] == 1
    assert [(x['attempt'], x['contaminated'], x['chosen']) for x in a['attempts']] == [('attempt-1', True, False), ('attempt-2', False, True)]
    # The unit GPU's samples inside each stage window, not the one outside every window.
    assert a['stages']['020_exec_classify']['gpu']['util_max'] == 60.0
    assert a['stages']['030_exec_compress']['gpu']['samples'] == 1
    assert a['load']['load_1m_max'] == 8.0
    # Other GPUs' peaks from the padded CSV, the cut last line skipped.
    assert a['gpus']['3'] == dict(util_max=100.0, util_mean=100.0, mem_max_mib=10785.0, samples=1)
    assert a['gpus']['1']['util_max'] == 70.0
    cpu = a['stages']['040_exec_detect']['cpu']
    assert cpu['others_on_node_mean'] == pytest.approx(0.3) and cpu['own_busy_mean'] == pytest.approx(0.2)
    # Foreign use subtracts the parallel reference's own share in the seconds both ran (020), not later (040), and
    # not the contaminated attempt's, which ran at another time.
    assert cpu['foreign_on_node_mean'] == pytest.approx(0.3)
    assert a['stages']['020_exec_classify']['cpu']['foreign_on_node_mean'] == pytest.approx(0.1)
    assert a['stages']['020_exec_classify']['cpu']['others_on_node_mean'] == pytest.approx(0.3)
    # Over the timed steps only (not the page-cache warm-up).
    assert a['cpu_timed']['samples'] == 5 and a['cpu_timed']['foreign_on_node_mean'] == pytest.approx((2 * 0.1 + 3 * 0.3) / 5)
    assert set(a['steps_cpu']) == {'preread'} and a['cpu_clean'] is False and a['gate'] is None
    # Stage timing: wall clock and logged ops; the time to the first and last output from the serial phase on.
    s = units[f'serial-{DATASET}-{A}']
    t = s['stages']['040_exec_detect']['time']
    assert t['wall_s'] == WALLS['040_exec_detect'] and t['source'] == 'rerun'
    assert t['ops_s'] == pytest.approx(sum(e['seconds'] for v in VIDEOS for e in op_seconds('040_exec_detect', v, ('clean', 'serial')).values()
                                           if e['counted']))
    assert (t['first_output_s'], t['last_output_s']) == pytest.approx((1.0, 2.0))
    assert a['stages']['040_exec_detect']['time']['first_output_s'] is None
    assert s['cpu_clean'] is True and s['gate'] == dict(node_busy=0.01)
    # p022 from the cache has no process to sample or time.
    b = units[f'serial-{DATASET}-{B}']
    assert b['p022_source'] == 'cache' and b['stages'][gpu.PRUNE_STAGE]['gpu'] is None
    assert b['stages'][gpu.PRUNE_STAGE]['time']['source'] == 'cache' and b['stages'][gpu.PRUNE_STAGE]['time']['first_output_s'] is None


def add_attempt(world, name: str, attempt: int, **kwargs) -> Path:
    """Another attempt of a unit, alone in its own time window."""
    raw, cache_root, p = world
    return write_attempt(raw, cache_root, unit_named(p, name), attempt, base_t=1.8e9 + 100000 * attempt, **kwargs)


def chosen(world) -> dict[str, dict]:
    raw, cache_root, _ = world
    return {u['unit']: u for u in ct.compute(raw, cache_root, None)['conditions.json']['units']}


def test_attempt_choice_prefers_an_idle_node_over_the_latest(world):
    # A later attempt on a busy node does not replace a clean one.
    add_attempt(world, f'serial-{DATASET}-reference', 2, node_busy=0.5)
    u = chosen(world)[f'serial-{DATASET}-reference']
    assert u['attempt'] == 'attempt-1' and u['cpu_clean'] is True
    assert [round(x['foreign_on_node_mean'], 6) for x in u['attempts']] == [0.02, 0.3]


def test_attempt_choice_takes_the_lowest_foreign_share(world):
    # Among clean attempts the quietest node wins, not the latest.
    add_attempt(world, f'serial-{DATASET}-reference', 2, node_busy=OWN_BUSY + 0.01)
    add_attempt(world, f'serial-{DATASET}-reference', 3, node_busy=OWN_BUSY + 0.04)
    assert chosen(world)[f'serial-{DATASET}-reference']['attempt'] == 'attempt-2'


def test_attempt_choice_prefers_an_uncontaminated_gpu(world):
    # A contaminated attempt loses even on a quieter node.
    add_attempt(world, f'serial-{DATASET}-reference', 2, node_busy=OWN_BUSY, contaminated=True)
    assert chosen(world)[f'serial-{DATASET}-reference']['attempt'] == 'attempt-1'


def test_only_contaminated_attempt_is_used_and_flagged(world):
    raw, cache_root, p = world
    # A's serial unit has only a contaminated attempt: it is used, and its dataset flagged.
    shutil.rmtree(raw / gpu.UNITS_DIR / f'serial-{DATASET}-{A}' / 'attempt-1')
    add_attempt(world, f'serial-{DATASET}-{A}', 2, contaminated=True)
    entry = ct.compute(raw, cache_root, None)['throughput.json']['datasets']['B3D1']
    assert entry['0.05']['contaminated'] is True and entry['0.10']['contaminated'] is False


def test_cpu_unclean_headline_is_flagged(world):
    raw, cache_root, _ = world
    shutil.rmtree(raw / gpu.UNITS_DIR / f'serial-{DATASET}-reference' / 'attempt-1')
    add_attempt(world, f'serial-{DATASET}-reference', 2, node_busy=0.5)
    entry = ct.compute(raw, cache_root, None)['throughput.json']['datasets']['B3D1']
    assert entry['0.05']['cpu_clean'] is False and entry['0.10']['cpu_clean'] is False


def test_failed_and_unfinished_attempts_are_skipped(world):
    raw, cache_root, _ = world
    # A failed later attempt, and one still running (no launcher summary), do not replace the successful one.
    for n, files in ((2, {gpu.MANIFEST: dict(status='failed'), gpu.LAUNCHER_SUMMARY: dict(contaminated=False, cpus='0-19,40-59')}),
                     (3, {gpu.MANIFEST: dict(status='ok')})):
        bad = raw / gpu.UNITS_DIR / f'serial-{DATASET}-reference' / f'attempt-{n}'
        bad.mkdir()
        for name, data in files.items():
            (bad / name).write_text(json.dumps(data))
    units = {u['unit']: u for u in ct.compute(raw, cache_root, None)['conditions.json']['units']}
    assert units[f'serial-{DATASET}-reference']['attempt'] == 'attempt-1'
    assert [x['status'] for x in units[f'serial-{DATASET}-reference']['attempts']] == ['ok', 'failed', 'ok']


def test_missing_plan_and_unit(tmp_path, world):
    with pytest.raises(AssertionError, match='plan, stage, and run'):
        ct.compute(tmp_path / 'nothing', tmp_path, None)
    raw, cache_root, _ = world
    with pytest.raises(AssertionError, match='not in the plan'):
        ct.compute(raw, cache_root, ['caldot1-y05'])
    # Every phase's units are required, the serial ones included.
    shutil.rmtree(raw / gpu.UNITS_DIR / f'serial-{DATASET}-{B}')
    with pytest.raises(AssertionError, match=f'serial-{DATASET}-{B} has no successful attempt'):
        ct.compute(raw, cache_root, None)


def test_planned_unit_mismatch(world):
    raw, cache_root, _ = world
    path = raw / gpu.UNITS_DIR / f'solo-{DATASET}-{A}' / 'attempt-1' / gpu.MANIFEST
    manifest = json.loads(path.read_text())
    manifest['unit']['videos'] = list(VIDEOS)
    path.write_text(json.dumps(manifest))
    with pytest.raises(AssertionError, match='different unit than planned'):
        ct.compute(raw, cache_root, None)


def test_port_check_catches_changed_cache(world):
    raw, cache_root, _ = world
    # A cached log changed after p121 ran no longer reproduces its table.
    write_log(cache_root / rel_path('040_exec_detect', 'te01.mp4', A), '040_exec_detect', 'te01.mp4', ('clean', 'solo'), False)
    with pytest.raises(AssertionError, match='do not reproduce p121'):
        ct.compute(raw, cache_root, None)


def test_paper_throughput_must_match_tradeoff(world):
    raw, cache_root, p = world
    p['datasets'][DATASET]['reference']['fps'] *= 1.01
    (raw / gpu.PLAN_FILE).write_text(json.dumps(p))
    with pytest.raises(AssertionError, match='tradeoff table'):
        ct.compute(raw, cache_root, None)


def append_line(world, unit: str, stage: str, video: str, variant: str | None, data):
    """Append one JSON line to a unit's harvested runtime log."""
    raw = world[0]
    path = raw / gpu.UNITS_DIR / unit / 'attempt-1' / gpu.LOGS_DIR / f'{rel_path(stage, video, variant)}.gz'
    with gzip.open(path, 'at') as f:
        f.write(json.dumps(data) + '\n')


@pytest.mark.parametrize('stage, data, message', [
    ('040_exec_detect', [{'op': 'detect', 'time': -1.0}], 'negative op times'),
    # p121 checks the sign before it drops its excluded ops, so a negative p020 read fails too.
    ('020_exec_classify', [{'op': 'read', 'time': -1.0}], 'negative op times'),
    ('040_exec_detect', [{'op': 'detect', 'time': True}], 'boolean op times'),
    ('040_exec_detect', [{'time': 1.0}], 'timed items without an op'),
    ('060_exec_track', {'frame_idx': 9}, 'runtime lines without an op list'),
    ('060_exec_track', {'frame_idx': 9, 'runtime': 3}, 'runtime lines without an op list'),
])
def test_invalid_runtime_items(world, stage, data, message):
    raw, cache_root, _ = world
    append_line(world, f'serial-{DATASET}-{A}', stage, 'te01.mp4', A, data)
    with pytest.raises(AssertionError, match=f'^{message}: '):
        ct.compute(raw, cache_root, None)


def test_negative_ignored_op_is_allowed(world):
    raw, cache_root, _ = world
    # p121 skips ignored ops before its sign check, so a negative save_canvas is no error.
    append_line(world, f'serial-{DATASET}-{A}', '030_exec_compress', 'te01.mp4', A,
                {'step': 'x', 'runtime': [{'op': 'save_canvas', 'time': -1.0}]})
    ct.compute(raw, cache_root, None)


def test_decode_frame_mismatch(world):
    raw, cache_root, _ = world
    path = raw / gpu.UNITS_DIR / f'solo-{DATASET}-reference' / 'attempt-1' / gpu.DECODE_FILE
    gpu.write_jsonl_gz(path, [dict(video='te01.mp4', frames=2, read_s=0.1, open_s=0.0, total_s=0.1)])
    with pytest.raises(AssertionError, match='other frame counts'):
        ct.compute(raw, cache_root, None)


def rewrite_decode(world, keep):
    """Keep only the serial reference's decode records that `keep` accepts."""
    path = world[0] / gpu.UNITS_DIR / f'serial-{DATASET}-reference' / 'attempt-1' / gpu.DECODE_FILE
    rows = [json.loads(x) for x in gzip.open(path, 'rt')]
    gpu.write_jsonl_gz(path, [r for r in rows if keep(r)])


def test_decode_pass_missing_a_video(world):
    raw, cache_root, _ = world
    rewrite_decode(world, lambda r: not (r['decode_pass'] == 1 and r['video'] == 'te04.mp4'))
    with pytest.raises(AssertionError, match='not all decoded in every pass'):
        ct.compute(raw, cache_root, None)


def test_video_without_decode(world):
    raw, cache_root, _ = world
    rewrite_decode(world, lambda r: r['video'] != 'te04.mp4')
    with pytest.raises(AssertionError, match='without a measured decode in every pass'):
        ct.compute(raw, cache_root, None)


def test_systems_on_different_videos(world):
    raw, cache_root, _ = world
    # A Tetris unit whose manifest lacks one video's outputs timed fewer videos than its reference (a parallel unit, so
    # the paper's side, which follows the serial units, still matches p121).
    path = raw / gpu.UNITS_DIR / f'parallel-{DATASET}-{A}' / 'attempt-2' / gpu.MANIFEST
    manifest = json.loads(path.read_text())
    for by_video in manifest['outputs'].values():
        by_video.pop('te04.mp4')
    path.write_text(json.dumps(manifest))
    with pytest.raises(AssertionError, match='different videos'):
        ct.compute(raw, cache_root, None)


def test_stage_without_wall_or_log(world):
    raw, cache_root, _ = world
    # A stage recorded as not run with no log to stand in for its process.
    path = raw / gpu.UNITS_DIR / f'serial-{DATASET}-{A}' / 'attempt-1' / gpu.MANIFEST
    manifest = json.loads(path.read_text())
    manifest['stages'].append(dict(stage='099_missing', ran=False, source='cache'))
    path.write_text(json.dumps(manifest))
    with pytest.raises(AssertionError, match='stages without a wall clock or logged time'):
        ct.compute(raw, cache_root, None)


def test_wall_clock_steady_needs_two_writes_per_stage(world):
    raw, cache_root, _ = world
    # A stage with one video's write time cannot separate its startup: no estimate for that unit.
    path = raw / gpu.UNITS_DIR / f'serial-{DATASET}-{A}' / 'attempt-1' / gpu.MANIFEST
    manifest = json.loads(path.read_text())
    manifest['outputs']['040_exec_detect']['te04.mp4'].pop('mtime_ns')
    path.write_text(json.dumps(manifest))
    entry = ct.compute(raw, cache_root, None)['throughput.json']['datasets']['B3D1']
    assert entry['0.05']['clean']['wall_clock_steady'] is None and entry['0.10']['clean']['wall_clock_steady'] is not None
    assert entry['0.05']['clean']['wall_clock'] is not None


def test_reference_must_process_every_frame(world):
    raw, cache_root, _ = world
    rel = rel_path('001_preprocess_groundtruth_detection', 'te04.mp4', None)
    write_log(raw / gpu.UNITS_DIR / f'parallel-{DATASET}-reference' / 'attempt-1' / gpu.LOGS_DIR / f'{rel}.gz',
              '001_preprocess_groundtruth_detection', 'te04.mp4', ('clean', 'parallel'), True, frames=1)
    with pytest.raises(AssertionError, match='did not process every frame'):
        ct.compute(raw, cache_root, None)


def test_dataset_label_fallback():
    assert ct.label('jnc0') == 'B3D1' and ct.label('other') == 'other'


def test_foreign_cpu_ignores_units_on_other_cpus(world):
    raw, cache_root, _ = world
    # The parallel reference ran on the other NUMA node: its own share is not subtracted from A's node.
    path = raw / gpu.UNITS_DIR / f'parallel-{DATASET}-reference' / 'attempt-1' / gpu.LAUNCHER_SUMMARY
    path.write_text(json.dumps(dict(json.loads(path.read_text()), numa_node=1, cpus='20-39,60-79')))
    units = {u['unit']: u for u in ct.compute(raw, cache_root, None)['conditions.json']['units']}
    assert units[f'parallel-{DATASET}-{A}']['stages']['020_exec_classify']['cpu']['foreign_on_node_mean'] == pytest.approx(0.3)
    assert units[f'parallel-{DATASET}-reference']['cpu_timed']['foreign_on_node_mean'] == pytest.approx(0.3)


def test_reports_are_reproducible(world):
    raw, cache_root, _ = world
    # Two runs over the same inputs write the same reports.
    first, second = (json.dumps(ct.rounded(ct.compute(raw, cache_root, None)), sort_keys=True) for _ in range(2))
    assert first == second
