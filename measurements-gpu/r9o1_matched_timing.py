#!/usr/local/bin/python
"""R9.O1 matched timing (GPU): the relevance classifier and the reference detector on the same frames and GPU.

measurements-results/r9o1_classifier_cost/ compares the two from pipeline logs written six weeks apart under
concurrent multi-GPU load (REVISION.md finding 4). This job times both models in one process, on one idle GPU,
pinned to that GPU's NUMA node, on the same decoded frames of the same test videos.

Per dataset:
  operating point  the fastest Tetris test configuration within --loss of the reference pipeline's HOTA
                   (common.fastest_tetris, as measurements/r9o1_classifier_cost.py chooses it); its classifier
                   and tile size are timed.
  classifier       loaded and prepared as scripts/p020_exec_classify.py's classify_all does (load_model, the
                   optimization its model_compilation.jsonl selects, ImageNet normalization, the always-relevant
                   mask, tile positions) and run through p020's own classify_batch on every frame, each paired
                   with its previous frame as p020 pairs them at sample rate 1.
  detector         polyis.models.detector.get_detector(dataset, batch_size=1), as
                   preprocess/p002_preprocess_naive_detection.py loads the reference pipeline's detector.
  runs             per video, in this order, on the same decoded frames (RUNS; a call's time is split evenly
                   over its frames):
                     decode             cv2 decoding of the whole video to BGR frames, one interval.
                     rgb_copy           p020's BGR-to-RGB contiguous copy of every frame, one interval. Only
                                        Tetris pays it (the detector takes BGR frames); p121 leaves it untimed.
                     classifier         classify_batch on 16 consecutive frames per call (p020's batching; a
                                        video's last call holds the rest).
                     classifier_1       classify_batch on one frame's tiles per call.
                     classifier_stream  classify_batch over the video in p020's batches, synchronized only
                                        before the first and after the last call, as p020 runs (one interval).
                     detector           detect() on one BGR frame per call (p002, batch size 1).
                     detector_16        detect_batch() on 16 consecutive BGR frames per call (same batching as
                                        the classifier; p002 itself runs batch size 1).
  timing           torch.cuda.synchronize() before and after every timed call and time.perf_counter_ns between,
                   so a call's time holds all of its GPU work. Synchronizing after every call also removes the
                   overlap of one batch's host work with the previous batch's GPU tail that p020 gets, so the
                   per-call classifier figure is an upper bound on its steady-state cost; classifier_stream is
                   the cross-check without that synchronization.
  warm-up          untimed, before the first timed video: every run over that video, except that both detector
                   runs see only its first 32 frames (p002's warm-up).
  GPU              the job refuses to start unless nvidia-smi reports 0% utilization and 0 MiB used on --gpu,
                   unless cuda:0 is that GPU (UUID), and unless it runs on the GPU's NUMA node (CPU affinity
                   within the GPU's local CPUs and numactl membind to its node). It samples every GPU of the host
                   every 500 ms; each sample carries the phase running when it arrived and the phase running
                   UTILIZATION_WINDOW_MS earlier, since utilization is a trailing average. Inside a container
                   nvidia-smi does not list other containers' processes; the launcher's host-side sampler does.

Writes measurements-results/r9o1_matched_timing/raw/ (never the cache):
  <dataset>.jsonl.gz      one line per timed call: dataset, video, run, frame_idx (its first frame), frames, ms,
                          start_ns (epoch), ops (p020's host-side op intervals; per-call classifier runs only)
  <dataset>.gpu.jsonl.gz  nvidia-smi samples of every GPU: t_ns, phase, window_phase, gpu_index, utilization,
                          memory, clocks, power, temperature, clock event reasons
  manifest.json           per finished dataset: operating point, models and weights, runs and their frames per
                          call, videos (frames, file hash, anonymous memory pages per NUMA node), warm-up, start
                          and end, the launch's idle check, and the environment (placement included)
  host_compute_apps.<start ms>.jsonl.gz  the launcher's host-side samples (t_ms, apps on the GPU, apps on all
                          GPUs, 1-minute load average), one file per launch
measurements/r9o1_matched_timing.py aggregates them.

Run with measurements-gpu/r9o1_matched_timing.sh <staging dir> <physical GPU index> (see that file): it checks
the GPU from the host, starts the host-side sampler, and runs this script in the polyis image with
CUDA_VISIBLE_DEVICES=<gpu>, CUDA_DEVICE_ORDER=PCI_BUS_ID, and numactl bound to the GPU's NUMA node.
"""
import argparse
import bisect
import collections
import gzip
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from functools import partial
from pathlib import Path

import cv2
import detectron2
import duckdb
import numpy as np
import torch
import ultralytics

import polyis.models.detector
import polyis.models.retinanet_b3d
from measurements.common import (DATASET_LABELS, fastest_tetris, file_record, host_info, iso_utc, load_test_tradeoff,
                                 load_tradeoff_points, results_dir, write_json)
from measurements.r9o1_matched_timing import MANIFEST, RAW_DIR
from polyis.io import cache, store
from polyis.train.select_model_optimization import select_model_optimization
from scripts.p020_exec_classify import BATCH_SIZE, classify_batch, load_model


# Reference detector frames run before timing, as p002 warms up.
DETECTOR_WARMUP_FRAMES = 32
# Interval between nvidia-smi samples.
SAMPLE_INTERVAL_MS = 500
# Longest period nvidia-smi's utilization averages over (1/6 s to 1 s depending on the GPU).
UTILIZATION_WINDOW_MS = 1000
# nvidia-smi fields sampled, in query order, with their names in the sample records.
GPU_FIELDS = {'index': 'gpu_index', 'utilization.gpu': 'utilization_gpu_pct', 'utilization.memory': 'utilization_memory_pct',
              'memory.used': 'memory_used_mib', 'clocks.sm': 'sm_clock_mhz', 'clocks.mem': 'memory_clock_mhz',
              'power.draw': 'power_w', 'temperature.gpu': 'temperature_c', 'pstate': 'pstate',
              'clocks_event_reasons.active': 'clock_event_reasons'}
# nvidia-smi fields identifying the GPU, in query order.
GPU_IDENTITY = ('name', 'uuid', 'pci.bus_id', 'driver_version', 'vbios_version')
# Timed runs per video, in execution order, with their frames per call (None: the whole video in one interval).
RUNS = {'decode': None, 'rgb_copy': None, 'classifier': BATCH_SIZE, 'classifier_1': 1, 'classifier_stream': None,
        'detector': 1, 'detector_16': BATCH_SIZE}
# What each run times, for the manifest.
RUN_DESCRIPTIONS = {
    'decode': 'cv2 decoding of the whole video to BGR frames (one interval)',
    'rgb_copy': "p020's BGR-to-RGB contiguous copy of every frame (one interval; Tetris only, untimed in p121)",
    'classifier': f"p020's classify_batch, {BATCH_SIZE} consecutive frames per call (p020's batching)",
    'classifier_1': 'classify_batch, one frame per call',
    'classifier_stream': (f"classify_batch over the video in {BATCH_SIZE}-frame calls, synchronized only before the first "
                          'and after the last call (one interval), as p020 runs'),
    'detector': 'polyis.models.detector.detect, one BGR frame per call (p002, batch size 1)',
    'detector_16': f'polyis.models.detector.detect_batch, {BATCH_SIZE} consecutive BGR frames per call',
}
# Runs whose calls return p020's op intervals, which the raw lines keep.
OPS_RUNS = ('classifier', 'classifier_1')
# Runs of the reference detector, which only sees the first DETECTOR_WARMUP_FRAMES frames during warm-up.
DETECTOR_RUNS = ('detector', 'detector_16')
# The device the models use; CUDA_VISIBLE_DEVICES maps it to --gpu.
DEVICE = 'cuda:0'
# The launcher this job is started by (recorded in the manifest).
LAUNCHER = Path(__file__).with_suffix('.sh')
# Where sysfs lists PCI devices, with their NUMA node and local CPUs.
SYSFS_PCI = Path('/sys/bus/pci/devices')


def parse_args():
    parser = argparse.ArgumentParser(description='R9.O1: classifier and reference detector timed on the same frames and idle GPU')
    parser.add_argument('--datasets', nargs='+', choices=list(DATASET_LABELS), default=list(DATASET_LABELS))
    parser.add_argument('--loss', type=float, default=0.05)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--videos', type=int, default=3)
    return parser.parse_args()


def number(text: str):
    """An nvidia-smi value as a float, a string when it is not numeric (pstate, hex flags), or None when unavailable."""
    # nvidia-smi writes unavailable values in brackets, e.g. [N/A] or [Not Supported].
    if text.startswith('['):
        return None
    # Numbers become floats; anything else stays text.
    try:
        return float(text)
    except ValueError:
        return text


def parse_sample(line: str) -> dict:
    """One nvidia-smi CSV line of GPU_FIELDS (no header, no units) as a record keyed by GPU_FIELDS' names."""
    # Comma-separated values in query order.
    values = [v.strip() for v in line.split(',')]
    # A line with another field count means the query or the driver's output changed.
    assert len(values) == len(GPU_FIELDS), f'unexpected nvidia-smi line {line!r}'
    record = {name: number(v) for name, v in zip(GPU_FIELDS.values(), values)}
    # The GPU index is an integer key, not a measurement.
    record['gpu_index'] = int(record['gpu_index'])
    return record


def query_gpu(gpu: int, fields) -> str:
    """nvidia-smi's CSV line (no header, no units) of `fields` for physical GPU `gpu`."""
    # nvidia-smi only queries the driver; it starts nothing on the GPU.
    return subprocess.run(['nvidia-smi', f'--query-gpu={",".join(fields)}', '--format=csv,noheader,nounits', '-i', str(gpu)],
                          capture_output=True, text=True, check=True).stdout.strip()


def idle_check(gpu: int) -> dict:
    """The GPU's sample before this job touches it; fails unless it shows 0% utilization and 0 MiB used."""
    # One sample of the sampled fields.
    sample = parse_sample(query_gpu(gpu, GPU_FIELDS))
    # Another job on the GPU would contend with both models.
    assert sample['utilization_gpu_pct'] == 0 and sample['memory_used_mib'] == 0, f'GPU {gpu} is not idle: {sample}'
    return dict(sample, t=iso_utc(time.time()))


class GpuSampler:
    """nvidia-smi samples of every GPU every SAMPLE_INTERVAL_MS, labeled with the phases running when they were taken."""

    def __init__(self):
        # Phases in the order the job enters them, each with its start time (epoch ns).
        self.phases: list[tuple[int, str]] = [(time.time_ns(), 'start')]
        # Samples received so far.
        self.samples: list[dict] = []
        # nvidia-smi's own sampling loop over all GPUs, one CSV line per GPU and sample.
        self.process = subprocess.Popen(['nvidia-smi', f'--query-gpu={",".join(GPU_FIELDS)}', '--format=csv,noheader,nounits',
                                         '-lms', str(SAMPLE_INTERVAL_MS)], stdout=subprocess.PIPE, text=True)
        # Read its lines in the background.
        self.thread = threading.Thread(target=self.read, daemon=True)
        self.thread.start()

    def set_phase(self, phase: str):
        """Start a phase now; samples taken from here on carry it."""
        # Appending is atomic, so the reader thread sees either the old or the new list end.
        self.phases.append((time.time_ns(), phase))

    def phase_at(self, t_ns: int) -> str:
        """The phase running at t_ns (the first phase for earlier times)."""
        # The last phase that started at or before t_ns.
        i = bisect.bisect_right(self.phases, t_ns, key=lambda p: p[0]) - 1
        return self.phases[max(i, 0)][1]

    def label(self, t_ns: int, line: str) -> dict:
        """A sample line arriving at t_ns, with the phase then and the phase at the start of its averaging window."""
        # A sample whose two phases differ straddles a phase boundary; the aggregation drops it.
        return dict(t_ns=t_ns, phase=self.phase_at(t_ns), window_phase=self.phase_at(t_ns - UTILIZATION_WINDOW_MS * 1_000_000),
                    **parse_sample(line))

    def read(self):
        """Label every non-blank line as it arrives."""
        # nvidia-smi writes until it is terminated.
        for line in self.process.stdout:
            # Blank lines separate nothing useful.
            if line.strip():
                self.samples.append(self.label(time.time_ns(), line))

    def stop(self) -> list[dict]:
        """End sampling and return the samples."""
        # Stop nvidia-smi, then let the reader drain what it wrote.
        self.process.terminate()
        self.process.wait()
        self.thread.join()
        return self.samples


def select_videos(videos: list[str], n: int) -> list[str]:
    """n of the sorted videos spread evenly from the first to the last (all of them when there are at most n)."""
    # A dataset without test videos, or no video asked for, has nothing to time.
    assert videos and n > 0, 'no videos to select'
    videos = sorted(videos)
    # How many to take, and their evenly spaced indices (first and last included).
    k = min(n, len(videos))
    return [videos[i * (len(videos) - 1) // (k - 1)] for i in range(k)] if k > 1 else videos[:1]


def operating_point(dataset: str, loss: float) -> dict:
    """The dataset's operating point as measurements/r9o1_classifier_cost.py chooses it, with its classifier and tile size."""
    # The test split's Tetris configurations and reference row.
    tetris, naive = load_test_tradeoff(dataset)
    # The fastest Tetris configuration within the bound, by the shared rule.
    con = duckdb.connect()
    load_tradeoff_points(con, 'configs', {dataset: (tetris, naive)})
    chosen = fastest_tetris(con, 'configs', loss)[dataset]
    # Its parameters from the tradeoff row.
    row = tetris[tetris['variant_id'] == chosen['config']].iloc[0]
    return dict(config=chosen['config'], classifier=str(row['classifier']), tilesize=int(row['tilesize']),
                sample_rate=int(row['sample_rate']), hota=float(chosen['accuracy']),
                reference_hota=float(chosen['reference_accuracy']), loss=loss)


def video_size(path: Path) -> tuple[int, int]:
    """(width, height) of a video from its header, as p020 reads it."""
    # Open the container without decoding frames.
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), f'{path}: cannot open'
    # The header's frame size.
    size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return size


def decode(path: Path) -> list[np.ndarray]:
    """Every frame of a video, decoded with cv2 (BGR), as p002 and p020 read them."""
    # Open the video for sequential decoding.
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), f'{path}: cannot open'
    # The frame count the header declares, which p020 checks the decoded frames against.
    declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    # Read to the end of the stream.
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    # A stream shorter or longer than its header would time frames p020 never sees, or miss some.
    assert len(frames) == declared, f'{path}: decoded {len(frames)} of {declared} frames'
    # p020 pairs the first frame with the second, so a video needs two frames.
    assert len(frames) >= 2, f'{path}: fewer than two frames'
    return frames


def to_rgb(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Contiguous RGB copies of BGR frames, as p020 makes them while reading."""
    # p020's read: np.ascontiguousarray(frame[:, :, ::-1]) per frame.
    return [np.ascontiguousarray(f[:, :, ::-1]) for f in frames]


def load_classifier(dataset: str, classifier: str, tilesize: int, width: int, height: int) -> dict:
    """The classifier prepared as p020's classify_all prepares it, and classify(batch_frames, batch_prev_frames)."""
    # Trained weights in half precision (p020's load_model), on the GPU.
    model = load_model(dataset, tilesize, classifier, DEVICE).to(DEVICE)
    # The optimization p020 applies: the fastest one in the compilation benchmark, for one frame's tiles.
    benchmark_path = cache.index(dataset, 'training', 'results', f'{classifier}_{tilesize}', 'model_compilation.jsonl')
    with open(benchmark_path) as f:
        benchmark = [json.loads(line) for line in f]
    # The tile grid; pixels beyond the last full tile are dropped.
    grid_w, grid_h = width // tilesize, height // tilesize
    model, method = select_model_optimization(model, benchmark, DEVICE, tilesize, grid_w * grid_h)
    # ImageNet normalization of the frame and difference channels, in half precision (p020's constants; a test
    # checks they still appear in p020's source).
    mean = torch.tensor([0.485, 0.456, 0.406] * 2, device=DEVICE, dtype=torch.float16).view(1, 6, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225] * 2, device=DEVICE, dtype=torch.float16).view(1, 6, 1, 1)
    # Tiles never relevant in training are prepared but skipped at inference.
    mask_path = cache.index(dataset, 'never-relevant', f'{tilesize}_all.npy')
    mask = np.load(mask_path)
    assert mask.shape == (grid_h, grid_w), f'{dataset}: mask {mask.shape} does not match the {grid_h}x{grid_w} grid'
    # Tile positions (row, column) in row-major order, as p020's classify builds them.
    y = torch.arange(grid_h, device=DEVICE, dtype=torch.uint8).repeat_interleave(grid_w)
    x = torch.arange(grid_w, device=DEVICE, dtype=torch.uint8).repeat(grid_h)
    # p020's classify_batch with everything but the frames bound.
    classify = partial(classify_batch, grid_width=grid_w, grid_height=grid_h, positions=torch.stack([y, x], dim=1).float(),
                       model=model, tile_size=tilesize, device=DEVICE, normalize_mean=mean, normalize_std=std,
                       always_relevant_mask=torch.from_numpy(mask.flatten()).to(DEVICE).to(torch.uint8))
    # What the manifest records about the classifier.
    info = dict(name=classifier, tilesize=tilesize, optimization=method, grid=[grid_h, grid_w], tiles=int(mask.size),
                tiles_inferred_at_most=int(np.count_nonzero(mask)), batch_size=BATCH_SIZE,
                weights=file_record(cache.index(dataset, 'training', 'results', f'{classifier}_{tilesize}', 'model_best.pth')),
                compilation_benchmark=file_record(benchmark_path), mask=file_record(mask_path))
    return dict(classify=classify, model=model, info=info)


def detector_weights(dataset: str) -> Path:
    """The reference detector's weights file, as polyis.models.detector.get_detector resolves it."""
    # The dataset's entry in configs/detectors.yaml.
    info = polyis.models.detector.get_detector_info(dataset)
    # RetinaNet (B3D) takes its weights from configs/retinanet.json instead.
    if info['detector'] == 'retina':
        with open(polyis.models.retinanet_b3d.CONFIG) as f:
            return Path(json.load(f)['weights'])
    return Path(info['model_path'])


def timed(fn, *args, **kwargs) -> tuple[object, float, int]:
    """fn's result, its time in ms with the GPU synchronized before and after, and its start (epoch ns)."""
    # Nothing queued earlier may run inside the interval.
    torch.cuda.synchronize()
    start_ns, t0 = time.time_ns(), time.perf_counter_ns()
    out = fn(*args, **kwargs)
    # All of the call's GPU work finishes inside the interval.
    torch.cuda.synchronize()
    return out, (time.perf_counter_ns() - t0) / 1e6, start_ns


def anon_pages_by_node() -> dict[str, int]:
    """This process's anonymous memory pages per NUMA node from /proc/self/numa_maps, e.g. {'N1': 123}."""
    pages = collections.Counter()
    with open('/proc/self/numa_maps') as f:
        # Anonymous mappings (heap, decoded frames) carry anon=; file-backed ones (libraries) live in the page cache.
        for line in f:
            if ' anon=' not in line:
                continue
            # Fields N<node>=<pages> say where the mapping's pages are.
            for field in line.split():
                if field.startswith('N') and '=' in field:
                    node, count = field.split('=')
                    pages[node] += int(count)
    return dict(sorted(pages.items()))


def run_video(dataset: str, video: str, path: Path, size: tuple[int, int], classify, detector, sampler,
              warmup: bool) -> tuple[list[dict], dict]:
    """Every run of RUNS over a video's frames, decoded here; one record per timed call, and the video's record.

    With warmup, the sampler's phases are prefixed 'warmup_' and the detector runs see only the first
    DETECTOR_WARMUP_FRAMES frames; the caller drops the records.
    """
    def phase(run: str):
        # Label the GPU samples with the run about to execute.
        sampler.set_phase(f'warmup_{run}' if warmup else run)

    def record(run: str, frame_idx: int, frames: int, ms: float, start_ns: int, **extra) -> dict:
        # One timed call.
        return dict(dataset=dataset, video=video, run=run, frame_idx=frame_idx, frames=frames, ms=ms, start_ns=start_ns, **extra)

    # Decode the whole video before any model runs, as p020 does.
    phase('decode')
    frames, decode_ms, decode_ns = timed(decode, path)
    # p020 takes the frame size from the first video and assumes it for every video.
    assert frames[0].shape[:2] == (size[1], size[0]), f'{dataset}/{video}: frame size differs from the first video'
    records = [record('decode', 0, len(frames), decode_ms, decode_ns)]
    # The classifier's input: RGB copies as p020 makes them, timed because only Tetris pays for them.
    phase('rgb_copy')
    rgb, copy_ms, copy_ns = timed(to_rgb, frames)
    records.append(record('rgb_copy', 0, len(rgb), copy_ms, copy_ns))
    # Each frame's previous frame (the second for the first), as p020 pairs them at sample rate 1.
    prev = [rgb[i - 1] if i > 0 else rgb[i + 1] for i in range(len(rgb))]

    def stream():
        # The whole video in p020's batches with no synchronization between calls, as p020 runs.
        for start in range(0, len(rgb), BATCH_SIZE):
            classify(batch_frames=rgb[start:start + BATCH_SIZE], batch_prev_frames=prev[start:start + BATCH_SIZE])

    # Per-call classifier runs, then the stream, under no_grad as p020 runs.
    with torch.no_grad():
        for run in OPS_RUNS:
            phase(run)
            # Consecutive calls of RUNS[run] frames; the last call holds the rest.
            for start in range(0, len(rgb), RUNS[run]):
                (_, ops), ms, start_ns = timed(classify, batch_frames=rgb[start:start + RUNS[run]],
                                               batch_prev_frames=prev[start:start + RUNS[run]])
                records.append(record(run, start, len(rgb[start:start + RUNS[run]]), ms, start_ns, ops=ops))
        phase('classifier_stream')
        _, ms, start_ns = timed(stream)
        records.append(record('classifier_stream', 0, len(rgb), ms, start_ns))
    # The reference detector on the BGR frames, one per call (p002) and in the classifier's batches.
    detect = {'detector': lambda batch: polyis.models.detector.detect(batch[0], detector),
              'detector_16': lambda batch: polyis.models.detector.detect_batch(batch, detector)}
    detector_frames = frames[:DETECTOR_WARMUP_FRAMES] if warmup else frames
    for run in DETECTOR_RUNS:
        phase(run)
        # Consecutive calls of RUNS[run] frames; the last call holds the rest.
        for start in range(0, len(detector_frames), RUNS[run]):
            batch = detector_frames[start:start + RUNS[run]]
            _, ms, start_ns = timed(detect[run], batch)
            records.append(record(run, start, len(batch), ms, start_ns))
    # Where the decoded frames and their copies live, while both are still in memory.
    return records, dict(frames=len(frames), anon_pages_by_node=anon_pages_by_node())


def cpu_model() -> str | None:
    """The host CPU's model name from /proc/cpuinfo, or None where it is unavailable."""
    try:
        # The first model name line; every socket of these hosts has the same model.
        with open('/proc/cpuinfo') as f:
            return next((line.split(':', 1)[1].strip() for line in f if line.startswith('model name')), None)
    except OSError:
        # Hosts without /proc (not Linux) say nothing about their CPU.
        return None


def sysfs_bus_id(bus_id: str) -> str:
    """nvidia-smi's PCI bus id (00000000:B2:00.0) as sysfs names the device (0000:b2:00.0)."""
    # nvidia-smi pads the domain to 8 hex digits and writes upper case; sysfs uses 4 digits and lower case.
    domain, rest = bus_id.split(':', 1)
    return f'{int(domain, 16):04x}:{rest.lower()}'


def parse_cpulist(text: str) -> list[int]:
    """A kernel CPU list such as '20-39,60-79' as the sorted CPU numbers."""
    cpus = []
    # Comma-separated single CPUs and inclusive ranges.
    for part in text.strip().split(','):
        first, _, last = part.partition('-')
        cpus += range(int(first), int(last or first) + 1)
    return sorted(cpus)


def numactl_show() -> dict:
    """numactl --show for this process (policy, membind, cpubind, ...), keyed by its field names."""
    # numactl is installed by the launcher; without it the job cannot say where its memory is bound.
    assert shutil.which('numactl'), 'numactl not found: start the job with measurements-gpu/r9o1_matched_timing.sh'
    out = subprocess.run(['numactl', '--show'], capture_output=True, text=True, check=True).stdout
    # One 'name: value' line per field.
    return {k.strip(): v.strip() for k, v in (line.split(':', 1) for line in out.splitlines() if ':' in line)}


def placement(bus_id: str) -> dict:
    """Where the job runs (CPU affinity, memory policy) and where the GPU sits (NUMA node, local CPUs)."""
    # The GPU's PCI device in sysfs.
    device = SYSFS_PCI / sysfs_bus_id(bus_id)
    return dict(gpu_numa_node=int((device / 'numa_node').read_text()),
                gpu_local_cpus=parse_cpulist((device / 'local_cpulist').read_text()),
                cpus=sorted(os.sched_getaffinity(0)), numactl=numactl_show())


def environment(gpu: int) -> dict:
    """Host, GPU, placement, driver, and library versions, so the timings say where they were taken."""
    # The GPU's identity from the driver, and the device the models see.
    identity = dict(zip(GPU_IDENTITY, (v.strip() for v in query_gpu(gpu, GPU_IDENTITY).split(','))))
    props = torch.cuda.get_device_properties(DEVICE)
    return dict(host_info(), cpu=cpu_model(), gpu_index=gpu, gpu=identity, torch_device=props.name,
                torch_device_uuid=f'GPU-{props.uuid}', placement=placement(identity['pci.bus_id']),
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), cuda_device_order=os.environ.get('CUDA_DEVICE_ORDER'),
                torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                ultralytics=ultralytics.__version__, detectron2=detectron2.__version__, opencv=cv2.__version__,
                python=platform.python_version(), container_image=os.environ.get('POLYIS_IMAGE', 'polyis (unrecorded tag)'),
                argv=sys.argv, launcher=file_record(LAUNCHER))


def check_environment(env: dict):
    """Fail unless cuda:0 is the checked GPU and the job's CPUs and memory are on that GPU's NUMA node."""
    # CUDA_DEVICE_ORDER and CUDA_VISIBLE_DEVICES must make cuda:0 the GPU that nvidia-smi checked and samples.
    assert env['torch_device_uuid'] == env['gpu']['uuid'], f"cuda:0 is {env['torch_device_uuid']}, not GPU {env['gpu_index']} ({env['gpu']['uuid']})"
    p = env['placement']
    # Host work (copies, kernel launches) must run on CPUs local to the GPU.
    assert set(p['cpus']) <= set(p['gpu_local_cpus']), f"CPUs {p['cpus']} are not all local to the GPU ({p['gpu_local_cpus']})"
    # Frames must be allocated on the GPU's node.
    assert p['numactl'].get('membind') == str(p['gpu_numa_node']), f"memory bound to {p['numactl'].get('membind')}, not node {p['gpu_numa_node']}"


def write_jsonl_gz(path: Path, records: list[dict]):
    """One JSON line per record, gzip-compressed."""
    # Text mode so json.dumps' strings are written directly.
    with gzip.open(path, 'wt') as f:
        f.writelines(json.dumps(r) + '\n' for r in records)


def time_dataset(dataset: str, args, out_dir: Path, idle: dict, env: dict) -> dict:
    """Load both models for one dataset, warm up, time every frame of the chosen videos, write the raw files; returns the manifest entry."""
    started = time.time()
    # Sample the GPUs for the whole dataset, starting with model loading.
    sampler = GpuSampler()
    sampler.set_phase('load')
    # The operating point's classifier and tile size.
    point = operating_point(dataset, args.loss)
    # The videos to time, spread over the test split, and their frame size (p020 takes it from the first video).
    videos = select_videos([v for v in os.listdir(store.dataset(dataset, 'test')) if v.endswith('.mp4')], args.videos)
    size = video_size(store.dataset(dataset, 'test', videos[0]))
    classifier = load_classifier(dataset, point['classifier'], point['tilesize'], *size)
    # The reference pipeline's detector, as p002 loads it.
    detector = polyis.models.detector.get_detector(dataset, 0, batch_size=1)
    records, video_records = [], {}
    for i, video in enumerate(videos):
        path = store.dataset(dataset, 'test', video)
        # Warm every run up on the first video, untimed.
        if i == 0:
            run_video(dataset, video, path, size, classifier['classify'], detector, sampler, warmup=True)
        timed_records, video_record = run_video(dataset, video, path, size, classifier['classify'], detector, sampler, warmup=False)
        records += timed_records
        # The video's frames and placement, and which file was timed.
        video_records[video] = dict(video_record, file=file_record(path))
        print(f'{dataset} {video}: {video_record["frames"]} frames', flush=True)
    sampler.set_phase('end')
    samples = sampler.stop()
    finished = time.time()
    # The raw timings and GPU samples.
    write_jsonl_gz(out_dir / f'{dataset}.jsonl.gz', records)
    write_jsonl_gz(out_dir / f'{dataset}.gpu.jsonl.gz', samples)
    # Free both models before the next dataset's.
    polyis.models.detector.delete(detector)
    info = classifier.pop('info')
    del classifier
    torch.cuda.empty_cache()
    return dict(
        label=DATASET_LABELS[dataset], operating_point=point, classifier=info,
        detector=dict(polyis.models.detector.get_detector_info(dataset), batch_size=1, weights=file_record(detector_weights(dataset))),
        runs=RUNS, videos=video_records, frames=sum(v['frames'] for v in video_records.values()),
        warmup=dict(video=videos[0], runs='every run over every frame', detector_frames=DETECTOR_WARMUP_FRAMES),
        started=iso_utc(started), finished=iso_utc(finished), started_ns=int(started * 1e9), finished_ns=int(finished * 1e9),
        seconds=round(finished - started, 1), samples=len(samples), idle_before_launch=idle, environment=env)


def main():
    # Datasets, the operating point's loss, the GPU, and videos per dataset.
    args = parse_args()
    # The models use cuda:0, which must be the GPU that is checked and sampled (nvidia-smi numbers GPUs by PCI bus).
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(args.gpu), f'set CUDA_VISIBLE_DEVICES={args.gpu}'
    assert os.environ.get('CUDA_DEVICE_ORDER') == 'PCI_BUS_ID', 'set CUDA_DEVICE_ORDER=PCI_BUS_ID'
    # Refuse a busy GPU before anything touches it.
    idle = idle_check(args.gpu)
    # Where the job runs; refuse the wrong device or a placement off the GPU's NUMA node.
    env = environment(args.gpu)
    check_environment(env)
    # The raw directory inside the measurement's result directory (same script name).
    out_dir = results_dir(__file__) / RAW_DIR
    out_dir.mkdir(exist_ok=True)
    # Keep the records of datasets finished by an earlier run.
    manifest_path = out_dir / MANIFEST
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else dict(datasets={})
    manifest['method'] = dict(
        operating_point=f'fastest Tetris test configuration with HOTA >= reference HOTA x (1 - {args.loss}) (common.fastest_tetris)',
        classifier=('scripts/p020_exec_classify.py load_model, select_model_optimization, and classify_batch, prepared as '
                    'classify_all; every frame paired with its previous frame (sample rate 1)'),
        runs=RUN_DESCRIPTIONS,
        timing=('torch.cuda.synchronize() before and after every timed call; time.perf_counter_ns; a call\'s time split '
                'evenly over its frames. Per-call synchronization removes p020\'s overlap of host work with the previous '
                'batch\'s GPU tail, so the per-call classifier runs are an upper bound on its steady-state cost; '
                'classifier_stream synchronizes once per video'),
        order=f'per video: {", ".join(RUNS)}, on the same decoded frames',
        warmup=f'untimed: every run over the first video, the detector runs on its first {DETECTOR_WARMUP_FRAMES} frames only',
        videos=f'{args.videos} test videos per dataset, evenly spaced in name order, every frame',
        placement=('CPU affinity and memory bound to the GPU\'s NUMA node (numactl --cpunodebind --membind), checked at '
                   'start; anonymous pages per node recorded per video'),
        gpu_samples=(f'nvidia-smi on every GPU every {SAMPLE_INTERVAL_MS} ms, labeled with the phase on arrival and '
                     f'{UTILIZATION_WINDOW_MS} ms earlier'),
        sampling='none', estimates='none')
    for dataset in args.datasets:
        # Time the dataset, then record it and save the manifest, so a later failure keeps it.
        manifest['datasets'][dataset] = time_dataset(dataset, args, out_dir, idle, env)
        write_json(out_dir, MANIFEST, manifest)


if __name__ == '__main__':
    main()
