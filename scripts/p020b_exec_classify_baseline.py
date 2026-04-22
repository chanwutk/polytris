#!/usr/local/bin/python

import argparse
import itertools
import json
import os
from typing import Callable, cast
import cv2
import torch
import numpy as np
import time
import shutil
import multiprocessing as mp
from functools import partial

from polyis.images import ImgNHWC, splitNHWC

from polyis.io import cache, store
from polyis.train.select_model_optimization import select_baseline_model_optimization
from polyis.utilities import format_time, ProgressBar, get_config


config = get_config()
TILE_SIZES: list[int] = config['EXEC']['TILE_SIZES']
CLASSIFIERS: list[str] = [c for c in config['EXEC']['CLASSIFIERS'] if c != 'Perfect']
DATASETS: list[str] = config['EXEC']['DATASETS']
# Baseline comparison is only run at sample_rate=1 (see evaluation/p205).
SAMPLE_RATES: list[int] = [1]
BATCH_SIZE: int = 16

# Source classifier names that have a trained baseline in p017 (must match p017 BASELINE_MODEL_ZOO).
BASELINE_CLASSIFIER_SOURCES: frozenset[str] = frozenset({'ShuffleNet05'})


def parse_args():
    parser = argparse.ArgumentParser(
        description='Execute tile classification using image-only baseline models (no diff, no position)'
    )
    parser.add_argument('--test', action='store_true', help='Process test videoset')
    parser.add_argument('--valid', action='store_true', help='Process valid videoset')
    return parser.parse_args()


def load_model(dataset_name: str, tile_size: int, classifier_name: str, device: str) -> torch.nn.Module:
    baseline_key = f'{classifier_name}Baseline_{tile_size}'
    results_path = cache.index(dataset_name, 'training', 'results', baseline_key)
    model_path = results_path / 'model_best.pth'

    if os.path.exists(model_path):
        model = torch.load(model_path, map_location='cpu', weights_only=False)
        model.eval()
        model.half()
        return model

    raise FileNotFoundError(f"No trained baseline model found for {baseline_key} in {results_path}")


def classify_batch(
    grid_width: int,
    grid_height: int,
    batch_frames: list[np.ndarray],
    model: torch.nn.Module,
    tile_size: int,
    device: str,
    normalize_mean: torch.Tensor,
    normalize_std: torch.Tensor,
    always_relevant_mask: torch.Tensor,
    method_name: str = 'baseline',
) -> tuple[torch.Tensor, list[dict]]:
    batch_size = len(batch_frames)
    num_tiles = grid_height * grid_width

    send_start = time.time_ns() / 1e6
    frames_gpu = [torch.from_numpy(f).to(device, non_blocking=True) for f in batch_frames]
    frames_tensor = torch.stack(frames_gpu, dim=0)
    send_runtime = time.time_ns() / 1e6 - send_start

    resize_start = time.time_ns() / 1e6
    target_h = tile_size * grid_height
    target_w = tile_size * grid_width
    current_h, current_w = frames_tensor.shape[1:3]

    if (current_h, current_w) != (target_h, target_w):
        frames_tensor = torch.nn.functional.interpolate(
            frames_tensor.permute(0, 3, 1, 2).half(),
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False,
        ).to(torch.uint8).permute(0, 2, 3, 1)
    resize_runtime = time.time_ns() / 1e6 - resize_start

    reshape_start = time.time_ns() / 1e6
    tiles_nghwc = splitNHWC(cast(ImgNHWC, frames_tensor), tile_size, tile_size)
    tiles_flat = tiles_nghwc.reshape(batch_size * num_tiles, tile_size, tile_size, 3)
    reshape_runtime = time.time_ns() / 1e6 - reshape_start

    mask_start = time.time_ns() / 1e6
    non_black_mask = tiles_flat.reshape(batch_size * num_tiles, -1).any(dim=1).to(torch.bool)
    always_relevant_expanded = (
        always_relevant_mask.to(torch.bool).unsqueeze(0).expand(batch_size, -1).reshape(-1)
    )
    valid_flat_idx = (non_black_mask & always_relevant_expanded).nonzero().squeeze(1)
    tiles_valid = tiles_flat[valid_flat_idx].float() / 255.0
    tiles_nchw_valid = tiles_valid.permute(0, 3, 1, 2)
    tiles_nchw_valid = (tiles_nchw_valid - normalize_mean) / normalize_std
    mask_runtime = (time.time_ns() / 1e6) - mask_start

    inference_start = time.time_ns() / 1e6
    all_tiles = tiles_nchw_valid.half()
    if method_name in ('channels_last', 'torch_compile_channels_last'):
        all_tiles = all_tiles.to(memory_format=torch.channels_last)  # type: ignore
    raw = model(all_tiles)
    if raw.dim() == 2 and raw.shape[1] == 1:
        predictions = torch.sigmoid(raw)
    else:
        predictions = torch.sigmoid(raw.unsqueeze(1) if raw.dim() == 1 else raw)
    inference_runtime = time.time_ns() / 1e6 - inference_start

    collect_start = time.time_ns() / 1e6
    predictions_uint8 = (predictions * 255).to(torch.uint8)
    probabilities_full = torch.zeros(batch_size * num_tiles, 1, device=device, dtype=torch.uint8)
    probabilities_full[valid_flat_idx] = predictions_uint8
    probabilities_per_frame = probabilities_full.reshape(batch_size, grid_height, grid_width)
    collect_runtime = time.time_ns() / 1e6 - collect_start

    runtime = format_time(
        inference=inference_runtime,
        collect=collect_runtime,
        reshape=reshape_runtime,
        mask=mask_runtime,
        diff=0.0,
        send=send_runtime,
        resize=resize_runtime,
    )
    return probabilities_per_frame, runtime


def classify(
    dataset: str,
    videoset: str,
    video: str,
    classifier: str,
    tile_size: int,
    sample_rate: int,
    device: str,
    model: torch.nn.Module,
    normalize_mean: torch.Tensor,
    normalize_std: torch.Tensor,
    always_relevant_mask: torch.Tensor,
    method_name: str = 'baseline',
):
    video_path = store.dataset(dataset, videoset, video)
    output_dir = cache.exec(dataset, 'relevancy', video)

    out_name = f'{classifier}Baseline_{tile_size}_{sample_rate}'
    classifier_dir = os.path.join(output_dir, out_name)
    if os.path.exists(classifier_dir):
        shutil.rmtree(classifier_dir)
    os.makedirs(classifier_dir)

    score_dir = os.path.join(classifier_dir, 'score')
    os.makedirs(score_dir)
    output_path = os.path.join(score_dir, 'score.jsonl')
    runtime_path = os.path.join(score_dir, 'runtime.jsonl')

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Could not open video {video_path}"

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    grid_width = width // tile_size
    grid_height = height // tile_size

    with open(output_path, 'w') as f, open(runtime_path, 'w') as fr, torch.no_grad():
        read_start = time.time_ns() / 1e6
        frames: list[np.ndarray] = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(np.ascontiguousarray(frame[:, :, ::-1]))
        cap.release()
        read_runtime = time.time_ns() / 1e6 - read_start
        assert len(frames) == frame_count, f"Expected {frame_count} frames, got {len(frames)}"

        sampled_indices = [idx for idx in range(len(frames)) if idx % sample_rate == 0]
        last_idx = len(frames) - 1
        if last_idx >= 0 and (not sampled_indices or sampled_indices[-1] != last_idx):
            sampled_indices.append(last_idx)
        sampled_frames = [frames[idx] for idx in sampled_indices]

        frame_idx_in_sampled = 0
        all_probs = []
        runtimes = []
        while frame_idx_in_sampled < len(sampled_frames):
            batch_end = min(frame_idx_in_sampled + BATCH_SIZE, len(sampled_frames))
            batch_frames = [sampled_frames[i] for i in range(frame_idx_in_sampled, batch_end)]
            batch_indices = [sampled_indices[i] for i in range(frame_idx_in_sampled, batch_end)]

            probs, runtime = classify_batch(
                grid_width,
                grid_height,
                batch_frames,
                model,
                tile_size,
                device,
                normalize_mean,
                normalize_std,
                always_relevant_mask,
                method_name,
            )
            runtimes.append(runtime)
            all_probs.append((probs, batch_indices))
            frame_idx_in_sampled += BATCH_SIZE

        for runtime in runtimes:
            fr.write(json.dumps(runtime) + '\n')

        retrieve_start = time.time_ns() / 1e6
        entries = []
        for probs, batch_indices in all_probs:
            for j, relevance_grid in enumerate(probs.cpu().numpy()):
                absolute_idx = batch_indices[j]
                frame_entry = {
                    "classification_size": relevance_grid.shape,
                    "classification_hex": relevance_grid.flatten().tobytes().hex(),
                    "idx": absolute_idx,
                }
                entries.append(frame_entry)
        retrieve_runtime = time.time_ns() / 1e6 - retrieve_start
        fr.write(json.dumps(format_time(read=read_runtime, retrieve=retrieve_runtime)) + '\n')

        for frame_entry in entries:
            f.write(json.dumps(frame_entry) + '\n')


def classify_all(
    dataset: str,
    videoset: str,
    videos: list[str],
    classifier: str,
    tile_size: int,
    sample_rate: int,
    gpu_id: int,
    command_queue: mp.Queue,
):
    device = f'cuda:{gpu_id}'

    model = load_model(dataset, tile_size, classifier, device)
    model = model.to(device)

    first_cap = cv2.VideoCapture(store.dataset(dataset, videoset, videos[0]))
    assert first_cap.isOpened(), f"Could not open video {videos[0]}"
    width = int(first_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    compilation_path = cache.index(
        dataset, 'training', 'results', f'{classifier}_{tile_size}', 'model_compilation.jsonl',
    )
    if os.path.exists(compilation_path):
        with open(compilation_path, 'r') as f:
            benchmark_results = [json.loads(line) for line in f]
        model, method_name = select_baseline_model_optimization(
            model,
            benchmark_results,
            device,
            tile_size,
            (width // tile_size) * (height // tile_size),
        )
    else:
        method_name = 'baseline'

    normalize_mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float16).view(1, 3, 1, 1)
    normalize_std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float16).view(1, 3, 1, 1)

    always_relevant_path = cache.index(dataset, 'never-relevant', f'{tile_size}_all.npy')
    assert os.path.exists(always_relevant_path), f"Always relevant bitmap not found for {dataset} {tile_size}"
    always_relevant_bitmap = np.load(always_relevant_path)
    always_relevant_mask = torch.from_numpy(always_relevant_bitmap.flatten()).to(device).to(torch.uint8)

    grid_width = width // tile_size
    grid_height = height // tile_size

    first_frames: list[np.ndarray] = []
    while first_cap.isOpened():
        ret, frame = first_cap.read()
        if not ret:
            break
        first_frames.append(np.ascontiguousarray(frame[:, :, ::-1]))
    first_cap.release()
    sampled_indices = [idx for idx in range(len(first_frames)) if idx % sample_rate == 0]
    last_idx = len(first_frames) - 1
    if last_idx >= 0 and (not sampled_indices or sampled_indices[-1] != last_idx):
        sampled_indices.append(last_idx)
    warmup_start = max(0, len(sampled_indices) - BATCH_SIZE)
    warmup_batch = [first_frames[idx] for idx in sampled_indices[warmup_start:]]

    command_queue.put((device, {'completed': 0, 'total': 16, 'description': 'Warm up (baseline)'}))
    with torch.no_grad():
        for warmup_i in range(16):
            classify_batch(
                grid_width,
                grid_height,
                warmup_batch,
                model,
                tile_size,
                device,
                normalize_mean,
                normalize_std,
                always_relevant_mask,
                method_name,
            )
            command_queue.put((device, {'completed': warmup_i + 1, 'total': 16, 'description': 'Warm up (baseline)'}))
    torch.cuda.synchronize()

    description = f"{dataset} {classifier}Baseline_{tile_size} sr{sample_rate} [{method_name}]"
    command_queue.put((device, {'completed': 0, 'total': len(videos), 'description': description}))
    for i, video in enumerate(videos):
        classify(
            dataset,
            videoset,
            video,
            classifier,
            tile_size,
            sample_rate,
            device,
            model,
            normalize_mean,
            normalize_std,
            always_relevant_mask,
            method_name,
        )
        command_queue.put((device, {'completed': i + 1, 'total': len(videos), 'description': description}))


def main():
    args = parse_args()

    selected_videosets = []
    if args.test:
        selected_videosets.append('test')
    if args.valid:
        selected_videosets.append('valid')

    if not selected_videosets:
        selected_videosets = ['valid']

    mp.set_start_method('spawn', force=True)

    funcs: list[Callable[[int, mp.Queue], None]] = []
    for dataset, videoset in itertools.product(DATASETS, selected_videosets):
        videoset_dir = store.dataset(dataset, videoset)
        assert os.path.exists(videoset_dir), f"Videoset directory {videoset_dir} does not exist"

        videos = [f for f in os.listdir(videoset_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        for classifier, tile_size, sample_rate in itertools.product(CLASSIFIERS, TILE_SIZES, SAMPLE_RATES):
            if classifier not in BASELINE_CLASSIFIER_SOURCES:
                continue
            func = partial(classify_all, dataset, videoset, sorted(videos), classifier, tile_size, sample_rate)
            funcs.append(func)

    if not funcs:
        print("No baseline classify tasks (check CLASSIFIERS vs BASELINE_CLASSIFIER_SOURCES).")
        return

    num_gpus = torch.cuda.device_count()
    assert num_gpus > 0, "No GPUs available"
    ProgressBar(num_workers=num_gpus, num_tasks=len(funcs)).run_all(funcs)


if __name__ == '__main__':
    main()
