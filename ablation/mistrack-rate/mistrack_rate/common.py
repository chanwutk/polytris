from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from polyis.io import cache, store
from polyis.utilities import (
    get_video_frame_count,
    get_video_resolution,
    load_tracking_results,
)


GRID_ROWS = 3
GRID_COLS = 3
RATE_CHOICES = (1, 2, 4)
HEURISTIC_THRESHOLDS = (30, 40, 50, 60, 70, 80, 90, 95, 100)


@dataclass
class PreparedVideoData:
    video: str
    width: int
    height: int
    total_frames: int
    source_tracks: dict[int, list[list[float]]]
    source_detections: dict[int, np.ndarray]
    gt_tracks: dict[int, list[list[float]]]
    relevance_bitmaps: np.ndarray
    tile_to_polyomino_id: np.ndarray
    polyomino_lengths: list[list[int]]


def list_split_videos(dataset: str, videoset: str) -> list[str]:
    videoset_dir = store.dataset(dataset, videoset)
    assert videoset_dir.exists(), f'Videoset directory does not exist: {videoset_dir}'

    return sorted(
        video.name
        for video in videoset_dir.iterdir()
        if video.suffix in {'.mp4', '.avi', '.mov', '.mkv'}
    )


def subsample_videos(videos: list[str], divisor: int, remainder: int = 0) -> list[str]:
    if divisor <= 0:
        raise ValueError(f'Expected positive divisor, got {divisor}')

    if remainder < 0 or remainder >= divisor:
        raise ValueError(f'Expected remainder in [0, {divisor}), got {remainder}')

    return [
        video
        for index, video in enumerate(videos)
        if index % divisor == remainder
    ]


def ablation_root(dataset: str, tracker_name: str, *parts: str) -> Path:
    path = cache.root(dataset, 'ablation', 'mistrack-rate', f'{GRID_ROWS}x{GRID_COLS}', tracker_name)
    for part in parts:
        path /= part
    return path


def heuristic_dir(dataset: str, tracker_name: str) -> Path:
    return ablation_root(dataset, tracker_name, 'train')


def evaluation_dir(dataset: str, tracker_name: str) -> Path:
    return ablation_root(dataset, tracker_name, 'test')


def plots_dir(dataset: str, tracker_name: str) -> Path:
    return evaluation_dir(dataset, tracker_name) / 'plots'


def results_csv_path(dataset: str, tracker_name: str) -> Path:
    return evaluation_dir(dataset, tracker_name) / 'results.csv'


def heuristic_counts_path(dataset: str, tracker_name: str) -> Path:
    return heuristic_dir(dataset, tracker_name) / 'counts.npy'


def heuristic_accuracy_path(dataset: str, tracker_name: str) -> Path:
    return heuristic_dir(dataset, tracker_name) / 'accuracy.npy'


def heuristic_rate_table_path(dataset: str, tracker_name: str) -> Path:
    return heuristic_dir(dataset, tracker_name) / 'max_rate_table.npy'


def heuristic_metadata_path(dataset: str, tracker_name: str) -> Path:
    return heuristic_dir(dataset, tracker_name) / 'metadata.json'


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def encode_rate_grid(rate_grid: np.ndarray) -> str:
    flat_values = [str(int(value)) for value in rate_grid.flatten()]
    return '-'.join(flat_values)


def decode_rate_grid(encoded_grid: str) -> np.ndarray:
    flat_values = [int(value) for value in encoded_grid.split('-') if value]
    assert len(flat_values) == GRID_ROWS * GRID_COLS, (
        f'Expected {GRID_ROWS * GRID_COLS} values in encoded grid, got {len(flat_values)}'
    )
    return np.asarray(flat_values, dtype=np.int32).reshape((GRID_ROWS, GRID_COLS))


def rate_grid_json(rate_grid: np.ndarray) -> str:
    return json.dumps(rate_grid.astype(int).tolist())


def iter_rate_grids() -> Iterator[np.ndarray]:
    for flat_values in itertools.product(RATE_CHOICES, repeat=GRID_ROWS * GRID_COLS):
        yield np.asarray(flat_values, dtype=np.int32).reshape((GRID_ROWS, GRID_COLS))


def make_variant_id(method: str, rate_grid: np.ndarray, heuristic_threshold: int | None = None) -> str:
    encoded_grid = encode_rate_grid(rate_grid)
    if heuristic_threshold is None:
        return f'{method}_{encoded_grid}'
    return f'{method}_t{heuristic_threshold:03d}_{encoded_grid}'


def load_tracking_file(path: Path) -> dict[int, list[list[float]]]:
    frame_tracks: dict[int, list[list[float]]] = {}
    if not path.exists():
        raise FileNotFoundError(f'Tracking file not found: {path}')

    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            frame_tracks[int(row['frame_idx'])] = row['tracks']

    return frame_tracks


def save_tracking_results_with_total_frames(
    frame_tracks: dict[int, list[list[float]]],
    total_frames: int,
    output_path: Path,
) -> None:
    ensure_dir(output_path.parent)

    with open(output_path, 'w', encoding='utf-8') as file:
        for frame_idx in range(total_frames):
            row = {
                'frame_idx': frame_idx,
                'tracks': frame_tracks.get(frame_idx, []),
            }
            file.write(json.dumps(row) + '\n')


def cell_width(width: int, num_cols: int = GRID_COLS) -> float:
    return float(width) / float(num_cols)


def cell_height(height: int, num_rows: int = GRID_ROWS) -> float:
    return float(height) / float(num_rows)


def _clip_box_to_frame(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    clipped_x1 = max(0.0, min(float(width), float(x1)))
    clipped_y1 = max(0.0, min(float(height), float(y1)))
    clipped_x2 = max(0.0, min(float(width), float(x2)))
    clipped_y2 = max(0.0, min(float(height), float(y2)))

    if clipped_x2 <= clipped_x1 or clipped_y2 <= clipped_y1:
        return None

    return clipped_x1, clipped_y1, clipped_x2, clipped_y2


def get_overlapping_rect_cells(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
    num_rows: int = GRID_ROWS,
    num_cols: int = GRID_COLS,
) -> tuple[int, int, int, int] | None:
    clipped_box = _clip_box_to_frame(x1, y1, x2, y2, width, height)
    if clipped_box is None:
        return None

    clipped_x1, clipped_y1, clipped_x2, clipped_y2 = clipped_box
    epsilon = 1e-9
    col_size = cell_width(width, num_cols)
    row_size = cell_height(height, num_rows)

    col_start = int(clipped_x1 // col_size)
    row_start = int(clipped_y1 // row_size)
    col_end = int(min(float(width) - epsilon, clipped_x2 - epsilon) // col_size)
    row_end = int(min(float(height) - epsilon, clipped_y2 - epsilon) // row_size)

    col_start = max(0, min(num_cols - 1, col_start))
    row_start = max(0, min(num_rows - 1, row_start))
    col_end = max(0, min(num_cols - 1, col_end))
    row_end = max(0, min(num_rows - 1, row_end))

    return row_start, row_end, col_start, col_end


def center_cell_for_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
    num_rows: int = GRID_ROWS,
    num_cols: int = GRID_COLS,
) -> tuple[int, int] | None:
    clipped_box = _clip_box_to_frame(x1, y1, x2, y2, width, height)
    if clipped_box is None:
        return None

    clipped_x1, clipped_y1, clipped_x2, clipped_y2 = clipped_box
    center_x = (clipped_x1 + clipped_x2) / 2.0
    center_y = (clipped_y1 + clipped_y2) / 2.0
    epsilon = 1e-9
    col_size = cell_width(width, num_cols)
    row_size = cell_height(height, num_rows)

    col_idx = int(min(float(width) - epsilon, center_x) // col_size)
    row_idx = int(min(float(height) - epsilon, center_y) // row_size)

    col_idx = max(0, min(num_cols - 1, col_idx))
    row_idx = max(0, min(num_rows - 1, row_idx))

    return row_idx, col_idx


def mark_boxes_on_grid(
    boxes: Iterable[list[float]],
    width: int,
    height: int,
    num_rows: int = GRID_ROWS,
    num_cols: int = GRID_COLS,
    bbox_slice: slice = slice(-4, None),
) -> np.ndarray:
    bitmap = np.zeros((num_rows, num_cols), dtype=np.uint8)

    for box in boxes:
        x1, y1, x2, y2 = [float(value) for value in box[bbox_slice]]
        overlap = get_overlapping_rect_cells(x1, y1, x2, y2, width, height, num_rows, num_cols)
        if overlap is None:
            continue

        row_start, row_end, col_start, col_end = overlap
        bitmap[row_start:row_end + 1, col_start:col_end + 1] = 1

    return bitmap


def track_center_is_active(
    track: list[float],
    active_bitmap: np.ndarray,
    width: int,
    height: int,
    bbox_slice: slice = slice(1, 5),
) -> bool:
    x1, y1, x2, y2 = [float(value) for value in track[bbox_slice]]
    center_cell = center_cell_for_box(x1, y1, x2, y2, width, height, active_bitmap.shape[0], active_bitmap.shape[1])
    if center_cell is None:
        return False

    row_idx, col_idx = center_cell
    return bool(active_bitmap[row_idx, col_idx] > 0)


def tracks_to_detection_arrays(
    frame_tracks: dict[int, list[list[float]]],
) -> dict[int, np.ndarray]:
    detections: dict[int, np.ndarray] = {}

    for frame_idx, tracks in frame_tracks.items():
        if not tracks:
            detections[frame_idx] = np.empty((0, 5), dtype=np.float64)
            continue

        rows: list[list[float]] = []
        for track in tracks:
            _, x1, y1, x2, y2 = track[:5]
            rows.append([float(x1), float(y1), float(x2), float(y2), 1.0])
        detections[frame_idx] = np.asarray(rows, dtype=np.float64)

    return detections


def build_relevance_bitmaps(
    frame_tracks: dict[int, list[list[float]]],
    total_frames: int,
    width: int,
    height: int,
    num_rows: int = GRID_ROWS,
    num_cols: int = GRID_COLS,
) -> np.ndarray:
    bitmaps = np.zeros((total_frames, num_rows, num_cols), dtype=np.uint8)

    for frame_idx in range(total_frames):
        tracks = frame_tracks.get(frame_idx, [])
        bitmaps[frame_idx] = mark_boxes_on_grid(
            tracks,
            width=width,
            height=height,
            num_rows=num_rows,
            num_cols=num_cols,
            bbox_slice=slice(1, 5),
        )

    return bitmaps


def filter_tracks_to_detection_arrays(
    frame_tracks: dict[int, list[list[float]]],
    active_bitmaps: np.ndarray,
    width: int,
    height: int,
) -> dict[int, np.ndarray]:
    detections: dict[int, np.ndarray] = {}

    for frame_idx, tracks in frame_tracks.items():
        if frame_idx >= len(active_bitmaps):
            break

        active_bitmap = active_bitmaps[frame_idx]
        rows: list[list[float]] = []
        for track in tracks:
            if not track_center_is_active(track, active_bitmap, width, height):
                continue

            _, x1, y1, x2, y2 = track[:5]
            rows.append([float(x1), float(y1), float(x2), float(y2), 1.0])

        detections[frame_idx] = (
            np.asarray(rows, dtype=np.float64)
            if rows
            else np.empty((0, 5), dtype=np.float64)
        )

    return detections


def count_active_cells(bitmaps: np.ndarray) -> int:
    return int(bitmaps.astype(np.int64).sum())


def retention_rate(original_bitmaps: np.ndarray, pruned_bitmaps: np.ndarray) -> float:
    original_count = count_active_cells(original_bitmaps)
    pruned_count = count_active_cells(pruned_bitmaps)
    if original_count == 0:
        return 0.0
    return float(pruned_count) / float(original_count)


def load_naive_tracking_source(dataset: str, video: str) -> dict[int, list[list[float]]]:
    return load_tracking_results(dataset, video)


def frame_metadata(dataset: str, video: str) -> tuple[int, int, int]:
    width, height = get_video_resolution(dataset, video)
    total_frames = get_video_frame_count(dataset, video)
    return width, height, total_frames
