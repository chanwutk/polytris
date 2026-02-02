#!/usr/local/bin/python

import argparse
import os

import cv2
import numpy as np

from polyis.utilities import get_config

config = get_config()
CACHE_DIR = config["DATA"]["CACHE_DIR"]
DATASETS_DIR = config["DATA"]["DATASETS_DIR"]
DATASETS_TO_TEST = config["EXEC"]["DATASETS"]
TILE_SIZES = config["EXEC"]["TILE_SIZES"]

VIDEO_SUBSETS = ("train", "valid", "test")
RED_TINT_ADD = 0xAA


def load_combined_always_relevant(
    cache_dir: str, always_relevant_dir: str, tile_size: int
) -> np.ndarray | None:
    """
    Load the combined always-relevant tiles bitmap for a dataset and tile size.
    Uses precomputed {tile_size}_all.npy if present, otherwise builds it by OR-ing
    per-video bitmaps (same logic as scripts/p013_tune_optimize_training_data.py).
    """
    all_path = os.path.join(always_relevant_dir, f"{tile_size}_all.npy")
    if os.path.exists(all_path):
        return np.load(all_path)
    relevancy_files = [
        f
        for f in os.listdir(always_relevant_dir)
        if f.endswith(".npy") and f.startswith(f"{tile_size}_") and f != f"{tile_size}_all.npy"
    ]
    if not relevancy_files:
        return None
    combined = None
    for relevancy_file in relevancy_files:
        path = os.path.join(always_relevant_dir, relevancy_file)
        if not os.path.exists(path):
            continue
        relevancy = np.load(path)
        if combined is None:
            combined = relevancy.copy()
        else:
            assert combined.shape == relevancy.shape, f"Shape mismatch for {relevancy_file}"
            combined |= relevancy
    return combined


def get_first_video_path(dataset_name: str, always_relevant_dir: str, tile_size: int) -> str | None:
    """
    Resolve the path to the first video used for this dataset/tile_size.
    Video names are inferred from always_relevant filenames: {tile_size}_{video_file}.npy.
    """
    base_dir = os.path.join(DATASETS_DIR, dataset_name, 'test')
    return os.path.join(base_dir, os.listdir(base_dir)[0])
    # prefix = f"{tile_size}_"
    # for f in sorted(os.listdir(always_relevant_dir)):
    #     if not f.endswith(".npy") or not f.startswith(prefix) or f == f"{tile_size}_all.npy":
    #         continue
    #     video_file = f[len(prefix) :]
    #     dataset_dir = os.path.join(DATASETS_DIR, dataset_name)
    #     for subset in VIDEO_SUBSETS:
    #         video_path = os.path.join(dataset_dir, subset, video_file)
    #         if os.path.exists(video_path):
    #             return video_path
    # return None


def read_first_frame(video_path: str) -> np.ndarray | None:
    """Read the first frame of a video as BGR (H, W, 3)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    return frame


def draw_grid_and_never_relevant_overlay(
    frame: np.ndarray,
    tile_size: int,
    combined_always_relevant_tiles: np.ndarray,
) -> np.ndarray:
    """
    Draw a grid with cell size tile_size and tint never-relevant tiles red by adding
    #AA0000 to each pixel RGB (cap at 255). Never-relevant tiles are those where
    combined_always_relevant_tiles == 0 (same as scripts/p013_tune_optimize_training_data.py:76).
    """
    out = frame.copy()
    h, w = frame.shape[:2]
    tiles_y, tiles_x = combined_always_relevant_tiles.shape
    assert tiles_y == h // tile_size and tiles_x == w // tile_size, (
        f"Frame shape ({h}, {w}) does not match bitmap ({tiles_y}, {tiles_x}) for tile_size {tile_size}"
    )
    never_relevant_positions = np.where(combined_always_relevant_tiles == 0)
    for i in range(len(never_relevant_positions[0])):
        ty = int(never_relevant_positions[0][i])
        tx = int(never_relevant_positions[1][i])
        y1 = ty * tile_size
        x1 = tx * tile_size
        y2 = min(y1 + tile_size, h)
        x2 = min(x1 + tile_size, w)
        roi = out[y1:y2, x1:x2, 2]
        out[y1:y2, x1:x2, 2] = np.minimum(
            255, roi.astype(np.int32) + RED_TINT_ADD
        ).astype(np.uint8)
    for x in range(0, w, tile_size):
        cv2.line(out, (x, 0), (x, h), (128, 128, 128), 1)
    for y in range(0, h, tile_size):
        cv2.line(out, (0, y), (w, y), (128, 128, 128), 1)
    return out


def render_and_save_first_frame(
    dataset_name: str,
    tile_size: int,
    output_dir: str,
) -> None:
    """
    For one dataset and tile_size: load combined always-relevant bitmap, get first frame
    of a video, draw grid and never-relevant overlay, and save the image.
    """
    cache_dir = os.path.join(CACHE_DIR, dataset_name)
    always_relevant_dir = os.path.join(cache_dir, "indexing", "always_relevant")
    if not os.path.isdir(always_relevant_dir):
        print(f"Always relevant directory {always_relevant_dir} does not exist")
        return
    combined = load_combined_always_relevant(cache_dir, always_relevant_dir, tile_size)
    if combined is None:
        print(f"Combined always-relevant bitmap is None for {dataset_name} with tile size {tile_size}")
        return
    video_path = get_first_video_path(dataset_name, always_relevant_dir, tile_size)
    if video_path is None:
        print(f"First video path is None for {dataset_name} with tile size {tile_size}")
        return
    frame = read_first_frame(video_path)
    if frame is None:
        print(f"First frame is None for {dataset_name} with tile size {tile_size}")
        return
    h, w = frame.shape[:2]
    expected_tiles_y = h // tile_size
    expected_tiles_x = w // tile_size
    if combined.shape[0] != expected_tiles_y or combined.shape[1] != expected_tiles_x:
        print(f"Combined always-relevant bitmap shape {combined.shape} does not match expected tiles {expected_tiles_y}x{expected_tiles_x} for {dataset_name} with tile size {tile_size}")
        return
    out = draw_grid_and_never_relevant_overlay(frame, tile_size, combined)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"first_frame_tile_size_{tile_size}.png")
    cv2.imwrite(out_path, out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Save the first frame of a video for each dataset with a tile grid and "
            "never-relevant tiles overlaid in tinted red (for classifier optimization visualization)."
        )
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Base directory for output images. Default: CACHE_DIR/<dataset>/indexing/always_relevant/visualization"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for dataset_name in DATASETS_TO_TEST:
        for tile_size in TILE_SIZES:
            print(f"Processing {dataset_name} with tile size {tile_size}")
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, dataset_name)
            else:
                output_dir = os.path.join(
                    CACHE_DIR,
                    dataset_name,
                    "indexing",
                    "always_relevant",
                    "visualization",
                )
            render_and_save_first_frame(dataset_name, tile_size, output_dir)


if __name__ == "__main__":
    main()
