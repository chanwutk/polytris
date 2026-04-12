#!/usr/local/bin/python

import argparse
import os

import cv2
import numpy as np

from polyis.io import cache, store
from polyis.utilities import get_config, load_detection_results, mark_detections
from evaluation.manifests import list_split_videos


CONFIG = get_config()
DATASETS = CONFIG['EXEC']['DATASETS']
TILE_SIZES = CONFIG['EXEC']['TILE_SIZES']

# Per-dataset tile-coordinate crop bounds as ((y1, x1), (y2, x2)), or None for no crop.
# Coordinates are exclusive on the upper bound: crop covers tiles [y1, y2) x [x1, x2).
DATASET_CROP: dict[str, tuple[tuple[int, int], tuple[int, int]] | None] = {
    'caldot1-y05': None,
    'caldot2-y05': None,
    'jnc0': None,
    'jnc2': None,
    'jnc6': None,
    'jnc7': None,
    'ams-y05': None,
}

# Precompute a 256-entry viridis BGR lookup table.
# Low relevance = yellow, high relevance = dark purple.
import matplotlib
_viridis = matplotlib.colormaps.get_cmap('viridis').resampled(256)
VIRIDIS_BGR_LUT = np.array([
    [int(c[2] * 255), int(c[1] * 255), int(c[0] * 255)]
    for c in (_viridis(i) for i in range(255, -1, -1))
], dtype=np.uint8)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame_idx', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=0.4)
    parser.add_argument('--split', type=str, default='test')
    return parser.parse_args()


def load_frame(video_path: str, frame_idx: int) -> np.ndarray:
    """Load a single BGR frame from a video file at the specified index."""
    # Open the video file.
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Could not open video: {video_path}"

    # Verify the requested frame index is within bounds.
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert 0 <= frame_idx < total_frames, (
        f"frame_idx {frame_idx} out of range [0, {total_frames}) for {video_path}"
    )

    # Seek to the requested frame.
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    # Read the frame.
    ret, frame = cap.read()
    cap.release()
    assert ret and frame is not None, f"Failed to read frame {frame_idx} from {video_path}"

    return frame


def crop_to_tiles(
    image: np.ndarray,
    tile_size: int,
    crop: tuple[tuple[int, int], tuple[int, int]] | None,
) -> np.ndarray:
    """Crop a rendered image to the given tile-coordinate bounds.

    Returns the original image unchanged if crop is None.
    """
    if crop is None:
        return image
    (y1, x1), (y2, x2) = crop
    # Convert tile coordinates to pixel coordinates.
    return image[y1 * tile_size:y2 * tile_size, x1 * tile_size:x2 * tile_size]


def compute_relevance_frequency(
    dataset: str,
    tile_size: int,
    split: str,
) -> tuple[np.ndarray, list[float]]:
    """Compute the fraction of frames each tile is relevant across all videos in a split.

    Uses groundtruth tracking data to determine tile relevance.
    Returns:
        relevance_freq: (grid_h, grid_w) float array in [0, 1]
        per_frame_irrelevance: list of per-frame irrelevant-tile fractions
    """
    videos = list_split_videos(dataset, split)
    assert len(videos) > 0, f"No {split} videos found for {dataset}"

    # Get video dimensions from the first video to determine grid size.
    video_path = str(store.dataset(dataset, split, videos[0]))
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Could not open video: {video_path}"
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Compute tile-aligned dimensions.
    target_w = (width // tile_size) * tile_size
    target_h = (height // tile_size) * tile_size
    grid_h = target_h // tile_size
    grid_w = target_w // tile_size
    total_tiles = grid_h * grid_w

    # Accumulate per-tile relevance counts across all videos.
    total_frames = 0
    relevance_sum = np.zeros((grid_h, grid_w), dtype=np.float64)
    per_frame_irrelevance: list[float] = []

    for video in videos:
        # Load groundtruth tracking results for this video.
        tracking_results = load_detection_results(
            dataset, video, tracking=True, groundtruth=True)

        for frame_data in tracking_results:
            tracks = frame_data.get('tracks', [])

            # Scale detection coordinates if needed.
            scaled_tracks = tracks
            if (width, height) != (target_w, target_h):
                scale_x = target_w / width
                scale_y = target_h / height
                scaled_tracks = [
                    [t[0], t[1] * scale_x, t[2] * scale_y, t[3] * scale_x, t[4] * scale_y]
                    for t in tracks
                ]

            # Mark which tiles have detections in this frame.
            bitmap = mark_detections(
                scaled_tracks, target_w, target_h, tile_size)
            relevance_sum += bitmap.astype(np.float64)
            total_frames += 1

            # Track the fraction of irrelevant tiles in this frame.
            n_relevant = int(bitmap.sum())
            per_frame_irrelevance.append(
                100.0 * (total_tiles - n_relevant) / total_tiles)

    assert total_frames > 0, f"No frames found for {dataset} {split}"

    # Compute per-tile fraction of frames with relevance.
    return relevance_sum / total_frames, per_frame_irrelevance


def render_relevance_overlay(
    frame: np.ndarray,
    relevance_freq: np.ndarray,
    tile_size: int,
    alpha: float,
) -> np.ndarray:
    """Overlay per-tile relevance frequency on a video frame using viridis colormap."""
    # Copy the frame to avoid mutating the original.
    result = frame.copy()

    grid_h, grid_w = relevance_freq.shape

    # Compute dynamic colormap range from nonzero relevance values.
    nonzero_vals = relevance_freq[relevance_freq > 0]
    if nonzero_vals.size > 0:
        freq_low = float(nonzero_vals.min())
        freq_high = float(nonzero_vals.max())
    else:
        freq_low = 0.0
        freq_high = 0.0

    # Tint each tile using viridis based on relevance frequency.
    # Tiles with 0% relevance get no overlay.
    for r in range(grid_h):
        for c in range(grid_w):
            freq = relevance_freq[r, c]
            if freq <= 0:
                # No overlay for tiles that are never relevant.
                continue
            if freq_low == freq_high:
                # All nonzero values are the same; map to middle of LUT.
                lut_idx = 128
            else:
                # Map frequency to 0-255 LUT index.
                clamped = max(freq_low, min(freq, freq_high))
                lut_idx = int(255 * (clamped - freq_low) / (freq_high - freq_low))
            color = VIRIDIS_BGR_LUT[lut_idx].astype(np.float32)
            # Inset the overlay by 1px on each side so adjacent borders don't overlap.
            y1 = r * tile_size + 1
            x1 = c * tile_size + 1
            y2 = y1 + tile_size - 2
            x2 = x1 + tile_size - 2
            # Blend the inset tile region toward the viridis color.
            roi = result[y1:y2, x1:x2].astype(np.float32)
            roi = roi * (1.0 - alpha) + color * alpha
            result[y1:y2, x1:x2] = np.clip(roi, 0, 255).astype(np.uint8)

            # Draw a border in the same color at moderate opacity.
            border_alpha = 0.6
            border_color = VIRIDIS_BGR_LUT[lut_idx].astype(np.float32)
            border_mask = np.zeros(result.shape[:2], dtype=np.uint8)
            cv2.rectangle(border_mask, (x1, y1), (x2 - 1, y2 - 1), 255, 2)
            bm = border_mask == 255
            result[bm] = np.clip(
                result[bm].astype(np.float32) * (1.0 - border_alpha)
                + border_color * border_alpha, 0, 255).astype(np.uint8)

    # Compute adaptive font scale so text fits smaller tiles.
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.3, tile_size / 90.0)
    thickness_outline = max(3, int(font_scale * 5))
    thickness_fill = max(1, int(font_scale * 2))

    # Draw the relevance percentage centered in each tile.
    for r in range(grid_h):
        for c in range(grid_w):
            freq = relevance_freq[r, c]
            # Display as integer percentage.
            text = str(int(round(freq * 100)))

            # Compute text size to center it within the tile.
            (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness_fill)
            y1 = r * tile_size
            x1 = c * tile_size
            tx = x1 + (tile_size - tw) // 2
            ty = y1 + (tile_size + th) // 2

            # Draw white stroke for contrast.
            cv2.putText(result, text, (tx, ty), font, font_scale,
                        (255, 255, 255), thickness_outline, cv2.LINE_AA)
            # Draw black fill on top.
            cv2.putText(result, text, (tx, ty), font, font_scale,
                        (0, 0, 0), thickness_fill, cv2.LINE_AA)

    return result


def main(args: argparse.Namespace) -> None:
    """Compute per-tile relevance frequency and render heatmap visualizations."""
    out_dir = os.path.join('paper', 'figures', 'generated')
    os.makedirs(out_dir, exist_ok=True)

    # Collect statistics across all datasets for the .tex output.
    all_per_frame_irrelevance: list[float] = []
    all_relevance_freqs: list[float] = []

    for dataset in DATASETS:
        # Look up the optional per-dataset tile crop bounds.
        crop = DATASET_CROP.get(dataset)

        for tile_size in TILE_SIZES:
            # Compute per-tile relevance frequency across all videos.
            print(f"  Computing relevance frequency for {dataset} tile={tile_size}...")
            relevance_freq, per_frame_irrelevance = compute_relevance_frequency(
                dataset, tile_size, args.split)

            # Accumulate cross-dataset statistics.
            all_per_frame_irrelevance.extend(per_frame_irrelevance)
            all_relevance_freqs.extend(relevance_freq.flatten().tolist())

            # Get the first video to extract a background frame.
            videos = list_split_videos(dataset, args.split)
            video_path = str(store.dataset(dataset, args.split, videos[0]))
            frame = load_frame(video_path, args.frame_idx)

            # Compute tile-aligned dimensions.
            height, width = frame.shape[:2]
            target_h = (height // tile_size) * tile_size
            target_w = (width // tile_size) * tile_size

            # Resize the frame to tile-aligned dimensions.
            frame = cv2.resize(frame, (target_w, target_h))

            # Verify grid dimensions match.
            grid_h, grid_w = relevance_freq.shape
            assert grid_h == target_h // tile_size and grid_w == target_w // tile_size, (
                f"Grid mismatch: freq ({grid_h}, {grid_w}) vs "
                f"frame ({target_h // tile_size}, {target_w // tile_size})"
            )

            # Render the relevance frequency overlay.
            result = render_relevance_overlay(
                frame, relevance_freq, tile_size, args.alpha)

            # Crop and save the visualization as PNG.
            result = crop_to_tiles(result, tile_size, crop)
            out_path = os.path.join(out_dir, f'{dataset}_tile_irrelevance.png')
            cv2.imwrite(out_path, result)
            print(f"  Saved {out_path}")

    # Compute aggregate statistics for the paper.
    # Mean percentage of irrelevant tiles per frame across all datasets.
    mean_irrelevance = float(np.mean(all_per_frame_irrelevance))
    # Mean per-tile relevance frequency (fraction of frames a tile is relevant).
    mean_tile_relevance_pct = float(np.mean(all_relevance_freqs)) * 100

    print(f"\n=== Statistics ===")
    print(f"  Mean irrelevant tiles per frame: {mean_irrelevance:.1f}%")
    print(f"  Mean per-tile relevance frequency: {mean_tile_relevance_pct:.1f}%")

    # Save statistics as LaTeX macros.
    tex_path = os.path.join(out_dir, 'p003_tile_irrelevance_visualize.tex')
    with open(tex_path, 'w') as f:
        f.write(f'% Auto-generated by evaluation/p003_tile_irrelevance_visualize.py\n')
        f.write(f'% Mean per-tile relevance frequency across all datasets/tiles.\n')
        f.write(f'\\newcommand{{\\tileRelevanceMeanPct}}{{\\autogen{{{mean_tile_relevance_pct:.1f}}}}}\n')
        f.write(f'% Mean percentage of irrelevant tiles per frame across all datasets.\n')
        f.write(f'\\newcommand{{\\tileIrrelevanceMeanPct}}{{\\autogen{{{mean_irrelevance:.1f}}}}}\n')
    print(f"  Saved {tex_path}")

    print("\nAll tile irrelevance visualizations completed!")


if __name__ == '__main__':
    main(parse_args())
