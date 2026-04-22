#!/usr/local/bin/python

import argparse
import json
import os

import cv2
import numpy as np

from polyis.io import cache, store
from polyis.utilities import get_config
from evaluation.manifests import list_split_videos


CONFIG = get_config()
DATASETS = CONFIG['EXEC']['DATASETS']
TRACKERS = CONFIG['EXEC']['TRACKERS']
TILE_SIZES = CONFIG['EXEC']['TILE_SIZES']

# Must match the thresholds defined in scripts/p016_tune_track_rate.py.
ACCURACY_THRESHOLDS = [30, 40, 50, 60, 70, 80, 90, 95, 100]

# Per-dataset tile-coordinate crop bounds as ((y1, x1), (y2, x2)), or None for no crop.
# Coordinates are inclusive tile indices: the crop covers tiles [y1, y2) x [x1, x2).
DATASET_CROP: dict[str, tuple[tuple[int, int], tuple[int, int]] | None] = {
    'caldot1-y05': None,
    'caldot2-y05': ((3, 0), (7, 6)),
    'jnc0': None,
    'jnc2': None,
    'jnc6': None,
    'jnc7': ((6, 5), (12, 14)),
    'ams-y05': None,
}

# Colormap range bounds for the overlay (rate 1 = no overlay).
RATE_LOW = 2
RATE_HIGH = 16

# Precompute a 256-entry reversed viridis BGR lookup table so that
# low rates map to yellow and high rates map to dark purple.
import matplotlib
_viridis = matplotlib.colormaps.get_cmap('viridis_r').resampled(256)
VIRIDIS_BGR_LUT = np.array([
    [int(c[2] * 255), int(c[1] * 255), int(c[0] * 255)]
    for c in (_viridis(i) for i in range(256))
], dtype=np.uint8)


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


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize per-tile max sampling rates from track rate analysis')
    parser.add_argument('--frame_idx', type=int, default=0,
                        help='Frame index to extract from the training video (default: 0)')
    parser.add_argument('--alpha', type=float, default=0.2,
                        help='Opacity of the color overlay (0.0=transparent, 1.0=opaque, default: 0.2)')
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


def render_rate_overlay(
    frame: np.ndarray,
    max_rate_slice: np.ndarray,
    tile_size: int,
    threshold_pct: int,
    alpha: float,
) -> np.ndarray:
    """Overlay colored tiles and rate numbers on a video frame for one threshold."""
    # Copy the frame to avoid mutating the original.
    result = frame.copy()

    grid_h, grid_w = max_rate_slice.shape

    # Tint each tile using viridis, with rate clamped to [RATE_LOW, RATE_HIGH].
    # Rate 1 gets no overlay; rates >= RATE_LOW are blended at the given alpha.
    for r in range(grid_h):
        for c in range(grid_w):
            rate = int(max_rate_slice[r, c])
            if rate <= 1:
                # No overlay for rate=1 (every frame processed).
                continue
            # Clamp rate into [RATE_LOW, RATE_HIGH] and map to 0–255 LUT index.
            clamped = max(RATE_LOW, min(rate, RATE_HIGH))
            lut_idx = int(255 * (clamped - RATE_LOW) / (RATE_HIGH - RATE_LOW))
            color = VIRIDIS_BGR_LUT[lut_idx].astype(np.float32)
            # Inset the overlay by 2px on each side so adjacent borders don't overlap.
            y1 = r * tile_size + 1
            x1 = c * tile_size + 1
            y2 = y1 + tile_size - 2
            x2 = x1 + tile_size - 2
            # Blend the inset tile region toward the viridis color.
            roi = result[y1:y2, x1:x2].astype(np.float32)
            roi = roi * (1.0 - alpha) + color * alpha
            result[y1:y2, x1:x2] = np.clip(roi, 0, 255).astype(np.uint8)

            # Draw a border in the same color at moderately higher opacity.
            border_alpha = 0.6
            border_color = VIRIDIS_BGR_LUT[lut_idx].astype(np.float32)
            # Draw on a temporary copy to extract border pixels only.
            border_mask = np.zeros(result.shape[:2], dtype=np.uint8)
            cv2.rectangle(border_mask, (x1, y1), (x2 - 1, y2 - 1), 255, 2)
            bm = border_mask == 255
            result[bm] = np.clip(
                result[bm].astype(np.float32) * (1.0 - border_alpha)
                + border_color * border_alpha, 0, 255).astype(np.uint8)

    # Compute adaptive font scale so text fits smaller tiles.
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, tile_size / 60.0)
    thickness_outline = max(3, int(font_scale * 5))
    thickness_fill = max(1, int(font_scale * 2))

    # Draw the rate number centered in each tile with a black outline for readability.
    for r in range(grid_h):
        for c in range(grid_w):
            rate = int(max_rate_slice[r, c])
            text = str(rate)

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


def _render_mistrack_base(
    frame: np.ndarray,
    mistrack_pct: np.ndarray,
    zero_count_mask: np.ndarray,
    tile_size: int,
    alpha: float,
    tile_texts: np.ndarray,
    font_scale_factor: float = 1.0,
) -> np.ndarray:
    """Shared overlay logic for mistrack visualizations.

    Applies viridis color overlay based on mistrack_pct, draws grid lines,
    and renders tile_texts centered in each tile ('x' for zero-count tiles).
    Each entry in tile_texts is a tuple of strings (one per line).
    """
    # Copy the frame to avoid mutating the original.
    result = frame.copy()

    grid_h, grid_w = mistrack_pct.shape

    # Compute the dynamic colormap range from tiles that have data and nonzero mistrack.
    valid_values = mistrack_pct[~zero_count_mask & (mistrack_pct > 0)]
    if valid_values.size > 0:
        rate_low = int(valid_values.min())
        rate_high = int(valid_values.max())
    else:
        # All valid tiles have 0% mistrack; no color overlay needed.
        rate_low = 0
        rate_high = 0

    # Tint each tile using viridis, with mistrack_pct clamped to [rate_low, rate_high].
    # Tiles with 0% mistrack or zero-count get no overlay.
    for r in range(grid_h):
        for c in range(grid_w):
            pct = int(mistrack_pct[r, c])
            if pct <= 0 or zero_count_mask[r, c]:
                # No overlay for 0% mistrack or zero-count tiles.
                continue
            if rate_low == rate_high:
                # All nonzero values are the same; map to middle of LUT.
                lut_idx = 128
            else:
                # Clamp pct into [rate_low, rate_high] and map to 0–255 LUT index.
                clamped = max(rate_low, min(pct, rate_high))
                lut_idx = int(255 * (clamped - rate_low) / (rate_high - rate_low))
            color = VIRIDIS_BGR_LUT[lut_idx].astype(np.float32)
            # Inset the overlay by 2px on each side so adjacent borders don't overlap.
            y1 = r * tile_size + 1
            x1 = c * tile_size + 1
            y2 = y1 + tile_size - 2
            x2 = x1 + tile_size - 2
            # Blend the inset tile region toward the viridis color.
            roi = result[y1:y2, x1:x2].astype(np.float32)
            roi = roi * (1.0 - alpha) + color * alpha
            result[y1:y2, x1:x2] = np.clip(roi, 0, 255).astype(np.uint8)

            # Draw a border in the same color at moderately higher opacity.
            border_alpha = 0.6
            border_color = VIRIDIS_BGR_LUT[lut_idx].astype(np.float32)
            # Draw on a temporary mask to extract border pixels only.
            border_mask = np.zeros(result.shape[:2], dtype=np.uint8)
            cv2.rectangle(border_mask, (x1, y1), (x2 - 1, y2 - 1), 255, 2)
            bm = border_mask == 255
            result[bm] = np.clip(
                result[bm].astype(np.float32) * (1.0 - border_alpha)
                + border_color * border_alpha, 0, 255).astype(np.uint8)

    # Compute adaptive font scale so text fits smaller tiles.
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, tile_size / 60.0) * font_scale_factor
    thickness_outline = max(3, int(font_scale * 5))
    thickness_fill = max(1, int(font_scale * 2))

    # Draw the tile text centered in each tile, supporting multiple lines.
    for r in range(grid_h):
        for c in range(grid_w):
            lines = tile_texts[r, c]
            num_lines = len(lines)

            # Measure each line to compute total text block height.
            line_sizes = []
            for line in lines:
                (tw, th), baseline = cv2.getTextSize(
                    line, font, font_scale, thickness_fill)
                line_sizes.append((tw, th))

            # Compute the vertical spacing between line baselines.
            max_th = max(th for _, th in line_sizes)
            line_gap = int(max_th * 0.4)
            total_text_h = sum(th for _, th in line_sizes) + line_gap * (num_lines - 1)

            # Starting y for the first line baseline (vertically centered in tile).
            y1 = r * tile_size
            x1 = c * tile_size
            base_y = y1 + (tile_size - total_text_h) // 2 + line_sizes[0][1]

            # Draw each line horizontally centered.
            for i, line in enumerate(lines):
                tw, th = line_sizes[i]
                tx = x1 + (tile_size - tw) // 2
                ty = base_y + i * (max_th + line_gap)

                # Draw white stroke for contrast.
                cv2.putText(result, line, (tx, ty), font, font_scale,
                            (255, 255, 255), thickness_outline, cv2.LINE_AA)
                # Draw black fill on top.
                cv2.putText(result, line, (tx, ty), font, font_scale,
                            (0, 0, 0), thickness_fill, cv2.LINE_AA)

    return result


def render_mistrack_overlay(
    frame: np.ndarray,
    mistrack_pct: np.ndarray,
    zero_count_mask: np.ndarray,
    tile_size: int,
    alpha: float,
) -> np.ndarray:
    """Overlay colored tiles and mistrack percentages on a video frame for one sample rate."""
    # Build text labels: ('x',) for zero-count tiles, (percentage,) otherwise.
    grid_h, grid_w = mistrack_pct.shape
    tile_texts = np.empty((grid_h, grid_w), dtype=object)
    for r in range(grid_h):
        for c in range(grid_w):
            if zero_count_mask[r, c]:
                tile_texts[r, c] = ('x',)
            else:
                tile_texts[r, c] = (str(int(mistrack_pct[r, c])),)

    return _render_mistrack_base(
        frame, mistrack_pct, zero_count_mask, tile_size, alpha, tile_texts)


def render_mistrack_count_overlay(
    frame: np.ndarray,
    mistrack_pct: np.ndarray,
    zero_count_mask: np.ndarray,
    incorrect: np.ndarray,
    total: np.ndarray,
    tile_size: int,
    alpha: float,
) -> np.ndarray:
    """Overlay colored tiles and incorrect/total counts on a video frame for one sample rate."""
    # Build text labels: ('x',) for zero-count, (incorrect, total) on two lines otherwise.
    grid_h, grid_w = mistrack_pct.shape
    tile_texts = np.empty((grid_h, grid_w), dtype=object)
    for r in range(grid_h):
        for c in range(grid_w):
            if zero_count_mask[r, c]:
                tile_texts[r, c] = ('x',)
            else:
                tile_texts[r, c] = (str(int(incorrect[r, c])),
                                    str(int(total[r, c])))

    return _render_mistrack_base(
        frame, mistrack_pct, zero_count_mask, tile_size, alpha, tile_texts,
        font_scale_factor=0.6)


def main(args: argparse.Namespace) -> None:
    """Iterate over all config combinations and render max-rate visualizations."""
    for dataset in DATASETS:
        # Look up the optional per-dataset tile crop bounds.
        crop = DATASET_CROP.get(dataset)

        for tracker_name in TRACKERS:
            for tile_size in TILE_SIZES:
                # Resolve the track_rates data directory for this combination.
                data_dir = cache.index(dataset, 'track_rates', f'{tracker_name}_{tile_size}')
                table_path = data_dir / 'max_rate_table.npy'

                # Skip combinations where p016 has not been run yet.
                if not table_path.exists():
                    print(f"  Skipping {dataset} {tracker_name} tile={tile_size}: "
                          f"max_rate_table.npy not found at {table_path}")
                    continue

                # Load the precomputed max-rate lookup table.
                max_rate_table = np.load(str(table_path))
                grid_h, grid_w, num_thresholds = max_rate_table.shape
                assert num_thresholds == len(ACCURACY_THRESHOLDS), (
                    f"Threshold count mismatch: table has {num_thresholds}, "
                    f"expected {len(ACCURACY_THRESHOLDS)}"
                )

                # Get the first training video for this dataset.
                videos = list_split_videos(dataset, 'train')
                assert len(videos) > 0, f"No training videos found for {dataset}"
                video_file = videos[0]
                video_path = str(store.dataset(dataset, 'train', video_file))

                # Load the requested frame from the video.
                frame = load_frame(video_path, args.frame_idx)

                # Compute tile-aligned dimensions matching p016's logic.
                height, width = frame.shape[:2]
                target_h = (height // tile_size) * tile_size
                target_w = (width // tile_size) * tile_size

                # Resize the frame to tile-aligned dimensions.
                frame = cv2.resize(frame, (target_w, target_h))

                # Verify grid dimensions match the loaded table.
                expected_grid_h = target_h // tile_size
                expected_grid_w = target_w // tile_size
                assert grid_h == expected_grid_h and grid_w == expected_grid_w, (
                    f"Grid mismatch: table ({grid_h}, {grid_w}) vs "
                    f"frame ({expected_grid_h}, {expected_grid_w})"
                )

                # Create the evaluation output directory for this combination.
                vis_dir = cache.eval(dataset, 'track-rate-vis', f'{tracker_name}_{tile_size}')
                os.makedirs(str(vis_dir), exist_ok=True)

                # Render and save one PNG per accuracy threshold.
                for t_idx, threshold_pct in enumerate(ACCURACY_THRESHOLDS):
                    # Extract the max-rate slice for this threshold.
                    max_rate_slice = max_rate_table[:, :, t_idx]

                    # Render the overlay on the frame.
                    result = render_rate_overlay(
                        frame, max_rate_slice, tile_size, threshold_pct, args.alpha)

                    # Crop and save the visualization as PNG.
                    result = crop_to_tiles(result, tile_size, crop)
                    out_path = str(vis_dir / f'{dataset}_max_rate_{threshold_pct:03d}.png')
                    cv2.imwrite(out_path, result)

                print(f"  Saved {len(ACCURACY_THRESHOLDS)} max-rate images to {vis_dir}")

                # Load per-tile mistrack counts if available.
                counts_path = data_dir / 'counts.npy'
                rates_path = data_dir / 'sample_rates.json'
                if not counts_path.exists() or not rates_path.exists():
                    print(f"  Skipping mistrack visualization for {dataset} "
                          f"{tracker_name} tile={tile_size}: "
                          f"counts.npy or sample_rates.json not found")
                    continue

                # Read the sample rates from the saved JSON.
                with open(str(rates_path), 'r') as f:
                    sample_rates = json.load(f)

                # Load the per-rate per-tile correct/incorrect counts.
                counts = np.load(str(counts_path))
                num_rates = counts.shape[0]
                assert num_rates == len(sample_rates), (
                    f"Rate count mismatch: counts has {num_rates}, "
                    f"sample_rates.json has {len(sample_rates)}"
                )
                assert counts.shape[1] == grid_h and counts.shape[2] == grid_w, (
                    f"Grid mismatch: counts ({counts.shape[1]}, {counts.shape[2]}) vs "
                    f"table ({grid_h}, {grid_w})"
                )

                # Render and save one PNG per sample rate.
                for rate_idx, rate in enumerate(sample_rates):
                    # Extract correct and incorrect counts for this rate.
                    correct = counts[rate_idx, :, :, 0]
                    incorrect = counts[rate_idx, :, :, 1]
                    total = correct + incorrect

                    # Identify tiles with no detections.
                    zero_count_mask = (total == 0)

                    # Compute integer mistrack percentage; zero-count tiles stay at 0.
                    mistrack_pct = np.zeros((grid_h, grid_w), dtype=np.int32)
                    valid = ~zero_count_mask
                    mistrack_pct[valid] = np.round(
                        100.0 * incorrect[valid] / total[valid]).astype(np.int32)

                    # Render the mistrack percentage overlay on the frame.
                    result = render_mistrack_overlay(
                        frame, mistrack_pct, zero_count_mask, tile_size, args.alpha)

                    # Crop and save the visualization as PNG.
                    result = crop_to_tiles(result, tile_size, crop)
                    out_path = str(vis_dir / f'{dataset}_mistrack_rate_{rate:03d}.png')
                    cv2.imwrite(out_path, result)

                    # Render the mistrack count overlay on the frame.
                    result = render_mistrack_count_overlay(
                        frame, mistrack_pct, zero_count_mask,
                        incorrect, total, tile_size, args.alpha)

                    # Crop and save the count visualization as PNG.
                    result = crop_to_tiles(result, tile_size, crop)
                    out_path = str(vis_dir / f'{dataset}_mistrack_count_{rate:03d}.png')
                    cv2.imwrite(out_path, result)

                print(f"  Saved {len(sample_rates)} mistrack-rate and "
                      f"{len(sample_rates)} mistrack-count images to {vis_dir}")

    print("All track rate visualizations completed!")


if __name__ == '__main__':
    main(parse_args())
