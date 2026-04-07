#!/usr/local/bin/python

import argparse
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


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize per-tile max sampling rates from track rate analysis')
    parser.add_argument('--frame_idx', type=int, default=0,
                        help='Frame index to extract from the training video (default: 0)')
    parser.add_argument('--alpha', type=float, default=0.4,
                        help='Opacity of the color overlay (0.0=transparent, 1.0=opaque, default: 0.4)')
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
            y1 = r * tile_size
            x1 = c * tile_size
            y2 = y1 + tile_size
            x2 = x1 + tile_size
            # Blend the tile region toward the viridis color.
            roi = result[y1:y2, x1:x2].astype(np.float32)
            roi = roi * (1.0 - alpha) + color * alpha
            result[y1:y2, x1:x2] = np.clip(roi, 0, 255).astype(np.uint8)

    # Draw gray grid lines at tile boundaries.
    h, w = result.shape[:2]
    for x in range(0, w + 1, tile_size):
        cv2.line(result, (x, 0), (x, h), (128, 128, 128), 1)
    for y in range(0, h + 1, tile_size):
        cv2.line(result, (0, y), (w, y), (128, 128, 128), 1)

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


def main(args: argparse.Namespace) -> None:
    """Iterate over all config combinations and render max-rate visualizations."""
    for dataset in DATASETS:
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

                    # Save the visualization as PNG.
                    out_path = str(vis_dir / f'max_rate_{threshold_pct:03d}.png')
                    cv2.imwrite(out_path, result)

                print(f"  Saved {len(ACCURACY_THRESHOLDS)} images to {vis_dir}")

    print("All track rate visualizations completed!")


if __name__ == '__main__':
    main(parse_args())
