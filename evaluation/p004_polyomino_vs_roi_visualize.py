#!/usr/local/bin/python

import argparse
import os

import cv2
import numpy as np
from scipy import ndimage

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
    'jnc7': ((4, 6), (12, 15)),
    'ams-y05': None,
}

# 4-connectivity structure for connected component labeling.
STRUCTURE_4CONN = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]], dtype=np.int32)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame_idx', type=int, default=500)
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


def get_video_grid_dims(
    dataset: str, split: str, tile_size: int,
) -> tuple[int, int, int, int]:
    """Return (width, height, target_w, target_h) for the first video in the split."""
    videos = list_split_videos(dataset, split)
    assert len(videos) > 0, f"No {split} videos found for {dataset}"
    video_path = str(store.dataset(dataset, split, videos[0]))
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Could not open video: {video_path}"
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    target_w = (width // tile_size) * tile_size
    target_h = (height // tile_size) * tile_size
    return width, height, target_w, target_h


def get_component_bboxes(
    labeled: np.ndarray,
    n_components: int,
) -> list[tuple[int, int, int, int]]:
    """Return tile-coordinate bounding boxes for labeled 4-connected components."""
    # Initialize the output list in component-id order.
    component_bboxes: list[tuple[int, int, int, int]] = []

    # Compute one bounding box per connected component.
    for comp_id in range(1, n_components + 1):
        # Extract the binary mask for the current component.
        comp_mask = (labeled == comp_id)

        # Find which rows and columns contain at least one active tile.
        rows = np.any(comp_mask, axis=1)
        cols = np.any(comp_mask, axis=0)

        # Resolve the inclusive tile bounds of the component.
        row_min, row_max = np.where(rows)[0][[0, -1]]
        col_min, col_max = np.where(cols)[0][[0, -1]]

        # Append the bounding box using inclusive tile coordinates.
        component_bboxes.append((row_min, row_max, col_min, col_max))

    return component_bboxes


def compute_bbox_union_area(
    grid_shape: tuple[int, int],
    component_bboxes: list[tuple[int, int, int, int]],
) -> int:
    """Return the union area of all component bounding boxes in tile units."""
    # Allocate a frame-level mask so overlapping bbox-only tiles count once.
    bbox_union_mask = np.zeros(grid_shape, dtype=bool)

    # Fill each component's bounding box into the shared union mask.
    for row_min, row_max, col_min, col_max in component_bboxes:
        bbox_union_mask[row_min:row_max + 1, col_min:col_max + 1] = True

    # Count the unique tiles covered by any bounding box.
    return int(bbox_union_mask.sum())


def compute_polyomino_vs_roi_stats(
    dataset: str,
    tile_size: int,
    split: str,
) -> dict:
    """Compute per-frame polyomino area vs bounding-box area statistics.

    For each frame, finds connected components of relevant tiles (4-connectivity),
    computes each component's area and its axis-aligned bounding box area, and
    accumulates statistics across all frames and videos.

    Returns a dict with:
        total_polyomino_area: sum of all connected component tile counts
        total_bbox_area: sum of frame-level unions of component bounding boxes
        num_components: total number of connected components
        num_frames: total number of frames processed
    """
    videos = list_split_videos(dataset, split)
    width, height, target_w, target_h = get_video_grid_dims(dataset, split, tile_size)

    total_polyomino_area = 0
    total_bbox_area = 0
    num_components = 0
    num_frames = 0

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
                    [t[0], t[1] * scale_x, t[2] * scale_y,
                     t[3] * scale_x, t[4] * scale_y]
                    for t in tracks
                ]

            # Mark which tiles have detections.
            bitmap = mark_detections(
                scaled_tracks, target_w, target_h, tile_size)
            binary = bitmap.astype(np.int32)

            # Skip frames with no relevant tiles.
            if binary.sum() == 0:
                num_frames += 1
                continue

            # Label connected components using 4-connectivity.
            labeled, n_components = ndimage.label(binary, structure=STRUCTURE_4CONN)

            # Compute all component bounding boxes once for this frame.
            component_bboxes = get_component_bboxes(labeled, n_components)

            # Count unique tiles covered by any component bounding box in this frame.
            total_bbox_area += compute_bbox_union_area(binary.shape, component_bboxes)

            for comp_id in range(1, n_components + 1):
                # Extract the component mask.
                comp_mask = (labeled == comp_id)
                polyomino_area = int(comp_mask.sum())

                total_polyomino_area += polyomino_area
                num_components += 1

            num_frames += 1

    return {
        'total_polyomino_area': total_polyomino_area,
        'total_bbox_area': total_bbox_area,
        'num_components': num_components,
        'num_frames': num_frames,
    }


def render_polyomino_vs_roi_overlay(
    frame: np.ndarray,
    binary_grid: np.ndarray,
    tile_size: int,
    alpha: float,
) -> np.ndarray:
    """Render an overlay showing polyomino tiles vs bounding-box overhead on a frame.

    Polyomino tiles are tinted green. Bounding-box-only tiles (overhead) are tinted red.
    """
    # Copy the frame to avoid mutating the original.
    result = frame.copy()

    grid_h, grid_w = binary_grid.shape

    # Label connected components.
    labeled, n_components = ndimage.label(binary_grid, structure=STRUCTURE_4CONN)

    # Compute all component bounding boxes once so rendering matches stats semantics.
    component_bboxes = get_component_bboxes(labeled, n_components)

    # Build a mask of all bounding-box-only tiles (overhead).
    bbox_only_mask = np.zeros_like(binary_grid, dtype=bool)
    for comp_id, (row_min, row_max, col_min, col_max) in enumerate(component_bboxes, start=1):
        # Find bounding box of this component.
        comp_mask = (labeled == comp_id)

        # Mark tiles inside bounding box but outside the polyomino as overhead.
        for r in range(row_min, row_max + 1):
            for c in range(col_min, col_max + 1):
                if not comp_mask[r, c]:
                    bbox_only_mask[r, c] = True

    # Draw gray grid lines at tile boundaries.
    h, w = result.shape[:2]
    for x in range(0, w + 1, tile_size):
        cv2.line(result, (x, 0), (x, h), (128, 128, 128), 1)
    for y in range(0, h + 1, tile_size):
        cv2.line(result, (0, y), (w, y), (128, 128, 128), 1)

    # Colors in BGR.
    green = np.array([0, 200, 0], dtype=np.float32)
    red = np.array([0, 0, 200], dtype=np.float32)

    # Tint polyomino tiles green.
    for r in range(grid_h):
        for c in range(grid_w):
            if binary_grid[r, c] <= 0:
                continue
            # Inset by 1px.
            y1 = r * tile_size + 1
            x1 = c * tile_size + 1
            y2 = y1 + tile_size - 2
            x2 = x1 + tile_size - 2
            roi = result[y1:y2, x1:x2].astype(np.float32)
            roi = roi * (1.0 - alpha) + green * alpha
            result[y1:y2, x1:x2] = np.clip(roi, 0, 255).astype(np.uint8)

            # Draw green border at moderate opacity.
            border_alpha = 0.6
            border_mask = np.zeros(result.shape[:2], dtype=np.uint8)
            cv2.rectangle(border_mask, (x1, y1), (x2 - 1, y2 - 1), 255, 2)
            bm = border_mask == 255
            result[bm] = np.clip(
                result[bm].astype(np.float32) * (1.0 - border_alpha)
                + green * border_alpha, 0, 255).astype(np.uint8)

    # Tint bounding-box-only tiles red.
    for r in range(grid_h):
        for c in range(grid_w):
            if not bbox_only_mask[r, c]:
                continue
            # Inset by 1px.
            y1 = r * tile_size + 1
            x1 = c * tile_size + 1
            y2 = y1 + tile_size - 2
            x2 = x1 + tile_size - 2
            roi = result[y1:y2, x1:x2].astype(np.float32)
            roi = roi * (1.0 - alpha) + red * alpha
            result[y1:y2, x1:x2] = np.clip(roi, 0, 255).astype(np.uint8)

            # Draw red border at moderate opacity.
            border_alpha = 0.6
            border_mask = np.zeros(result.shape[:2], dtype=np.uint8)
            cv2.rectangle(border_mask, (x1, y1), (x2 - 1, y2 - 1), 255, 2)
            bm = border_mask == 255
            result[bm] = np.clip(
                result[bm].astype(np.float32) * (1.0 - border_alpha)
                + red * border_alpha, 0, 255).astype(np.uint8)

    # Draw dashed bounding box rectangle for each connected component.
    bbox_color = (0, 0, 255)  # Red in BGR.
    dash_len = 10  # Length of each dash in pixels.
    gap_len = 6    # Length of each gap in pixels.
    for row_min, row_max, col_min, col_max in component_bboxes:
        # Convert tile coordinates to pixel coordinates.
        px1 = col_min * tile_size
        py1 = row_min * tile_size
        px2 = (col_max + 1) * tile_size - 1
        py2 = (row_max + 1) * tile_size - 1
        # Draw dashed lines along each edge of the bounding box.
        for edge in [
            ((px1, py1), (px2, py1)),  # Top edge.
            ((px1, py2), (px2, py2)),  # Bottom edge.
            ((px1, py1), (px1, py2)),  # Left edge.
            ((px2, py1), (px2, py2)),  # Right edge.
        ]:
            (ex1, ey1), (ex2, ey2) = edge
            # Compute edge length and direction.
            dx = ex2 - ex1
            dy = ey2 - ey1
            length = max(abs(dx), abs(dy))
            if length == 0:
                continue
            # Step along the edge drawing dashes.
            pos = 0
            while pos < length:
                # Start and end of this dash segment.
                t0 = pos / length
                t1 = min((pos + dash_len) / length, 1.0)
                sx = int(ex1 + dx * t0)
                sy = int(ey1 + dy * t0)
                ex = int(ex1 + dx * t1)
                ey = int(ey1 + dy * t1)
                cv2.line(result, (sx, sy), (ex, ey), bbox_color, 2)
                pos += dash_len + gap_len

    return result


def main(args: argparse.Namespace) -> None:
    """Compute polyomino vs ROI stats and render visual examples."""
    out_dir = os.path.join('paper', 'figures', 'generated')
    os.makedirs(out_dir, exist_ok=True)

    # Print aggregate statistics across all datasets.
    print("=== Polyomino vs Bounding-Box Area Statistics ===\n")
    grand_poly_area = 0
    grand_bbox_area = 0
    grand_components = 0

    for dataset in DATASETS:
        # Look up the optional per-dataset tile crop bounds.
        crop = DATASET_CROP.get(dataset)

        for tile_size in TILE_SIZES:
            # Compute aggregate statistics.
            print(f"  Computing polyomino vs ROI stats for {dataset} tile={tile_size}...")
            stats = compute_polyomino_vs_roi_stats(
                dataset, tile_size, args.split)

            poly_area = stats['total_polyomino_area']
            bbox_area = stats['total_bbox_area']
            n_comp = stats['num_components']
            n_frames = stats['num_frames']

            # Compute the overhead percentage.
            if poly_area > 0:
                overhead_pct = 100.0 * (bbox_area - poly_area) / poly_area
            else:
                overhead_pct = 0.0

            print(f"    {dataset}: {n_comp} components across {n_frames} frames")
            print(f"    Polyomino area: {poly_area}, BBox area: {bbox_area}, "
                  f"Overhead: {overhead_pct:.1f}%")

            grand_poly_area += poly_area
            grand_bbox_area += bbox_area
            grand_components += n_comp

            # Render a visual example using the requested frame of the first video.
            videos = list_split_videos(dataset, args.split)
            video_path = str(store.dataset(dataset, args.split, videos[0]))
            frame = load_frame(video_path, args.frame_idx)

            # Compute tile-aligned dimensions.
            height, width = frame.shape[:2]
            target_h = (height // tile_size) * tile_size
            target_w = (width // tile_size) * tile_size
            frame = cv2.resize(frame, (target_w, target_h))

            # Load the groundtruth tracking data for the example frame.
            tracking_results = load_detection_results(
                dataset, videos[0], tracking=True, groundtruth=True)

            # Find the frame matching args.frame_idx.
            example_tracks = []
            for r in tracking_results:
                if r['frame_idx'] == args.frame_idx:
                    example_tracks = r.get('tracks', [])
                    break

            # Scale detections if needed.
            if (width, height) != (target_w, target_h):
                scale_x = target_w / width
                scale_y = target_h / height
                example_tracks = [
                    [t[0], t[1] * scale_x, t[2] * scale_y,
                     t[3] * scale_x, t[4] * scale_y]
                    for t in example_tracks
                ]

            # Build the binary grid from detections.
            binary_grid = mark_detections(
                example_tracks, target_w, target_h, tile_size).astype(np.int32)

            # Render the polyomino vs ROI overlay.
            result = render_polyomino_vs_roi_overlay(
                frame, binary_grid, tile_size, args.alpha)

            # Crop and save.
            result = crop_to_tiles(result, tile_size, crop)
            out_path = os.path.join(out_dir, f'{dataset}_polyomino_vs_roi.png')
            cv2.imwrite(out_path, result)
            print(f"    Saved {out_path}")

    # Print grand totals.
    print(f"\n=== Grand Totals ===")
    print(f"  Total components: {grand_components}")
    print(f"  Total polyomino area: {grand_poly_area}")
    print(f"  Total bbox area: {grand_bbox_area}")
    grand_overhead = 0.0
    if grand_poly_area > 0:
        grand_overhead = 100.0 * (grand_bbox_area - grand_poly_area) / grand_poly_area
        print(f"  Overall overhead: {grand_overhead:.1f}%")

    # Save statistics as LaTeX macros.
    tex_path = os.path.join(out_dir, 'p004_polyomino_vs_roi_visualize.tex')
    with open(tex_path, 'w') as f:
        f.write(f'% Auto-generated by evaluation/p004_polyomino_vs_roi_visualize.py\n')
        f.write(f'% Average bounding-box overhead over polyomino area across all datasets.\n')
        f.write(f'\\newcommand{{\\bboxOverheadPct}}{{\\autogen{{{grand_overhead:.0f}}}}}\n')
        f.write(f'% Total connected components across all datasets.\n')
        f.write(f'\\newcommand{{\\bboxTotalComponents}}{{\\autogen{{{grand_components}}}}}\n')
        f.write(f'% Total polyomino tile area across all datasets.\n')
        f.write(f'\\newcommand{{\\bboxTotalPolyArea}}{{\\autogen{{{grand_poly_area}}}}}\n')
        f.write(f'% Total bounding-box tile area across all datasets.\n')
        f.write(f'\\newcommand{{\\bboxTotalBboxArea}}{{\\autogen{{{grand_bbox_area}}}}}\n')
    print(f"  Saved {tex_path}")

    print("\nAll polyomino vs ROI visualizations completed!")


if __name__ == '__main__':
    main(parse_args())
