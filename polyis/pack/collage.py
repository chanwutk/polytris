"""Collage rendering and detection unpacking shared by scripts/p030, scripts/p050 and execution2."""

from typing import NamedTuple

import numpy as np

from polyis import dtypes


class PolyominoPosition(NamedTuple):
    oy: int
    ox: int
    py: int
    px: int
    frame: int
    shape: np.ndarray


OffsetLookup = tuple[tuple[int, int], tuple[int, int], int]


Det = list[float]
Dets = list[Det]
FrameIdToDets = dict[int, Dets]
UnpackedDets = tuple[FrameIdToDets, Dets, Dets]


def _compute_polyomino_tile_boundaries(
    oy: int,
    ox: int,
    py: int,
    px: int,
    i_coords: np.ndarray,
    j_coords: np.ndarray,
    src_grid_y_starts: np.ndarray,
    src_grid_y_ends: np.ndarray,
    src_grid_x_starts: np.ndarray,
    src_grid_x_ends: np.ndarray,
    dst_grid_y_starts: np.ndarray,
    dst_grid_y_ends: np.ndarray,
    dst_grid_x_starts: np.ndarray,
    dst_grid_x_ends: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Compute source row boundaries via lookup to avoid repeated integer arithmetic.
    sy_starts = src_grid_y_starts[oy + i_coords]
    # Compute source column boundaries via lookup to avoid repeated integer arithmetic.
    sx_starts = src_grid_x_starts[ox + j_coords]
    # Compute source row end boundaries via lookup to avoid repeated integer arithmetic.
    sy_ends = src_grid_y_ends[oy + i_coords]
    # Compute source column end boundaries via lookup to avoid repeated integer arithmetic.
    sx_ends = src_grid_x_ends[ox + j_coords]
    # Compute destination row boundaries via lookup to avoid repeated integer arithmetic.
    dy_starts = dst_grid_y_starts[py + i_coords]
    # Compute destination column boundaries via lookup to avoid repeated integer arithmetic.
    dx_starts = dst_grid_x_starts[px + j_coords]
    # Compute destination row end boundaries via lookup to avoid repeated integer arithmetic.
    dy_ends = dst_grid_y_ends[py + i_coords]
    # Compute destination column end boundaries via lookup to avoid repeated integer arithmetic.
    dx_ends = dst_grid_x_ends[px + j_coords]
    return sy_starts, sx_starts, sy_ends, sx_ends, dy_starts, dx_starts, dy_ends, dx_ends


def _copy_same_shape_tiles(
    canvas: np.ndarray,
    frame: np.ndarray,
    sy_starts: np.ndarray,
    sx_starts: np.ndarray,
    sy_ends: np.ndarray,
    sx_ends: np.ndarray,
    dy_starts: np.ndarray,
    dx_starts: np.ndarray,
    dy_ends: np.ndarray,
    dx_ends: np.ndarray,
    same_shape: np.ndarray,
):
    # Iterate over tile indices where source and destination shapes already match.
    for idx in np.flatnonzero(same_shape):
        # Copy a same-size source tile directly into the destination tile.
        canvas[dy_starts[idx]:dy_ends[idx], dx_starts[idx]:dx_ends[idx]] = frame[sy_starts[idx]:sy_ends[idx], sx_starts[idx]:sx_ends[idx]]


def _copy_mismatched_tiles_with_edge_repeat(
    canvas: np.ndarray,
    frame: np.ndarray,
    sy_starts: np.ndarray,
    sx_starts: np.ndarray,
    sy_ends: np.ndarray,
    sx_ends: np.ndarray,
    dy_starts: np.ndarray,
    dx_starts: np.ndarray,
    dy_ends: np.ndarray,
    dx_ends: np.ndarray,
    src_hs: np.ndarray,
    src_ws: np.ndarray,
    dst_hs: np.ndarray,
    dst_ws: np.ndarray,
    same_shape: np.ndarray,
):
    # Iterate over tile indices where source and destination shapes are different.
    for idx in np.flatnonzero(~same_shape):
        # Read source row bounds for this tile.
        sy_start = sy_starts[idx]
        # Read source row end for this tile.
        sy_end = sy_ends[idx]
        # Read source column bounds for this tile.
        sx_start = sx_starts[idx]
        # Read source column end for this tile.
        sx_end = sx_ends[idx]
        # Read destination row bounds for this tile.
        dy_start = dy_starts[idx]
        # Read destination row end for this tile.
        dy_end = dy_ends[idx]
        # Read destination column bounds for this tile.
        dx_start = dx_starts[idx]
        # Read destination column end for this tile.
        dx_end = dx_ends[idx]

        # Read source tile height for this tile.
        src_h = src_hs[idx]
        # Read source tile width for this tile.
        src_w = src_ws[idx]
        # Read destination tile height for this tile.
        dst_h = dst_hs[idx]
        # Read destination tile width for this tile.
        dst_w = dst_ws[idx]

        # Skip degenerate tiles to avoid invalid indexing.
        if src_h <= 0 or src_w <= 0 or dst_h <= 0 or dst_w <= 0:
            continue

        # Compute copied height as the overlap between source and destination heights.
        copy_h = min(src_h, dst_h)
        # Compute copied width as the overlap between source and destination widths.
        copy_w = min(src_w, dst_w)

        # Create a destination tile view for in-place writes.
        dst_tile = canvas[dy_start:dy_end, dx_start:dx_end]
        # Create a source tile view for reading source pixels.
        src_tile = frame[sy_start:sy_end, sx_start:sx_end]

        # Copy the overlapping source area into the destination tile.
        dst_tile[:copy_h, :copy_w] = src_tile[:copy_h, :copy_w]

        # Extend the last copied row downward when destination height is larger.
        if dst_h > copy_h:
            dst_tile[copy_h:dst_h, :copy_w] = src_tile[copy_h - 1:copy_h, :copy_w]

        # Extend the last copied column rightward when destination width is larger.
        if dst_w > copy_w:
            dst_tile[:, copy_w:dst_w] = dst_tile[:, copy_w - 1:copy_w]


def _copy_polyomino_tiles_to_canvas(
    canvas: np.ndarray,
    frame: np.ndarray,
    sy_starts: np.ndarray,
    sx_starts: np.ndarray,
    sy_ends: np.ndarray,
    sx_ends: np.ndarray,
    dy_starts: np.ndarray,
    dx_starts: np.ndarray,
    dy_ends: np.ndarray,
    dx_ends: np.ndarray,
):
    # Precompute destination tile heights.
    dst_hs = (dy_ends - dy_starts)
    # Precompute destination tile widths.
    dst_ws = (dx_ends - dx_starts)
    # Precompute source tile heights.
    src_hs = (sy_ends - sy_starts)
    # Precompute source tile widths.
    src_ws = (sx_ends - sx_starts)
    # Identify tiles where source and destination shapes already match.
    same_shape = (src_hs == dst_hs) & (src_ws == dst_ws)

    # Copy same-shape tiles with direct slicing.
    _copy_same_shape_tiles(
        canvas=canvas,
        frame=frame,
        sy_starts=sy_starts,
        sx_starts=sx_starts,
        sy_ends=sy_ends,
        sx_ends=sx_ends,
        dy_starts=dy_starts,
        dx_starts=dx_starts,
        dy_ends=dy_ends,
        dx_ends=dx_ends,
        same_shape=same_shape,
    )

    # Copy mismatched tiles with in-place edge-repeat behavior.
    _copy_mismatched_tiles_with_edge_repeat(
        canvas=canvas,
        frame=frame,
        sy_starts=sy_starts,
        sx_starts=sx_starts,
        sy_ends=sy_ends,
        sx_ends=sx_ends,
        dy_starts=dy_starts,
        dx_starts=dx_starts,
        dy_ends=dy_ends,
        dx_ends=dx_ends,
        src_hs=src_hs,
        src_ws=src_ws,
        dst_hs=dst_hs,
        dst_ws=dst_ws,
        same_shape=same_shape,
    )


def _update_polyomino_metadata(
    index_map: dtypes.IndexMap,
    offset_lookup: list[OffsetLookup],
    gid: int,
    py: int,
    px: int,
    oy: int,
    ox: int,
    frame_idx: int,
    i_coords: np.ndarray,
    j_coords: np.ndarray,
):
    # Write the group id into index_map positions covered by this polyomino.
    index_map[py + i_coords, px + j_coords] = gid
    # Append the mapping tuple for decompression lookup.
    offset_lookup.append(((py, px), (oy, ox), frame_idx))


def unpack_detections(detections: list[list[float]], index_map: np.ndarray,
                      offset_lookup: list[tuple[tuple[int, int], tuple[int, int], int]],
                      tilesize: int, drop_center_outside: bool = False) -> UnpackedDets:
    """
    Unpack detections from compressed coordinates back to original frame coordinates.

    Args:
        detections (list[list[float]]): list of bounding boxes in compressed coordinates [x1, y1, x2, y2]
        index_map (np.ndarray): Index map from the mapping file (2D array with group_ids)
        offset_lookup (list[tuple[tuple[int, int], tuple[int, int], int]]): Offset lookup from the mapping file
        tilesize (int): Size of tiles used for compression
        drop_center_outside (bool): Drop detections whose center falls on an empty canvas cell instead of
            assigning them to the polyomino under one of their corners (off by default; the tracker can
            discard such boxes, while detection-only evaluations may drop them)

    Returns:
        tuple[
            dict[int, list[list[float]]],
            list[list[float]],
            list[list[float]],
        ]:
            dictionary mapping frame indices to lists of bounding boxes in original frame coordinates,
            list of detections that are not in any tile,
            list of detections that are in the center of a tile but not in any tile
    """
    # Initialize dictionary to store detections per frame
    frame_detections: dict[int, list[list[float]]] = {}
    not_in_any_tile_detections: list[list[float]] = []
    center_not_in_any_tile_detections: list[list[float]] = []

    # Process each detection
    for x1, y1, x2, y2, *_ in detections:
        # Get the center point of the detection in compressed coordinates
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

        # center, top-left, top-right, bottom-left, bottom-right
        xs = [center_x, x1, x2, x1, x2]
        ys = [center_y, y1, y1, y2, y2]

        group_id: int | None = None
        frame_idx: int | None = None
        center_in_any_tile: bool = True
        for x, y in zip(xs, ys):
            # Convert to tile coordinates in the compressed image
            tile_x = int(x // tilesize)
            tile_y = int(y // tilesize)

            # Ensure tile coordinates are within bounds
            if (tile_y < 0 or tile_y >= index_map.shape[0] or
                tile_x < 0 or tile_x >= index_map.shape[1]):
                continue

            # Get the group ID for this tile
            group_id_ = int(index_map[tile_y, tile_x])

            if group_id_ != 0:
                group_id = group_id_
                break
            center_in_any_tile = False

        if not center_in_any_tile:
            center_not_in_any_tile_detections.append([x1, y1, x2, y2, *_])
            # Optionally drop a box centered on empty canvas space rather than map it through a corner.
            if drop_center_outside:
                continue

        if group_id is None:
            not_in_any_tile_detections.append([x1, y1, x2, y2, *_])
            continue

        # Convert group_id to 0-based index
        group_id -= 1

        # Get the offset information for this group
        assert 0 <= group_id < len(offset_lookup), f"Group {group_id} not found in offset lookup"
        (packed_y, packed_x), (original_offset_y, original_offset_x), frame_idx = offset_lookup[group_id]

        # Calculate the offset to convert from compressed to original coordinates
        # The offset represents how much the tile was moved during compression
        offset_x = (original_offset_x - packed_x) * tilesize
        offset_y = (original_offset_y - packed_y) * tilesize

        # Convert detection back to original frame coordinates
        original_det = [
            x1 + offset_x,
            y1 + offset_y,
            x2 + offset_x,
            y2 + offset_y,
            *_
        ]

        # Add to frame detections
        if frame_idx not in frame_detections:
            frame_detections[frame_idx] = []
        frame_detections[frame_idx].append(original_det)

    return frame_detections, not_in_any_tile_detections, center_not_in_any_tile_detections
