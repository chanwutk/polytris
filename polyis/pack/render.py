"""CPU canvas rendering helpers shared by scripts/p030 and execution/compress_stage."""

from typing import Callable
import numpy as np

from polyis import dtypes


OffsetLookup = tuple[tuple[int, int], tuple[int, int], int]


def precompute_grid_boundaries(
    grid_height: int, grid_width: int, tile_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute tile pixel boundaries for a grid_height x grid_width grid at a fixed tile_size.

    Returns (y_starts, y_ends, x_starts, x_ends), each a 1D int32 ndarray.
    """
    # Build grid row indices.
    rows = np.arange(grid_height, dtype=np.int32)
    # Build grid column indices.
    cols = np.arange(grid_width, dtype=np.int32)
    # Precompute row starts as exact fixed-size tile boundaries.
    y_starts = rows * tile_size
    # Precompute row ends as exact fixed-size tile boundaries.
    y_ends = y_starts + tile_size
    # Precompute column starts as exact fixed-size tile boundaries.
    x_starts = cols * tile_size
    # Precompute column ends as exact fixed-size tile boundaries.
    x_ends = x_starts + tile_size
    return y_starts, y_ends, x_starts, x_ends


def compute_polyomino_tile_boundaries(
    oy: int,
    ox: int,
    py: int,
    px: int,
    i_coords: np.ndarray,
    j_coords: np.ndarray,
    src_y_starts: np.ndarray,
    src_y_ends: np.ndarray,
    src_x_starts: np.ndarray,
    src_x_ends: np.ndarray,
    dst_y_starts: np.ndarray,
    dst_y_ends: np.ndarray,
    dst_x_starts: np.ndarray,
    dst_x_ends: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Source row boundaries via lookup.
    sy_starts = src_y_starts[oy + i_coords]
    # Source column boundaries via lookup.
    sx_starts = src_x_starts[ox + j_coords]
    # Source row end boundaries via lookup.
    sy_ends = src_y_ends[oy + i_coords]
    # Source column end boundaries via lookup.
    sx_ends = src_x_ends[ox + j_coords]
    # Destination row boundaries via lookup.
    dy_starts = dst_y_starts[py + i_coords]
    # Destination column boundaries via lookup.
    dx_starts = dst_x_starts[px + j_coords]
    # Destination row end boundaries via lookup.
    dy_ends = dst_y_ends[py + i_coords]
    # Destination column end boundaries via lookup.
    dx_ends = dst_x_ends[px + j_coords]
    return sy_starts, sx_starts, sy_ends, sx_ends, dy_starts, dx_starts, dy_ends, dx_ends


def copy_same_shape_tiles(
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
    # Direct slicing for tiles where source/destination shapes match.
    for idx in np.flatnonzero(same_shape):
        canvas[dy_starts[idx]:dy_ends[idx], dx_starts[idx]:dx_ends[idx]] = \
            frame[sy_starts[idx]:sy_ends[idx], sx_starts[idx]:sx_ends[idx]]


def copy_mismatched_tiles_with_edge_repeat(
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
    # Edge-repeat copy for boundary tiles where shapes differ.
    for idx in np.flatnonzero(~same_shape):
        sy_start = sy_starts[idx]
        sy_end = sy_ends[idx]
        sx_start = sx_starts[idx]
        sx_end = sx_ends[idx]
        dy_start = dy_starts[idx]
        dy_end = dy_ends[idx]
        dx_start = dx_starts[idx]
        dx_end = dx_ends[idx]

        src_h = src_hs[idx]
        src_w = src_ws[idx]
        dst_h = dst_hs[idx]
        dst_w = dst_ws[idx]

        # Skip degenerate tiles to avoid invalid indexing.
        if src_h <= 0 or src_w <= 0 or dst_h <= 0 or dst_w <= 0:
            continue

        # Overlap region between source and destination tile sizes.
        copy_h = min(src_h, dst_h)
        copy_w = min(src_w, dst_w)

        # Destination tile view for in-place writes.
        dst_tile = canvas[dy_start:dy_end, dx_start:dx_end]
        # Source tile view for reads.
        src_tile = frame[sy_start:sy_end, sx_start:sx_end]

        # Copy the overlapping area.
        dst_tile[:copy_h, :copy_w] = src_tile[:copy_h, :copy_w]

        # Extend the last copied row downward when destination is taller.
        if dst_h > copy_h:
            dst_tile[copy_h:dst_h, :copy_w] = src_tile[copy_h - 1:copy_h, :copy_w]

        # Extend the last copied column rightward when destination is wider.
        if dst_w > copy_w:
            dst_tile[:, copy_w:dst_w] = dst_tile[:, copy_w - 1:copy_w]


def copy_polyomino_tiles_to_canvas(
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
    # Tile heights/widths.
    dst_hs = (dy_ends - dy_starts)
    dst_ws = (dx_ends - dx_starts)
    src_hs = (sy_ends - sy_starts)
    src_ws = (sx_ends - sx_starts)
    # Identify tiles where source and destination shapes already match.
    same_shape = (src_hs == dst_hs) & (src_ws == dst_ws)

    copy_same_shape_tiles(
        canvas=canvas, frame=frame,
        sy_starts=sy_starts, sx_starts=sx_starts,
        sy_ends=sy_ends, sx_ends=sx_ends,
        dy_starts=dy_starts, dx_starts=dx_starts,
        dy_ends=dy_ends, dx_ends=dx_ends,
        same_shape=same_shape,
    )

    copy_mismatched_tiles_with_edge_repeat(
        canvas=canvas, frame=frame,
        sy_starts=sy_starts, sx_starts=sx_starts,
        sy_ends=sy_ends, sx_ends=sx_ends,
        dy_starts=dy_starts, dx_starts=dx_starts,
        dy_ends=dy_ends, dx_ends=dx_ends,
        src_hs=src_hs, src_ws=src_ws,
        dst_hs=dst_hs, dst_ws=dst_ws,
        same_shape=same_shape,
    )


def update_polyomino_metadata(
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


def render_collage_cpu(
    canvas: np.ndarray,
    index_map: np.ndarray,
    offset_lookup: list[OffsetLookup],
    collage: list,
    fetch_frame: Callable[[int], np.ndarray],
    src_y_starts: np.ndarray,
    src_y_ends: np.ndarray,
    src_x_starts: np.ndarray,
    src_x_ends: np.ndarray,
    dst_y_starts: np.ndarray,
    dst_y_ends: np.ndarray,
    dst_x_starts: np.ndarray,
    dst_x_ends: np.ndarray,
) -> None:
    """Render all polyominoes of one collage onto canvas.

    canvas, index_map, offset_lookup are mutated in place.
    collage is a list of items unpacking as (oy, ox, py, px, frame_idx, shape).
    fetch_frame(abs_frame_idx) returns a [H, W, 3] uint8 numpy array (view OK).
    src_* and dst_* are precomputed tile pixel boundary arrays for the source
    and destination grids respectively (see precompute_grid_boundaries).
    """
    for gid, poly_pos in enumerate(collage, start=1):
        # Unpack the polyomino position record (works for tuple or NamedTuple).
        oy, ox, py, px, frame_idx, shape = poly_pos

        # Resolve the source frame for this polyomino.
        frame = fetch_frame(frame_idx)

        # Polyomino tile coordinates within its own bounding box.
        i_coords = shape[:, 0]
        j_coords = shape[:, 1]

        # Compute source/destination tile pixel boundaries for every tile.
        sy_starts, sx_starts, sy_ends, sx_ends, dy_starts, dx_starts, dy_ends, dx_ends = \
            compute_polyomino_tile_boundaries(
                oy=oy, ox=ox, py=py, px=px,
                i_coords=i_coords, j_coords=j_coords,
                src_y_starts=src_y_starts, src_y_ends=src_y_ends,
                src_x_starts=src_x_starts, src_x_ends=src_x_ends,
                dst_y_starts=dst_y_starts, dst_y_ends=dst_y_ends,
                dst_x_starts=dst_x_starts, dst_x_ends=dst_x_ends,
            )

        # Copy every tile of this polyomino onto canvas, with edge-repeat for mismatches.
        copy_polyomino_tiles_to_canvas(
            canvas=canvas, frame=frame,
            sy_starts=sy_starts, sx_starts=sx_starts,
            sy_ends=sy_ends, sx_ends=sx_ends,
            dy_starts=dy_starts, dx_starts=dx_starts,
            dy_ends=dy_ends, dx_ends=dx_ends,
        )

        # Update index_map and offset_lookup for this polyomino.
        update_polyomino_metadata(
            index_map=index_map,
            offset_lookup=offset_lookup,
            gid=gid, py=py, px=px, oy=oy, ox=ox,
            frame_idx=frame_idx,
            i_coords=i_coords, j_coords=j_coords,
        )
