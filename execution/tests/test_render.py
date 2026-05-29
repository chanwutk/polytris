"""Unit tests for ``polyis.pack.render``.

Verifies the CPU canvas rendering helper used by both scripts/p030 and
execution/compress_stage produces correct output for hand-constructed
collages (same-shape tile copy, edge-repeat for boundary mismatches, and
index_map / offset_lookup population).
"""

from __future__ import annotations

import numpy as np

from polyis.pack.render import (
    compute_polyomino_tile_boundaries,
    copy_polyomino_tiles_to_canvas,
    precompute_grid_boundaries,
    render_collage_cpu,
    update_polyomino_metadata,
)


def test_precompute_grid_boundaries_fixed_tiles():
    """precompute_grid_boundaries returns evenly spaced tile pixel bounds."""
    y_starts, y_ends, x_starts, x_ends = precompute_grid_boundaries(3, 4, tile_size=10)
    np.testing.assert_array_equal(y_starts, [0, 10, 20])
    np.testing.assert_array_equal(y_ends, [10, 20, 30])
    np.testing.assert_array_equal(x_starts, [0, 10, 20, 30])
    np.testing.assert_array_equal(x_ends, [10, 20, 30, 40])


def test_compute_polyomino_tile_boundaries_lookup():
    """compute_polyomino_tile_boundaries looks up bounds via integer arithmetic."""
    src_y_starts, src_y_ends, src_x_starts, src_x_ends = precompute_grid_boundaries(4, 4, 10)
    dst_y_starts, dst_y_ends, dst_x_starts, dst_x_ends = precompute_grid_boundaries(4, 4, 10)
    # A polyomino with two tiles at (oy=1, ox=2)+(0,0) and (0,1) placed at (py=0, px=0).
    i_coords = np.array([0, 0], dtype=np.int64)
    j_coords = np.array([0, 1], dtype=np.int64)
    sy_s, sx_s, sy_e, sx_e, dy_s, dx_s, dy_e, dx_e = compute_polyomino_tile_boundaries(
        oy=1, ox=2, py=0, px=0,
        i_coords=i_coords, j_coords=j_coords,
        src_y_starts=src_y_starts, src_y_ends=src_y_ends,
        src_x_starts=src_x_starts, src_x_ends=src_x_ends,
        dst_y_starts=dst_y_starts, dst_y_ends=dst_y_ends,
        dst_x_starts=dst_x_starts, dst_x_ends=dst_x_ends,
    )
    np.testing.assert_array_equal(sy_s, [10, 10])
    np.testing.assert_array_equal(sx_s, [20, 30])
    np.testing.assert_array_equal(sy_e, [20, 20])
    np.testing.assert_array_equal(sx_e, [30, 40])
    np.testing.assert_array_equal(dy_s, [0, 0])
    np.testing.assert_array_equal(dx_s, [0, 10])


def test_copy_polyomino_tiles_same_shape_block():
    """Same-shape src/dst tiles are copied byte-for-byte by direct slicing."""
    tile_size = 4
    frame = np.arange(tile_size * tile_size * 3 * 4, dtype=np.uint8).reshape(
        tile_size, tile_size * 4, 3,
    )
    canvas = np.zeros((tile_size, tile_size * 4, 3), dtype=np.uint8)

    # Single tile at src (0, 8..12) -> dst (0, 0..4).
    copy_polyomino_tiles_to_canvas(
        canvas=canvas, frame=frame,
        sy_starts=np.array([0]), sx_starts=np.array([8]),
        sy_ends=np.array([4]), sx_ends=np.array([12]),
        dy_starts=np.array([0]), dx_starts=np.array([0]),
        dy_ends=np.array([4]), dx_ends=np.array([4]),
    )
    np.testing.assert_array_equal(canvas[:, :4, :], frame[:, 8:12, :])


def test_update_polyomino_metadata_indexes_correctly():
    """update_polyomino_metadata sets index_map cells and appends to offset_lookup."""
    index_map = np.zeros((4, 4), dtype=np.uint16)
    offset_lookup: list = []
    i_coords = np.array([0, 0, 1], dtype=np.int64)
    j_coords = np.array([0, 1, 0], dtype=np.int64)

    update_polyomino_metadata(
        index_map=index_map, offset_lookup=offset_lookup,
        gid=7, py=1, px=2, oy=0, ox=3, frame_idx=42,
        i_coords=i_coords, j_coords=j_coords,
    )

    # Index map entries at (py+i, px+j) should equal gid; others stay zero.
    expected_map = np.zeros((4, 4), dtype=np.uint16)
    expected_map[1, 2] = 7
    expected_map[1, 3] = 7
    expected_map[2, 2] = 7
    np.testing.assert_array_equal(index_map, expected_map)
    assert offset_lookup == [((1, 2), (0, 3), 42)]


def test_render_collage_cpu_single_polyomino_round_trip():
    """End-to-end render of a one-polyomino collage onto a zero canvas."""
    tile_size = 4
    src_h, src_w = 2, 3   # 2 rows x 3 cols of tiles
    dst_h, dst_w = 2, 3
    src_y_s, src_y_e, src_x_s, src_x_e = precompute_grid_boundaries(src_h, src_w, tile_size)
    dst_y_s, dst_y_e, dst_x_s, dst_x_e = precompute_grid_boundaries(dst_h, dst_w, tile_size)

    # Build a recognisable source frame: each tile has its (i, j) tile index
    # encoded as a constant value to make assertions clear.
    frame_h = src_h * tile_size
    frame_w = src_w * tile_size
    frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    for i in range(src_h):
        for j in range(src_w):
            frame[i * tile_size:(i + 1) * tile_size,
                  j * tile_size:(j + 1) * tile_size, :] = (i * src_w + j + 1)

    # Polyomino with two tiles: (0,0) and (0,1), placed at (py=0, px=1).
    shape = np.array([[0, 0], [0, 1]], dtype=np.int64)
    collage = [(0, 0, 0, 1, 42, shape)]

    canvas = np.zeros((dst_h * tile_size, dst_w * tile_size, 3), dtype=np.uint8)
    index_map = np.zeros((dst_h, dst_w), dtype=np.uint16)
    offset_lookup: list = []

    render_collage_cpu(
        canvas=canvas, index_map=index_map, offset_lookup=offset_lookup,
        collage=collage,
        fetch_frame=lambda fidx: frame,
        src_y_starts=src_y_s, src_y_ends=src_y_e,
        src_x_starts=src_x_s, src_x_ends=src_x_e,
        dst_y_starts=dst_y_s, dst_y_ends=dst_y_e,
        dst_x_starts=dst_x_s, dst_x_ends=dst_x_e,
    )

    # Destination tile (0,1) should equal source tile (0,0) which has value 1.
    np.testing.assert_array_equal(
        canvas[0:tile_size, tile_size:2 * tile_size, :], 1,
    )
    # Destination tile (0,2) should equal source tile (0,1) which has value 2.
    np.testing.assert_array_equal(
        canvas[0:tile_size, 2 * tile_size:3 * tile_size, :], 2,
    )
    # Outside the polyomino: still zero.
    np.testing.assert_array_equal(canvas[0:tile_size, 0:tile_size, :], 0)
    np.testing.assert_array_equal(canvas[tile_size:, :, :], 0)

    # Metadata.
    assert index_map[0, 1] == 1
    assert index_map[0, 2] == 1
    assert offset_lookup == [((0, 1), (0, 0), 42)]
