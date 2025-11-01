# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

cimport numpy as cnp
import cython
from cython.parallel import prange

ctypedef cnp.uint8_t DTYPE_t


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
def render(
    cnp.uint8_t[:, :, :] canvas,
    list positions,
    cnp.uint8_t[:, :, :] frame,
    int tile_size,
):
    """
    Render packed polyominoes onto the canvas.
    
    Args:
        canvas: The canvas to render onto
        positions: List of packed polyominoe positions
        frame: Source frame
        tile_size: Size of each tile/chunk
        
    Returns:
        np.ndarray: Updated canvas
    """
    cdef int i, y, x, yfrom, xfrom, n_masks, offset_h, offset_w
    cdef tuple position, offset
    cdef int ts = tile_size

    for i in range(len(positions)):
        position = positions[i]
        y = position[0]
        x = position[1]
        mask = position[2]
        offset = position[3]
        offset_h = offset[0]
        offset_w = offset[1]

        yfrom = y * ts
        xfrom = x * ts

        n_masks = mask.shape[0] // 2
        
        # Process all coordinates at once using vectorized operations
        for k in range(n_masks):
            mask_x = mask[k << 1]
            mask_y = mask[(k << 1) + 1]

            sx = xfrom + (ts * mask_x)
            sy = yfrom + (ts * mask_y)

            dx = (mask_y + offset_h) * ts
            dy = (mask_x + offset_w) * ts

            canvas[sy: sy + ts, sx: sx + ts, :] = frame[dy: dy + ts, dx: dx + ts, :]
