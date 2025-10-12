# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

cimport numpy as cnp
import cython

ctypedef cnp.uint8_t DTYPE_t


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
def render(
    cnp.uint8_t[:, :, :] canvas,
    list positions,
    cnp.uint8_t[:, :, :] frame,
    int chunk_size,
):
    """
    Render packed polyominoes onto the canvas.
    
    Args:
        canvas: The canvas to render onto
        positions: List of packed polyominoe positions
        frame: Source frame
        chunk_size: Size of each tile/chunk
        
    Returns:
        np.ndarray: Updated canvas
    """
    cdef int i, j
    cdef int y, x, yfrom, xfrom, n
    cdef long offset_h, offset_w
    cdef tuple position, offset

    n = len(positions)

    for i in range(n):
        position = positions[i]
        y = position[0]
        x = position[1]
        mask = position[2]
        offset = position[3]
        offset_h = offset[0]
        offset_w = offset[1]

        yfrom = y * chunk_size
        xfrom = x * chunk_size
        
        # mask is a 1D array of positions in format [x, y, x, y, x, y, ...]
        # Reshape to get x and y coordinates as separate arrays
        coords = mask.reshape(-1, 2)  # Shape: (n_coords, 2) where each row is [x, y]
        mask_x = coords[:, 0]  # All x coordinates
        mask_y = coords[:, 1]  # All y coordinates
        
        # Process all coordinates at once using vectorized operations
        for k in range(len(mask_x)):
            canvas[
                yfrom + (chunk_size * mask_y[k]): yfrom + (chunk_size * mask_y[k]) + chunk_size,
                xfrom + (chunk_size * mask_x[k]): xfrom + (chunk_size * mask_x[k]) + chunk_size,
            ] = frame[
                (mask_y[k] + offset_h) * chunk_size:(mask_y[k] + offset_h + 1) * chunk_size,
                (mask_x[k] + offset_w) * chunk_size:(mask_x[k] + offset_w + 1) * chunk_size,
            ]