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
    cdef long mask_h, mask_w, offset_h, offset_w
    cdef cnp.uint8_t[:, :] mask_view
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
        mask_h = mask.shape[0]
        mask_w = mask.shape[1]
        mask_view = mask

        yfrom = y * chunk_size
        xfrom = x * chunk_size
        
        # Get mask indices where True
        for i in range(mask_h):
            for j in range(mask_w):
                if mask_view[i, j]:
                    canvas[
                        yfrom + (chunk_size * i): yfrom + (chunk_size * i) + chunk_size,
                        xfrom + (chunk_size * j): xfrom + (chunk_size * j) + chunk_size,
                    ] = frame[
                        (i + offset_h) * chunk_size:(i + offset_h + 1) * chunk_size,
                        (j + offset_w) * chunk_size:(j + offset_w + 1) * chunk_size,
                    ]