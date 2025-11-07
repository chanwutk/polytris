# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

cimport numpy as cnp
import numpy as np
import cython


# Declare C structures from utilities.h
cdef extern from "c/utilities.h":
    ctypedef struct ShortArray:
        unsigned short *data  # type: ignore
        int size
        int capacity

    ctypedef struct Polyomino:
        ShortArray mask
        int offset_i
        int offset_j

    ctypedef struct PolyominoArray:
        Polyomino *data
        int size
        int capacity

    # Declare utility functions
    void PolyominoArray_cleanup(PolyominoArray *array)


# Declare C functions from group_tiles.h
cdef extern from "c/group_tiles.h":
    # Main function to group tiles into polyominoes
    # bitmap_input: 2D array (flattened) of uint8_t representing the grid of tiles
    #               where 1 indicates a tile with detection and 0 indicates no detection
    # width: width of the bitmap
    # height: height of the bitmap
    # tilepadding_mode: The mode of tile padding to apply
    #                   - 0: No padding
    #                   - 1: Disconnected padding
    #                   - 2: Connected padding
    # Returns: Pointer to PolyominoArray containing all found polyominoes
    PolyominoArray* group_tiles_(
        unsigned char *bitmap_input,
        short width,
        short height,
        char tilepadding_mode
    )

    # Free a polyomino array allocated by group_tiles
    # Returns the number of polyominoes that were freed
    int free_polyomino_array_(PolyominoArray *polyomino_array)


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
def group_tiles(cnp.uint8_t[:, :] bitmap_input, int tilepadding_mode) -> np.uint64:
    """
    Group connected tiles into polyominoes using C implementation.

    Parameters:
        bitmap_input: 2D numpy array of uint8 representing the tile grid
                     where 1 indicates a tile with detection and 0 indicates no detection
                     must be contiguous
        tilepadding_mode: The mode of tile padding to apply
                         - 0: No padding
                         - 1: Disconnected padding
                         - 2: Connected padding

    Returns:
        numpy.uint64: Memory address (as uint64) pointing to a PolyominoArray
    """
    cdef short height = <short>bitmap_input.shape[0]
    cdef short width = <short>bitmap_input.shape[1]
    cdef PolyominoArray* result_ptr = group_tiles_(&bitmap_input[0, 0], width, height, <char>tilepadding_mode)
    # Convert pointer to numpy.uint64 for safe type handling
    return np.uint64(<cnp.uint64_t>result_ptr)


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
def free_polyomino_array(cnp.uint64_t polyomino_array_addr) -> int:
    """
    Free a polyomino array allocated by group_tiles.
    
    Parameters:
        polyomino_array: Memory address as numpy.uint64, int, or compatible type
    """
    return free_polyomino_array_(<PolyominoArray*>polyomino_array_addr)