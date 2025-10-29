# cython: language_level=3
# Type stub for group_tiles module

cimport numpy as cnp
from polyis.binpack.utilities cimport IntStack, Polyomino, PolyominoStack

cdef int compare_polyomino_by_mask_length(const void *a, const void *b) noexcept nogil

cdef IntStack _find_connected_tiles(
    unsigned int* bitmap,
    unsigned short h,
    unsigned short w,
    unsigned short start_i,
    unsigned short start_j,
    cnp.uint8_t[:, :] bitmap_input,
    int mode
) noexcept nogil

cdef void _add_padding(
    cnp.uint8_t[:, :] bitmap,
    unsigned short h,
    unsigned short w
) noexcept nogil

def group_tiles(cnp.uint8_t[:, :] bitmap_input, int tilepadding_mode)

def free_polyimino_stack(unsigned long long polyomino_stack_ptr) -> int
