# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

cimport numpy as cnp


# cdef unsigned long long group_tiles(cnp.uint8_t[:, :] bitmap_input) noexcept nogil