# cython: language_level=3
# Type declarations for greedy pruning algorithm

from libc.stdint cimport int32_t, int64_t, uint8_t, uint64_t

# Structure to track deadline information for each tile position
cdef struct TileDeadline:
    int32_t row  # Tile row position
    int32_t col  # Tile column position
    int32_t deadline  # Frame index by which this tile must be sampled
    int32_t last_selected  # Last frame where this tile was selected

# Structure for priority queue entry
cdef struct PQEntry:
    int32_t deadline  # The deadline value (for priority)
    int32_t row  # Tile row
    int32_t col  # Tile column