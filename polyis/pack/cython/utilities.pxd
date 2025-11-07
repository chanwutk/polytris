# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

cdef struct IntStack:
    short *data
    int top
    int capacity

cdef struct Polyomino:
    IntStack mask
    int offset_i
    int offset_j

cdef struct PolyominoStack:
    Polyomino *mo_data
    int top
    int capacity

cdef int IntStack_init(IntStack *stack, int initial_capacity) noexcept nogil
cdef int IntStack_push(IntStack *stack, short value) noexcept nogil
cdef void IntStack_cleanup(IntStack *stack) noexcept nogil

cdef void Polyomino_cleanup(Polyomino *polyomino) noexcept nogil

cdef int PolyominoStack_init(PolyominoStack *stack, int initial_capacity) noexcept nogil
cdef int PolyominoStack_push(PolyominoStack *stack, Polyomino value) noexcept nogil
cdef void PolyominoStack_cleanup(PolyominoStack *stack) noexcept nogil
