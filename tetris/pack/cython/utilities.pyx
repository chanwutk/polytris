# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

from libc.stdlib cimport malloc, free, realloc
import cython


# Coordinate structure to represent a 2D point
cdef struct Coordinate:
    short y
    short x


cdef struct IntStack:
    short *data
    int top
    int capacity


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef int IntStack_init(IntStack *stack, int initial_capacity) noexcept nogil:
    """Initialize an integer vector with initial capacity"""
    # if not stack:
    #     return -1

    stack.data = <short*>malloc(<size_t>initial_capacity * sizeof(short))
    # if not stack.data:
    #     return -1

    stack.top = 0
    stack.capacity = initial_capacity
    return 0


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef int IntStack_push(IntStack *stack, short value) noexcept nogil:
    """Push a value onto the vector, expanding if necessary"""
    cdef int new_capacity
    cdef short *new_data

    # if not stack:
    #     return -1

    # Check if we need to expand
    if stack.top >= stack.capacity:
        new_capacity = stack.capacity * 2
        new_data = <short*>realloc(<void*>stack.data,
                                            <size_t>new_capacity * sizeof(short))
        # if not new_data:
        #     return -1  # Memory allocation failed

        stack.data = new_data
        stack.capacity = new_capacity

    # Push the value
    stack.data[stack.top] = value  # type: ignore
    stack.top += 1
    return 0


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void IntStack_cleanup(IntStack *stack) noexcept nogil:
    """Free the stack's data array (stack itself is on stack memory)"""
    if stack:
        if stack.data:
            free(<void*>(stack.data))
            stack.data = NULL
        stack.top = 0
        stack.capacity = 0


# CoordinateStack structure - dynamic array of Coordinate structs
cdef struct CoordinateStack:
    Coordinate *data
    int top
    int capacity


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef int CoordinateStack_init(CoordinateStack *stack, int initial_capacity) noexcept nogil:
    """Initialize a coordinate stack with initial capacity"""
    stack.data = <Coordinate*>malloc(<size_t>initial_capacity * sizeof(Coordinate))
    stack.top = 0
    stack.capacity = initial_capacity
    return 0


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef int CoordinateStack_push(CoordinateStack *stack, Coordinate value) noexcept nogil:
    """Push a coordinate onto the stack, expanding if necessary"""
    cdef int new_capacity
    cdef Coordinate *new_data

    # Check if we need to expand
    if stack.top >= stack.capacity:
        new_capacity = stack.capacity * 2
        new_data = <Coordinate*>realloc(<void*>stack.data,
                                        <size_t>new_capacity * sizeof(Coordinate))
        stack.data = new_data
        stack.capacity = new_capacity

    # Push the value
    stack.data[stack.top] = value  # type: ignore
    stack.top += 1
    return 0


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void CoordinateStack_cleanup(CoordinateStack *stack) noexcept nogil:
    """Free the coordinate stack's data array"""
    if stack:
        if stack.data:
            free(<void*>(stack.data))
            stack.data = NULL
        stack.top = 0
        stack.capacity = 0


cdef struct Polyomino:
    CoordinateStack mask
    int offset_i
    int offset_j


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void Polyomino_cleanup(Polyomino *polyomino) noexcept nogil:
    """Free the stack's data array (stack itself is on stack memory)"""
    if polyomino:
        CoordinateStack_cleanup(&(polyomino.mask))


cdef struct PolyominoStack:
    Polyomino *mo_data
    int top
    int capacity


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef int PolyominoStack_init(
    PolyominoStack *stack,
    int initial_capacity
) noexcept nogil:
    """Initialize an mask and offset pointer vector with initial capacity"""
    # if not stack:
    #     return -1
    
    stack.mo_data = <Polyomino*>malloc(<size_t>initial_capacity * sizeof(Polyomino))
    # if not stack.mo_data:
    #     return -1
    
    stack.top = 0
    stack.capacity = initial_capacity
    return 0


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef int PolyominoStack_push(
    PolyominoStack *stack,
    Polyomino value
) noexcept nogil:
    """Push a value onto the vector, expanding if necessary"""
    cdef int new_capacity
    cdef Polyomino *new_data
    
    # if not stack:
    #     return -1
    
    # Check if we need to expand
    if stack.top >= stack.capacity:
        new_capacity = stack.capacity * 2
        new_data = <Polyomino*>realloc(<void*>stack.mo_data,
                                            <size_t>new_capacity * sizeof(Polyomino))
        # if not new_data:
        #     return -1  # Memory allocation failed
        
        stack.mo_data = new_data
        stack.capacity = new_capacity
    
    # Push the value
    stack.mo_data[stack.top] = value  # type: ignore
    stack.top += 1
    return 0


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void PolyominoStack_cleanup(PolyominoStack *stack) noexcept nogil:
    """Free the stack's data array (stack itself is on stack memory)"""
    cdef int i
    if stack:
        if stack.mo_data:
            for i in range(stack.top):
                Polyomino_cleanup(&(stack.mo_data[i]))  # type: ignore
            free(<void*>stack.mo_data)
            stack.mo_data = NULL
        stack.top = 0
        stack.capacity = 0