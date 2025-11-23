# Force compiling with Python 3 
# cython: language_level=3

import numpy as np
cimport numpy as cnp
cimport cython
from libc.stdlib cimport malloc, free


cdef extern from "lapjv.h" nogil:
    ctypedef signed int int_t
    ctypedef unsigned int uint_t

    int lapjv_internal(const uint_t n, double *cost[], int_t *x, int_t *y)


@cython.boundscheck(False)
@cython.wraparound(False)
def lapjv(cnp.ndarray[cnp.float64_t, ndim=2] cost not None):
    """Solve linear assignment problem using Jonker-Volgenant algorithm.

    Parameters
    ----------
    cost: (N,N) ndarray
        Cost matrix. Entry `cost[i, j]` is the cost of assigning row `i` to
        column `j`.

    Returns
    -------
    x: (N,) ndarray
        Assignment. `x[i]` specifies the column to which row `i` is assigned.
    y: (N,) ndarray
        Assignment. `y[j]` specifies the row to which column `j` is assigned.

    Notes
    -----
    For non-square matrices, there will be unmatched rows, columns in the solution `x`, `y`.
    All such entries are set to -1.
    """
    cdef cnp.ndarray[cnp.double_t, ndim=2, mode='c'] cost_c = \
        np.ascontiguousarray(cost, dtype=np.double)
    cdef cnp.ndarray[cnp.double_t, ndim=2, mode='c'] cost_c_extended
    cdef uint_t n_rows = cost_c.shape[0]
    cdef uint_t n_cols = cost_c.shape[1]
    cdef uint_t n = 0
    if n_rows == n_cols:
        n = n_rows

    n = max(n_rows, n_cols)
    cost_c_extended = np.zeros((n, n), dtype=np.double)
    cost_c_extended[:n_rows, :n_cols] = cost_c
    cost_c = cost_c_extended

    cdef double **cost_ptr
    cost_ptr = <double **> malloc(n * sizeof(double *))
    cdef int i
    for i in range(n):
        cost_ptr[i] = &cost_c[i, 0]

    cdef cnp.ndarray[int_t, ndim=1, mode='c'] x_c = \
        np.empty((n,), dtype=np.int32)
    cdef cnp.ndarray[int_t, ndim=1, mode='c'] y_c = \
        np.empty((n,), dtype=np.int32)

    cdef int ret = lapjv_internal(n, cost_ptr, &x_c[0], &y_c[0])
    free(cost_ptr)
    if ret != 0:
        if ret == -1:
            raise MemoryError('Out of memory.')
        raise RuntimeError('Unknown error (lapjv_internal returned %d).' % ret)

    x_c[x_c >= n_cols] = -1
    y_c[y_c >= n_rows] = -1
    x_c = x_c[:n_rows]
    y_c = y_c[:n_cols]

    return x_c, y_c

