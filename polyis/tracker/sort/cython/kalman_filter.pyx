# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

from libc.string cimport memset, memcpy
from libc.math cimport fabs
from polyis.tracker.sort.cython.kalman_filter cimport KalmanFilter
import cython

# Global identity matrix
cdef double I7[49]

# Helper functions for matrix operations
@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void matmul(double *A, double *B, double *C, int m, int n, int k) noexcept nogil:
    """C = A * B where A is (m x n), B is (n x k)"""
    cdef int i, j, l
    cdef double val
    for i in range(m):
        for j in range(k):
            val = 0.0
            for l in range(n):
                val += A[i * n + l] * B[l * k + j]
            C[i * k + j] = val

@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void mat_transpose(double *A, double *B, int m, int n) noexcept nogil:
    """B = A^T where A is (m x n)"""
    cdef int i, j
    for i in range(m):
        for j in range(n):
            B[j * m + i] = A[i * n + j]

@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void mat_add(double *A, double *B, double *C, int size) noexcept nogil:
    """C = A + B"""
    cdef int i
    for i in range(size):
        C[i] = A[i] + B[i]

@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void mat_sub(double *A, double *B, double *C, int size) noexcept nogil:
    """C = A - B"""
    cdef int i
    for i in range(size):
        C[i] = A[i] - B[i]

@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void mat_eye(double *A, int n) noexcept nogil:
    """Set A to identity matrix of size n x n"""
    memset(A, 0, n * n * sizeof(double))
    cdef int i
    for i in range(n):
        A[i * n + i] = 1.0

@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void mat_zeros(double *A, int size) noexcept nogil:
    """Set A to zeros"""
    memset(A, 0, size * sizeof(double))

@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef int mat_inv_4x4(double *A, double *inv) noexcept nogil:
    """
    Invert 4x4 matrix A using Gauss-Jordan elimination.
    Returns 1 on success, 0 on failure (singular).
    """
    cdef double mat[16]
    memcpy(mat, A, 16 * sizeof(double))
    mat_eye(inv, 4)
    
    cdef int i, j, k, pivot_idx
    cdef double pivot, temp, factor
    
    for i in range(4):
        pivot = mat[i*4 + i]
        pivot_idx = i
        for j in range(i + 1, 4):
            if fabs(mat[j*4 + i]) > fabs(pivot):
                pivot = mat[j*4 + i]
                pivot_idx = j
        
        if fabs(pivot) < 1e-9:
            return 0 # Singular
            
        if pivot_idx != i:
            for k in range(4):
                temp = mat[i*4 + k]
                mat[i*4 + k] = mat[pivot_idx*4 + k]
                mat[pivot_idx*4 + k] = temp
                
                temp = inv[i*4 + k]
                inv[i*4 + k] = inv[pivot_idx*4 + k]
                inv[pivot_idx*4 + k] = temp
        
        pivot = mat[i*4 + i]
        for k in range(4):
            mat[i*4 + k] /= pivot
            inv[i*4 + k] /= pivot
            
        for j in range(4):
            if i != j:
                factor = mat[j*4 + i]
                for k in range(4):
                    mat[j*4 + k] -= factor * mat[i*4 + k]
                    inv[j*4 + k] -= factor * inv[i*4 + k]
    return 1

# Initialize global I7
mat_eye(I7, 7)

# Implementation of public functions declared in .pxd

@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void kf_init(KalmanFilter *kf) noexcept nogil:
    # Initialize state vector (7x1)
    mat_zeros(kf.x, 7)
    # Initialize uncertainty covariance (7x7)
    mat_eye(<double*>kf.P, 7)
    # Initialize process uncertainty (7x7)
    mat_eye(<double*>kf.Q, 7)
    # Initialize state transition matrix (7x7)
    mat_eye(<double*>kf.F, 7)
    # Initialize measurement function (4x7)
    mat_zeros(<double*>kf.H, 28)
    # Initialize state uncertainty (4x4)
    mat_eye(<double*>kf.R, 4)

@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void kf_predict(KalmanFilter *kf) noexcept nogil:
    # x = F * x
    cdef double new_x[7]
    matmul(<double*>kf.F, kf.x, new_x, 7, 7, 1)
    memcpy(kf.x, new_x, 7 * sizeof(double))
    
    # P = F * P * F^T + Q
    cdef double FP[49] # 7x7
    matmul(<double*>kf.F, <double*>kf.P, FP, 7, 7, 7)
    
    cdef double FT[49]
    mat_transpose(<double*>kf.F, FT, 7, 7)
    
    cdef double FPFt[49]
    matmul(FP, FT, FPFt, 7, 7, 7)
    
    mat_add(FPFt, <double*>kf.Q, <double*>kf.P, 49)

@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void kf_update(KalmanFilter *kf, double *z) noexcept nogil:
    # y = z - H * x
    cdef double Hx[4]
    matmul(<double*>kf.H, kf.x, Hx, 4, 7, 1)
    
    cdef double y[4]
    mat_sub(z, Hx, y, 4)
    
    # S = H * P * H^T + R
    cdef double HT[28] # 7x4
    mat_transpose(<double*>kf.H, HT, 4, 7)
    
    cdef double PHt[28] # 7x7 * 7x4 -> 7x4
    matmul(<double*>kf.P, HT, PHt, 7, 7, 4)
    
    cdef double HPHt[16] # 4x7 * 7x4 -> 4x4
    matmul(<double*>kf.H, PHt, HPHt, 4, 7, 4)
    
    cdef double S[16]
    mat_add(HPHt, <double*>kf.R, S, 16)
    
    # SI = inv(S)
    cdef double SI[16]
    if mat_inv_4x4(S, SI) == 0:
        return 
        
    # K = P * H^T * SI = PHt * SI
    cdef double K[28] # 7x4 * 4x4 -> 7x4
    matmul(PHt, SI, K, 7, 4, 4)
    
    # x = x + K * y
    cdef double Ky[7] # 7x4 * 4x1 -> 7x1
    matmul(K, y, Ky, 7, 4, 1)
    
    cdef double new_x[7]
    mat_add(kf.x, Ky, new_x, 7)
    memcpy(kf.x, new_x, 7 * sizeof(double))
    
    # P = (I - K * H) * P * (I - K * H)^T + K * R * K^T
    # I_KH = I - K * H
    cdef double KH[49] # 7x4 * 4x7 -> 7x7
    matmul(K, <double*>kf.H, KH, 7, 4, 7)
    
    # Use global I7
    cdef double I_KH[49]
    mat_sub(I7, KH, I_KH, 49)
    
    # Term 1: I_KH * P * I_KH^T
    cdef double I_KH_P[49]
    matmul(I_KH, <double*>kf.P, I_KH_P, 7, 7, 7)
    
    cdef double I_KH_T[49]
    mat_transpose(I_KH, I_KH_T, 7, 7)
    
    cdef double Term1[49]
    matmul(I_KH_P, I_KH_T, Term1, 7, 7, 7)
    
    # Term 2: K * R * K^T
    cdef double KR[28] # 7x4 * 4x4 -> 7x4
    matmul(K, <double*>kf.R, KR, 7, 4, 4)
    
    cdef double KT[28] # 4x7
    mat_transpose(K, KT, 7, 4)
    
    cdef double Term2[49] # 7x4 * 4x7 -> 7x7
    matmul(KR, KT, Term2, 7, 4, 7)
    
    # P = Term1 + Term2
    mat_add(Term1, Term2, <double*>kf.P, 49)
