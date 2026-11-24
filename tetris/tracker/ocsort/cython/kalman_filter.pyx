# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

from libc.string cimport memset, memcpy
from libc.math cimport fabs, sqrt
from polyis.tracker.ocsort.cython.kalman_filter cimport KalmanFilter
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
    # Initialize freeze/unfreeze state
    kf.has_saved = 0
    kf.observed = 0

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
    # Handle NULL z (no observation)
    if z is NULL:
        kf.observed = 0
        return
    
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
    
    # Mark as observed
    kf.observed = 1

@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void kf_freeze(KalmanFilter *kf) noexcept nogil:
    """Save the parameters before non-observation forward"""
    memcpy(kf.x_saved, kf.x, 7 * sizeof(double))
    memcpy(<double*>kf.P_saved, <double*>kf.P, 49 * sizeof(double))
    kf.has_saved = 1

@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void kf_unfreeze(KalmanFilter *kf, double *history_obs, int history_len, int max_history) noexcept nogil:
    """
    Unfreeze and perform online smoothing.
    history_obs is a flat array of observations, each observation is 4 doubles (x, y, s, r).
    Only valid observations (non-None) are stored, with None represented as all zeros.
    """
    if kf.has_saved == 0:
        return
    
    # Restore saved state
    memcpy(kf.x, kf.x_saved, 7 * sizeof(double))
    memcpy(<double*>kf.P, <double*>kf.P_saved, 49 * sizeof(double))
    
    # Find last two valid observations
    cdef int i, j
    cdef int index1 = -1
    cdef int index2 = -1
    cdef int valid_count = 0
    cdef int is_valid
    
    # Count valid observations and find last two
    for i in range(history_len - 1, -1, -1):
        # Check if observation is valid (not all zeros)
        is_valid = 0
        for j in range(4):
            if fabs(history_obs[i * 4 + j]) > 1e-9:
                is_valid = 1
                break
        
        if is_valid:
            valid_count += 1
            if index2 == -1:
                index2 = i
            elif index1 == -1:
                index1 = i
                break
    
    if index1 == -1 or index2 == -1:
        return
    
    # Declare all variables at the top
    cdef double x1, y1, s1, r1, w1, h1
    cdef double x2, y2, s2, r2, w2, h2
    cdef int time_gap
    cdef double dx, dy, dw, dh
    cdef double x, y, w, h, s, r
    cdef double new_box[4]
    
    # Extract box1 and box2
    x1 = history_obs[index1 * 4 + 0]
    y1 = history_obs[index1 * 4 + 1]
    s1 = history_obs[index1 * 4 + 2]
    r1 = history_obs[index1 * 4 + 3]
    w1 = sqrt(s1 * r1)
    h1 = s1 / w1
    
    x2 = history_obs[index2 * 4 + 0]
    y2 = history_obs[index2 * 4 + 1]
    s2 = history_obs[index2 * 4 + 2]
    r2 = history_obs[index2 * 4 + 3]
    w2 = sqrt(s2 * r2)
    h2 = s2 / w2
    
    time_gap = index2 - index1
    if time_gap <= 0:
        return
    
    dx = (x2 - x1) / time_gap
    dy = (y2 - y1) / time_gap
    dw = (w2 - w1) / time_gap
    dh = (h2 - h1) / time_gap
    
    # Generate virtual trajectory
    for i in range(time_gap):
        x = x1 + (i + 1) * dx
        y = y1 + (i + 1) * dy
        w = w1 + (i + 1) * dw
        h = h1 + (i + 1) * dh
        s = w * h
        r = w / h
        
        new_box[0] = x
        new_box[1] = y
        new_box[2] = s
        new_box[3] = r
        
        # Update with virtual observation
        kf_update(kf, new_box)
        
        # Predict for next step (except last)
        if i < time_gap - 1:
            kf_predict(kf)
    
    kf.has_saved = 0

