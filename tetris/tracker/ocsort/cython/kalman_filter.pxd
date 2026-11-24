# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

cdef struct KalmanFilter:
    double x[7]
    double P[7][7]
    double Q[7][7]
    double F[7][7]
    double H[4][7]
    double R[4][4]
    # For freeze/unfreeze functionality
    double x_saved[7]
    double P_saved[7][7]
    int has_saved
    int observed

cdef void kf_init(KalmanFilter *kf) noexcept nogil
cdef void kf_predict(KalmanFilter *kf) noexcept nogil
cdef void kf_update(KalmanFilter *kf, double *z) noexcept nogil
cdef void kf_freeze(KalmanFilter *kf) noexcept nogil
cdef void kf_unfreeze(KalmanFilter *kf, double *history_obs, int history_len, int max_history) noexcept nogil

