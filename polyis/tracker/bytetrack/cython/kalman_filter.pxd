# cython: language_level=3

# Kalman Filter structure for ByteTrack (8-dimensional state space)
cdef struct KalmanFilter:
    # State vector (8 dimensional: x, y, a, h, vx, vy, va, vh)
    double x[8]
    # Covariance matrix (8x8)
    double P[8][8]
    # State transition matrix (8x8)
    double F[8][8]
    # Observation matrix (4x8)
    double H[4][8]
    # Process noise covariance (8x8)
    double Q[8][8]
    # Measurement noise covariance (4x4)
    double R[4][4]
    # Motion matrix for prediction (8x8)
    double motion_mat[8][8]
    # Update matrix for observation (4x8)
    double update_mat[4][8]
    # Standard deviation weights
    double std_weight_position
    double std_weight_velocity

# Kalman Filter functions
cdef void kf_init(KalmanFilter *kf) noexcept nogil
cdef void kf_predict(KalmanFilter *kf) noexcept nogil
cdef void kf_update(KalmanFilter *kf, double *measurement) noexcept nogil
cdef void kf_initiate(KalmanFilter *kf, double *measurement, double *mean_out, double *cov_out) noexcept nogil
