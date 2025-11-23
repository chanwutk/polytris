cdef struct KalmanFilter:
    double x[7]
    double P[7][7]
    double Q[7][7]
    double F[7][7]
    double H[4][7]
    double R[4][4]

cdef void kf_init(KalmanFilter *kf)
cdef void kf_predict(KalmanFilter *kf)
cdef void kf_update(KalmanFilter *kf, double *z)
