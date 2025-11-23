# Force compiling with Python 3 
# cython: language_level=3

import numpy as np
cimport numpy as cnp
cimport cython
from numpy cimport ndarray

# Global identity matrix for 7x7 (as Python object, not buffer)
I7 = np.eye(7, dtype=np.float64)


cdef public class KalmanFilter7x4 [object KalmanFilter7x4Object, type KalmanFilter7x4Type]:
    """
    Cython implementation of KalmanFilter7x4 for tracking.
    """
    cdef public object x  # state
    cdef public object P  # uncertainty covariance
    cdef public object Q  # process uncertainty
    cdef public object F  # state transition matrix
    cdef public object H  # Measurement function
    cdef public object R  # state uncertainty
    cdef public object y  # residual

    def __init__(self):
        # Initialize state vector (7x1)
        self.x = np.zeros((7, 1), dtype=np.float64)
        # Initialize uncertainty covariance (7x7)
        self.P = np.eye(7, dtype=np.float64)
        # Initialize process uncertainty (7x7)
        self.Q = np.eye(7, dtype=np.float64)
        # Initialize state transition matrix (7x7)
        self.F = np.eye(7, dtype=np.float64)
        # Initialize measurement function (4x7)
        self.H = np.zeros((4, 7), dtype=np.float64)
        # Initialize state uncertainty (4x4)
        self.R = np.eye(4, dtype=np.float64)
        # Initialize residual (4x1)
        self.y = np.zeros((4, 1), dtype=np.float64)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def predict(self):
        """
        Predict next state using state transition matrix.
        """
        # x = Fx + Bu (no control input, so just Fx)
        self.x = np.dot(self.F, self.x)
        # P = FPF' + Q
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def update(self, object z):
        """
        Update state with measurement z.
        
        Args:
            z: Measurement vector (4x1)
        """
        # y = z - Hx (residual between measurement and prediction)
        self.y = z - np.dot(self.H, self.x)
        
        # Common subexpression for speed: PHT = P * H'
        cdef cnp.ndarray[cnp.float64_t, ndim=2] PHT = np.dot(self.P, self.H.T)
        
        # S = HPH' + R (project system uncertainty into measurement space)
        cdef cnp.ndarray[cnp.float64_t, ndim=2] S = np.dot(self.H, PHT) + self.R
        cdef cnp.ndarray[cnp.float64_t, ndim=2] SI = np.linalg.inv(S).astype(np.float64)
        
        # K = PH'inv(S) (map system uncertainty into kalman gain)
        cdef cnp.ndarray[cnp.float64_t, ndim=2] K = np.dot(PHT, SI)
        
        # x = x + Ky (predict new x with residual scaled by the kalman gain)
        self.x = self.x + np.dot(K, self.y)
        
        # P = (I-KH)P(I-KH)' + KRK'
        # This is more numerically stable and works for non-optimal K
        cdef cnp.ndarray[cnp.float64_t, ndim=2] I_KH = I7 - np.dot(K, self.H)
        self.P = np.dot(np.dot(I_KH, self.P), I_KH.T) + np.dot(np.dot(K, self.R), K.T)
