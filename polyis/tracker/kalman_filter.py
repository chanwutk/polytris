from copy import deepcopy
import numpy as np
import numpy.typing as npt
from numpy import dot, zeros, eye


class KalmanFilter:
    def __init__(self, dim_x: int, dim_z: int) -> None:
        if dim_x < 1:
            raise ValueError('dim_x must be 1 or greater')
        if dim_z < 1:
            raise ValueError('dim_z must be 1 or greater')

        self.dim_x: int = dim_x
        self.dim_z: int = dim_z

        self.x: npt.NDArray[np.float64] = zeros((dim_x, 1))        # state
        self.P: npt.NDArray[np.float64] = eye(dim_x)               # uncertainty covariance
        self.Q: npt.NDArray[np.float64] = eye(dim_x)               # process uncertainty
        self.F: npt.NDArray[np.float64] = eye(dim_x)               # state transition matrix
        self.H: npt.NDArray[np.float64] = zeros((dim_z, dim_x))    # Measurement function
        self.R: npt.NDArray[np.float64] = eye(dim_z)               # state uncertainty

        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self.y: npt.NDArray[np.float64] = zeros((dim_z, 1))

        # identity matrix. Do not alter this.
        self._I: npt.NDArray[np.float64] = np.eye(dim_x)


    def predict(self) -> None:
        F: npt.NDArray[np.float64] = self.F
        Q: npt.NDArray[np.float64] = self.Q

        # x = Fx + Bu
        self.x = dot(F, self.x)

        # P = FPF' + Q
        self.P = dot(dot(F, self.P), F.T) + Q


    def update(self, z: npt.NDArray[np.float64]) -> None:
        R: npt.NDArray[np.float64] = self.R
        H: npt.NDArray[np.float64] = self.H

        # y = z - Hx
        # error (residual) between measurement and prediction
        self.y = z - dot(H, self.x)

        # common subexpression for speed
        PHT: npt.NDArray[np.float64] = dot(self.P, H.T)

        # S = HPH' + R
        # project system uncertainty into measurement space
        S: npt.NDArray[np.float64] = dot(H, PHT) + R
        SI: npt.NDArray[np.float64] = np.linalg.inv(S).astype(np.float64)
        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        K: npt.NDArray[np.float64] = dot(PHT, SI)

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self.x = self.x + dot(K, self.y)

        # P = (I-KH)P(I-KH)' + KRK'
        # This is more numerically stable
        # and works for non-optimal K vs the equation
        # P = (I-KH)P usually seen in the literature.

        I_KH: npt.NDArray[np.float64] = self._I - dot(K, H)
        self.P = dot(dot(I_KH, self.P), I_KH.T) + dot(dot(K, R), K.T)