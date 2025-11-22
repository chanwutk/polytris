from copy import deepcopy
import numpy as np
import numpy.typing as npt
from numpy import dot, zeros, eye


def reshape_z(z: npt.NDArray[np.float64] | npt.NDArray[np.integer], dim_z: int, ndim: int) -> npt.NDArray[np.float64] | float:
    """ ensure z is a (dim_z, 1) shaped vector"""
    z_arr: npt.NDArray[np.float64] = np.atleast_2d(z).astype(np.float64)
    if z_arr.shape[1] == dim_z:
        z_arr = z_arr.T

    if z_arr.shape != (dim_z, 1):
        raise ValueError('z must be convertible to shape ({}, 1)'.format(dim_z))

    if ndim == 1:
        z_result: npt.NDArray[np.float64] = z_arr[:, 0]
        return z_result

    if ndim == 0:
        z_result_scalar: float = float(z_arr[0, 0])
        return z_result_scalar

    return z_arr


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
        self.M: npt.NDArray[np.float64] = np.zeros((dim_z, dim_z)) # process-measurement cross correlation
        self.z: npt.NDArray[np.float64] = np.array([[None]*self.dim_z]).T

        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self.K: npt.NDArray[np.float64] = np.zeros((dim_x, dim_z)) # kalman gain
        self.y: npt.NDArray[np.float64] = zeros((dim_z, 1))
        self.S: npt.NDArray[np.float64] = np.zeros((dim_z, dim_z)) # system uncertainty
        self.SI: npt.NDArray[np.float64] = np.zeros((dim_z, dim_z)) # inverse system uncertainty

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
        z_reshaped: npt.NDArray[np.float64] | float = reshape_z(z, self.dim_z, self.x.ndim)
        # Ensure z_reshaped is an array for subsequent operations
        if isinstance(z_reshaped, np.ndarray):
            z_arr: npt.NDArray[np.float64] = z_reshaped
        else:
            z_arr = np.array([[z_reshaped]])

        R: npt.NDArray[np.float64] = self.R
        H: npt.NDArray[np.float64] = self.H

        # y = z - Hx
        # error (residual) between measurement and prediction
        self.y = z_arr - dot(H, self.x)

        # common subexpression for speed
        PHT: npt.NDArray[np.float64] = dot(self.P, H.T)

        # S = HPH' + R
        # project system uncertainty into measurement space
        self.S = dot(H, PHT) + R
        self.SI = np.linalg.inv(self.S).astype(np.float64)
        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        self.K = dot(PHT, self.SI)

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self.x = self.x + dot(self.K, self.y)

        # P = (I-KH)P(I-KH)' + KRK'
        # This is more numerically stable
        # and works for non-optimal K vs the equation
        # P = (I-KH)P usually seen in the literature.

        I_KH: npt.NDArray[np.float64] = self._I - dot(self.K, H)
        self.P = dot(dot(I_KH, self.P), I_KH.T) + dot(dot(self.K, R), self.K.T)

        # save measurement and posterior state
        self.z = z_arr