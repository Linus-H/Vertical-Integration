import numpy as np

from operators.derivative import diff_forward_n1_e1
from operators.derivative import diff_backward_n1_e1
from operators.derivative import diff_n1_e2
from operators.derivative import diff_n1_e4
from operators.derivative import diff_fft
from utils import TimeDerivative


def apply_operation(u, v, c, delta_x):
    du = np.real(diff_fft(v,delta_x))#diff_n1_e4(v, delta_x)
    dv = c * c * np.real(diff_fft(u,delta_x))#diff_n1_e4(u, delta_x)
    return du, dv


class PeriodicWaveTimeDerivative(TimeDerivative):
    def __init__(self, delta_x, c):
        self.delta_x = delta_x
        self.c = c

    def __call__(self, z, t):
        u = z[0]
        v = z[1]

        du, dv = apply_operation(u, v, self.c, self.delta_x)

        # generate output
        dz = np.stack((du, dv), axis=-1)
        dz = dz.transpose()
        return dz

    def __str__(self):
        return "wave_wrap_around_time_derivative"


class TimeDerivativeMatrix(TimeDerivative):
    def __init__(self, delta_x, c):
        self.delta_x = delta_x
        self.c = c

    def __call__(self, z, t):
        num_vars, length, *_ = z.shape

        A = -np.eye(length)  # this imitates the u and v state vector as a matrix to extract the operation
        B = np.zeros((length, length))  # this is a blank padding matrix to get to a square matrix at the end

        du, dv = apply_operation(A, A, self.c, self.delta_x)

        # generate output
        du = np.concatenate((B, du), axis=1)
        dv = np.concatenate((dv, B), axis=1)

        dz = np.stack((du, dv), axis=0)

        return dz

    def __str__(self):
        return "wave_wrap_around_time_derivative_matrix"