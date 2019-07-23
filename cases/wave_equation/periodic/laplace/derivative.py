import numpy as np

import operators.laplace
import operators.derivative
from utils import TimeDerivative


class PeriodicWaveLaplace(TimeDerivative):
    def __init__(self, delta_x, c):
        self.delta_x = delta_x
        self.c = c

    def __call__(self, z, t):
        u = z[0]
        v = z[1]

        du = v
        dv = operators.laplace.diff_n2_e4(u, self.delta_x) * self.c ** 2

        # generate output
        dz = np.stack((du, dv), axis=-1)
        dz = dz.transpose()
        return dz

    def __str__(self):
        return "wave_wrap_around_laplace_time_derivative"
