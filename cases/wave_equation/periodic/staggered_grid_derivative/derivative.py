import numpy as np

import operators.derivative
from utils import TimeDerivative


class PeriodicWaveStaggered(TimeDerivative):
    def __init__(self, delta_x, c):
        self.delta_x = delta_x
        self.c = c

    def __call__(self, z, t):
        u = z[0]
        v = z[1]
        du = operators.derivative.diff_backward_n1_e1(v, self.delta_x)
        dv = self.c * self.c * operators.derivative.diff_forward_n1_e1(u, self.delta_x)

        # generate output
        dz = np.stack((du, dv), axis=-1)
        dz = dz.transpose()
        return dz

    def __str__(self):
        return "wave_wrap_around_staggered_time_derivative"