import numpy as np

import operators.laplace


class TimeDerivativeLaplace:
    def __init__(self, delta_x, c):
        self.delta_x = delta_x
        self.c = c

    def __call__(self, z, t):
        du = z[1]
        u = z[0]

        dv = operators.laplace.diff_n2_e2(u, self.delta_x) * self.c ** 2
        dv[0] = 0

        # generate output
        dz = np.stack((du, dv), axis=-1)
        dz = dz.transpose()
        return dz