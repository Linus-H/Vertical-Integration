import numpy as np

from operators.derivative import diff_n1_e2


class TimeDerivative:
    def __init__(self, delta_x, c):
        self.delta_x = delta_x
        self.c = c

    def __call__(self, z):
        u = z[0]
        v = z[1]
        du = diff_n1_e2(v, self.delta_x)
        dv = self.c * self.c * diff_n1_e2(u, self.delta_x)

        # generate output
        dz = np.stack((du, dv), axis=-1)
        dz = dz.transpose()
        return dz
