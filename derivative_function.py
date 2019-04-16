import numpy as np
import operators as op


class WaveFunctionWraparound:
    def __init__(self, delta_x, c):
        self.delta_x = delta_x
        self.c = c

    def __call__(self, z):
        du = z[1]
        u = z[0]
        dv = op.laplace(u, self.delta_x) * self.c ** 2

        # generate output
        dz = np.stack((du, dv), axis=-1)
        dz = dz.transpose()
        return dz


class WaveFunctionFixedEnd:
    def __init__(self, delta_x, c):
        self.delta_x = delta_x
        self.c = c

    def __call__(self, z):
        du = z[1]
        u = z[0]

        dv = op.laplace(u, self.delta_x) * self.c ** 2
        dv[0] = dv[-1] = 0

        # generate output
        dz = np.stack((du, dv), axis=-1)
        dz = dz.transpose()
        return dz


class WaveFunctionLooseEnd():
    def __init__(self, delta_x, c):
        self.delta_x = delta_x
        self.c = c

    def __call__(self, z):
        du = z[1]
        u = z[0]

        dv = op.laplace(u, self.delta_x) * self.c ** 2
        dv[0] = dv[-1] = 0
        dv[0] = 2 * self.c * self.c * (u[1] - u[0]) / (self.delta_x ** 2)
        dv[-1] = 2 * self.c * self.c * (u[-2] - u[-1]) / (self.delta_x ** 2)

        # generate output
        dz = np.stack((du, dv), axis=-1)
        dz = dz.transpose()
        return dz
