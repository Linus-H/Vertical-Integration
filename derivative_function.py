import numpy as np


class WaveFunctionWraparound():
    def __init__(self, delta_x, c):
        self.delta_x = delta_x
        self.c = c

    def __call__(self, z):
        du = z[1]
        u = z[0]
        a = np.roll(u, 1) + np.roll(u, -1)
        b = 2 * u
        dv = self.c * self.c * (a - b) / (self.delta_x ** 2)
        dz = np.stack((du, dv), axis=-1)
        dz = dz.transpose()

        return dz


class WaveFunctionFixedEnd():
    def __init__(self, delta_x, c):
        self.delta_x = delta_x
        self.c = c

    def __call__(self, z):
        du = z[1]

        u = z[0]

        a = np.roll(u, 1) + np.roll(u, -1)
        b = 2 * u

        dv = self.c * self.c * (a - b) / (self.delta_x ** 2)
        dv[0] = dv[-1] = 0

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

        a = np.roll(u, 1) + np.roll(u, -1)
        b = 2 * u

        dv = self.c * self.c * (a - b) / (self.delta_x ** 2)
        dv[0] = dv[-1] = 0
        dv[0] = 2 * self.c * self.c * (dv[1] - dv[0]) / (self.delta_x ** 2)
        dv[-1] = 2 * self.c * self.c * (dv[-2] - dv[-1]) / (self.delta_x ** 2)

        dz = np.stack((du, dv), axis=-1)
        dz = dz.transpose()

        return dz
