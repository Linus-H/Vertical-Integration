import math
import numpy as np


class TimeDerivative:
    def __call__(self, z, t):
        y = z[0]

        dy = y * (math.sin(t) ** 2)
        # dy = y * math.sin(t)
        # dy = -3 * y
        #dy = t/(t*t+1)

        # generate output
        dz = np.resize(dy, (1, 1))
        return dz
