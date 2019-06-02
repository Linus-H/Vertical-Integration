import math
import numpy as np


class TimeDerivative:
    def __init__(self, case_num):
        self.case_num = case_num

    def __call__(self, z, t):
        y = z[0]

        cases = {
            0: lambda y, t: y * (math.sin(t) ** 2),
            1: lambda y, t: -3 * y,
            2: lambda y, t: t / (t * t + 1),
            3: lambda y, t: y * math.sin(t)
        }

        dy = cases[self.case_num](y, t)

        # generate output
        dz = np.resize(dy, (1, 1))
        return dz
