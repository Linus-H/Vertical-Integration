import math
import numpy as np

import utils
from utils import Solution


class CaseSolution(Solution):
    def __init__(self, dt, t0, start_value, case_num):
        super().__init__(t0, dt)
        self.start_value = start_value
        self.axes = np.zeros((1, 1))
        self.state = utils.State(num_vars=1, dim_vars=1, axes=self.axes)
        self.case_num = case_num

    def solution(self, t, new_object=False):
        # get object to return
        if new_object:
            state = utils.State(num_vars=1, dim_vars=1, axes=self.axes)
            state_vars = state.get_state_vars()
        else:
            state = self.state
            state_vars = state.get_state_vars()
            state_vars.fill(0.0)

        cases = {
            0: lambda t: self.start_value * math.exp(0.5 * (t - math.sin(t - self.t0) * math.cos(t - self.t0))),
            1: lambda t: self.start_value * math.exp(-3 * (t - self.t0)),
            2: lambda t: self.start_value + 0.5 * math.log(t * t + 1),
            3: lambda t: self.start_value * math.e * math.exp(-math.cos(t - self.t0))
        }

        state_vars[0] = cases[self.case_num](t)

        return state
