import numpy as np

import utils
from utils import Solution


def exp_curve(a, g):
    T = 273.15
    R = 8.314
    return np.exp(g * a / (R * T))


class StationarySolution(Solution):
    def __init__(self, num_grid_points, dt, domain_size, g):
        super().__init__(self.solution, 0, dt)
        self.num_grid_points = num_grid_points
        self.domain_size = domain_size
        self.axes = np.tile(np.linspace(0, domain_size, self.num_grid_points + 1)[:-1], (2, 1))

        self.axes[0] += 0.5 * domain_size / num_grid_points

        self.state = utils.State(num_vars=2, dim_vars=self.num_grid_points, axes=self.axes)
        self.g = g

    def solution(self, t, new_object=False):
        # get object to return
        if new_object:
            state = utils.State(num_vars=2, dim_vars=self.num_grid_points, axes=self.axes)
            state_vars = state.get_state_vars()
        else:
            state = self.state
            state_vars = state.get_state_vars()
            state_vars.fill(0.0)

        state_vars[0] = np.log(exp_curve(self.axes[0], self.g))
        state_vars[1] *= 0

        return state
