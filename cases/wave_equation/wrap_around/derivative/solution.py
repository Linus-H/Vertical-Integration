import numpy as np

import utils
from utils import Solution


class CaseSolution(Solution):
    def __init__(self, num_grid_points, dt, domain_size, c, f):
        super().__init__(self.solution, 0, dt)
        self.num_grid_points = num_grid_points
        self.domain_size = domain_size
        self.c = c
        self.f = f
        self.axes = np.tile(np.linspace(0, domain_size, self.num_grid_points + 1)[:-1], (2, 1))
        self.state = utils.State(num_vars=2, dim_vars=self.num_grid_points, axes=self.axes)

    def solution(self, t, new_object=False):
        # get object to return
        if new_object:
            state = utils.State(num_vars=2, dim_vars=self.num_grid_points, axes=self.axes)
            state_vars = state.get_state_vars()
        else:
            state = self.state
            state_vars = state.get_state_vars()
            state_vars.fill(0.0)

        travelled_dist = t * self.c
        travelled_dist = travelled_dist % self.domain_size  # get rid of wraparound

        # initialize with wave travelling right
        cond = travelled_dist <= self.axes[0]
        state_vars[0][cond] = self.f(self.axes[0][cond] - travelled_dist)
        state_vars[1][cond] = - self.f(self.axes[1][cond] - travelled_dist)

        # part of wave travelling right that has wrapped around
        cond = np.invert(cond)
        state_vars[0][cond] = self.f(self.axes[0][cond] + self.domain_size - travelled_dist)
        state_vars[1][cond] = - self.f(self.axes[1][cond] + self.domain_size - travelled_dist)

        # add wave travelling left
        cond = self.axes[0] <= self.domain_size - travelled_dist
        state_vars[0][cond] += self.f(self.axes[0][cond] + travelled_dist)
        state_vars[1][cond] += self.f(self.axes[1][cond] + travelled_dist)

        # part of wave travelling left that has wrapped around
        cond = np.invert(cond)
        state_vars[0][cond] += self.f(self.axes[0][cond] - self.domain_size + travelled_dist)
        state_vars[1][cond] += self.f(self.axes[1][cond] - self.domain_size + travelled_dist)

        state_vars[1] *= self.c

        return state
