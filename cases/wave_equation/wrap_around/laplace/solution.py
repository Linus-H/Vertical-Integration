import numpy as np

import utils
from utils import Solution


class CaseSolution(Solution):
    def __init__(self, c, num_grid_points, dt, f, df_dx):
        super().__init__(self.solution, dt)
        self.c = c
        self.num_grid_points = num_grid_points
        self.f = f
        self.df_dx = df_dx
        self.axes = np.tile(np.linspace(0, 1, self.num_grid_points + 1)[:-1], (2, 1))
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
        travelled_dist = travelled_dist % 1  # get rid of wraparound

        # initialize with wave travelling right
        cond = travelled_dist <= self.axes[0]
        state_vars[0][cond] = self.f(self.axes[0][cond] - travelled_dist)
        state_vars[1][cond] = - self.df_dx(self.axes[1][cond] - travelled_dist)

        # part of wave travelling right that has wrapped around
        cond = np.invert(cond)
        state_vars[0][cond] = self.f(self.axes[0][cond] + 1 - travelled_dist)
        state_vars[1][cond] = - self.df_dx(self.axes[1][cond] + 1 - travelled_dist)

        # add wave travelling left
        cond = self.axes[0] <= 1 - travelled_dist
        state_vars[0][cond] += self.f(self.axes[0][cond] + travelled_dist)
        state_vars[1][cond] += self.df_dx(self.axes[1][cond] + travelled_dist)

        # part of wave travelling left that has wrapped around
        cond = np.invert(cond)
        state_vars[0][cond] += self.f(self.axes[0][cond] - 1 + travelled_dist)
        state_vars[1][cond] += self.df_dx(self.axes[1][cond] - 1 + travelled_dist)

        state_vars[1] *= self.c

        return state
