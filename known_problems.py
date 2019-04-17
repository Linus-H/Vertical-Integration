import math
import numpy as np

import helpers


class Problem:
    def __init__(self, solution, dt):
        self._dt = dt
        self._solution = solution

    def get_initial_state(self):
        return self._solution(0, new_object=True)

    def __iter__(self):
        self.iteration_n = 0
        return self

    def __next__(self):
        self.iteration_n += 1
        return self._solution(self.iteration_n * self._dt)


class WraparoundSolution(Problem):
    def __init__(self, c, num_grid_points, dt, f, df_dx):
        super().__init__(self.solution, dt)
        self.c = c
        self.num_grid_points = num_grid_points
        self.f = f
        self.df_dx = df_dx
        self.axes = np.tile(np.linspace(0, 1, self.num_grid_points + 1)[:-1], (2, 1))
        self.state = helpers.State(num_vars=2, dim_vars=self.num_grid_points, axes=self.axes)

    def solution(self, t, new_object=False):
        # get object to return
        if new_object:
            state = helpers.State(num_vars=2, dim_vars=self.num_grid_points, axes=self.axes)
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
        state_vars[1][cond] = self.df_dx(self.axes[1][cond] - travelled_dist)

        # part of wave travelling right that has wrapped around
        cond = np.invert(cond)
        state_vars[0][cond] = self.f(self.axes[0][cond] + 1 - travelled_dist)
        state_vars[1][cond] = self.df_dx(self.axes[1][cond] + 1 - travelled_dist)

        # add wave travelling left
        cond = self.axes[0] <= 1 - travelled_dist
        state_vars[0][cond] += self.f(self.axes[0][cond] + travelled_dist)
        state_vars[1][cond] -= self.df_dx(self.axes[1][cond] + travelled_dist)

        # part of wave travelling left that has wrapped around
        cond = np.invert(cond)
        state_vars[0][cond] += self.f(self.axes[0][cond] - 1 + travelled_dist)
        state_vars[1][cond] -= self.df_dx(self.axes[1][cond] - 1 + travelled_dist)

        state_vars[1] *= self.c

        return state



class StandingWaveFixedEnd(Problem):
    def __init__(self, c, num_grid_points, dt, coefficients):
        """
        :param c: Wave speed from wave equation.
        :param coefficients: list of tuples (k,b_k), where k is the number of minimums/maximums of the corresponding sine-wave, and b_k is the coefficient of that sine-wave.
        :param num_grid_points: Number of samples in space-dimension.
        :param dt: Size of a time-step.
        """
        super().__init__(self.solution, dt)
        self._coefficients = coefficients
        self._c = c
        self._num_grid_points = num_grid_points
        self._axes = np.tile(np.linspace(0, 1, self._num_grid_points + 1)[:-1], (2, 1))
        self._state = helpers.State(num_vars=2, dim_vars=self._num_grid_points, axes=self._axes)

    def solution(self, t, new_object=False):
        # get object to return
        if new_object:
            state = helpers.State(num_vars=2, dim_vars=self._num_grid_points, axes=self._axes)
            state_vars = state.get_state_vars()
        else:
            state = self._state
            state_vars = state.get_state_vars()
            state_vars.fill(0.0)

        # put solution for time t into the object
        for k, b_k in self._coefficients:
            # u(x,t) of a standing wave
            state_vars[0] += b_k * math.cos(np.pi * k * self._c * t) * np.sin(np.pi * k * self._axes[0])
            # v = d u(x,t)/dt of the standing wave
            state_vars[1] += - b_k * np.pi * k * self._c * math.sin(np.pi * k * self._c * t) * np.sin(
                np.pi * k * self._axes[1])

        return state
