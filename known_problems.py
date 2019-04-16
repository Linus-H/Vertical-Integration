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


class StandingWaveFixedEnd(Problem):
    def __init__(self, c, coefficients, num_grid_points, dt):
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
        self._axes = np.tile(np.linspace(0, 1, self._num_grid_points), (2, 1))
        self._state = state = helpers.State(num_vars=2, dim_vars=self._num_grid_points, axes=self._axes)

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
