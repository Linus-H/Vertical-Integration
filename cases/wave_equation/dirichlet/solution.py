import math

import numpy as np

import utils
from utils import Solution


# TODO: make solution adaptive to the domain size
class StandingWaveFixedEnd(Solution):
    def __init__(self, num_grid_points, dt, domain_size, c, coefficients):
        """
        :param coefficients: list of tuples (k,b_k), where k is the number of minimums/maximums of the corresponding sine-wave, and b_k is the coefficient of that sine-wave.
        :param num_grid_points: Number of samples in space-dimension.
        :param dt: Size of a time-step.
        :param c: Wave speed from wave equation.
        :param domain_size: is currently ignored and assumed to be 1.
        """
        super().__init__(self.solution, 0, dt)
        self._coefficients = coefficients
        self._c = c
        self._num_grid_points = num_grid_points
        self._axes = np.tile(np.linspace(0, 1, self._num_grid_points + 1)[:-1], (2, 1))
        self._state = utils.State(num_vars=2, dim_vars=self._num_grid_points, axes=self._axes)

    def solution(self, t, new_object=False):
        # get object to return
        if new_object:
            state = utils.State(num_vars=2, dim_vars=self._num_grid_points, axes=self._axes)
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
