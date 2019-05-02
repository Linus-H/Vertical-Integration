import numpy as np


class State:
    def __init__(self, num_vars, dim_vars, axes, names=None):
        if names is None:
            names = [("", "")] * num_vars
        self._names = names
        self._axes = axes
        self._vars = np.zeros((num_vars, dim_vars))

    def get_state_vars(self):
        return self._vars

    def get_names(self):
        return self._names

    def get_axes(self):
        return self._axes

    def set_state_vars(self, vars):
        self._vars = vars


class Integrator:
    def __init__(self, state, stepper):
        """
        :param state: The state-object the integrator will operate on.
        :param stepper: The function the integrator has to call each iteration, passing the state-variable, which is implicitly modified.
        """
        self.state = state
        self._stepper = stepper

    def __iter__(self):
        return self

    def __next__(self):
        self._stepper(self.state.get_state_vars())
        return self.state


class Solution:
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
