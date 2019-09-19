from abc import ABCMeta, abstractmethod

import numpy as np

data_path = "" # TODO: set appropriate path to folder in which to store large amounts of data (avoid SSDs)


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

    def __sub__(self, other):
        if isinstance(other, State) and self._vars.shape == other._vars.shape:
            num_vars, dim_vars = self._vars.shape
            ret_val = State(num_vars, dim_vars, self._axes, self._names)
            ret_val._vars = self._vars - other._vars
            return ret_val
        else:
            return None


class TimeDerivative(metaclass=ABCMeta):
    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class Integrator(metaclass=ABCMeta):
    def __init__(self, state, time_derivative, t0, dt):
        """
        :param state: The state-object the integrator will operate on.
        :param stepper: The function the integrator has to call each iteration, passing the state-variable, which is implicitly modified.
        """
        self.state = state
        self.time_derivative = time_derivative
        self.t0 = t0
        self.dt = dt

    def __iter__(self):
        self.n = 0
        return self

    @abstractmethod
    def stepper(self, state_vars, t):
        pass

    def __next__(self):
        self.stepper(self.state.get_state_vars(), self.t0 + self.n * self.dt)
        self.n += 1
        return self.state

    def get_operator_matrix(self, t):
        num_vars, dim_vars = self.state.get_state_vars().shape
        # A = np.eye(dim_vars * num_vars)
        A = np.zeros((dim_vars * num_vars, dim_vars * num_vars))
        A = np.reshape(A, (num_vars, dim_vars, -1))
        self.stepper(A, t)
        A = np.reshape(A, (dim_vars * num_vars, -1))
        return A


class Solution(metaclass=ABCMeta):
    def __init__(self, t0, dt):
        self.t0 = t0
        self._dt = dt

    @abstractmethod
    def solution(self, time, new_object=False):
        pass

    def get_initial_state(self):
        return self.solution(self.t0, new_object=True)

    def __iter__(self):
        self.iteration_n = 0
        return self

    def __next__(self):
        self.iteration_n += 1
        return self.solution(self.t0 + self.iteration_n * self._dt)
