from matplotlib import pyplot as plt
import numpy as np


class State:
    def __init__(self, num_vars, dim_vars, axis, names=None):
        if names is None:
            names = [""] * num_vars
        self.names = names
        self.axis = axis
        self.vars = np.ndarray((num_vars, dim_vars))

    def get_vars(self):
        return self.vars

    def get_names(self):
        return self.names

    def get_axis(self):
        return self.axis

    def set_vars(self, vars):
        self.vars = vars


class Integrator:
    def __init__(self, state, stepper):
        self.state = state
        self._stepper = stepper

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        self.state.set_vars(self._stepper(self.state))
        return self.state


class StateVisualizer:
    def __init__(self, n_rows, n_cols):
        self.fig = plt.figure()
        self.axes = []
        for i in range(n_rows * n_cols):
            self.axes.append(self.fig.add_subplot(n_rows, n_cols, i + 1))

    def display(self, plot_index, state, state_index, y_min=None, y_max=None):
        ax = self.axes[plot_index - 1]
        ax.clear()
        data = state.get_vars()[state_index]
        name = state.get_names()[state_index]
        axis = state.get_axis()
        ax.plot(axis, data)
        if y_min is not None and y_max is not None:
            ax.set_ylim((y_min, y_max))
        ax.set_ylabel(name)
        plt.draw()
        plt.pause(1e-20)
