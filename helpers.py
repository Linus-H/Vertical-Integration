from matplotlib import pyplot as plt
import numpy as np


class State:
    def __init__(self, num_vars, dim_vars, axes, names=None):
        if names is None:
            names = [("", "")] * num_vars
        self._names = names
        self._axes = axes
        self._vars = np.ndarray((num_vars, dim_vars))

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
        self.state = state
        self._stepper = stepper

    def __iter__(self):
        return self

    def __next__(self):
        self._stepper(self.state.get_state_vars())
        return self.state


class StateVisualizer:
    def __init__(self, n_rows, n_cols):
        self.fig = plt.figure()
        self.axes = []
        for i in range(n_rows * n_cols):
            self.axes.append(self.fig.add_subplot(n_rows, n_cols, i + 1))

    def display(self, plot_index, state, state_index, y_min=None, y_max=None):
        # select and clear the graph to plot in
        ax = self.axes[plot_index - 1]
        ax.clear()

        # get data to plot
        state_var = state.get_state_vars()[state_index]
        x_label, y_label = state.get_names()[state_index]
        axis = state.get_axes()[state_index]

        # plot the data
        ax.plot(axis, state_var)

        # label and scale the axes
        if y_min is not None and y_max is not None:
            ax.set_ylim((y_min, y_max))
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)

        # make sure updated graph is actually plotted
        plt.draw()
        plt.pause(1e-20)  # necessary for matplotlib to display changes without pausing the program
