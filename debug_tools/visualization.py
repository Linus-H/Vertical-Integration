from matplotlib import pyplot as plt
import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
import numpy as np


class WindowManager:
    def __init__(self, n_rows, n_cols):
        self.fig = plt.figure()
        self.axes = []
        for i in range(n_rows * n_cols):
            self.axes.append(self.fig.add_subplot(n_rows, n_cols, i + 1))

    def get_axis(self, plot_index):
        return self.axes[plot_index - 1]

    def sync_axes(self, plot_index_list):
        grouper = self.axes[plot_index_list[0]].get_shared_x_axes()
        for i in plot_index_list[1:]:
            grouper.join(self.axes[plot_index_list[0]], self.axes[plot_index_list[i]])

    def show(self):
        self.fig.show()

    def display_state(self, plot_index, state, state_index, y_min=None, y_max=None, clear_axis=True, operations=None):
        """
        :param plot_index: index of the plot to be plotted in (index starts at 1)
        :param state: state whose variable is to be displayed.
        :param state_index: index of the state variable to be displayed (index starts at 0)
        :param y_min: lower bound of y-axis to be displayed.
        :param y_max: upper bound of y-axis to be displayed.
        :param clear_axis: whether the axis should be cleared before the new plot is drawn.
        :param operations: tuple of operations to apply to the axis and state variable vector before being displayed.
        """
        # select and clear the graph to plot in
        ax = self.axes[plot_index - 1]
        if clear_axis:
            ax.clear()

        # get data to plot
        state_var = state.get_state_vars()[state_index]
        axis = state.get_axes()[state_index]

        if operations is not None:
            axis_op, state_op = operations
            axis = axis_op(axis)
            state_var = state_op(state_var)
        x_label, y_label = state.get_names()[state_index]

        # plot the data
        ax.plot(axis, state_var, "k")

        # label and scale the axes
        if y_min is not None and y_max is not None:
            ax.set_ylim((y_min, y_max))
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)

        # make sure updated graph is actually plotted
        ax.grid()
        plt.draw()
        plt.pause(1e-20)  # necessary for matplotlib to display changes without pausing the program

    def display_error(self, plot_index, error_tracker, double_log, line_name="", clear_axis=False, linestyle=None):
        """
        :param plot_index: index of the plot to be plotted in (index starts at 1)
        :param error_tracker: the error-tracker to be displayed.
        :param double_log: whether to plot the line double-logarithmically.
        :param line_name: name of the line (for legend).
        :param clear_axis: whether the axis should be cleared before the new plot is drawn.
        """
        # select and clear the graph to plot in
        ax = self.axes[plot_index - 1]

        if clear_axis:
            ax.clear()

        # plot the data
        if double_log:
            if linestyle is None:
                ax.loglog(error_tracker.labels, error_tracker.abs_error, label=line_name)
            else:
                ax.loglog(error_tracker.labels, error_tracker.abs_error, label=line_name, c="k", ls=linestyle)
        else:
            if linestyle is None:
                ax.plot(error_tracker.labels, error_tracker.abs_error, label=line_name)
            else:
                ax.plot(error_tracker.labels, error_tracker.abs_error, label=line_name, c="k", ls=linestyle)

        # label the axes
        ax.set_ylabel(error_tracker.error_name)
        ax.set_xlabel(error_tracker.label_name)

        # make sure updated graph is actually plotted
        ax.grid()
        ax.legend()
        plt.draw()
        plt.pause(1e-20)  # necessary for matplotlib to display changes without pausing the program

    def draw_loglog_oder_line(self, plot_index, x_start, y_start, width, order):
        """
        Plots a double logarithmic line to visualize a certain order.
        :param plot_index: index of the plot to be plotted in (index starts at 1).
        :param x_start: x-coordinate of where the order-line should start.
        :param y_start: y-coordinate of where the order-line should start.
        :param width: width of the order-line (i.e. 10^width).
        :param order: order of the order line.
        :return:
        """
        ax = self.axes[plot_index - 1]

        x = [x_start, x_start * 10 ** width]
        y = [y_start, y_start / 10 ** (width * order)]

        ax.loglog(x, y, 'k-.')
        ax.text(x[-1], y[-1], "$\mathcal{O}" + "(\Delta x^{" + "{}".format(abs(order)) + "})$")

        # make sure updated graph is actually plotted
        ax.grid()
        ax.legend()
        plt.draw()
        plt.pause(1e-20)  # necessary for matplotlib to display changes without pausing the program
