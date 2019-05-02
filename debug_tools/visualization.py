from matplotlib import pyplot as plt


class WindowManager:
    def __init__(self, n_rows, n_cols):
        self.fig = plt.figure()
        self.axes = []
        for i in range(n_rows * n_cols):
            self.axes.append(self.fig.add_subplot(n_rows, n_cols, i + 1))

    def get_axis(self, plot_index):
        return self.axes[plot_index - 1]

    def display_state(self, plot_index, state, state_index, y_min=None, y_max=None):
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
        ax.grid()
        plt.draw()
        plt.pause(1e-20)  # necessary for matplotlib to display changes without pausing the program

    def display_error(self, plot_index, error_tracker, double_log, line_name="", clear_axis=False):
        # select and clear the graph to plot in
        ax = self.axes[plot_index - 1]

        if clear_axis:
            ax.clear()

        # plot the data
        if double_log:
            ax.loglog(error_tracker.labels, error_tracker.abs_error, label=line_name)
        else:
            ax.plot(error_tracker.labels, error_tracker.abs_error, label=line_name)

        # label the axes
        ax.set_ylabel(error_tracker.error_name)
        ax.set_xlabel(error_tracker.label_name)

        # make sure updated graph is actually plotted
        ax.grid()
        ax.legend()
        plt.draw()
        plt.pause(1e-20)  # necessary for matplotlib to display changes without pausing the program

    def draw_loglog_oder_line(self, plot_index, x_start, y_start, width, order):
        ax = self.axes[plot_index - 1]

        x = [x_start, x_start * 10 ** width]
        y = [y_start, y_start / 10 ** (width * order)]

        ax.loglog(x, y, 'k--')
        ax.text(x[-1], y[-1], "O(dx^{})".format(order))

        # make sure updated graph is actually plotted
        ax.grid()
        ax.legend()
        plt.draw()
        plt.pause(1e-20)  # necessary for matplotlib to display changes without pausing the program
