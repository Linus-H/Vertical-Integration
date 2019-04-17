import math
import numpy as np

from matplotlib import pyplot as plt

import operators
import debugging


def draw_oder_line(x_start, y_start, width, order):
    x = [x_start, x_start * 10 ** width]
    y = [y_start, y_start / 10 ** (width * order)]
    plt.loglog(x, y, 'k--')
    plt.text(x[-1], y[-1], "O(dx^{})".format(order))


# [operators.derivative_O1, operators.derivative_O2, operators.derivative_O4], ["Derivative Order: 1", "Derivative Order: 2", "Derivative Order: 4"]
# [operators.laplace_O2, operators.laplace_O4],["Laplace Order: 2", "Laplace Order: 4"]
for op, name in zip([operators.laplace_O2, operators.laplace_O4],["Laplace Order: 2", "Laplace Order: 4"]):
    print(name)
    tracker = debugging.TrackError(0, "l_inf")
    for resolution_exp in range(15):
        num_points = 5 * (2 ** resolution_exp)
        L = 1
        axis = np.linspace(0, L, num_points + 1)[:-1] - np.e
        dx = L * 1.0 / num_points

        factor = np.pi * 2 * 10 # * 500
        signal = np.sin(axis * factor)

        calculated_result = op(signal, dx)
        accurate_result = - (factor ** 2) * np.sin(axis * factor)
        #accurate_result = factor * np.cos(axis * factor)

        tracker.add_entry(num_points, accurate_result, calculated_result)
        print(str(num_points) + "\t" + str(tracker.abs_error[-1]) + "\t" + str(1 / dx ** 2))
    plt.loglog(tracker.timestamps, tracker.abs_error, label=name)
plt.xlabel("Number of Samples")
plt.ylabel("Error")

draw_oder_line(10, 1e-6, 1, 1)
draw_oder_line(10, 1e-6, 1, 2)
draw_oder_line(10, 1e-6, 1, 4)
plt.grid()
plt.legend()
plt.show()
"""plt.cla()
        plt.plot(axis, accurate_result - calculated_result, label="diff")
        plt.legend()
        plt.draw()
        plt.pause(0.002)"""

# ind = np.argmax(np.abs(accurate_result - calculated_result))
