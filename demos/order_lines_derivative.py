import numpy as np
from math import sqrt
from matplotlib import pyplot as plt

from debug_tools import error_tracking_tools
from debug_tools.visualization import WindowManager
from operators.derivative import *

window_manager = WindowManager(1, 1)

for op, name, ls in zip([diff_backward_n1_e1, diff_n1_e2, diff_n1_e4],
                        ["Derivative Order: 1", "Derivative Order: 2", "Derivative Order: 4"],
                        ["-", "--", ":"]):
    print(name)
    tracker = error_tracking_tools.ErrorTracker(num_points=None, x_label="Number of Samples", norm="l_inf")
    for resolution_exp in range(40):
        num_points = int(5 * (1.5 ** resolution_exp))
        L = 1

        # shifting axis by irrational number e to avoid 'random' perfect results at low resolutions
        axis = np.linspace(0, L, num_points + 1)[:-1] - np.e
        dx = L * 1.0 / num_points

        factor = np.pi * 2 * 10  # 2 pi to have a periodic signal, other factors to increase the frequency
        signal = np.sin(axis * factor)

        calculated_result = op(signal, dx)
        accurate_result = factor * np.cos(axis * factor)

        tracker.add_entry(num_points, accurate_result, calculated_result)
        print(str(num_points) + "\t" + str(tracker.abs_error[-1]) + "\t" + str(1 / dx ** 2))

    window_manager.display_error(1, tracker, double_log=True, line_name=name, clear_axis=False, linestyle=ls)

window_manager.draw_loglog_oder_line(1, 1e4, 1e+0, 1, 1)
window_manager.draw_loglog_oder_line(1, 1e4, 1e-2, 1, 2)
window_manager.draw_loglog_oder_line(1, sqrt(10)*1e3, 1e-4, 1, 4)
plt.plot([1e7],[1e2],"")
plt.grid()
plt.show()
